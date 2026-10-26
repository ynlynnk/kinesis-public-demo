"""AI progress report for a KINESIS patient: is the patient improving or declining?

Two layers, kept apart on purpose:

* Ground truth (deterministic, computed here): the session history up to the selected
  session, the rehab timeframe, per-metric trends, a rule-based trajectory and the risk
  flags. These are always shown and the language model cannot change them.
* Reasoning (Model 2): Gemini or Claude is given *tools* (function calling) that return
  that ground truth on demand - overview, history, one session's detail, fatigue history,
  risk flags, approved knowledge - and writes the interpretation. Without a provider, or
  if the call fails, a local report is built from the same evidence.

Only relative labels reach the model: sessions are "S1..Sn" and dates are "day N of
rehab"; no patient name, folder or calendar date is sent.
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import re
from typing import Any, Callable

try:
    from .claude_summary import _extract_json, _model2_provider
    from .kinesis_session_rag import session_facts
    from .knowledge_retriever import retrieve_knowledge
except ImportError:
    from claude_summary import _extract_json, _model2_provider
    from kinesis_session_rag import session_facts
    from knowledge_retriever import retrieve_knowledge


SESSION_ID = re.compile(r"^se(\d{8})_(\d{6})$")
MAX_TOOL_ROUNDS = 8
REPORT_FILE = "ai_progress_report.json"

# Change that counts as a real move for each metric (smaller is "stable").
METRICS = {
    "good_rate_pct": {"label": "Good-repetition rate (%)", "step": 5.0, "better": +1},
    "avg_peak_deg": {"label": "Average peak angle (°)", "step": 3.0, "better": +1},
    "fatigue_score": {"label": "End-of-session fatigue score", "step": 5.0, "better": -1},
    "completion_pct": {"label": "Planned repetitions completed (%)", "step": 10.0, "better": +1},
    "muscle_pct": {"label": "Muscle effort in good reps (% of reference)", "step": 10.0, "better": 0},
}
TRAJECTORY_WEIGHTS = {"good_rate_pct": 2, "avg_peak_deg": 1, "fatigue_score": 1, "completion_pct": 1}
COMPENSATION = ("FOREARM_LIFT", "WRIST_TWIST", "SIDE_BEND")
RISK_LEVELS = {"high": 3, "medium": 2, "low": 1}


# ------------------------------------------------------------------ ground truth

def _finite(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _r(v: Any, nd: int = 1) -> float | None:
    return round(float(v), nd) if _finite(v) else None


def session_time(session_id: str) -> datetime | None:
    m = SESSION_ID.match(session_id)
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None


def _events(path: Path) -> dict[str, int]:
    counts = {"emergency_stops": 0, "disconnects": 0, "signal_pauses": 0}
    try:
        with open(path / "events.csv", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                arg = f"{row.get('event')}:{row.get('arg')}"
                if arg.startswith("session:pause:emergency"):
                    counts["emergency_stops"] += 1
                elif arg.startswith("session:pause:disconnect"):
                    counts["disconnects"] += 1
                elif arg.startswith("session:pause:signal"):
                    counts["signal_pauses"] += 1
    except OSError:
        pass
    return counts


def _session_row(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    totals = report.get("totals", {})
    session = report.get("session", {})
    fatigue = report.get("fatigue", {})
    reps = report.get("reps", [])
    attempts = totals.get("attempts") or 0
    counted = [r for r in reps if r.get("counted")]
    planned = session.get("planned", {})
    planned_reps = (planned.get("sets") or 0) * (planned.get("reps_per_set") or 0)
    faults = {f["code"]: f.get("reps", 0) for f in report.get("faults", [])}
    trend = [t.get("score") for t in fatigue.get("trend", []) if _finite(t.get("score"))]
    muscle = [r.get("muscle_pct_rvc") for r in counted if _finite(r.get("muscle_pct_rvc"))]
    return {
        "exercise": session.get("exercise"),
        "attempts": attempts,
        "good_reps": totals.get("good_reps") or 0,
        "good_rate_pct": _r(100 * totals["good_rate"]) if _finite(totals.get("good_rate")) else None,
        "avg_peak_deg": _r(totals.get("avg_peak_angle_deg")),
        "target_deg": session.get("target_deg"),
        "personal_rom_deg": session.get("personal_rom_deg"),
        "fatigue_score": _r(fatigue.get("final_score")) if fatigue.get("source") == "counted" else None,
        "fatigue_zone": fatigue.get("zone") if fatigue.get("source") == "counted" else None,
        "fatigue_source": fatigue.get("source"),
        "fatigue_within_session": (_r(trend[-1] - trend[0]) if len(trend) > 1 else None),
        "muscle_pct": _r(sum(muscle) / len(muscle)) if muscle else None,
        "completion_pct": _r(min(100.0, 100 * attempts / planned_reps)) if planned_reps and attempts else None,
        "fault_pct": {c: _r(100 * n / attempts) for c, n in faults.items()} if attempts else {},
        "top_fault": totals.get("most_common_fault"),
        **_events(path),
    }


def load_history(patient_dir: Path, upto_session: str) -> list[dict[str, Any]]:
    """Every session of the patient up to (and including) the selected one, oldest first."""

    rows = []
    for path in sorted(patient_dir.iterdir()):
        if not (path.is_dir() and SESSION_ID.match(path.name) and (path / "report.json").exists()):
            continue
        if path.name > upto_session:
            break
        try:
            report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        rows.append({"session_id": path.name, "when": session_time(path.name), "path": path,
                     "report": report, **_session_row(path, report)})
    first = next((r["when"] for r in rows if r["when"]), None)
    for i, row in enumerate(rows):
        row["label"] = f"S{i + 1}"
        row["day"] = (row["when"] - first).days if (row["when"] and first) else None
    return rows


def _direction(metric: str, change: float | None) -> str:
    spec = METRICS[metric]
    if change is None:
        return "not enough data"
    if abs(change) < spec["step"]:
        return "stable"
    if spec["better"] == 0:
        return "higher" if change > 0 else "lower"
    return "improving" if change * spec["better"] > 0 else "declining"


def _slope(points: list[tuple[int, float]]) -> float | None:
    """Least-squares change per session over the given (session index, value) points."""
    if len(points) < 2:
        return None
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    den = sum((x - mx) ** 2 for x, _ in points)
    return _r(sum((x - mx) * (y - my) for x, y in points) / den, 2) if den else None


def progress_overview(history: list[dict[str, Any]]) -> dict[str, Any]:
    """Timeframe, counts, per-metric trends for the selected session's exercise, and a
    rule-based trajectory (the reference the AI is compared against)."""

    current = history[-1]
    same = [h for h in history if h["exercise"] == current["exercise"] and h["attempts"]]
    previous = [h for h in history[:-1] if h["when"]]
    first_day, this_day = history[0].get("day"), current.get("day")
    span_days = (this_day - first_day) if (first_day is not None and this_day is not None) else None
    overview: dict[str, Any] = {
        "this_session": current["label"],
        "session_number": len(history),
        "sessions_to_date": len(history),
        "sessions_with_reps": sum(1 for h in history if h["attempts"]),
        "same_exercise_sessions": len(same),
        "exercise": current["exercise"],
        "rehab_day": this_day,
        "rehab_span_days": span_days,
        "rehab_span_weeks": _r(span_days / 7) if span_days is not None else None,
        "sessions_per_week": (
            _r(len(history) / max(span_days / 7, 1)) if span_days else None
        ),
        "days_since_previous": (
            (current["when"] - previous[-1]["when"]).days if previous and current["when"] else None
        ),
        "metrics": {},
    }
    score, counted = 0, 0
    for metric, spec in METRICS.items():
        series = [(i, h[metric]) for i, h in enumerate(same) if _finite(h.get(metric))]
        if not series:
            continue
        values = [v for _, v in series]
        cur = values[-1] if same and same[-1].get(metric) is not None else None
        prev_vals = values[:-1] if cur is not None else values
        baseline = sum(prev_vals[:2]) / len(prev_vals[:2]) if prev_vals else None
        recent = prev_vals[-3:]
        vs_recent = _r(cur - sum(recent) / len(recent)) if (cur is not None and recent) else None
        vs_first = _r(cur - baseline) if (cur is not None and baseline is not None) else None
        overview["metrics"][metric] = {
            "label": spec["label"],
            "by_session": [{"session": same[i]["label"], "day": same[i]["day"], "value": v} for i, v in series],
            "first": values[0],
            "current": cur,
            "change_vs_recent_sessions": vs_recent,
            "change_vs_start": vs_first,
            "slope_per_session": _slope(series[-5:]),
            "direction_recent": _direction(metric, vs_recent),
            "direction_since_start": _direction(metric, vs_first),
        }
        if metric in TRAJECTORY_WEIGHTS and vs_recent is not None:
            counted += 1
            d = overview["metrics"][metric]["direction_recent"]
            score += TRAJECTORY_WEIGHTS[metric] * (1 if d == "improving" else -1 if d == "declining" else 0)
    overview["rule_trajectory"] = (
        "insufficient_data" if len(same) < 2 or not counted
        else "improving" if score >= 2 else "declining" if score <= -2 else "stable"
    )
    overview["rule_trajectory_score"] = score
    return overview


def risk_flags(history: list[dict[str, Any]], overview: dict[str, Any]) -> dict[str, Any]:
    """Abnormal findings for the selected session, each with its evidence and level."""

    cur = history[-1]
    same = [h for h in history[:-1] if h["exercise"] == cur["exercise"] and h["attempts"]]
    flags: list[dict[str, Any]] = []

    def flag(code, level, title, evidence):
        flags.append({"code": code, "level": level, "title": title, "evidence": evidence})

    m = overview["metrics"]
    gr = m.get("good_rate_pct", {})
    if gr.get("change_vs_recent_sessions") is not None:
        drop = -gr["change_vs_recent_sessions"]
        if drop >= 20:
            flag("QUALITY_DROP", "high", "Sharp drop in movement quality",
                 f"Good-rep rate {gr['current']}%, {drop:.0f} points below the previous sessions.")
        elif drop >= 10:
            flag("QUALITY_DROP", "medium", "Movement quality falling",
                 f"Good-rep rate {gr['current']}%, {drop:.0f} points below the previous sessions.")
    ang = m.get("avg_peak_deg", {})
    best = max((h["avg_peak_deg"] for h in same if _finite(h.get("avg_peak_deg"))), default=None)
    if ang.get("current") is not None and best is not None:
        loss = best - ang["current"]
        if loss >= 10:
            flag("ROM_LOSS", "high", "Range of motion lost",
                 f"Average peak {ang['current']}°, {loss:.0f}° below this patient's best ({best}°).")
        elif loss >= 5:
            flag("ROM_LOSS", "medium", "Range of motion reduced",
                 f"Average peak {ang['current']}°, {loss:.0f}° below this patient's best ({best}°).")
    if cur.get("fatigue_zone") == "red":
        flag("FATIGUE_HIGH", "high", "High fatigue this session",
             f"Fatigue score {cur['fatigue_score']} (red zone, above 70).")
    elif cur.get("fatigue_zone") == "amber":
        flag("FATIGUE_HIGH", "medium", "Raised fatigue this session",
             f"Fatigue score {cur['fatigue_score']} (amber zone).")
    fat = [h["fatigue_score"] for h in same + [cur] if _finite(h.get("fatigue_score"))]
    if len(fat) >= 3 and fat[-3] < fat[-2] < fat[-1] and fat[-1] - fat[-3] >= 10:
        flag("FATIGUE_RISING", "medium", "Fatigue rising across sessions",
             f"End-of-session fatigue {fat[-3]} -> {fat[-2]} -> {fat[-1]} over the last three sessions.")
    if cur["emergency_stops"]:
        flag("EMERGENCY_STOP", "high", "Emergency stop pressed",
             f"The patient pressed the emergency stop {cur['emergency_stops']} time(s) in this session.")
    attempts = cur["attempts"]
    if attempts:
        fp = cur["fault_pct"]
        if (fp.get("NO_MUSCLE") or 0) >= 30:
            flag("LOW_MUSCLE_USE", "medium", "Target muscle often not doing the work",
                 f"{fp['NO_MUSCLE']}% of attempts failed the muscle check (momentum or assisted lifts).")
        comp = sum(fp.get(c) or 0 for c in COMPENSATION)
        if comp >= 30:
            flag("COMPENSATION", "medium", "Frequent compensation",
                 f"{comp:.0f}% of attempts used other joints (forearm lift / twist / side bend).")
        if (fp.get("OVER_FLEX") or 0) > 0 and cur["good_reps"] < attempts:
            n = round(fp["OVER_FLEX"] * attempts / 100)
            if n >= 2:
                flag("OVER_FLEX", "medium", "Repeatedly moving past the safe range",
                     f"{n} repetitions went beyond the target or personal range.")
        if (fp.get("SIGNAL_POOR") or 0) >= 20:
            flag("DATA_QUALITY", "low", "Unreliable sensor data",
                 f"{fp['SIGNAL_POOR']}% of attempts had poor signal; muscle findings are less certain.")
    if cur.get("completion_pct") is not None and cur["completion_pct"] < 50:
        flag("SESSION_CUT_SHORT", "medium", "Session cut short",
             f"Only {cur['completion_pct']}% of the planned repetitions were attempted.")
    if not attempts:
        flag("NO_REPETITIONS", "low", "No repetitions recorded",
             "The session ended before any repetition was recorded.")
    if (overview.get("days_since_previous") or 0) >= 7:
        flag("ADHERENCE_GAP", "low", "Long gap since the previous session",
             f"{overview['days_since_previous']} days since the previous session.")
    highs = sum(f["level"] == "high" for f in flags)
    mediums = sum(f["level"] == "medium" for f in flags)
    tag = (
        "HIGH RISK" if highs else "AT RISK" if mediums >= 2 else "MONITOR" if mediums or flags
        else "NO FLAGS"
    )
    flags.sort(key=lambda f: -RISK_LEVELS[f["level"]])
    return {"risk_tag": tag, "at_risk": tag in ("HIGH RISK", "AT RISK"), "flags": flags,
            "rule": "HIGH RISK = any high flag; AT RISK = two or more medium flags; "
                    "MONITOR = other flags."}


def ground_truth(patient_dir: Path, session_id: str) -> dict[str, Any]:
    history = load_history(patient_dir, session_id)
    if not history or history[-1]["session_id"] != session_id:
        raise FileNotFoundError(f"Session {session_id} not found in {patient_dir}")
    overview = progress_overview(history)
    return {"history": history, "overview": overview, "risk": risk_flags(history, overview)}


# ------------------------------------------------------------------ tools (function calling)

def _public_row(h: dict[str, Any]) -> dict[str, Any]:
    keep = ("label", "day", "exercise", "attempts", "good_reps", "good_rate_pct", "avg_peak_deg",
            "target_deg", "fatigue_score", "fatigue_zone", "fatigue_within_session", "muscle_pct",
            "completion_pct", "top_fault", "emergency_stops", "disconnects")
    return {k: h.get(k) for k in keep}


TOOL_SPECS = [
    {"name": "get_progress_overview",
     "description": "Rehab timeframe (session number, days since the first session, sessions per week), "
                    "per-metric trends for this exercise and the rule-based trajectory.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "list_session_history",
     "description": "One row per session up to and including the current one (S1..Sn, rehab day, reps, "
                    "good-rep rate, peak angle, fatigue, completion, top fault, emergency stops).",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_session_detail",
     "description": "Detailed facts of one session: rep timing, muscle effort, faults, first vs second "
                    "half quality, fatigue and the rules in force.",
     "parameters": {"type": "object", "properties": {
         "session": {"type": "string", "description": "Session label, e.g. 'S3'."}},
         "required": ["session"]}},
    {"name": "get_fatigue_history",
     "description": "Fatigue per session: end score, zone, rise within the session and the last rep's "
                    "change from baseline (MDF, RMS, lift time) for the current session.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_risk_flags",
     "description": "Rule-based abnormal findings for the current session with evidence and level, "
                    "and the patient's risk tag.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "search_knowledge",
     "description": "Search the approved rehabilitation knowledge library; returns sections to cite.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What to look up."}},
         "required": ["query"]}},
]


def make_tools(truth: dict[str, Any]) -> dict[str, Callable[..., Any]]:
    history = truth["history"]
    by_label = {h["label"]: h for h in history}
    cur = history[-1]

    def overview():
        return truth["overview"]

    def history_rows():
        return {"sessions": [_public_row(h) for h in history]}

    def detail(session: str = ""):
        h = by_label.get(str(session).strip().upper())
        if h is None:
            return {"error": f"Unknown session {session!r}; use S1..S{len(history)}."}
        facts = session_facts(h["report"])
        facts.pop("rules", None)
        return {"session": h["label"], "day": h["day"], **facts,
                "emergency_stops": h["emergency_stops"], "disconnects": h["disconnects"]}

    def fatigue():
        return {
            "zones": cur["report"].get("fatigue", {}).get("zones"),
            "by_session": [
                {"session": h["label"], "day": h["day"], "end_score": h["fatigue_score"],
                 "zone": h["fatigue_zone"], "rise_within_session": h["fatigue_within_session"],
                 "source": h["fatigue_source"]}
                for h in history
            ],
            "current_last_rep_change_from_baseline_pct":
                cur["report"].get("fatigue", {}).get("last_rep_changes_pct"),
            "note": "Scores from sessions with too few counted reps (source 'demo') are omitted.",
        }

    def flags():
        return truth["risk"]

    def knowledge(query: str = ""):
        try:
            hits = retrieve_knowledge(query=str(query) or "rehabilitation progress", top_k=3)
        except (FileNotFoundError, RuntimeError, ValueError):
            hits = []
        return {"results": [
            {"id": f"{h['document_title']} - {h['section_title']}", "text": str(h["text"])[:900]}
            for h in hits
        ]}

    return {"get_progress_overview": overview, "list_session_history": history_rows,
            "get_session_detail": detail, "get_fatigue_history": fatigue,
            "get_risk_flags": flags, "search_knowledge": knowledge}


SYSTEM_PROMPT = """You are KINESIS Model 2, writing a progress report for a physiotherapist about one
patient's home rehabilitation, recorded by a wearable motion + muscle (sEMG) sensor.
Use the tools to gather evidence before answering: always call get_progress_overview,
get_risk_flags and get_fatigue_history; call list_session_history, get_session_detail and
search_knowledge when they help. Then reason whether the patient is improving, stable or
declining in function, using movement quality, range of motion, completion and fatigue over
time, and how this session fits the timeframe of the rehab so far.
Rules: quote numbers exactly as the tools return them; never invent data; keep the rule-based
risk tag and flags unchanged (you may add your own concerns separately); no diagnosis or
prescription; refer to sessions as S1..Sn and time as rehab days.
When done, reply with ONLY this JSON:
{"trajectory": "improving|stable|declining|insufficient_data", "confidence": "low|medium|high",
 "headline": str, "reasoning": str (at most 200 words),
 "metric_progress": [{"metric": str, "direction": str, "evidence": str}],
 "fatigue_assessment": str, "risk_comment": str, "additional_concerns": [str],
 "suggested_review_actions": [str], "citations": [str]}"""


# ------------------------------------------------------------------ provider loops

REPORT_PROMPT = "Write the progress report for the current session."


def _run_claude(api_key: str, model: str, tools: dict, call_log: list, system: str = SYSTEM_PROMPT,
                tool_specs: list = TOOL_SPECS, prompt: str = REPORT_PROMPT, turns: list = ()) -> str:
    """Tool loop; `turns` are earlier (question, answer) pairs of a conversation."""
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key, timeout=60.0, max_retries=1)
    specs = [{"name": t["name"], "description": t["description"], "input_schema": t["parameters"]}
             for t in tool_specs]
    messages: list[dict[str, Any]] = []
    for q, a in turns:
        messages += [{"role": "user", "content": q}, {"role": "assistant", "content": a}]
    messages.append({"role": "user", "content": prompt})
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(model=model, max_tokens=1800, system=system,
                                      tools=specs, messages=messages)
        uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
        if not uses:
            return "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        messages.append({"role": "assistant", "content": resp.content})
        results = []
        for b in uses:
            out = _call_tool(tools, b.name, dict(b.input or {}), call_log)
            results.append({"type": "tool_result", "tool_use_id": b.id, "content": json.dumps(out, default=str)})
        messages.append({"role": "user", "content": results})
    raise RuntimeError("The model kept calling tools without writing the report.")


def _run_gemini(api_key: str, model: str, tools: dict, call_log: list, system: str = SYSTEM_PROMPT,
                tool_specs: list = TOOL_SPECS, prompt: str = REPORT_PROMPT, turns: list = ()) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(name=t["name"], description=t["description"],
                                      parameters_json_schema=t["parameters"])
            for t in tool_specs
        ])],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        max_output_tokens=1800,
    )
    contents: list[Any] = []
    for q, a in turns:
        contents += [types.Content(role="user", parts=[types.Part.from_text(text=q)]),
                     types.Content(role="model", parts=[types.Part.from_text(text=a)])]
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=prompt)]))
    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.models.generate_content(model=model, contents=contents, config=config)
        calls = resp.function_calls or []
        if not calls:
            return str(resp.text or "")
        contents.append(resp.candidates[0].content)
        parts = [
            types.Part.from_function_response(
                name=c.name, response={"result": _call_tool(tools, c.name, dict(c.args or {}), call_log)})
            for c in calls
        ]
        contents.append(types.Content(role="user", parts=parts))
    raise RuntimeError("The model kept calling tools without writing the report.")


def _call_tool(tools: dict, name: str, args: dict, call_log: list) -> Any:
    fn = tools.get(name)
    call_log.append({"tool": name, "args": args})
    if fn is None:
        return {"error": f"Unknown tool {name}"}
    try:
        return json.loads(json.dumps(fn(**args), default=str))
    except TypeError as exc:
        return {"error": f"Bad arguments for {name}: {exc}"}


# ------------------------------------------------------------------ local fallback

def local_report(truth: dict[str, Any]) -> dict[str, Any]:
    ov, risk = truth["overview"], truth["risk"]
    traj = ov["rule_trajectory"]
    rows = []
    for metric, data in ov["metrics"].items():
        rows.append({
            "metric": data["label"],
            "direction": data["direction_recent"],
            "evidence": (
                f"{data['first']} at the start -> {data['current']} now "
                f"(vs recent sessions: {data['change_vs_recent_sessions']:+})"
                if data["change_vs_recent_sessions"] is not None
                else f"{data['current']} (first session of this exercise)"
            ),
        })
    fat = ov["metrics"].get("fatigue_score", {})
    span = ov.get("rehab_span_days")
    when = f"day {ov['rehab_day']} of rehab" if ov.get("rehab_day") is not None else "this session"
    headline = {
        "improving": "Function is improving across sessions.",
        "declining": "Function is declining compared with recent sessions.",
        "stable": "Function is stable compared with recent sessions.",
        "insufficient_data": "Not enough comparable sessions yet to judge a trend.",
    }[traj]
    def named(direction: str, key: str = "direction_recent") -> str:
        """Metrics moving in `direction`, with the values compared (recent average or start)."""
        diff = "change_vs_recent_sessions" if key == "direction_recent" else "change_vs_start"
        return ", ".join(
            f"{d['label'].split(' (')[0].lower()} ({_r(d['current'] - d[diff])} -> {d['current']})"
            for d in ov["metrics"].values() if d[key] == direction and d.get(diff) is not None
        )

    parts = [
        f"Session {ov['session_number']} ({ov['this_session']}) on {when}"
        + (f", {span} days ({ov['rehab_span_weeks']} weeks) after the first session at "
           f"{ov['sessions_per_week']} sessions per week" if span else "")
        + f"; {ov['same_exercise_sessions']} session(s) of {ov['exercise']} with repetitions so far."
    ]
    if traj != "insufficient_data":
        for direction, text in (("improving", "Better than the average of the last sessions"),
                                ("declining", "Worse than the average of the last sessions")):
            if named(direction):
                parts.append(f"{text}: {named(direction)}.")
        if named("improving", "direction_since_start"):
            parts.append(f"Better than at the start: {named('improving', 'direction_since_start')}.")
    parts.append(
        f"Rule-based trajectory score {ov['rule_trajectory_score']} (good-rep rate counts double; "
        "+2 or more = improving, -2 or less = declining)."
    )
    reasoning = " ".join(parts)
    return {
        "trajectory": traj,
        "confidence": "low" if traj == "insufficient_data" or ov["same_exercise_sessions"] < 4 else "medium",
        "headline": headline,
        "reasoning": reasoning,
        "metric_progress": rows,
        "fatigue_assessment": (
            f"End-of-session fatigue {fat.get('first')} at the start -> {fat.get('current')} now "
            f"({fat.get('direction_since_start')})." if fat else "No counted fatigue score yet."
        ),
        "risk_comment": (
            f"Risk tag {risk['risk_tag']}: " + "; ".join(f["title"] for f in risk["flags"])
            if risk["flags"] else "No abnormal flags in this session."
        ),
        "additional_concerns": [],
        "suggested_review_actions": [f"Review: {f['title'].lower()} ({f['evidence']})"
                                     for f in risk["flags"] if f["level"] != "low"][:4],
        "citations": [],
    }


# ------------------------------------------------------------------ entry point

TRAJECTORIES = {"improving", "stable", "declining", "insufficient_data"}


def generate_progress_report(patient_dir: Path, session_id: str) -> dict[str, Any]:
    """Ground truth + the AI's reasoning over it (via tools), or a local report."""

    truth = ground_truth(Path(patient_dir), session_id)
    tools = make_tools(truth)
    provider, api_key, model = _model2_provider()
    call_log: list[dict[str, Any]] = []
    mode, error, ai = "Local rule-based report", None, None
    if api_key and provider in ("gemini", "claude"):
        try:
            text = (_run_gemini if provider == "gemini" else _run_claude)(api_key, model, tools, call_log)
            ai = _extract_json(text)
            if ai.get("trajectory") not in TRAJECTORIES or not str(ai.get("reasoning", "")).strip():
                raise ValueError("The model's report is incomplete.")
            mode = f"AI reasoning ({provider}: {model}, {len(call_log)} tool calls)"
        except Exception as exc:     # the panel must still show the ground truth
            error, ai = str(exc), None
    if ai is None:
        ai = local_report(truth)
    ov = truth["overview"]
    timeline = [
        {"label": h["label"], "session_id": h["session_id"], "day": h["day"],
         "date": h["when"].strftime("%Y-%m-%d") if h["when"] else None, **_public_row(h)}
        for h in truth["history"]
    ]
    return {
        "session_id": session_id,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "generation_mode": mode,
        "provider_error": error,
        "tool_calls": call_log,
        "ground_truth": {"overview": ov, "risk": truth["risk"], "timeline": timeline},
        "ai": ai,
        "safety_note": "Decision support from sensor data, not a diagnosis. The risk tag is rule-based; "
                       "clinical judgement remains with the physiotherapist.",
    }


def save_report(session_dir: Path, report: dict[str, Any]) -> Path:
    path = Path(session_dir) / REPORT_FILE
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


def load_report(session_dir: Path) -> dict[str, Any] | None:
    try:
        return json.loads((Path(session_dir) / REPORT_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
