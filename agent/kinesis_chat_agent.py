"""Chat agent for the specialist dashboard with memory of the patient's whole record.

The specialist chats while looking at one session, but the agent can reach every session
the patient has sent: it gets the patient's full history (S1..Sn) as tools, plus the
earlier turns of the conversation. It looks up what it needs - any session's detail, a
comparison of sessions, how often a fault has occurred, the progress and risk flags as
of any session - and answers with the sessions and knowledge it used.

Same rules as the progress report: numbers come from the tools (computed in
kinesis_progress_agent), only relative labels reach the model (S1..Sn, rehab day N; no
name, folder or calendar date), and without a provider a local evidence answer is built
from the same tools.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

try:
    from . import kinesis_progress_agent as pa
    from .claude_summary import _extract_json, _model2_provider
    from .kinesis_session_rag import SAFETY_NOTE, retrieve, session_facts
except ImportError:
    import kinesis_progress_agent as pa
    from claude_summary import _extract_json, _model2_provider
    from kinesis_session_rag import SAFETY_NOTE, retrieve, session_facts

MAX_TURNS = 6   # earlier turns of the conversation sent with each question

CHAT_TOOL_SPECS = [
    {"name": "list_session_history",
     "description": "Every session the patient has sent (S1..Sn, oldest first): rehab day, exercise, "
                    "reps, good-rep rate, peak angle, fatigue, completion, top fault, emergency stops.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "get_session_detail",
     "description": "Detailed facts of one session: rep timing, muscle effort, faults, first vs second "
                    "half quality, fatigue and signal problems.",
     "parameters": {"type": "object", "properties": {
         "session": {"type": "string", "description": "Session label, e.g. 'S3'."}},
         "required": ["session"]}},
    {"name": "compare_sessions",
     "description": "Side-by-side key numbers of two or more sessions.",
     "parameters": {"type": "object", "properties": {
         "sessions": {"type": "array", "items": {"type": "string"},
                      "description": "Session labels, e.g. ['S2', 'S5']."}},
         "required": ["sessions"]}},
    {"name": "get_progress_overview",
     "description": "Rehab timeframe, per-metric trends and the rule-based trajectory as of a session "
                    "(default: the session being viewed).",
     "parameters": {"type": "object", "properties": {
         "session": {"type": "string", "description": "Session label; omit for the viewed one."}}}},
    {"name": "get_risk_flags",
     "description": "Rule-based abnormal findings and risk tag as of a session (default: the viewed one).",
     "parameters": {"type": "object", "properties": {
         "session": {"type": "string", "description": "Session label; omit for the viewed one."}}}},
    {"name": "get_fatigue_history",
     "description": "Fatigue per session across the whole record: end score, zone, rise within session.",
     "parameters": {"type": "object", "properties": {}}},
    {"name": "find_fault",
     "description": "Which sessions had a given repetition fault, and in what share of attempts.",
     "parameters": {"type": "object", "properties": {
         "code": {"type": "string", "description": "Fault code, e.g. 'EARLY_DOWN', 'FOREARM_LIFT'."}},
         "required": ["code"]}},
    {"name": "search_knowledge",
     "description": "Search the approved rehabilitation knowledge library; returns sections to cite.",
     "parameters": {"type": "object", "properties": {
         "query": {"type": "string", "description": "What to look up."}},
         "required": ["query"]}},
]

CHAT_SYSTEM_PROMPT = """You are KINESIS Model 2, a chat assistant for a physiotherapist evaluating one
patient's home rehabilitation (wearable motion + muscle/sEMG sensor). You can see the
patient's whole record through tools: every session S1..Sn, oldest first. The message says
which session the physiotherapist is viewing and gives retrieved knowledge sections [K1..].
Use the tools to look up any session, compare sessions, trace faults or fatigue over time,
and get progress or risk flags as of any session. Earlier turns of this chat are included:
use them for follow-up questions.
You are a reference for the physiotherapist, who makes every diagnosis and clinical decision.
Rules: quote numbers exactly as the tools return them; never invent data; keep rule-based
risk tags unchanged; never diagnose, name a condition or prescribe treatment - point to what
the physiotherapist could check instead; say plainly when the evidence is thin (few sessions,
few repetitions, signal problems) or when you are unsure; refer to sessions as S1..Sn and
time as rehab days.
Style: clinical and compact. Answer the question directly, lead with the numbers, no filler,
no restating the question, no inline citation tags (list sources in the fields below).
When done, reply with ONLY this JSON:
{"answer": "1-2 sentences that answer the question, with the key numbers and session labels",
 "findings": [{"label": "2-3 word topic", "detail": "one line of evidence, e.g. 'S1 50.0% -> S5 91.7% -> S6 35.7%'"}],
 "table_sessions": ["S1", "..."],
 "check": "one thing the physiotherapist could check, or empty",
 "sessions_used": ["S1", "..."], "knowledge_ids_used": ["K1", "..."]}
`findings`: at most 4. `table_sessions`: up to 6 sessions whose measured numbers should be shown
side by side (the dashboard fills the table from the record), or [] when a table does not help."""

MAX_FINDINGS = 4
MAX_TABLE_ROWS = 6


def make_chat_tools(patient_dir: Path, history: list[dict[str, Any]], viewed: str) -> dict:
    """Tools over the patient's full history; `viewed` is the label of the session on screen."""

    base = pa.make_tools({"history": history, "overview": {}, "risk": {}})
    by_label = {h["label"]: h for h in history}
    as_of: dict[str, dict] = {}

    def truth_at(label: str | None) -> dict | str:
        h = by_label.get(str(label or viewed).strip().upper())
        if h is None:
            return f"Unknown session {label!r}; use S1..S{len(history)}."
        if h["label"] not in as_of:
            as_of[h["label"]] = pa.ground_truth(patient_dir, h["session_id"])
        return as_of[h["label"]]

    def overview(session: str = ""):
        t = truth_at(session or None)
        return t["overview"] if isinstance(t, dict) else {"error": t}

    def flags(session: str = ""):
        t = truth_at(session or None)
        return t["risk"] if isinstance(t, dict) else {"error": t}

    def compare(sessions: list | None = None):
        rows = []
        for s in sessions or []:
            h = by_label.get(str(s).strip().upper())
            rows.append(pa._public_row(h) if h else {"label": s, "error": "unknown session"})
        return {"sessions": rows}

    def find_fault(code: str = ""):
        code = str(code).strip().upper()
        hits = [{"session": h["label"], "day": h["day"], "exercise": h["exercise"],
                 "pct_of_attempts": h["fault_pct"][code], "attempts": h["attempts"]}
                for h in history if code in h.get("fault_pct", {})]
        return {"code": code, "sessions_with_fault": hits,
                "sessions_without": [h["label"] for h in history if h["attempts"] and code not in h["fault_pct"]]}

    return {"list_session_history": base["list_session_history"], "get_session_detail": base["get_session_detail"],
            "compare_sessions": compare, "get_progress_overview": overview, "get_risk_flags": flags,
            "get_fatigue_history": base["get_fatigue_history"], "find_fault": find_fault,
            "search_knowledge": base["search_knowledge"]}


def _mentioned(question: str, history: list[dict[str, Any]]) -> list[str]:
    labels = {h["label"] for h in history}
    return [s for s in dict.fromkeys(re.findall(r"\bS\d+\b", question.upper())) if s in labels]


def _series(rows: list[dict[str, Any]], key: str, unit: str = "") -> str:
    """'S1 50.0% → S2 58.3% → ...' (first and last four when long)."""

    vals = [f"{h['label']} {h[key]}{unit}" for h in rows if h.get(key) is not None]
    if len(vals) > 5:
        vals = vals[:1] + ["…"] + vals[-4:]
    return " → ".join(vals) if vals else "no data"


def session_table(history: list[dict[str, Any]], labels: list[str], viewed: str) -> list[dict[str, Any]]:
    """Measured numbers of the chosen sessions, oldest first, straight from the record."""

    wanted = set(labels)
    rows = [h for h in history if h["label"] in wanted]
    many = len({h["exercise"] for h in rows}) > 1
    return [{
        "Session": h["label"] + (" (viewed)" if h["label"] == viewed else ""),
        "Day": h["day"],
        **({"Exercise": h["exercise"]} if many else {}),
        "Good": f"{h['good_reps']}/{h['attempts']}"
                + (f" ({h['good_rate_pct']:.0f}%)" if h["good_rate_pct"] is not None else ""),
        "Peak°": h["avg_peak_deg"],
        "Fatigue": h["fatigue_score"],
        "Fault": h["top_fault"] or "-",
    } for h in rows]


def _local_answer(question: str, history: list[dict[str, Any]], viewed: dict[str, Any], tools: dict,
                  knowledge: list[dict[str, Any]]) -> dict[str, Any]:
    """Evidence answer without a language model, drawn from the same tools."""

    q = question.casefold()
    ov, risk = tools["get_progress_overview"](), tools["get_risk_flags"]()
    v, label = viewed, viewed["label"]
    rated = [h for h in history if h["attempts"] and h["good_rate_pct"] is not None]
    same = [h for h in rated if h["exercise"] == v["exercise"]]
    trend = str(ov.get("rule_trajectory", "n/a")).replace("_", " ")

    if v["attempts"]:
        lead = (f"{label} (day {v['day']}, {v['exercise']}): {v['good_reps']}/{v['attempts']} good reps "
                f"({v['good_rate_pct']}%)")
        prev = [h for h in same if h["session_id"] < v["session_id"]][-3:]
        if prev:
            avg = round(sum(h["good_rate_pct"] for h in prev) / len(prev), 1)
            lead += f", against {avg}% over the previous {len(prev)} session(s)"
    else:
        lead = f"{label} (day {v['day']}, {v['exercise']}) has no repetitions recorded"
    answer = f"{lead}. Trend: {trend} · risk tag: {risk.get('risk_tag')}."

    asks_fatigue = any(w in q for w in ("fatigue", "tired", "effort"))
    asks_angle = any(w in q for w in ("angle", "range", "rom", "peak"))
    asks_fault = any(w in q for w in ("fault", "not counted", "why", "technique", "wrong"))
    asks_compare = any(w in q for w in ("best", "worst", "compare"))
    codes = sorted({c for h in history for c in h.get("fault_pct", {})} | {
        c for c in re.findall(r"\b[A-Z]+_[A-Z_]+\b", question.upper())})
    named_codes = [c for c in codes if c.casefold() in q or c.replace("_", " ").casefold() in q]

    findings = [{"label": "Good-rep rate", "detail": _series(same, "good_rate_pct", "%")}]
    for code in named_codes:
        found = tools["find_fault"](code)["sessions_with_fault"]
        findings.append({"label": code, "detail": " · ".join(
            f"{f['session']} {f['pct_of_attempts']}%" for f in found) + " of attempts" if found
            else "not seen in any session"})
    if asks_fault and not named_codes and v["top_fault"]:
        found = tools["find_fault"](v["top_fault"])["sessions_with_fault"]
        findings.append({"label": f"Top fault {v['top_fault']}", "detail": " · ".join(
            f"{f['session']} {f['pct_of_attempts']}%" for f in found) + " of attempts"})
    if asks_compare and same:
        best = max(same, key=lambda h: h["good_rate_pct"])
        worst = min(same, key=lambda h: h["good_rate_pct"])
        findings.append({"label": "Best / worst", "detail": f"best {best['label']} {best['good_rate_pct']}% · "
                                                            f"worst {worst['label']} {worst['good_rate_pct']}%"})
    general = not (asks_fault or asks_compare or named_codes or asks_fatigue or asks_angle)
    if asks_fatigue or general:
        findings.append({"label": "Fatigue (end)", "detail": _series(same, "fatigue_score")})
    if asks_angle or general:
        findings.append({"label": "Peak angle", "detail": _series(same, "avg_peak_deg", "°")
                         + (f" (target {v['target_deg']}°)" if v.get("target_deg") else "")})
    if risk.get("flags"):
        findings.append({"label": "Risk flags", "detail": "; ".join(f["title"] for f in risk["flags"])})

    labels = [label, *_mentioned(question, history)]
    if asks_compare and same:
        labels += [best["label"], worst["label"]]
    if len(set(labels)) < 2:     # nothing specific: the most recent sessions of this exercise
        labels += [h["label"] for h in same if h["session_id"] <= v["session_id"]][-MAX_TABLE_ROWS:]
    labels = list(dict.fromkeys(labels))[:MAX_TABLE_ROWS]
    concern = next((f for f in risk.get("flags", []) if f["level"] != "low"), None)
    return {"answer": answer, "findings": findings[:MAX_FINDINGS], "table_sessions": labels,
            "check": f"{concern['title']}: {concern['evidence']}" if concern else "",
            "sessions_used": sorted(set(labels) | {h["label"] for h in same}, key=lambda s: int(s[1:])),
            "knowledge_ids_used": [k["id"] for k in knowledge[:2]]}


NUMBER = re.compile(r"(?<![A-Za-z\d.])-?\d+(?:\.\d+)?")


def unverified_numbers(reply: dict[str, Any], evidence: Any) -> list[str]:
    """Numbers in the model's reply that appear nowhere in the evidence it was given.

    A safeguard, not a proof: a number that matches can still be used wrongly. Session
    labels (S3) are skipped, and so are small whole numbers (counts like "3 sessions");
    a value may match after rounding (91.7 written as 92)."""

    known = [float(n) for n in NUMBER.findall(json.dumps(evidence, default=str))]
    text = " ".join([reply["answer"], reply["check"], *(f"{f['label']} {f['detail']}" for f in reply["findings"])])
    missing = []
    for raw in dict.fromkeys(NUMBER.findall(text)):
        value = float(raw)
        if "." not in raw and abs(value) <= 12:
            continue
        tolerance = 0.051 if "." in raw else 0.51
        if not any(abs(value - k) <= tolerance for k in known):
            missing.append(raw)
    return missing


def _shape(result: dict[str, Any], history: list[dict[str, Any]], viewed: str) -> dict[str, Any]:
    """Keeps the reply to the compact format, whatever the model returned."""

    labels = {h["label"] for h in history}
    findings = []
    for f in result.get("findings") or []:
        if isinstance(f, dict) and str(f.get("detail", "")).strip():
            findings.append({"label": str(f.get("label", "")).strip(), "detail": str(f["detail"]).strip()})
        elif isinstance(f, str) and f.strip():
            findings.append({"label": "", "detail": f.strip()})
    if not findings:   # older format
        findings = [{"label": "", "detail": str(p)} for p in result.get("key_points", []) if str(p).strip()]
    table = [s for s in dict.fromkeys(str(s).strip().upper() for s in result.get("table_sessions") or [])
             if s in labels][:MAX_TABLE_ROWS]
    return {
        "answer": str(result.get("answer", "")).strip(),
        "findings": findings[:MAX_FINDINGS],
        "table": session_table(history, table, viewed) if table else [],
        "check": str(result.get("check") or "").strip(),
        "sessions_used": [s for s in dict.fromkeys(str(s).strip().upper() for s in result.get("sessions_used", []))
                          if s in labels],
        "knowledge_ids_used": [str(k) for k in result.get("knowledge_ids_used", [])],
    }


def chat_answer(patient_dir: Path, session_id: str, question: str,
                conversation: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Answer a specialist's chat question with the patient's whole record available."""

    patient_dir = Path(patient_dir)
    question = " ".join(question.split()) or "Summarise this patient's progress."
    sessions = sorted(p.name for p in patient_dir.iterdir() if pa.SESSION_ID.match(p.name))
    history = pa.load_history(patient_dir, sessions[-1]) if sessions else []
    viewed = next((h for h in history if h["session_id"] == session_id), None)
    if viewed is None:
        raise FileNotFoundError(f"Session {session_id} not found in {patient_dir}")
    tools = make_chat_tools(patient_dir, history, viewed["label"])
    looked_up: list[Any] = []   # everything the tools returned: the evidence a reply's numbers must come from

    def recorded(fn):
        def run(*args, **kw):
            out = fn(*args, **kw)
            looked_up.append(out)
            return out
        return run

    tools = {name: recorded(fn) for name, fn in tools.items()}
    earlier = [h for h in history if h["session_id"] < session_id]
    facts = session_facts(viewed["report"], [
        {"good_rate_percent": h["good_rate_pct"], "avg_peak_angle_deg": h["avg_peak_deg"],
         "fatigue_score": h["fatigue_score"]} for h in earlier])
    turns = [(t["question"], "\n".join([t["answer"], *t.get("key_points", [])]))
             for t in (conversation or [])[-MAX_TURNS:]]
    # A short follow-up ("why?") retrieves poorly on its own; borrow the previous question's words.
    knowledge = retrieve(f"{turns[-1][0]} {question}" if turns and len(question) < 40 else question, facts)
    prompt = (
        f"The physiotherapist is viewing {viewed['label']} (rehab day {viewed['day']}, {viewed['exercise']}) "
        f"of {len(history)} sessions (S1..S{len(history)}).\n"
        + "Retrieved knowledge:\n"
        + "\n".join(f"[{k['id']}] {k['document_title']} - {k['section_title']}: {str(k['text'])[:900]}"
                    for k in knowledge)
        + f"\n\nQuestion: {question}"
    )
    provider, api_key, model = _model2_provider()
    call_log: list[dict[str, Any]] = []
    mode, error, result = "Local evidence summary", None, None
    if api_key and provider in ("gemini", "claude"):
        try:
            run = pa._run_gemini if provider == "gemini" else pa._run_claude
            result = _extract_json(run(api_key, model, tools, call_log, system=CHAT_SYSTEM_PROMPT,
                                       tool_specs=CHAT_TOOL_SPECS, prompt=prompt, turns=turns))
            if not str(result.get("answer", "")).strip():
                raise ValueError("empty answer")
            mode = f"AI agent ({provider}: {model}, {len(call_log)} tool calls)"
        except Exception as exc:   # the chat must keep working without the provider
            error, result = str(exc), None
    if result is None:
        call_log = []
        result = _local_answer(question, history, viewed, tools, knowledge)
    reply = _shape(result, history, viewed["label"])
    evidence = [looked_up, facts, [pa._public_row(h) for h in history], prompt]
    return {
        "question": question,
        **reply,
        "unverified_numbers": unverified_numbers(reply, evidence) if call_log else [],
        # Plain-text form of the findings: the memory sent with follow-up questions.
        "key_points": [f"{f['label']}: {f['detail']}" if f["label"] else f["detail"] for f in reply["findings"]]
                      + ([f"Check: {reply['check']}"] if reply["check"] else []),
        "knowledge": knowledge,
        "facts": facts,
        "tool_calls": call_log,
        "session_labels": {h["label"]: h["session_id"] for h in history},
        "viewed": viewed["label"],
        "generation_mode": mode,
        "provider_error": error,
        "safety_note": SAFETY_NOTE,
    }
