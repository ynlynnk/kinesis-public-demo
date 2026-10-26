"""RAG layer for sessions recorded by the KINESIS patient app (report/<patient>/se<id>/).

Retrieval has two parts:
  1. session evidence - deterministic facts computed here from report.json (and the
     patient's earlier sessions), never altered by the language model; and
  2. approved knowledge - TF-IDF retrieval over knowledge/*.md, queried with the
     question plus the session's own findings (faults, fatigue zone, signal problems).

The configured Model 2 provider (Gemini or Claude, see claude_summary) only words an
answer from those two inputs and must cite them. With no provider, or if the call
fails, a local deterministic answer is returned, so the dashboard always works.
Nothing identifying (patient name, folder, dates) is sent to the language model.
"""

from __future__ import annotations

import math
from statistics import mean
from typing import Any

try:
    from .claude_summary import _extract_json, _generate_model_text, _model2_provider
    from .knowledge_retriever import retrieve_knowledge
except ImportError:
    from claude_summary import _extract_json, _generate_model_text, _model2_provider
    from knowledge_retriever import retrieve_knowledge


SAFETY_NOTE = (
    "This is decision support from sensor data, not a diagnosis. Clinical "
    "decisions remain with the physiotherapist."
)
PATIENT_SAFETY_NOTE = (
    "This summary was checked by your physiotherapist. If anything hurts or "
    "feels wrong, stop and contact them before your next session."
)

# Plain-language advice for the patient, per rep check (codes come from report.json).
PATIENT_TIPS = {
    "INCOMPLETE_ROM": "Try to lift all the way to the target before holding.",
    "EARLY_DOWN": "At the top, pause and count 'one' before lowering.",
    "NO_MUSCLE": "Let the working muscle do the lift, not momentum or the other hand.",
    "FOREARM_LIFT": "Keep the rest of the arm still; only the joint should move.",
    "DROP_FAST": "Lower slowly, about as slowly as you lifted.",
    "SWING_FAST": "Lift smoothly instead of swinging.",
    "WRIST_TWIST": "Keep your palm facing the same way through the movement.",
    "SIDE_BEND": "Keep the movement in a straight line, not to the side.",
    "OVER_FLEX": "Stop at the target; going further is not needed.",
    "NOT_RETURNED": "Go all the way back to the start between repetitions.",
    "JERKY": "Move at a steady, even speed.",
    "SIGNAL_POOR": "Press the sensor firmly onto clean, dry skin before starting.",
    "TIMEOUT": "Keep each repetition flowing; rest between repetitions, not during them.",
}

# Question topic -> a short, precise retrieval query for that topic's own section.
TOPIC_QUERIES = (
    (("fatigue", "tired", "effort"), "fatigue score baseline median frequency smoothing"),
    (("signal", "quality", "electrode", "trust", "noise"), "SIGNAL_POOR electrode lift-off clipping mains hum"),
    (("progress", "compare", "earlier", "previous", "trend"), "compare earlier sessions progress good-repetition rate"),
)

SPECIALIST_SYSTEM_PROMPT = """You are KINESIS Model 2, assisting a physiotherapist who reviews one
rehabilitation session recorded by a wearable sEMG + IMU device.
Answer the question using ONLY the supplied `session_evidence` and `approved_knowledge`.
Rules:
- Never change, recompute or invent numbers; quote them as given.
- Separate measured findings from interpretation; say when evidence is missing or weak.
- Do not diagnose and do not prescribe treatment; you may suggest what to check or review.
- Cite knowledge you use by its id, e.g. [K1]; cite session facts as [S].
- Be concise (at most ~180 words in `answer`).
- `earlier_turns` (if any) is the chat so far: use it to resolve follow-ups ("why?", "and the
  last rep?"), but take every number from `session_evidence`.
Return JSON: {"answer": str, "key_points": [str, ...], "knowledge_ids_used": [str, ...]}"""

PATIENT_SYSTEM_PROMPT = """You rewrite a physiotherapist-approved session summary for the patient,
an older adult doing home rehabilitation. Use only the supplied draft and facts.
Rules:
- Plain, warm, encouraging language; short sentences; no jargon (no sEMG, RVC, MDF, IMU, EWMA).
- Do not add findings, numbers, diagnoses or new exercises; keep every number you keep exactly.
- Keep the physiotherapist's status and note unchanged in meaning.
Return JSON with the same keys as `draft`: {"title", "summary", "movement", "effort", "progress",
"tips": [str], "specialist_note"} (each value a short paragraph or list)."""


# ------------------------------------------------------------------ session evidence

def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _avg(values: list[Any]) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return round(mean(nums), 2) if nums else None


def session_facts(report: dict[str, Any], history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Deterministic, de-identified facts about one session (and its predecessors)."""

    session = report.get("session", {})
    totals = report.get("totals", {})
    fatigue = report.get("fatigue", {})
    reps = report.get("reps", [])
    counted = [r for r in reps if r.get("counted")]
    half = len(reps) // 2
    first, second = reps[:half], reps[half:]

    def rate(group):
        return round(100 * sum(r.get("counted", False) for r in group) / len(group), 1) if group else None

    faults = sorted(report.get("faults", []), key=lambda f: -f.get("reps", 0))
    facts: dict[str, Any] = {
        "exercise": session.get("exercise"),
        "target_angle_deg": session.get("target_deg"),
        "personal_range_deg": session.get("personal_rom_deg"),
        "load_kg": session.get("load_kg"),
        "muscle_reference_calibrated": session.get("muscle_reference_calibrated"),
        "attempts": totals.get("attempts"),
        "good_reps": totals.get("good_reps"),
        "not_counted": totals.get("not_counted"),
        "good_rate_percent": (
            round(100 * totals["good_rate"], 1) if totals.get("good_rate") is not None else None
        ),
        "average_peak_angle_deg": totals.get("avg_peak_angle_deg"),
        "faults": [
            {"code": f.get("code"), "tip": f.get("tip"), "reps": f.get("reps")} for f in faults
        ],
        "good_rate_first_half_percent": rate(first),
        "good_rate_second_half_percent": rate(second),
        "counted_rep_means": {
            "lift_s": _avg([r.get("lift_s") for r in counted]),
            "hold_s": _avg([r.get("hold_s") for r in counted]),
            "lower_s": _avg([r.get("lower_s") for r in counted]),
            "muscle_pct_of_reference": _avg([r.get("muscle_pct_rvc") for r in counted]),
            "peak_angle_deg": _avg([r.get("peak_deg") for r in counted]),
        },
        "signal_problem_reps": sum("SIGNAL_POOR" in (r.get("faults") or []) for r in reps),
        "fatigue": {
            "final_score_0_100": fatigue.get("final_score"),
            "zone": fatigue.get("zone"),
            "source": fatigue.get("source"),
            "zones": fatigue.get("zones"),
            "last_rep_change_from_baseline_pct": fatigue.get("last_rep_changes_pct"),
            "trend": [t.get("score") for t in fatigue.get("trend", [])],
        },
        "rules": [
            {"check": c.get("code"), "rule": c.get("rule")}
            for c in report.get("criteria", []) if c.get("enabled", True)
        ],
    }
    # Sessions without reps (ended during calibration) have no rate: not a comparison point.
    earlier = [h for h in (history or []) if _finite(h.get("good_rate_percent"))]
    if earlier:
        prev = earlier[-1]
        facts["history"] = {
            "earlier_sessions": len(earlier),
            "previous_good_rate_percent": prev.get("good_rate_percent"),
            "previous_average_peak_angle_deg": prev.get("avg_peak_angle_deg"),
            "previous_fatigue_score": prev.get("fatigue_score"),
            "good_rate_by_session": [h.get("good_rate_percent") for h in earlier],
        }
    else:
        facts["history"] = {"earlier_sessions": 0}
    return facts


def knowledge_query(question: str, facts: dict[str, Any]) -> str:
    """The question plus the topics this session's findings make relevant."""

    topics = [question, "rehabilitation session monitoring"]
    codes = {f["code"] for f in facts.get("faults", [])}
    # The fault codes themselves match the KINESIS repetition-check document section by section.
    topics += [f"{f['code']} {f.get('tip') or ''}" for f in facts.get("faults", [])[:4]]
    if codes & {"NO_MUSCLE"}:
        topics += ["muscle activation", "surface EMG amplitude"]
    if codes & {"SWING_FAST", "DROP_FAST", "JERKY", "EARLY_DOWN", "INCOMPLETE_ROM", "OVER_FLEX"}:
        topics += ["movement quality", "kinematic measurements", "range of motion"]
    if codes & {"FOREARM_LIFT", "WRIST_TWIST", "SIDE_BEND"}:
        topics += ["compensation", "movement asymmetry"]
    if facts.get("signal_problem_reps"):
        topics += ["signal quality", "electrode contact"]
    zone = (facts.get("fatigue") or {}).get("zone")
    if zone in ("amber", "red"):
        topics += ["fatigue indicators", "muscle fatigue", "median frequency"]
    return " ".join(t for t in topics if t)


def retrieve(question: str, facts: dict[str, Any], top_k: int = 6) -> list[dict[str, Any]]:
    """The section for the question's topic, the section for each of the session's top
    faults, then broader question-driven hits, in that order of precision.

    One long query dilutes rare terms, so topics and fault codes get their own short
    queries (codes are named in the KINESIS repetition-check document)."""

    q = question.casefold()
    queries = [(query, 1) for words, query in TOPIC_QUERIES if any(w in q for w in words)]
    # Code only: the patient tip's everyday words ("keep your elbow...") match other documents.
    queries += [(f["code"], 1) for f in facts.get("faults", [])[:2]]
    queries.append((knowledge_query(question, facts), 3))
    hits, seen = [], set()
    for query, k in queries:
        try:
            found = retrieve_knowledge(query=query, top_k=k)
        except (FileNotFoundError, RuntimeError, ValueError):
            continue
        for hit in found:
            ident = (hit["source"], hit["section_title"])
            if ident not in seen:
                seen.add(ident)
                hits.append(hit)
    return [dict(hit, id=f"K{i + 1}") for i, hit in enumerate(hits[:top_k])]


# ------------------------------------------------------------------ specialist Q&A

def _local_answer(question: str, facts: dict[str, Any], knowledge: list[dict[str, Any]]) -> dict[str, Any]:
    q = question.casefold()
    f = facts
    points = [
        f"{f['good_reps']} of {f['attempts']} attempts counted ({f['good_rate_percent']}%) "
        f"for {f['exercise']}, target {f['target_angle_deg']}°. [S]",
    ]
    if f["faults"]:
        top = ", ".join(f"{x['code']} ×{x['reps']}" for x in f["faults"][:3])
        points.append(f"Most frequent rep problems: {top}. [S]")
    m = f["counted_rep_means"]
    if any(m.values()):
        points.append(
            f"Counted reps averaged lift {m['lift_s']} s, hold {m['hold_s']} s, lower {m['lower_s']} s, "
            f"muscle effort {m['muscle_pct_of_reference']}% of reference. [S]"
        )
    fat = f["fatigue"]
    if any(w in q for w in ("fatigue", "tired", "effort", "summary", "overall")) or not f["faults"]:
        points.append(
            f"Fatigue score {fat['final_score_0_100']} ({fat['zone'] or 'no zone'}; source: "
            f"{fat['source'] or 'n/a'}). [S]"
        )
    if f["good_rate_first_half_percent"] is not None:
        points.append(
            f"Good-rep rate {f['good_rate_first_half_percent']}% in the first half vs "
            f"{f['good_rate_second_half_percent']}% in the second half. [S]"
        )
    h = f["history"]
    if h.get("earlier_sessions"):
        points.append(
            f"Previous session: {h['previous_good_rate_percent']}% good reps, fatigue "
            f"{h['previous_fatigue_score']}. [S]"
        )
    if f["signal_problem_reps"]:
        points.append(f"{f['signal_problem_reps']} rep(s) had a signal-quality problem. [S]")
    for k in knowledge[:2]:
        points.append(f"See {k['document_title']} - {k['section_title']}. [{k['id']}]")
    return {
        "answer": "Local evidence summary (no language model connected):",
        "key_points": points,
        "knowledge_ids_used": [k["id"] for k in knowledge[:2]],
    }


def answer_specialist_question(report: dict[str, Any], question: str,
                               history: list[dict[str, Any]] | None = None,
                               conversation: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Grounded answer for the specialist, with the evidence and sources it used.

    `conversation` is the earlier chat ({question, answer} turns, oldest first); the last
    few go to the model so follow-up questions make sense."""

    question = " ".join(question.split()) or "Summarise this session."
    facts = session_facts(report, history)
    earlier = [{"question": t["question"], "answer": str(t.get("answer", ""))[:600]}
               for t in (conversation or [])[-4:]]
    # A short follow-up ("why?") retrieves poorly on its own; borrow the previous question's words.
    knowledge = retrieve(" ".join([earlier[-1]["question"], question]) if earlier and len(question) < 40
                         else question, facts)
    provider, api_key, model = _model2_provider()
    payload = {
        "question": question,
        "earlier_turns": earlier,
        "session_evidence": facts,
        "approved_knowledge": [
            {"id": k["id"], "title": k["document_title"], "section": k["section_title"],
             "text": str(k["text"])[:1200]}
            for k in knowledge
        ],
    }
    mode, error = "Local evidence summary", None
    result = None
    if api_key:
        try:
            result = _extract_json(
                _generate_model_text(provider, api_key, model, SPECIALIST_SYSTEM_PROMPT, payload, 900)
            )
            if not str(result.get("answer", "")).strip():
                raise ValueError("empty answer")
            mode = f"AI-generated ({provider}: {model})"
        except Exception as exc:   # the dashboard must keep working without the provider
            error, result = str(exc), None
    if result is None:
        result = _local_answer(question, facts, knowledge)
    return {
        "question": question,
        "answer": str(result.get("answer", "")),
        "key_points": [str(p) for p in result.get("key_points", [])],
        "knowledge_ids_used": [str(k) for k in result.get("knowledge_ids_used", [])],
        "knowledge": knowledge,
        "facts": facts,
        "generation_mode": mode,
        "provider_error": error,
        "safety_note": SAFETY_NOTE,
    }


# ------------------------------------------------------------------ patient-facing log

def draft_patient_log(report: dict[str, Any], review: dict[str, Any] | None,
                      history: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Plain-language log of the session for the patient; the specialist edits and releases it."""

    f = session_facts(report, history)
    review = review or {}
    status_label = review.get("status_label", "Waiting for your physiotherapist's review")
    good, attempts = f["good_reps"] or 0, f["attempts"] or 0
    if attempts:
        summary = (
            f"You did {attempts} repetitions of {f['exercise'] or 'your exercise'} and {good} of them "
            f"were done well."
        )
    else:
        summary = "This session ended before any repetitions were recorded."
    movement = ""
    if f["average_peak_angle_deg"] is not None and f["target_angle_deg"]:
        reached = f["average_peak_angle_deg"] >= f["target_angle_deg"] - 2
        movement = (
            f"On average you reached {round(f['average_peak_angle_deg'])}° "
            f"(target {round(f['target_angle_deg'])}°) - "
            + ("you reached your target." if reached else "a little more range will come with practice.")
        )
    zone = (f["fatigue"] or {}).get("zone")
    effort = {
        "green": "Your muscle stayed fresh through the session.",
        "amber": "Your muscle started to get tired towards the end - that is normal, rest well.",
        "red": "Your muscle got quite tired. Take a proper rest before the next session.",
    }.get(zone, "There was not enough data to judge tiredness this time.")
    progress = "This is your first recorded session - next time we can compare."
    h = f["history"]
    if h.get("earlier_sessions") and _finite(h.get("previous_good_rate_percent")) and _finite(f["good_rate_percent"]):
        diff = f["good_rate_percent"] - h["previous_good_rate_percent"]
        progress = (
            "More of your repetitions were done well than last time." if diff > 5
            else "About the same quality as last time - steady progress." if diff >= -5
            else "Fewer repetitions were done well than last time; that can happen on a tiring day."
        )
    tips = [PATIENT_TIPS[x["code"]] for x in f["faults"] if x["code"] in PATIENT_TIPS][:3]
    draft = {
        "title": f"Your {f['exercise'] or 'exercise'} session",
        "summary": summary,
        "movement": movement,
        "effort": effort,
        "progress": progress,
        "tips": tips or ["Keep going at the same steady pace."],
        "specialist_note": review.get("notes", ""),
    }
    provider, api_key, model = _model2_provider()
    mode = "Local plain-language draft"
    if api_key:
        try:
            worded = _extract_json(
                _generate_model_text(
                    provider, api_key, model, PATIENT_SYSTEM_PROMPT,
                    {"draft": draft, "status": status_label,
                     "facts": {k: f[k] for k in ("good_reps", "attempts", "average_peak_angle_deg",
                                                  "target_angle_deg")}},
                    700,
                )
            )
            for key in draft:
                if key == "tips" and isinstance(worded.get("tips"), list) and worded["tips"]:
                    draft["tips"] = [str(t) for t in worded["tips"]][:4]
                elif key != "tips" and str(worded.get(key, "")).strip():
                    draft[key] = str(worded[key]).strip()
            mode = f"AI-worded ({provider}: {model})"
        except Exception:
            pass
    return {
        **draft,
        "status_label": status_label,
        "level": review.get("level", "yellow"),
        "generation_mode": mode,
        "safety_note": PATIENT_SAFETY_NOTE,
    }
