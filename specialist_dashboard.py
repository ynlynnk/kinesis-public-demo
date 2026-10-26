"""Specialist workspace for sessions sent by the KINESIS patient app.

Reads report/<patient>/se<session>/ (see kinesis_reports.py) without the MCP backend:
overview and progress, whole-session signals, every repetition, the fatigue score,
a floating grounded RAG chat (bottom-right), and the review that turns into a patient-facing log.
"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

import kinesis_reports as kr
from agent.claude_summary import model2_status
from agent.kinesis_progress_agent import (
    generate_progress_report,
    ground_truth,
    load_report,
    save_report,
)
from agent.kinesis_chat_agent import chat_answer
from agent.kinesis_session_rag import draft_patient_log


STATUS_BADGES = {
    "awaiting_review": ("🟡", "Awaiting review"),
    "reviewed": ("🔵", "Reviewed, not released"),
    "released": ("🟢", "Released to patient"),
}
LEVEL_ICON = {"green": "🟢", "yellow": "🟡", "red": "🔴"}
RISK_STYLE = {   # tag -> (icon, streamlit callout)
    "HIGH RISK": ("⛔", "error"),
    "AT RISK": ("⚠️", "warning"),
    "MONITOR": ("👁️", "info"),
    "NO FLAGS": ("✅", "success"),
}
FLAG_ICON = {"high": "🔴", "medium": "🟠", "low": "🟡"}
TRAJECTORY_ICON = {"improving": "Improving", "stable": "Stable", "declining": "Declining",
                   "insufficient_data": "Too early"}
SUGGESTED_QUESTIONS = [
    "How has this patient progressed across all sessions?",
    "Why were repetitions not counted here, and has that fault happened before?",
    "Compare this session with the patient's best session.",
    "Is fatigue building up from session to session?",
]


def _rep_frame(report: dict, t0: float | None) -> pd.DataFrame:
    rows = []
    for r in report.get("reps", []):
        start, end = r.get("t_start"), r.get("t_end")
        rows.append(
            {
                "#": r.get("index"),
                "Set": r.get("set"),
                "Counted": "Yes" if r.get("counted") else "No",
                "Faults": ", ".join(r.get("faults") or []) or "-",
                "Peak °": r.get("peak_deg"),
                "Lift s": r.get("lift_s"),
                "Hold s": r.get("hold_s"),
                "Lower s": r.get("lower_s"),
                "Up °/s": r.get("peak_speed_up_dps"),
                "Down °/s": r.get("peak_speed_down_dps"),
                "Muscle %": r.get("muscle_pct_rvc"),
                "Onset lag s": r.get("muscle_onset_lag_s"),
                "SPARC": r.get("smoothness_sparc"),
                "MDF Hz": r.get("mdf_hz"),
                "Fatigue": r.get("fatigue_score"),
                "start": (start - t0) if (start is not None and t0 is not None) else None,
                "end": (end - t0) if (end is not None and t0 is not None) else None,
            }
        )
    return pd.DataFrame(rows)


def _rep_bands(reps: pd.DataFrame) -> alt.Chart | None:
    bands = reps.dropna(subset=["start", "end"]) if not reps.empty else reps
    if bands.empty:
        return None
    return (
        alt.Chart(bands)
        .mark_rect(opacity=0.18)
        .encode(
            x="start:Q",
            x2="end:Q",
            color=alt.Color(
                "Counted:N",
                scale=alt.Scale(domain=["Yes", "No"], range=["#16a34a", "#dc2626"]),
                legend=alt.Legend(title="Rep counted"),
            ),
            tooltip=["#:Q", "Counted:N", "Faults:N", "Peak °:Q", "Hold s:Q"],
        )
    )


def _line(df: pd.DataFrame, y: str, title: str, color: str, rule: float | None = None,
          bands: alt.Chart | None = None, top: float | None = None) -> alt.Chart:
    """`top` caps the y axis (values above are clipped), so one spike cannot flatten the rest."""
    scale = alt.Scale(domain=[0, top], clamp=True) if top is not None else alt.Scale()
    line = (
        alt.Chart(df)
        .mark_line(color=color, strokeWidth=1.4)
        .encode(x=alt.X("t_s:Q", title="time (s)"), y=alt.Y(f"{y}:Q", title=title, scale=scale))
    )
    layers = [bands] if bands is not None else []
    layers.append(line)
    if rule is not None:
        layers.append(
            alt.Chart(pd.DataFrame({"y": [rule]}))
            .mark_rule(strokeDash=[5, 4], color="#6b7280")
            .encode(y="y:Q")
        )
    return alt.layer(*layers).properties(height=190)


def _render_overview(session: dict, trend: pd.DataFrame) -> None:
    report = session["report"]
    totals, fatigue = report.get("totals", {}), report.get("fatigue", {})
    cols = st.columns(5)
    cols[0].metric("Good reps", f"{totals.get('good_reps', 0)} / {totals.get('attempts', 0)}")
    rate = totals.get("good_rate")
    cols[1].metric("Good-rep rate", f"{100 * rate:.0f}%" if rate is not None else "-")
    cols[2].metric("Average peak", f"{totals.get('avg_peak_angle_deg') or '-'}°",
                   help=f"Target {report.get('session', {}).get('target_deg')}°")
    cols[3].metric("Fatigue score", fatigue.get("final_score", "-"),
                   help=f"Zone: {fatigue.get('zone')}; source: {fatigue.get('source')}")
    cols[4].metric("Sets", totals.get("sets_done", 0))
    if fatigue.get("source") == "demo":
        st.warning("Too few reps were counted: the fatigue score is a demonstration estimate over all attempts.")

    left, right = st.columns(2)
    with left:
        st.markdown("**Why reps were not counted**")
        faults = pd.DataFrame(report.get("faults", []))
        if faults.empty:
            st.success("No faults in this session.")
        else:
            st.altair_chart(
                alt.Chart(faults)
                .mark_bar(color="#dc2626")
                .encode(
                    x=alt.X("reps:Q", title="reps", axis=alt.Axis(tickMinStep=1, format="d")),
                    y=alt.Y("code:N", sort="-x", title=None, axis=alt.Axis(labelOverlap=False)),
                    tooltip=["code", "tip", "reps"],
                )
                .properties(height=max(120, 36 * len(faults))),
                width="stretch",
            )
    with right:
        st.markdown("**Progress across sessions**")
        if len(trend) < 2:
            st.info("Progress appears after the patient's second session.")
        else:
            long = trend.melt(
                id_vars="session_id",
                value_vars=["good_rate_percent", "fatigue_score"],
                var_name="measure",
                value_name="value",
            ).dropna()
            st.altair_chart(
                alt.Chart(long)
                .mark_line(point=True)
                .encode(
                    x=alt.X("session_id:N", title=None, sort=None),
                    y=alt.Y("value:Q", title="%  /  score"),
                    color=alt.Color("measure:N", title=None),
                    tooltip=["session_id", "measure", "value"],
                )
                .properties(height=220),
                width="stretch",
            )
    with st.expander("All sessions of this patient"):
        st.dataframe(trend, hide_index=True, width="stretch")


def _render_signals(session: dict, reps: pd.DataFrame) -> None:
    signals = session["report"].get("signals") or {}
    if not signals.get("t_s"):
        st.info("No signal trace was recorded (the session ended before a set started).")
        return
    df = pd.DataFrame(
        {"t_s": signals["t_s"], "angle": signals["angle_deg"], "muscle": signals["muscle_pct_rvc"]}
    )
    target = session["report"].get("session", {}).get("target_deg")
    bands = _rep_bands(reps)
    st.altair_chart(_line(df, "angle", "Angle ° (IMU)", "#2563eb", target, bands), width="stretch")
    top = max(120.0, float(df["muscle"].quantile(0.99))) if df["muscle"].notna().any() else 120.0
    st.altair_chart(
        _line(df, "muscle", "Muscle % (sEMG)", "#b45309", 25, bands, top=round(top)),
        width="stretch",
    )
    t0 = signals.get("t0")
    t1 = t0 + signals["t_s"][-1] if t0 is not None else None
    raw = kr.raw_semg_envelope(session["path"], t0, t1)
    if raw is None:
        st.caption("Raw sEMG is not available for this session (no emg.csv).")
        return
    band = (
        alt.Chart(raw)
        .mark_area(color="#b45309", opacity=0.6)
        .encode(x=alt.X("t_s:Q", title="time (s)"), y=alt.Y("min:Q", title="Raw sEMG (ADC)"),
                y2="max:Q")
    )
    layers = [bands, band] if bands is not None else [band]
    st.altair_chart(alt.layer(*layers).properties(height=190), width="stretch")
    st.caption("Shaded bands are repetitions: green counted, red not counted (hover for the reason).")


def _render_reps(session: dict, reps: pd.DataFrame) -> None:
    if reps.empty:
        st.info("No repetitions were recorded.")
    else:
        st.dataframe(reps.drop(columns=["start", "end"]), hide_index=True, width="stretch")
    st.markdown("**Rules in force for this session**")
    counts = {f["code"]: f["reps"] for f in session["report"].get("faults", [])}
    rules = pd.DataFrame(
        [
            {"Check": c["code"], "Tip shown": c["tip"],
             "Rule": c["rule"] if c.get("enabled", True) else "Off for this exercise",
             "Failed": counts.get(c["code"], 0)}
            for c in session["report"].get("criteria", [])
        ]
    )
    st.dataframe(rules, hide_index=True, width="stretch")


def _render_fatigue(session: dict) -> None:
    fatigue = session["report"].get("fatigue", {})
    trend = pd.DataFrame(fatigue.get("trend", []))
    if trend.empty:
        st.info("No fatigue trend for this session.")
    else:
        zones = fatigue.get("zones", {"green_below": 40, "red_above": 70})
        bands = pd.DataFrame(
            [
                {"lo": 0, "hi": zones["green_below"], "zone": "green"},
                {"lo": zones["green_below"], "hi": zones["red_above"], "zone": "amber"},
                {"lo": zones["red_above"], "hi": 100, "zone": "red"},
            ]
        )
        zone_layer = alt.Chart(bands).mark_rect(opacity=0.12).encode(
            y="lo:Q", y2="hi:Q",
            color=alt.Color("zone:N", scale=alt.Scale(domain=["green", "amber", "red"],
                                                      range=["#16a34a", "#d97706", "#dc2626"]), legend=None),
        )
        line = alt.Chart(trend).mark_line(point=True, color="#2563eb").encode(
            x=alt.X("rep:Q", title="rep"), y=alt.Y("score:Q", title="fatigue score", scale=alt.Scale(domain=[0, 100])),
            tooltip=["rep", "score"],
        )
        st.altair_chart(alt.layer(zone_layer, line).properties(height=240), width="stretch")
    left, right = st.columns(2)
    with left:
        st.markdown("**Last scored rep vs. baseline (% change, + = more tired)**")
        st.dataframe(
            pd.DataFrame(fatigue.get("last_rep_changes_pct", {}).items(), columns=["term", "change %"]),
            hide_index=True, width="stretch",
        )
    with right:
        st.markdown("**Weights and full scale**")
        weights = fatigue.get("weights", {})
        full = fatigue.get("full_scale_pct", {})
        st.dataframe(
            pd.DataFrame([{"term": k, "weight": v, "full scale %": full.get(k)} for k, v in weights.items()]),
            hide_index=True, width="stretch",
        )
    st.caption(
        f"Baseline = first {fatigue.get('baseline_reps')} scored reps; smoothing α = "
        f"{fatigue.get('ewma_alpha')}. See the full report for the step-by-step explanation."
    )


CHAT_CSS = """<style>
.st-key-kin_chat_fab {position: fixed; right: 1.5rem; bottom: 1.5rem; z-index: 1001; width: auto !important;}
.st-key-kin_chat_fab button {border-radius: 999px; padding: 0.55rem 1.1rem;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);}
.st-key-kin_chat_panel {position: fixed; right: 1.5rem; bottom: 4.8rem; z-index: 1000;
    width: min(%(width)spx, calc(100vw - 2rem)) !important; max-height: calc(100vh - 7rem); overflow-y: auto;
    background: %(bg)s; border: 1px solid %(line)s; border-radius: 14px; padding: 0.9rem 1rem 0.6rem;
    box-shadow: 0 10px 34px rgba(0, 0, 0, %(shadow)s);}
</style>"""
CHAT_WIDTH, CHAT_HEIGHT = 460, 420   # original panel width, message area height (px)


def _table_width(table: list[dict]) -> int:
    """Rough pixel width a reply's table needs: widest text per column, plus the panel's
    chat indent and padding."""

    if not table:
        return 0
    cols = [max(len(str(c)), *(len(str(r.get(c, ""))) for r in table)) for c in table[0]]
    return sum(8 * n + 26 for n in cols) + 110
AI_CAUTION = ("Reference only: the AI can be wrong or miss context. Check its points against the session "
              "data; the diagnosis and every clinical decision are yours.")


def _toggle_chat() -> None:
    st.session_state["kin_chat_open"] = not st.session_state.get("kin_chat_open", False)


def _ask(text: str) -> None:
    st.session_state["kin_chat_pending"] = text


def _render_turn(result: dict) -> None:
    with st.chat_message("user"):
        st.markdown(result["question"])
    with st.chat_message("assistant"):
        st.markdown(result["answer"])
        if result.get("findings"):
            st.markdown("\n".join(f"- **{f['label']}** · {f['detail']}" if f["label"] else f"- {f['detail']}"
                                  for f in result["findings"]))
        elif result.get("key_points"):
            st.markdown("\n".join(f"- {p}" for p in result["key_points"]))
        if result.get("table"):
            st.dataframe(pd.DataFrame(result["table"]), hide_index=True, width="stretch",
                         column_config={"Peak°": st.column_config.NumberColumn(format="%.1f"),
                                        "Fatigue": st.column_config.NumberColumn(format="%.1f")})
            st.caption("Table: measured values from the record.")
        if result.get("check"):
            st.markdown(f"**Check:** {result['check']}")
        if result.get("unverified_numbers"):
            st.warning(f"Not found in the patient's record: {', '.join(result['unverified_numbers'])}. "
                       "Verify these before relying on them.")
        used = result.get("sessions_used") or []
        ai = bool(result.get("tool_calls"))
        st.caption(" · ".join(filter(None, [
            "⚠️ AI reference, verify before acting" if ai else "Computed from the record",
            result["generation_mode"],
            f"sessions {', '.join(used)}" if used else "",
            f"provider unavailable ({result['provider_error'][:60]})" if result.get("provider_error") else "",
        ])))
        with st.expander("Sources"):
            labels = result.get("session_labels", {})
            if result.get("sessions_used"):
                st.markdown("**Sessions:** " + ", ".join(
                    f"{s} = `{labels.get(s, '?')}`" for s in result["sessions_used"]))
            if result.get("tool_calls"):
                st.markdown("**Looked up:** " + ", ".join(
                    f"`{c['tool']}({', '.join(str(v) for v in c['args'].values())})`" for c in result["tool_calls"]))
            for k in result["knowledge"]:
                used = " ✓ cited" if k["id"] in result["knowledge_ids_used"] else ""
                st.markdown(f"**[{k['id']}]** {k['document_title']} - {k['section_title']}{used}")
                st.caption(str(k["text"])[:300] + ("..." if len(str(k["text"])) > 300 else ""))
            if not result["knowledge"]:
                st.caption("No knowledge section matched.")
            st.markdown(f"**[S]** facts of the viewed session ({result.get('viewed', '')})")
            st.json(result["facts"], expanded=False)


def _render_chat(patient: str, session_id: str) -> None:
    """Floating agent chat in the bottom-right corner with the patient's whole record.

    It is fixed-positioned on top of the page, so it stays open while the specialist moves
    between tabs and sessions; the thread belongs to the patient, so a follow-up can move
    from one session to another, and the agent looks up any session through its tools."""

    dark = st.context.theme.type == "dark"
    thread = st.session_state.setdefault(f"kin_chat_{patient}", [])
    fit = max([_table_width(t.get("table") or []) for t in thread] + [0])
    wide = st.session_state.get("kin_chat_wide", False) and fit > CHAT_WIDTH
    width = fit if wide else CHAT_WIDTH
    st.html(CHAT_CSS % {"bg": "#0e1117" if dark else "#ffffff", "line": "#3a3f4b" if dark else "#d9dce3",
                        "shadow": "0.55" if dark else "0.18", "width": width})
    is_open = st.session_state.get("kin_chat_open", False)
    with st.container(key="kin_chat_fab"):
        st.button("✕ Close chat" if is_open else "💬 Ask the agent", type="primary", key="kin_chat_toggle",
                  on_click=_toggle_chat)
    if not is_open:
        return
    n_sessions = len(kr.list_sessions(patient))
    with st.container(key="kin_chat_panel"):
        head, sizer, clear = st.columns([6, 1.6, 1.2], vertical_alignment="center")
        head.markdown(f"**KINESIS agent** · {patient} · all {n_sessions} sessions")
        if fit > CHAT_WIDTH:   # a reply has a table wider than the original panel
            sizer.button("⤡ Shrink" if wide else "⤢ Fit table", key="kin_chat_size", type="tertiary",
                         help="Back to the original size" if wide else "Widen the chat to fit the tables",
                         on_click=lambda: st.session_state.update(kin_chat_wide=not wide))
        if thread and clear.button("Clear", key=f"kin_chat_clear_{patient}", type="tertiary"):
            thread.clear()
            st.session_state["kin_chat_wide"] = False
        if not st.session_state.get("kin_chat_ack"):
            st.warning("**Before you use the agent**\n\n" + AI_CAUTION + " It does not diagnose, and a number "
                       "it quotes that is not in the record is flagged.")
            st.button("I understand: the agent is a reference, I make the diagnosis", type="primary",
                      width="stretch", key="kin_chat_ack_btn",
                      on_click=lambda: st.session_state.update(kin_chat_ack=True))
            return
        status = model2_status()
        st.caption(f"Viewing {session_id} · "
                   + (f"{status['provider']} ({status['model']})" if status["configured"]
                      else "no API key: local evidence answers")
                   + f"  \n⚠️ {AI_CAUTION}")
        messages = st.container(height=CHAT_HEIGHT, border=False)
        typed = st.chat_input("Ask about this patient's sessions...", key=f"kin_chat_in_{patient}")
        question = typed or st.session_state.pop("kin_chat_pending", None)
        with messages:
            if not thread and not question:
                st.caption("The agent can look up every session this patient has sent (S1 = oldest), "
                           "compare them and trace faults and fatigue, citing sessions and knowledge. Try:")
                for i, q in enumerate(SUGGESTED_QUESTIONS):
                    st.button(q, key=f"kin_chat_suggest_{patient}_{i}", width="stretch", on_click=_ask, args=(q,))
            for result in thread:
                _render_turn(result)
            if question:
                with st.chat_message("user"):
                    st.markdown(question)
                with st.spinner("Looking through the patient's sessions..."):
                    result = chat_answer(kr.session_dir(patient, session_id).parent, session_id, question, thread)
                thread.append(result)
                if _table_width(result.get("table") or []) > CHAT_WIDTH:
                    st.session_state["kin_chat_wide"] = True   # fit the new table; "Shrink" undoes it
                st.rerun()   # redraw the thread in order, with the new turn's sources


def _render_review(session: dict, trend: pd.DataFrame, patient: str, session_id: str, key: str) -> None:
    review = session["review"] or {}
    st.markdown("#### 1. Specialist review")
    options = list(kr.REVIEW_STATUSES)
    with st.form(f"review_form_{key}"):
        status = st.radio(
            "Decision",
            options,
            index=options.index(review["status"]) if review.get("status") in options else 0,
            format_func=lambda s: f"{LEVEL_ICON[kr.REVIEW_STATUSES[s][0]]} {kr.REVIEW_STATUSES[s][1]}",
            horizontal=True,
        )
        notes = st.text_area(
            "Clinical notes (shown to the patient as 'Note from your physiotherapist')",
            value=review.get("notes", ""), height=110,
        )
        reviewer = st.text_input("Reviewer", value=review.get("reviewer", ""))
        if st.form_submit_button("Save review", type="primary"):
            review = kr.save_review(patient, session_id, status, notes, reviewer)
            st.session_state.pop(f"log_draft_{key}", None)
            st.success("Review saved.")
    if review:
        st.caption(f"Last saved {review.get('reviewed_at')} by {review.get('reviewer')}.")

    st.markdown("#### 2. Patient-facing log")
    if not review:
        st.info("Save the review first: the log carries your decision and note to the patient.")
        return
    draft_key = f"log_draft_{key}"
    if st.button("Draft from session data", key=f"draft_{key}"):
        earlier = trend[trend["session_id"] < session_id].to_dict("records") if not trend.empty else []
        with st.spinner("Writing a plain-language draft..."):
            st.session_state[draft_key] = draft_patient_log(session["report"], review, earlier)
    log = st.session_state.get(draft_key) or session["patient_log"]
    if not log:
        st.caption("No log yet. Draft one, edit it, then release it to the patient.")
        return
    st.caption(f"{log.get('generation_mode', '')} - edit anything before releasing.")
    with st.form(f"log_form_{key}"):
        edited = dict(log)
        edited["title"] = st.text_input("Title", log.get("title", ""))
        for field, label in (("summary", "How the session went"), ("movement", "Your movement"),
                             ("effort", "Muscle effort and tiredness"), ("progress", "Your progress")):
            edited[field] = st.text_area(label, log.get(field, ""), height=80)
        edited["tips"] = [
            t.strip() for t in st.text_area("For next time (one tip per line)", "\n".join(log.get("tips", []))).splitlines()
            if t.strip()
        ]
        edited["specialist_note"] = st.text_area("Note from your physiotherapist", log.get("specialist_note", ""))
        edited["status_label"] = review.get("status_label", log.get("status_label"))
        edited["level"] = review.get("level", log.get("level"))
        save_col, release_col = st.columns(2)
        save = save_col.form_submit_button("Save draft")
        release = release_col.form_submit_button("Release to patient", type="primary")
    if save or release:
        saved = kr.save_patient_log(patient, session_id, edited, released=release)
        st.session_state[draft_key] = saved
        st.success("Released: the patient sees it in the KINESIS app the next time they enter their name "
                   "(also under 'Messages from your physiotherapist')." if release else "Draft saved.")
    current = st.session_state.get(draft_key) or log
    read = session.get("patient_read") or {}
    if current.get("released") and not (save or release):
        if read.get("read_at") and read.get("released_at") == current.get("released_at"):
            st.caption(f"✓ Read by the patient in the app on {read['read_at'].replace('T', ' ')}.")
        else:
            st.caption("Released, not opened by the patient yet"
                       + (" (they read an earlier version)." if read.get("read_at") else "."))
    with st.expander("Preview as the patient will read it", expanded=bool(release)):
        st.markdown(kr.patient_log_markdown(st.session_state.get(draft_key) or log))


@st.cache_data(show_spinner=False)
def _truth(patient_dir: str, session_id: str, _stamp: float) -> dict:
    """Ground truth for one session; `_stamp` (folder mtime) refreshes it when files change."""
    from pathlib import Path

    truth = ground_truth(Path(patient_dir), session_id)
    truth["history"] = [{k: v for k, v in h.items() if k not in ("report", "path")} for h in truth["history"]]
    return truth


def _risk_of(patient: str, session_id: str) -> dict:
    path = kr.session_dir(patient, session_id)
    return _truth(str(path.parent), session_id, path.stat().st_mtime)


def _render_progress(patient: str, session: dict, session_id: str, key: str) -> None:
    truth = _risk_of(patient, session_id)
    ov, risk = truth["overview"], truth["risk"]
    icon, callout = RISK_STYLE[risk["risk_tag"]]
    flag_text = "; ".join(f["title"] for f in risk["flags"]) or "No abnormal findings in this session."
    getattr(st, callout)(f"**{icon} {risk['risk_tag']}** - {flag_text}")

    span = ov.get("rehab_span_days")
    cols = st.columns(5)
    cols[0].metric("Session", f"{ov['session_number']} of {ov['sessions_to_date']}",
                   help=f"{ov['sessions_with_reps']} with repetitions; {ov['same_exercise_sessions']} of "
                        f"{ov['exercise']}")
    cols[1].metric("Rehab day", ov.get("rehab_day") if ov.get("rehab_day") is not None else "-")
    many = ov["sessions_to_date"] > 1
    cols[2].metric(
        "Timeframe",
        f"{ov['rehab_span_weeks']} wk" if span else ("same day" if many else "first session"),
        help=f"{span} days since the first recorded session" if span is not None else None,
    )
    cols[3].metric("Sessions / week", ov.get("sessions_per_week") or "-")
    cols[4].metric("Rule trajectory", TRAJECTORY_ICON[ov["rule_trajectory"]],
                   help=f"Score {ov['rule_trajectory_score']}: good-rep rate counts double; "
                        "+2 or more improving, -2 or less declining. Needs two sessions of this "
                        "exercise with repetitions.")
    history = truth["history"]
    first, this = history[0].get("when"), history[-1].get("when")
    st.caption(
        f"{ov['sessions_to_date']} session(s) recorded up to this one, "
        f"{ov['sessions_with_reps']} with repetitions ({ov['same_exercise_sessions']} of {ov['exercise']})"
        + (f" · first {first:%d %b %Y %H:%M} -> this {this:%d %b %Y %H:%M}" if first and this else "")
        + ("" if ov["same_exercise_sessions"] >= 2 else
           " · a trend needs at least two sessions of this exercise with repetitions")
    )

    if risk["flags"]:
        with st.container(border=True):
            st.markdown("**Risk flags (rule-based)**")
            for f in risk["flags"]:
                st.markdown(f"{FLAG_ICON[f['level']]} **{f['title']}** `{f['code']}` - {f['evidence']}")
            st.caption(risk["rule"])

    rows = [
        {"session": b["session"], "day": b["day"], "metric": d["label"], "value": b["value"],
         "current": b["session"] == ov["this_session"]}
        for d in ov["metrics"].values() for b in d["by_session"]
    ]
    if rows:
        st.markdown("**Progress by metric** (sessions of this exercise up to this one)")
        df = pd.DataFrame(rows)
        base = alt.Chart(df).encode(x=alt.X("day:Q", title="rehab day"), y=alt.Y("value:Q", title=None))
        chart = alt.layer(
            base.mark_line(color="#2563eb"),
            base.mark_point(filled=True, size=70).encode(
                color=alt.condition("datum.current", alt.value("#dc2626"), alt.value("#2563eb")),
                tooltip=["session", "day", "value"]),
        ).properties(height=140, width=230).facet(
            facet=alt.Facet("metric:N", title=None, header=alt.Header(labelFontWeight="bold"),
                            sort=[d["label"] for d in ov["metrics"].values()]), columns=3,
        ).resolve_scale(y="independent")
        st.altair_chart(chart)
        table = pd.DataFrame([
            {"Metric": d["label"], "Start": d["first"], "Now": d["current"],
             "vs recent": d["change_vs_recent_sessions"], "Recent trend": d["direction_recent"],
             "Since start": d["direction_since_start"], "Per session": d["slope_per_session"]}
            for d in ov["metrics"].values()
        ])
        st.dataframe(table, hide_index=True, width="stretch")

    st.markdown("#### AI progress report")
    status = model2_status()
    cached = load_report(session["path"])
    label = "Regenerate AI report" if cached else "Generate AI report"
    if st.button(label, type="primary", key=f"progress_{key}"):
        with st.spinner("The AI is gathering session, history and fatigue evidence through its tools..."):
            cached = generate_progress_report(session["path"].parent, session_id)
            save_report(session["path"], cached)
    if not cached:
        st.caption(
            f"Model 2: {status['provider']} ({status['model']}). The AI calls tools for the overview, "
            "history, session details, fatigue history, risk flags and knowledge, then reasons over them."
            + ("" if status["configured"] else " No API key: a local rule-based report is produced.")
        )
        return
    ai = cached["ai"]
    with st.container(border=True):
        top = st.columns([3, 1, 1])
        top[0].markdown(f"### {ai.get('headline', '')}")
        top[1].metric("AI trajectory", TRAJECTORY_ICON.get(ai.get("trajectory"), ai.get("trajectory")))
        top[2].metric("Confidence", str(ai.get("confidence", "-")).title())
        if ai.get("trajectory") != cached["ground_truth"]["overview"]["rule_trajectory"]:
            st.caption("The AI's trajectory differs from the rule-based one - read its reasoning.")
        st.markdown(ai.get("reasoning", ""))
        if ai.get("metric_progress"):
            st.dataframe(pd.DataFrame(ai["metric_progress"]), hide_index=True, width="stretch")
        left, right = st.columns(2)
        with left:
            st.markdown("**Fatigue trend**")
            st.write(ai.get("fatigue_assessment", "-"))
            st.markdown("**Risk**")
            st.write(ai.get("risk_comment", "-"))
        with right:
            if ai.get("additional_concerns"):
                st.markdown("**Additional concerns (AI, not rule-based)**")
                for c in ai["additional_concerns"]:
                    st.markdown(f"- {c}")
            if ai.get("suggested_review_actions"):
                st.markdown("**Suggested review actions**")
                for a in ai["suggested_review_actions"]:
                    st.markdown(f"- {a}")
        if ai.get("citations"):
            st.caption("Sources: " + "; ".join(map(str, ai["citations"])))
        st.caption(f"{cached['generation_mode']} · generated {cached['generated_at']} · {cached['safety_note']}")
        if cached.get("provider_error"):
            st.caption(f"AI provider unavailable, local report used ({cached['provider_error'][:160]}).")
    if cached.get("tool_calls"):
        with st.expander(f"Function calls made by the AI ({len(cached['tool_calls'])})"):
            for i, call in enumerate(cached["tool_calls"], 1):
                st.markdown(f"{i}. `{call['tool']}({json.dumps(call['args'])})`")


def render_specialist_dashboard() -> None:
    st.caption("Sessions sent by the KINESIS patient app after every summary. Specialist view: all data.")
    patients = kr.list_patients()
    if not patients:
        st.info(
            "No sessions yet. Finish a session (or press 'Skip to results (demo)') in the KINESIS "
            f"patient app; it is delivered to `{kr.REPORT_ROOT}`."
        )
        return
    latest_tag = {}
    for p in patients:
        latest = kr.list_sessions(p)
        latest_tag[p] = _risk_of(p, latest[-1]["session_id"])["risk"]["risk_tag"] if latest else "NO FLAGS"
    with st.sidebar:
        st.header("Patient session")
        patient = st.selectbox(
            "Patient", patients,
            format_func=lambda p: f"{RISK_STYLE[latest_tag[p]][0]} {p}"
            + (f"  ·  {latest_tag[p]}" if latest_tag[p] in ("HIGH RISK", "AT RISK") else ""),
        )
        sessions = kr.list_sessions(patient)
        labels = {
            s["session_id"]: f"{STATUS_BADGES[s['review_status']][0]} {s['session_id']} · {s['exercise']}"
            for s in sessions
        }
        session_id = st.selectbox(
            "Session", list(labels)[::-1], format_func=labels.get,
        )
        awaiting = sum(s["review_status"] == "awaiting_review" for s in sessions)
        st.caption(f"{len(sessions)} session(s), {awaiting} awaiting review.")
    session = kr.load_session(patient, session_id)
    trend = kr.session_trend(patient)
    manifest = session["manifest"]
    status = next((s["review_status"] for s in sessions if s["session_id"] == session_id), "awaiting_review")
    icon, status_text = STATUS_BADGES[status]
    risk_tag = _risk_of(patient, session_id)["risk"]["risk_tag"]
    st.subheader(f"{patient} · {session_id} · {manifest.get('exercise', '')}")
    st.caption(
        f"{RISK_STYLE[risk_tag][0]} {risk_tag} · {icon} {status_text} · sent {manifest.get('published_at', '-')}"
        + ("" if manifest.get("raw_data", True) else " · report only (no raw recordings)")
    )
    signals = session["report"].get("signals") or {}
    reps = _rep_frame(session["report"], signals.get("t0"))
    key = f"{patient}_{session_id}"
    _render_chat(patient, session_id)
    tabs = st.tabs(["AI progress report", "Overview", "Signals", "Repetitions", "Fatigue",
                    "Review & patient log", "Full report"])
    with tabs[0]:
        _render_progress(patient, session, session_id, key)
    with tabs[1]:
        _render_overview(session, trend)
    with tabs[2]:
        _render_signals(session, reps)
    with tabs[3]:
        _render_reps(session, reps)
    with tabs[4]:
        _render_fatigue(session)
    with tabs[5]:
        _render_review(session, trend, patient, session_id, key)
    with tabs[6]:
        html_path = session["path"] / "report.html"
        if html_path.exists():
            # The app's report (all text HTML-escaped) plus the record and AI summary, which
            # full_report_html escapes too: no raw model output reaches the iframe.
            full = kr.full_report_html(html_path.read_text(encoding="utf-8"), _risk_of(patient, session_id),
                                       load_report(session["path"]))
            st.caption("Session report from the patient app, with the rehabilitation record and the AI "
                       "progress summary added under its summary.")
            st.iframe(full, height=1600)
            st.download_button("Download full report (HTML)", full.encode("utf-8"),
                               f"{patient}_{session_id}_full_report.html", "text/html", key=f"dl_html_{key}")
        else:
            st.info("No HTML report in this session folder.")
        st.download_button(
            "Download report.json", json.dumps(session["report"], indent=2).encode("utf-8"),
            f"{patient}_{session_id}_report.json", "application/json", key=f"dl_{key}",
        )
