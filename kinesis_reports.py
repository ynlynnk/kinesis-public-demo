"""File-based store for sessions handed over by the KINESIS patient app.

The patient app writes each finished session (also demo "Skip to results") to

    report/<patient>/se<session log>/
        manifest.json  report.json  report.html
        emg.csv  imu.csv  reps.csv  events.csv  calibration.json  meta.json  config.yaml

This dashboard only adds files next to them:

    review.json        the specialist's assessment (status, clinical notes)
    patient_log.json   the patient-facing explanation, and patient_log.md once released

The folder is git-ignored: it holds patient data.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd


REPORT_ROOT = Path(
    os.getenv(
        "KINESIS_REPORT_DIR",
        str(Path(__file__).resolve().parent / "report"),
    )
)
SESSION_PATTERN = re.compile(r"^se[0-9A-Za-z_-]{1,64}$")
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
REVIEW_STATUSES = {
    "good_to_continue": ("green", "Good to continue"),
    "needs_attention": ("yellow", "Continue with changes"),
    "follow_up": ("red", "Physiotherapist follow-up needed"),
}


def _check(name: str, pattern: re.Pattern, label: str) -> str:
    if not pattern.fullmatch(str(name)):
        raise ValueError(f"Invalid {label}: {name!r}")
    return str(name)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def session_dir(patient: str, session_id: str, root: Path | None = None) -> Path:
    return (
        (root or REPORT_ROOT)
        / _check(patient, NAME_PATTERN, "patient")
        / _check(session_id, SESSION_PATTERN, "session")
    )


def list_patients(root: Path | None = None) -> list[str]:
    base = root or REPORT_ROOT
    if not base.exists():
        return []
    return sorted(
        p.name
        for p in base.iterdir()
        if p.is_dir() and NAME_PATTERN.fullmatch(p.name)
        and any(_is_session(s) for s in p.iterdir())
    )


def _is_session(path: Path) -> bool:
    return (
        path.is_dir()
        and SESSION_PATTERN.fullmatch(path.name) is not None
        and (path / "report.json").exists()
    )


def list_sessions(patient: str, root: Path | None = None) -> list[dict[str, Any]]:
    """Sessions of one patient, oldest first, with their manifest and review status."""

    folder = (root or REPORT_ROOT) / _check(patient, NAME_PATTERN, "patient")
    if not folder.exists():
        return []
    out = []
    for path in sorted(folder.iterdir()):
        if not _is_session(path):
            continue
        manifest = _read_json(path / "manifest.json") or {}
        review = _read_json(path / "review.json") or {}
        log = _read_json(path / "patient_log.json") or {}
        out.append(
            {
                "patient": patient,
                "session_id": path.name,
                "exercise": manifest.get("exercise", "Unknown exercise"),
                "published_at": manifest.get("published_at"),
                "totals": manifest.get("totals", {}),
                "review_status": (
                    "released" if log.get("released")
                    else "reviewed" if review
                    else manifest.get("review_status", "awaiting_review")
                ),
                "path": path,
            }
        )
    return out


def load_session(patient: str, session_id: str, root: Path | None = None) -> dict[str, Any]:
    path = session_dir(patient, session_id, root)
    report = _read_json(path / "report.json")
    if report is None:
        raise FileNotFoundError(f"No readable report.json in {path}")
    return {
        "path": path,
        "manifest": _read_json(path / "manifest.json") or {},
        "report": report,
        "review": _read_json(path / "review.json"),
        "patient_log": _read_json(path / "patient_log.json"),
        # Written by the patient app when the patient opens the released log.
        "patient_read": _read_json(path / "patient_read.json"),
        "has_html": (path / "report.html").exists(),
    }


def raw_semg_envelope(path: Path, t0: float | None = None, t1: float | None = None,
                      bins: int = 1200) -> pd.DataFrame | None:
    """Raw sEMG (ADC counts) as min/max per time bin, so every spike survives.

    t0/t1 (device clock) crop to the exercise part of the session; `t_s` is then seconds
    since t0, the same axis as report.json `signals`."""

    csv = path / "emg.csv"
    if not csv.exists():
        return None
    data = pd.read_csv(csv)
    if data.empty or not {"t", "adc"} <= set(data.columns):
        return None
    if t0 is not None:
        data = data[data["t"] >= t0]
    if t1 is not None:
        data = data[data["t"] <= t1]
    if len(data) < 2:
        return None
    t = data["t"].to_numpy(float)
    origin = t0 if t0 is not None else t[0]
    edges = np.linspace(t[0], t[-1], bins + 1)
    idx = np.clip(np.searchsorted(edges, t, side="right") - 1, 0, bins - 1)
    grouped = pd.DataFrame({"bin": idx, "adc": data["adc"].to_numpy(float)}).groupby("bin")["adc"]
    centres = (edges[:-1] + edges[1:]) / 2 - origin
    out = pd.DataFrame({"min": grouped.min(), "max": grouped.max()})
    out["t_s"] = centres[out.index.to_numpy()]
    return out.reset_index(drop=True)


def session_trend(patient: str, root: Path | None = None) -> pd.DataFrame:
    """One row per session of this patient, for progress over time."""

    rows = []
    for item in list_sessions(patient, root):
        report = _read_json(item["path"] / "report.json") or {}
        totals = report.get("totals", {})
        fatigue = report.get("fatigue", {})
        rows.append(
            {
                "session_id": item["session_id"],
                "exercise": item["exercise"],
                "attempts": totals.get("attempts"),
                "good_reps": totals.get("good_reps"),
                "good_rate_percent": (
                    round(100 * totals["good_rate"], 1)
                    if totals.get("good_rate") is not None else None
                ),
                "avg_peak_angle_deg": totals.get("avg_peak_angle_deg"),
                "fatigue_score": fatigue.get("final_score"),
                "most_common_fault": totals.get("most_common_fault"),
            }
        )
    return pd.DataFrame(rows)


def save_review(patient: str, session_id: str, status: str, notes: str, reviewer: str,
                root: Path | None = None) -> dict[str, Any]:
    if status not in REVIEW_STATUSES:
        raise ValueError(f"Unknown review status: {status}")
    review = {
        "status": status,
        "status_label": REVIEW_STATUSES[status][1],
        "level": REVIEW_STATUSES[status][0],
        "notes": notes.strip(),
        "reviewer": reviewer.strip() or "Specialist",
        "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    _write_json(session_dir(patient, session_id, root) / "review.json", review)
    return review


def save_patient_log(patient: str, session_id: str, log: dict[str, Any], released: bool,
                     root: Path | None = None) -> dict[str, Any]:
    """Store the patient-facing log; releasing it also writes a readable patient_log.md."""

    path = session_dir(patient, session_id, root)
    log = dict(log, released=bool(released))
    log["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if released:
        log["released_at"] = log["saved_at"]
        (path / "patient_log.md").write_text(patient_log_markdown(log), encoding="utf-8")
    _write_json(path / "patient_log.json", log)
    return log


def patient_log_markdown(log: dict[str, Any]) -> str:
    lines = [f"# {log.get('title', 'Your session')}", "", f"**{log.get('status_label', '')}**", ""]
    for key, heading in (
        ("summary", "How the session went"),
        ("movement", "Your movement"),
        ("effort", "Muscle effort and tiredness"),
        ("progress", "Your progress"),
    ):
        if log.get(key):
            lines += [f"## {heading}", str(log[key]), ""]
    if log.get("tips"):
        lines += ["## For next time", *[f"- {tip}" for tip in log["tips"]], ""]
    if log.get("specialist_note"):
        lines += ["## Note from your physiotherapist", str(log["specialist_note"]), ""]
    lines += ["---", str(log.get("safety_note", "")), ""]
    return "\n".join(lines)


# ------------------------------------------------------------------ full report with history and AI summary

_RISK_COLOUR = {"HIGH RISK": "var(--bad)", "AT RISK": "var(--amber)", "MONITOR": "var(--accent)",
                "NO FLAGS": "var(--good)"}
_FULL_REPORT_CSS = """<style>
.kin-risk { border-left:5px solid var(--risk); background:color-mix(in srgb, var(--risk) 10%, var(--card)); }
.kin-risk b.tag { color:var(--risk); font-size:18px; }
tr.kin-cur td { font-weight:600; background:color-mix(in srgb, var(--accent) 9%, transparent); }
.kin-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:12px; margin-top:12px; }
.kin-grid h3 { margin:0 0 6px; } .kin-grid ul { margin:0; padding-left:18px; }
</style>"""
MAX_RECORD_ROWS = 12


def _cell(value: Any, unit: str = "") -> str:
    return "–" if value is None else f"{escape(str(value))}{unit}"


_DIRECTION_COLOUR = {"improving": "var(--good)", "declining": "var(--bad)"}


def _metric_chart(m: dict[str, Any]) -> str:
    """Small line chart of one metric across sessions (inline SVG, current session marked)."""

    pts = [p for p in m.get("by_session", []) if p.get("value") is not None]
    if not pts:
        return ""
    w, h, left, right, top, bottom = 300, 130, 34, 10, 10, 22
    values = [float(p["value"]) for p in pts]
    lo, hi = min(values), max(values)
    # At least 20% of the value as y-range, so a 1° wobble around 93° does not look like a swing.
    widen = max(0.0, max(abs(hi), 5.0) * 0.2 - (hi - lo)) / 2
    lo, hi = lo - widen, hi + widen
    pad = (hi - lo) * 0.1
    lo, hi = lo - pad, hi + pad

    def xy(i: int, v: float) -> tuple[float, float]:
        x = left + (w - left - right) * (i / (len(pts) - 1) if len(pts) > 1 else 0.5)
        return round(x, 1), round(top + (h - top - bottom) * (hi - v) / (hi - lo), 1)

    coords = [xy(i, v) for i, v in enumerate(values)]
    svg = [f"<svg viewBox='0 0 {w} {h}' class='chart' role='img' aria-label='{escape(m['label'])} by session'>",
           f"<text x='{left - 4}' y='{top + 4}' class='ax' text-anchor='end'>{hi - pad:.0f}</text>",
           f"<text x='{left - 4}' y='{h - bottom}' class='ax' text-anchor='end'>{lo + pad:.0f}</text>",
           f"<line x1='{left}' y1='{h - bottom}' x2='{w - right}' y2='{h - bottom}' class='ref'/>"]
    if len(coords) > 1:
        svg.append("<polyline class='line' points='" + " ".join(f"{x},{y}" for x, y in coords) + "'/>")
    for (x, y), p in zip(coords, pts):
        last = p is pts[-1]
        svg.append(f"<circle cx='{x}' cy='{y}' r='{5 if last else 3.5}' class='dot'"
                   + (" style='fill:var(--text)'" if last else "") + f"><title>{escape(str(p['session']))}: "
                   f"{escape(str(p['value']))}</title></circle>")
        svg.append(f"<text x='{x}' y='{h - 6}' class='ax' text-anchor='middle'>{escape(str(p['session']))}</text>")
    svg.append("</svg>")
    direction = str(m.get("direction_recent", ""))
    change = m.get("change_vs_recent_sessions")
    since = m.get("change_vs_start")
    return (f'<div class="card"><h3>{escape(m["label"])}</h3>{"".join(svg)}<div class="muted">'
            f'Now {escape(str(m.get("current")))} · <b style="color:{_DIRECTION_COLOUR.get(direction, "var(--muted)")}">'
            f'{escape(direction)}</b>'
            + (f" ({change:+} vs recent sessions" + (f", {since:+} since start" if since is not None else "") + ")"
               if change is not None else "")
            + "</div></div>")


def full_report_html(report_html: str, truth: dict[str, Any], ai_report: dict[str, Any] | None) -> str:
    """The app's session report with two sections added under its summary tiles: a short
    record of every session up to this one, and the key points of the AI progress summary.

    Every value is HTML-escaped (the AI's text included), so the result is safe to embed."""

    ov, risk, history = truth["overview"], truth["risk"], truth["history"]
    span = ov.get("rehab_span_days")
    parts = [_FULL_REPORT_CSS, "<h2>Rehabilitation record</h2>", '<div class="muted">' + " · ".join(filter(None, [
        f"Session {ov['session_number']} of {ov['sessions_to_date']}",
        f"rehab day {ov['rehab_day']}" if ov.get("rehab_day") is not None else "",
        f"over {ov['rehab_span_weeks']} weeks ({span} days)" if span else "",
        f"{ov['sessions_per_week']} sessions/week" if ov.get("sessions_per_week") else "",
        f"{ov['same_exercise_sessions']} session(s) of {ov['exercise']} with repetitions",
        "rule trajectory: " + str(ov.get("rule_trajectory", "")).replace("_", " "),
    ])) + "</div>"]
    rows = history[-MAX_RECORD_ROWS:]
    table = ["<tr><th>Session</th><th>Date</th><th>Day</th><th>Exercise</th><th>Good reps</th>"
             "<th>Peak angle</th><th>Fatigue</th><th>Top fault</th></tr>"]
    for h in rows:
        when = h.get("when")
        good = (f"{h['good_reps']}/{h['attempts']} ({h['good_rate_pct']:.0f}%)"
                if h.get("attempts") and h.get("good_rate_pct") is not None else "no reps")
        this = h is history[-1]
        table.append(
            ('<tr class="kin-cur">' if this else "<tr>")
            + f"<td>{escape(h['label'])}{' (this)' if this else ''}</td>"
            f"<td>{escape(when.strftime('%d %b %Y %H:%M')) if when else '–'}</td><td>{_cell(h.get('day'))}</td>"
            f"<td>{_cell(h.get('exercise'))}</td><td>{escape(good)}</td><td>{_cell(h.get('avg_peak_deg'), '°')}</td>"
            f"<td>{_cell(h.get('fatigue_score'))}</td><td>{_cell(h.get('top_fault'))}</td></tr>"
        )
    parts.append('<div class="card"><table>' + "".join(table) + "</table></div>")
    if len(history) > len(rows):
        parts.append(f'<div class="muted">{len(history) - len(rows)} earlier session(s) not shown.</div>')
    charts = [c for c in (_metric_chart(m) for m in ov.get("metrics", {}).values()) if c]
    if charts:
        parts += ["<h2>Progress by metric</h2>",
                  f'<div class="muted">Sessions of {escape(str(ov["exercise"]))} with repetitions, up to this one '
                  "(dark dot). Direction compares this session with the average of the previous sessions.</div>",
                  '<div class="kin-grid">' + "".join(charts) + "</div>"]

    tag = risk.get("risk_tag", "NO FLAGS")
    parts += ["<h2>AI progress summary</h2>",
              f'<div class="card kin-risk" style="--risk:{_RISK_COLOUR.get(tag, "var(--muted)")}">'
              f'<b class="tag">{escape(tag)}</b> <span class="muted">rule-based risk tag</span>'
              + ("<ul>" + "".join(f"<li><b>{escape(f['title'])}</b> ({escape(f['level'])}): {escape(f['evidence'])}</li>"
                                  for f in risk.get("flags", [])) + "</ul>" if risk.get("flags")
                 else "<div>No abnormal flags.</div>") + "</div>"]
    ai = (ai_report or {}).get("ai")
    if ai:
        parts.append(
            f'<div class="card" style="margin-top:12px"><b>{escape(str(ai.get("headline", "")))}</b>'
            f'<div class="muted">Trajectory: {escape(str(ai.get("trajectory", "")).replace("_", " "))} · '
            f'confidence: {escape(str(ai.get("confidence", "")))}</div>'
            f'<p>{escape(str(ai.get("reasoning", "")))}</p></div>'
        )
        metrics = "".join(f"<li><b>{escape(str(m.get('metric', '')))}</b>: {escape(str(m.get('direction', '')))}"
                          f" ({escape(str(m.get('evidence', '')))})</li>" for m in ai.get("metric_progress", [])[:5])
        lists = [("By metric", metrics),
                 ("Fatigue", f"<li>{escape(str(ai['fatigue_assessment']))}</li>" if ai.get("fatigue_assessment") else ""),
                 ("Concerns", "".join(f"<li>{escape(str(c))}</li>" for c in ai.get("additional_concerns", [])[:4])),
                 ("Suggested review", "".join(f"<li>{escape(str(a))}</li>"
                                               for a in ai.get("suggested_review_actions", [])[:4]))]
        parts.append('<div class="kin-grid">' + "".join(
            f'<div class="card"><h3>{title}</h3><ul>{items}</ul></div>' for title, items in lists if items) + "</div>")
        parts.append(f'<p class="muted">{escape(str(ai_report.get("generation_mode", "")))} · generated '
                     f'{escape(str(ai_report.get("generated_at", "")))}</p>')
    else:
        parts.append('<p class="muted">The AI summary has not been generated for this session yet '
                     '(AI progress report tab, "Generate AI report").</p>')
    parts.append('<div class="note">Decision support only: the AI summary and risk tag can be wrong. '
                 "The diagnosis and clinical decisions rest with the reviewing specialist.</div>")

    block = "\n".join(parts) + "\n"
    at = report_html.find("<h2>")
    if at < 0:
        at = report_html.find("<main>") + len("<main>") if "<main>" in report_html else len(report_html)
    return report_html[:at] + block + report_html[at:]
