import asyncio
from html import escape
import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from mcp import Client

from agent.claude_summary import model2_status
from forearm_fatigue_runtime import (
    CLASS_NAMES as FATIGUE_CLASS_NAMES,
    load_forearm_fatigue_model,
    predict_normalized_fatigue_window,
)
from forearm_movement_runtime import (
    CLASS_NAMES as FOREARM_CLASS_NAMES,
    load_forearm_movement_model,
    predict_normalized_forearm_window,
)
from model1_signal_pipeline import (
    DEVICE_IMU_RATE_HZ,
    DEVICE_SEMG_RATE_HZ,
    EMA_ALPHA,
    MAX_SESSION_SECONDS,
    MODEL_SEMG_RATE_HZ,
    UPLOAD_INTERVAL_SECONDS,
)
from live_model1_runtime import (
    RealtimeInputNotReady,
    predict_live_model1_session,
)
from quadriceps_runtime import (
    load_quadriceps_fatigue_model,
    load_quadriceps_movement_model,
    predict_quadriceps_fatigue_window,
    predict_quadriceps_movement_window,
)
from realtime_input_contract import assess_realtime_sensor_frame
from session_packet import (
    build_session_packet,
    packet_to_json,
    save_session_packet,
)
from session_report import (
    build_standard_session_report,
    report_to_json,
    report_to_markdown,
)
from sensor_feed import (
    calculate_feed_age_seconds,
    load_sensor_frame,
    prepare_sensor_frame,
    summarize_sensor_frame,
)
from synthetic_live_demo import (
    BUFFER_SAMPLES as SYNTHETIC_BUFFER_SAMPLES,
    SAMPLE_RATE_HZ as SYNTHETIC_SAMPLE_RATE_HZ,
    advance_demo_buffer,
    create_demo_buffer,
)


MCP_SERVER_URL = os.getenv(
    "KINESIS_MCP_URL",
    "http://127.0.0.1:8000/mcp",
)
PUBLIC_DEMO_MODE = os.getenv("KINESIS_PUBLIC_DEMO", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
SENSOR_FEED_URL = os.getenv("KINESIS_SENSOR_FEED_URL", "").strip()
SENSOR_FEED_FILE = Path(__file__).resolve().parent / "data" / "nexus_testing.csv"
SESSION_PACKET_ROOT = (
    Path(__file__).resolve().parent / "data" / "session_packets"
)
NEXUS_MODEL_DIRECTORY = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "models"
)
FOREARM_MOVEMENT_MODEL_FILE = (
    NEXUS_MODEL_DIRECTORY / "kinesis_forearm_movement_cnn.pt"
)
FOREARM_FATIGUE_MODEL_FILE = (
    NEXUS_MODEL_DIRECTORY / "kinesis_forearm_fatigue_cnn.pt"
)
QUADRICEPS_MOVEMENT_MODEL_FILE = (
    NEXUS_MODEL_DIRECTORY / "kinesis_quadriceps_movement_cnn.pt"
)
QUADRICEPS_FATIGUE_MODEL_FILE = (
    NEXUS_MODEL_DIRECTORY / "kinesis_quadriceps_fatigue_cnn.pt"
)
MUSCLE_GROUPS = {
    "Forearm": {
        "exercise": "Pronation and supination",
        "movement_model_file": "kinesis_forearm_movement_cnn.pt",
        "fatigue_model_file": "kinesis_forearm_fatigue_cnn.pt",
    },
    "Quadriceps": {
        "exercise": "Knee extension",
        "movement_model_file": "kinesis_quadriceps_movement_cnn.pt",
        "fatigue_model_file": "kinesis_quadriceps_fatigue_cnn.pt",
    },
}
SENSOR_REFRESH_SECONDS = max(
    2,
    int(os.getenv("KINESIS_SENSOR_REFRESH_SECONDS", "5")),
)
SENSOR_TIMEZONE = os.getenv(
    "KINESIS_SENSOR_TIMEZONE",
    "Asia/Kuala_Lumpur",
).strip()

st.set_page_config(
    page_title="KINESIS Rehabilitation AI",
    page_icon="K",
    layout="wide",
)

st.markdown(
    """
    <style>
    div[data-testid="stMetric"] {
        min-width: 0;
    }
    div[data-testid="stMetricValue"],
    div[data-testid="stMetricValue"] > div {
        font-size: clamp(1.55rem, 2.15vw, 2.6rem) !important;
        line-height: 1.15 !important;
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
        overflow-wrap: anywhere !important;
    }
    div[data-testid="stMetricDelta"],
    div[data-testid="stMetricDelta"] > div {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: clip !important;
    }
    .role-status-card {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 1rem 1.2rem;
        margin: 0.35rem 0 1rem;
        border: 1px solid;
        border-radius: 1rem;
    }
    .role-status-icon {
        display: inline-flex;
        flex: 0 0 3rem;
        width: 3rem;
        height: 3rem;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        color: #ffffff;
        font-size: 1.65rem;
        font-weight: 800;
    }
    .role-status-copy strong {
        display: block;
        margin-bottom: 0.2rem;
        font-size: 1.15rem;
    }
    .role-status-copy span {
        line-height: 1.45;
    }
    .role-status-green {
        background: rgba(22, 163, 74, 0.12);
        border-color: rgba(34, 197, 94, 0.7);
    }
    .role-status-green .role-status-icon { background: #16a34a; }
    .role-status-yellow {
        background: rgba(202, 138, 4, 0.14);
        border-color: rgba(234, 179, 8, 0.75);
    }
    .role-status-yellow .role-status-icon { background: #ca8a04; }
    .role-status-red {
        background: rgba(220, 38, 38, 0.12);
        border-color: rgba(239, 68, 68, 0.75);
    }
    .role-status-red .role-status-icon { background: #dc2626; }
    @media (max-width: 900px) {
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] > div {
            font-size: 1.65rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ROLE_NAMES = {
    "Physiotherapist": "physiotherapist",
    "Care-team clinician": "clinician",
    "Patient": "patient",
}


def question_modes_for_role(role: str) -> list[str]:
    """Return the assistant modes that the selected role may open."""

    modes = ["A selected session"]
    if role == "patient":
        modes.append("General rehabilitation")
    return modes


async def build_session_context(
    patient_id: str,
    session_id: str,
    user_query: str,
) -> dict:
    """Load the optional knowledge-summary path only when it is requested."""

    from agent.agent import build_session_context as build_context

    return await build_context(patient_id, session_id, user_query)


async def build_role_session_context(
    patient_id: str,
    session_id: str,
    role: str,
    requesting_patient_id: str | None,
    user_query: str,
) -> dict:
    """Build a role-limited Model 2 answer through the MCP workflow."""

    from agent.agent import build_role_session_context as build_context

    return await build_context(
        patient_id,
        session_id,
        role,
        requesting_patient_id,
        user_query,
    )


def build_general_question_context(
    role: str,
    user_query: str,
) -> dict:
    """Build a Model 2 answer without requesting a patient session."""

    from agent.agent import build_general_question_context as build_context

    return build_context(role, user_query)


async def call_mcp_tool(tool_name: str, arguments: dict) -> dict:
    """Send a request to the separately running MCP backend."""
    async with Client(MCP_SERVER_URL, raise_exceptions=True) as client:
        result = await client.call_tool(tool_name, arguments)
        if result.is_error or result.structured_content is None:
            raise RuntimeError(str(result.content))
        return result.structured_content


def run_mcp_tool(tool_name: str, arguments: dict) -> dict:
    return asyncio.run(call_mcp_tool(tool_name, arguments))


def load_session_catalog(
    role: str,
    requesting_patient_id: str | None,
) -> list[dict]:
    result = run_mcp_tool(
        "list_available_sessions",
        {
            "requesting_role": role,
            "requesting_patient_id": requesting_patient_id,
        },
    )
    return result["sessions"]


def retrieve_role_result(
    role: str,
    patient_id: str,
    session_id: str,
    requesting_patient_id: str | None,
) -> tuple[str, dict]:
    if role == "physiotherapist":
        tool_name = "get_model1_outputs"
        arguments = {
            "patient_id": patient_id,
            "session_id": session_id,
            "requesting_role": role,
        }
    elif role == "clinician":
        tool_name = "get_clinician_summary"
        arguments = {
            "patient_id": patient_id,
            "session_id": session_id,
            "requesting_role": role,
        }
    else:
        tool_name = "get_patient_summary"
        arguments = {
            "patient_id": patient_id,
            "session_id": session_id,
            "requesting_role": role,
            "requesting_patient_id": requesting_patient_id,
        }
    return tool_name, run_mcp_tool(tool_name, arguments)


def resolve_review_choice(
    choice: str,
    predicted_label: str,
    opposite_label: str,
) -> tuple[str, str | None]:
    if choice.startswith("Confirm"):
        return "confirmed", predicted_label
    if choice.startswith("Correct"):
        return "corrected", opposite_label
    return "uncertain", None


def show_workflow(
    role: str,
    patient_id: str,
    session_id: str,
    review_status: str,
) -> None:
    with st.expander("How this information was prepared", expanded=False):
        st.markdown(
            f"""
1. Your selected role determines which information you can access.
2. Information is requested for patient {patient_id}, session {session_id}.
3. The patient and session identifiers are checked.
4. Muscle activity and movement findings are considered together.
5. Other available sessions are requested for comparison.
6. The recorded professional review status is shown: {review_status}.
7. The information is presented for your role.
"""
        )


def show_service_error(message: str, error: Exception) -> None:
    st.error(message)
    with st.expander("Technical error details", expanded=False):
        st.code(str(error), language=None)


def render_browser_connection_indicator() -> None:
    st.iframe(
        """
<div id="connection-status" role="status" aria-live="polite">
  <span id="connection-dot"></span>
  <span id="connection-text">Checking internet connection</span>
</div>
<style>
  body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: transparent;
  }
  #connection-status {
    box-sizing: border-box;
    display: flex;
    align-items: center;
    gap: 10px;
    width: 100%;
    padding: 11px 14px;
    border: 1px solid #d1d5db;
    border-radius: 10px;
    color: #1f2937;
    background: #f8fafc;
    font-size: 15px;
  }
  #connection-dot {
    width: 11px;
    height: 11px;
    flex: 0 0 11px;
    border-radius: 50%;
    background: #64748b;
  }
  #connection-status.online {
    border-color: #86efac;
    background: #f0fdf4;
  }
  #connection-status.online #connection-dot {
    background: #16a34a;
  }
  #connection-status.offline {
    border-color: #fdba74;
    background: #fff7ed;
  }
  #connection-status.offline #connection-dot {
    background: #ea580c;
  }
</style>
<script>
  const statusBox = document.getElementById("connection-status");
  const statusText = document.getElementById("connection-text");
  let wasOffline = false;

  function updateConnectionStatus() {
    const isOnline = navigator.onLine;
    statusBox.className = isOnline ? "online" : "offline";

    if (!isOnline) {
      wasOffline = true;
      statusText.textContent =
        "Offline - new sessions cannot be loaded and reviews cannot be saved.";
    } else if (wasOffline) {
      statusText.textContent =
        "Back online - reload the session to retrieve and sync the latest data.";
    } else {
      statusText.textContent =
        "Online - live session data can be loaded and saved.";
    }
  }

  window.addEventListener("online", updateConnectionStatus);
  window.addEventListener("offline", updateConnectionStatus);
  updateConnectionStatus();
</script>
        """,
        height=58,
    )


@st.cache_data(ttl=SENSOR_REFRESH_SECONDS, show_spinner=False)
def load_dashboard_sensor_feed(
    source_url: str,
    local_path: str,
) -> tuple[pd.DataFrame, str, bool]:
    frame, source_name, is_live = load_sensor_frame(
        source_url,
        local_path,
    )
    return prepare_sensor_frame(frame), source_name, is_live


def render_feed_status(
    summary: dict,
    is_live: bool,
) -> None:
    if not is_live:
        st.info(
            "Showing the Nexus workbook snapshot you supplied. Configure a "
            "live feed URL to replace this snapshot automatically."
        )
        return

    try:
        age_seconds = calculate_feed_age_seconds(
            summary["latest_timestamp"],
            SENSOR_TIMEZONE,
        )
    except Exception:
        st.warning(
            "The live feed loaded, but its timestamp timezone could not be "
            "compared with the server clock."
        )
        return

    expected_interval = summary["median_interval_seconds"] or 5
    fresh_limit = max(15, expected_interval * 3)
    if -60 <= age_seconds <= fresh_limit:
        st.success(
            f"Live feed connected. Latest row received "
            f"{max(age_seconds, 0):.0f} seconds ago."
        )
    elif age_seconds < -60:
        st.warning(
            "Live feed connected, but the device timestamp is ahead of the "
            "dashboard clock. Check the firmware timezone."
        )
    else:
        st.warning(
            f"Feed connected, but the latest row is {age_seconds:.0f} "
            "seconds old. The device or internet connection may be delayed."
        )


def render_sensor_charts(sensor_data: pd.DataFrame) -> None:
    st.markdown("### Latest recording")
    latest_session = sensor_data["Session"].iloc[-1]
    latest_recording = sensor_data[
        sensor_data["Session"] == latest_session
    ].sort_values("Seq", kind="stable")
    st.caption(
        "Only the latest recording group is plotted. Its samples are shown "
        "by sequence number so separate dates are not joined into a false "
        "continuous signal."
    )
    emg_chart = (
        alt.Chart(latest_recording)
        .mark_line(point=True, color="#7C3AED")
        .encode(
            x=alt.X(
                "Seq:Q",
                title="Sample sequence",
                axis=alt.Axis(tickMinStep=1),
            ),
            y=alt.Y(
                "sEMG raw:Q",
                title="Raw sEMG count",
                scale=alt.Scale(zero=False),
            ),
            tooltip=[
                alt.Tooltip("Timestamp:T", title="Recorded"),
                alt.Tooltip("Seq:Q", title="Sequence"),
                alt.Tooltip("sEMG raw:Q", title="Raw sEMG"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(emg_chart, width="stretch")

    orientation_data = latest_recording.melt(
        id_vars=["Timestamp", "Session", "Seq"],
        value_vars=["Roll (deg)", "Pitch (deg)", "Yaw (deg)"],
        var_name="Orientation",
        value_name="Degrees",
    )
    orientation_chart = (
        alt.Chart(orientation_data)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Seq:Q",
                title="Sample sequence",
                axis=alt.Axis(tickMinStep=1),
            ),
            y=alt.Y("Degrees:Q", title="Orientation (degrees)"),
            color=alt.Color(
                "Orientation:N",
                scale=alt.Scale(
                    domain=["Roll (deg)", "Pitch (deg)", "Yaw (deg)"],
                    range=["#0284C7", "#16A34A", "#EA580C"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Timestamp:T", title="Recorded"),
                alt.Tooltip("Orientation:N"),
                alt.Tooltip("Degrees:Q", format=".2f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(orientation_chart, width="stretch")


def render_real_forearm_cnn_verification() -> None:
    """Show compact training evidence for both real-data Model 1 heads."""

    st.markdown("### Model 1 training status")
    st.caption(
        "The public datasets below were used to train and test the two CNN "
        "heads. Public participant recordings are not displayed on this live "
        "dashboard; future predictions will use the connected KINESIS feed."
    )
    st.markdown("#### Movement head — forearm direction")
    if not FOREARM_MOVEMENT_MODEL_FILE.is_file():
        st.warning("The real-data forearm CNN checkpoint is not installed.")
        return

    try:
        _, checkpoint = load_forearm_movement_model(
            str(FOREARM_MOVEMENT_MODEL_FILE)
        )
    except Exception as error:
        st.error("The real-data movement checkpoint could not be loaded.")
        with st.expander("Verification error details", expanded=False):
            st.exception(error)
        return

    metrics = checkpoint["test_metrics"]
    st.success(
        "Training and inference completed on labelled recordings from real "
        "participants, with the final test performed on participants that "
        "were not used for training or model tuning."
    )
    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Held-out accuracy",
        f"{metrics['accuracy']:.1%}",
    )
    metric_columns[1].metric(
        "Macro F1",
        f"{metrics['macro_f1']:.1%}",
    )
    metric_columns[2].metric(
        "Unseen participants",
        metrics["held_out_subject_count"],
    )
    metric_columns[3].metric(
        "Held-out windows",
        f"{metrics['sample_count']:,}",
    )

    st.warning(
        "This proves that the CNN training and inference pipeline works on "
        "real public forearm sEMG. The "
        f"{metrics['accuracy']:.1%} held-out accuracy is above the "
        "33.3% three-class chance level, but it is not high enough for a "
        "clinical or production claim."
    )

    with st.expander("Training and evaluation details", expanded=False):
        st.write(
            "Dataset: **Multi-Channel Electromyography Signal Acquisition "
            "of Forearm** (50 real participants, 200 Hz sEMG)."
        )
        st.markdown(
            "Source: [Mendeley Data — DOI 10.17632/p77jn92bzg.1]"
            "(https://data.mendeley.com/datasets/p77jn92bzg/1)"
        )
        st.write("Licence: **CC BY 4.0**")
        st.write(
            "Participant split: **35 training / 7 validation / 8 test**. "
            "No test participant appears in training or validation."
        )
        recall_table = pd.DataFrame({
            "Movement": list(metrics["per_class_recall"]),
            "Held-out recall": [
                f"{value:.1%}"
                for value in metrics["per_class_recall"].values()
            ],
        })
        st.dataframe(recall_table, hide_index=True, width="stretch")
        confusion = pd.DataFrame(
            metrics["confusion_matrix"],
            index=[f"Actual {name}" for name in FOREARM_CLASS_NAMES],
            columns=[f"Predicted {name}" for name in FOREARM_CLASS_NAMES],
        )
        st.write("Held-out confusion matrix")
        st.dataframe(confusion, width="stretch")
        st.caption(
            "The public dataset uses eight Myo-armband electrodes. Each "
            "electrode was treated as an independent one-channel example so "
            "the CNN architecture remains compatible with a one-electrode "
            "design. KINESIS-device validation still requires dense 200 Hz "
            "recordings and a neutral baseline from the actual wearable."
        )

    st.divider()
    st.markdown("#### Fatigue head — sustained-load fatigue pattern")
    if not FOREARM_FATIGUE_MODEL_FILE.is_file():
        st.warning("The real-data fatigue CNN checkpoint is not installed.")
        return

    try:
        _, fatigue_checkpoint = load_forearm_fatigue_model(
            str(FOREARM_FATIGUE_MODEL_FILE)
        )
    except Exception as error:
        st.error("The real-data fatigue checkpoint could not be loaded.")
        with st.expander("Fatigue verification error details", expanded=False):
            st.exception(error)
        return

    fatigue_metrics = fatigue_checkpoint["test_metrics"]
    st.success(
        "The fatigue CNN was trained on real upper-limb sEMG and tested on "
        "three participants who were never used for training or tuning."
    )
    fatigue_metric_columns = st.columns(5)
    fatigue_metric_columns[0].metric(
        "Held-out accuracy", f"{fatigue_metrics['accuracy']:.1%}"
    )
    fatigue_metric_columns[1].metric(
        "Macro F1", f"{fatigue_metrics['macro_f1']:.1%}"
    )
    fatigue_metric_columns[2].metric(
        "ROC AUC", f"{fatigue_metrics['roc_auc']:.1%}"
    )
    fatigue_metric_columns[3].metric(
        "Unseen participants", fatigue_metrics["held_out_subject_count"]
    )
    fatigue_metric_columns[4].metric(
        "Held-out windows", f"{fatigue_metrics['sample_count']:,}"
    )
    st.warning(
        "The fatigue score is the CNN's probability of a late, fatigue-like "
        "sEMG phase during a controlled 120-second, 6 kg hold. It is a real "
        "model output, but it is not a clinical severity score."
    )

    with st.expander("Fatigue training and evaluation details", expanded=False):
        st.write(
            "Dataset: **Multi-channel Surface EMG Dataset for Fatigue "
            "analysis** (15 real participants, eight sEMG channels, 200 Hz)."
        )
        st.markdown(
            "Source: [Zenodo — DOI 10.5281/zenodo.5189275]"
            "(https://zenodo.org/records/5189275)"
        )
        st.write("Licence: **CC BY 4.0**")
        st.write(
            "Protocol: participants held a 6 kg load with the elbow at 90°. "
            "Windows from 5–35 seconds were labelled low-fatigue phase; "
            "windows from 90–120 seconds were labelled fatigue-like phase."
        )
        st.write(
            "Participant split: **10 training / 2 validation / 3 test**. "
            "No test participant appears in training or validation."
        )
        fatigue_recall = pd.DataFrame({
            "Phase": list(fatigue_metrics["per_class_recall"]),
            "Held-out recall": [
                f"{value:.1%}"
                for value in fatigue_metrics["per_class_recall"].values()
            ],
        })
        st.dataframe(fatigue_recall, hide_index=True, width="stretch")
        fatigue_confusion = pd.DataFrame(
            fatigue_metrics["confusion_matrix"],
            index=[f"Actual {name}" for name in FATIGUE_CLASS_NAMES],
            columns=[f"Predicted {name}" for name in FATIGUE_CLASS_NAMES],
        )
        st.write("Held-out confusion matrix")
        st.dataframe(fatigue_confusion, width="stretch")
        st.caption(
            "This phase-based model evaluates fatigue-pattern scoring on "
            "real sEMG. Final KINESIS validation still requires dense 200 Hz "
            "recordings from the actual forearm placement plus ground-truth "
            "fatigue labels collected under the intended exercise protocol."
        )

    st.info(
        "The trained CNN checkpoints are retained. Movement and fatigue "
        "predictions will appear only when a compatible continuous KINESIS "
        "recording is connected; public participant examples are not used as "
        "session results."
    )


def render_compact_model1_status(show_muscle_selector: bool = False) -> bool:
    """Show the compact operational status for Model 1."""

    st.markdown("### Model 1 CNN")
    selected_muscle_group = (
        st.session_state.get("model1_muscle_group", "Forearm")
        if show_muscle_selector
        else "Forearm"
    )
    if selected_muscle_group not in MUSCLE_GROUPS:
        selected_muscle_group = "Forearm"

    if selected_muscle_group == "Forearm":
        movement_model_file = FOREARM_MOVEMENT_MODEL_FILE
        fatigue_model_file = FOREARM_FATIGUE_MODEL_FILE
        load_movement = load_forearm_movement_model
        load_fatigue = load_forearm_fatigue_model
    else:
        movement_model_file = QUADRICEPS_MOVEMENT_MODEL_FILE
        fatigue_model_file = QUADRICEPS_FATIGUE_MODEL_FILE
        load_movement = load_quadriceps_movement_model
        load_fatigue = load_quadriceps_fatigue_model

    if not movement_model_file.is_file() or not fatigue_model_file.is_file():
        st.error(
            f"One or both trained {selected_muscle_group.lower()} CNN "
            "checkpoints are missing."
        )
        return False

    try:
        _, movement_checkpoint = load_movement(str(movement_model_file))
        _, fatigue_checkpoint = load_fatigue(str(fatigue_model_file))
        if selected_muscle_group == "Forearm":
            movement_check = predict_normalized_forearm_window(
                np.zeros(256, dtype=np.float32),
                str(movement_model_file),
            )
            fatigue_check = predict_normalized_fatigue_window(
                np.zeros(512, dtype=np.float32),
                str(fatigue_model_file),
            )
        else:
            identity_quaternions = np.zeros((64, 4), dtype=np.float32)
            identity_quaternions[:, 0] = 1.0
            movement_check = predict_quadriceps_movement_window(
                np.zeros(256, dtype=np.float32),
                identity_quaternions,
                str(movement_model_file),
            )
            fatigue_check = predict_quadriceps_fatigue_window(
                np.zeros(512, dtype=np.float32),
                str(fatigue_model_file),
            )
        for result_name, result in (
            ("Movement", movement_check),
            ("Fatigue", fatigue_check),
        ):
            if not np.isclose(
                sum(result["probabilities"].values()),
                1.0,
                atol=1e-5,
            ):
                raise ValueError(
                    f"{result_name} probabilities do not sum to one."
                )
    except Exception as error:
        st.error("A trained CNN checkpoint failed its loading or inference check.")
        with st.expander("Model check error", expanded=False):
            st.exception(error)
        return False

    checkpoint_ready = True
    st.success(
        f"{selected_muscle_group} movement and fatigue checkpoint inference "
        "checks passed."
    )

    model_columns = st.columns(5)
    model_columns[0].metric(
        "CNN status",
        "Ready",
    )
    model_columns[1].metric("Device sEMG", f"{DEVICE_SEMG_RATE_HZ} Hz")
    model_columns[2].metric("IMU stream", f"{DEVICE_IMU_RATE_HZ} Hz")
    model_columns[3].metric("CNN input", f"{MODEL_SEMG_RATE_HZ} Hz")
    if show_muscle_selector:
        with model_columns[4]:
            selected_muscle_group = st.selectbox(
                "Muscle group",
                list(MUSCLE_GROUPS),
                index=list(MUSCLE_GROUPS).index(selected_muscle_group),
                disabled=bool(
                    st.session_state.get("synthetic_demo_running", False)
                ),
                key="model1_muscle_group",
                help="Select the placement before starting the session.",
            )
    else:
        model_columns[4].metric("Validated placement", "Forearm")

    with st.expander("Model evidence and sources", expanded=False):
        if selected_muscle_group == "Forearm":
            st.write(
                "Movement model: 256-sample windows; Neutral, Pronation, "
                "and Supination."
            )
            st.write(
                "Fatigue model: 512-sample windows; Low-fatigue phase and "
                "Fatigue-like phase."
            )
        else:
            st.write(
                "Movement model: a 256-sample rectus-femoris sEMG window "
                "combined with a 64-sample IMU motion window; Correct "
                "extension, Incomplete extension, and Compensatory leg lift."
            )
            st.write(
                "Fatigue model: a 512-sample quadriceps sEMG window; "
                "Low-fatigue phase and Fatigue-like phase."
            )
        st.write(
            f"Signal preparation: {DEVICE_SEMG_RATE_HZ} Hz device sEMG, "
            f"EMA filtering (alpha {EMA_ALPHA}), neutral-baseline scaling, "
            f"then resampling to {MODEL_SEMG_RATE_HZ} Hz for the CNN."
        )
        if selected_muscle_group == "Forearm":
            st.write(
                f"Orientation is monitored independently at {DEVICE_IMU_RATE_HZ} "
                "Hz. The forearm checkpoints make predictions from sEMG only."
            )
            st.markdown(
                "Movement dataset: [Mendeley Data — DOI 10.17632/p77jn92bzg.1]"
                "(https://data.mendeley.com/datasets/p77jn92bzg/1)"
            )
            st.markdown(
                "Fatigue dataset: [Zenodo — DOI 10.5281/zenodo.5189275]"
                "(https://zenodo.org/records/5189275)"
            )
        else:
            st.write(
                "The movement checkpoint was evaluated on held-out KneE-PAD "
                "participants and still requires validation with the final "
                "KINESIS device and rehabilitation protocol."
            )
            st.write(
                "The quadriceps fatigue source contains only three "
                "participants, so this checkpoint needs broader validation."
            )
            st.markdown(
                "Movement dataset: [KneE-PAD — DOI 10.5281/zenodo.12112951]"
                "(https://zenodo.org/records/12112951)"
            )
            st.markdown(
                "Fatigue dataset: [Muscle Fatigue in Cyclist — CC BY 4.0]"
                "(https://huggingface.co/datasets/YominE/Muscle_Fatigue_Cycling)"
            )
        st.caption(
            "Detailed evaluation metrics are maintained in the development "
            "reports rather than displayed in the operational dashboard."
        )
    return True


@st.fragment(run_every="1s")
def render_synthetic_model1_demo() -> None:
    """Run both trained CNN heads on the prepared in-memory sensor stream."""

    running = bool(st.session_state.get("synthetic_demo_running", False))
    muscle_group = st.session_state.get("model1_muscle_group", "Forearm")
    if muscle_group not in MUSCLE_GROUPS:
        muscle_group = "Forearm"
    exercise = MUSCLE_GROUPS[muscle_group]["exercise"]
    if muscle_group == "Forearm":
        movement_model_file = FOREARM_MOVEMENT_MODEL_FILE
        fatigue_model_file = FOREARM_FATIGUE_MODEL_FILE
    else:
        movement_model_file = QUADRICEPS_MOVEMENT_MODEL_FILE
        fatigue_model_file = QUADRICEPS_FATIGUE_MODEL_FILE
    checkpoint_ready = (
        movement_model_file.is_file() and fatigue_model_file.is_file()
    )

    st.markdown(f"### Live {muscle_group.lower()} analysis")
    pattern = "Sustained contraction"

    state = st.session_state.get("synthetic_demo_state")
    if (
        state is None
        or state.get("pattern") != pattern
        or state.get("muscle_group") != muscle_group
    ):
        state = create_demo_buffer(pattern)
        state["muscle_group"] = muscle_group
        st.session_state["synthetic_demo_state"] = state
        st.session_state["synthetic_demo_running"] = False

    running = bool(st.session_state.get("synthetic_demo_running", False))
    status_columns = st.columns(3)
    session_status = (
        "Running" if running else "Stopped"
    ) if checkpoint_ready else "Not ready"
    status_columns[0].metric("Session status", session_status)
    status_columns[1].metric("Muscle group", muscle_group)
    status_columns[2].metric("Exercise", exercise)
    acquisition_columns = st.columns(2)
    acquisition_columns[0].metric(
        "sEMG acquisition",
        f"{DEVICE_SEMG_RATE_HZ} Hz",
    )
    acquisition_columns[1].metric(
        "IMU acquisition",
        f"{DEVICE_IMU_RATE_HZ} Hz",
    )

    start_column, stop_column, note_column = st.columns([1, 1, 4])
    with start_column:
        if st.button(
            "Start",
            type="primary",
            disabled=running or not checkpoint_ready,
            width="stretch",
            key="start_synthetic_demo",
        ):
            st.session_state["synthetic_demo_running"] = True
            st.rerun()
    with stop_column:
        if st.button(
            "Stop",
            disabled=not running,
            width="stretch",
            key="stop_synthetic_demo",
        ):
            st.session_state["synthetic_demo_running"] = False
            st.rerun()
    with note_column:
        st.caption(
            "Input source: Simulator · Session data is not stored"
        )

    if not checkpoint_ready:
        st.warning(
            f"The trained {muscle_group.lower()} movement and fatigue "
            "checkpoints are unavailable. Install both checkpoints before "
            "starting this session."
        )
        return

    if running:
        if state["elapsed_seconds"] >= MAX_SESSION_SECONDS:
            st.session_state["synthetic_demo_running"] = False
            running = False
            st.warning("The session stopped automatically at the 20-minute limit.")
        else:
            state = advance_demo_buffer(state)
            st.session_state["synthetic_demo_state"] = state

    signal = np.asarray(state["signal"], dtype=np.float32)
    try:
        if muscle_group == "Forearm":
            movement_prediction = predict_normalized_forearm_window(
                signal[-256:],
                str(movement_model_file),
            )
            fatigue_prediction = predict_normalized_fatigue_window(
                signal[-SYNTHETIC_BUFFER_SAMPLES:],
                str(fatigue_model_file),
            )
        else:
            movement_prediction = predict_quadriceps_movement_window(
                signal[-256:],
                np.asarray(state["imu_quaternion"], dtype=np.float32)[-64:],
                str(movement_model_file),
            )
            fatigue_prediction = predict_quadriceps_fatigue_window(
                signal[-SYNTHETIC_BUFFER_SAMPLES:],
                str(fatigue_model_file),
            )
    except Exception as error:
        st.error("The CNN could not process the current input window.")
        with st.expander("CNN execution error", expanded=False):
            st.exception(error)
        return

    execution_columns = st.columns(5)
    execution_columns[0].metric("CNN execution", "Successful")
    execution_columns[1].metric(
        "CNN window (samples)", f"{len(signal)} / {SYNTHETIC_BUFFER_SAMPLES}"
    )
    execution_columns[2].metric(
        "Session time", f"{state['elapsed_seconds']:.1f} s"
    )
    execution_columns[3].metric("CNN input rate", f"{MODEL_SEMG_RATE_HZ} Hz")
    execution_columns[4].metric("Analysis", "Active" if running else "Ready")

    st.caption(
        f"Signal path: {DEVICE_SEMG_RATE_HZ} Hz sEMG → EMA alpha "
        f"{EMA_ALPHA} → baseline scaling → {MODEL_SEMG_RATE_HZ} Hz CNN input. "
        f"Device-cloud batch interval: {UPLOAD_INTERVAL_SECONDS} seconds. "
        "The 20-minute safety stop is active."
    )

    relative_time = (
        np.arange(len(signal), dtype=np.float32) - (len(signal) - 1)
    ) / SYNTHETIC_SAMPLE_RATE_HZ
    chart_frame = pd.DataFrame(
        {
            "Time (seconds)": relative_time,
            "Normalised sEMG": signal,
        }
    )
    signal_chart = (
        alt.Chart(chart_frame)
        .mark_line(color="#7C3AED")
        .encode(
            x=alt.X("Time (seconds):Q", title="Seconds (latest sample = 0)"),
            y=alt.Y("Normalised sEMG:Q", title="Normalised sEMG"),
            tooltip=[
                alt.Tooltip("Time (seconds):Q", format=".3f"),
                alt.Tooltip("Normalised sEMG:Q", format=".3f"),
            ],
        )
        .properties(height=260)
    )
    st.altair_chart(signal_chart, width="stretch")

    quaternions = np.asarray(state["imu_quaternion"], dtype=np.float32)
    w = quaternions[:, 0]
    x = quaternions[:, 1]
    y = quaternions[:, 2]
    z = quaternions[:, 3]
    roll_degrees = np.degrees(
        np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    )
    pitch_degrees = np.degrees(
        np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    )
    yaw_degrees = np.degrees(
        np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    )
    imu_time = (
        np.arange(len(roll_degrees), dtype=np.float32)
        - (len(roll_degrees) - 1)
    ) / DEVICE_IMU_RATE_HZ
    imu_frame = pd.DataFrame(
        {
            "Time (seconds)": imu_time,
            "Roll": roll_degrees,
            "Pitch": pitch_degrees,
            "Yaw": yaw_degrees,
        }
    ).melt(
        id_vars="Time (seconds)",
        var_name="Orientation",
        value_name="Degrees",
    )
    imu_chart = (
        alt.Chart(imu_frame)
        .mark_line()
        .encode(
            x=alt.X("Time (seconds):Q", title="Seconds (latest sample = 0)"),
            y=alt.Y("Degrees:Q", title="IMU orientation (degrees)"),
            color=alt.Color(
                "Orientation:N",
                scale=alt.Scale(
                    domain=["Roll", "Pitch", "Yaw"],
                    range=["#0284C7", "#16A34A", "#EA580C"],
                ),
            ),
            tooltip=[
                alt.Tooltip("Time (seconds):Q", format=".3f"),
                "Orientation:N",
                alt.Tooltip("Degrees:Q", format=".2f"),
            ],
        )
        .properties(
            height=220,
            title=(
                "Forearm IMU orientation"
                if muscle_group == "Forearm"
                else "Knee and thigh IMU orientation"
            ),
        )
    )
    st.altair_chart(imu_chart, width="stretch")
    if muscle_group == "Forearm":
        st.caption(
            "The 50 Hz IMU stream is monitored for movement context. The "
            "forearm checkpoints make their predictions from sEMG only."
        )
    else:
        st.caption(
            "The quadriceps movement CNN combines the 200 Hz rectus-femoris "
            "sEMG window with the matching 50 Hz IMU motion window. The "
            "quadriceps fatigue CNN uses sEMG."
        )

    movement_column, fatigue_column = st.columns(2)
    with movement_column:
        st.markdown("#### Movement CNN")
        st.metric(
            "Prediction",
            movement_prediction["label"],
            f"{movement_prediction['confidence']:.1%} model confidence",
        )
        for label, probability in movement_prediction["probabilities"].items():
            st.progress(probability, text=f"{label}: {probability:.1%}")

    with fatigue_column:
        st.markdown("#### Fatigue CNN")
        st.metric(
            "Prediction",
            fatigue_prediction["label"],
            f"{fatigue_prediction['fatigue_score']:.1%} fatigue score",
        )
        for label, probability in fatigue_prediction["probabilities"].items():
            st.progress(probability, text=f"{label}: {probability:.1%}")

    st.caption(
        "Model scores support professional review and are not a medical "
        "diagnosis."
    )

    with st.expander("Hardware handoff readiness", expanded=False):
        st.info(
            "The interface contract is ready, but this screen is using the "
            "simulator. Live sensor-quality status will appear after the "
            "hardware feed is connected."
        )
        readiness_columns = st.columns(3)
        readiness_columns[0].metric(
            "Signal-quality checks",
            "Ready for live input",
        )
        readiness_columns[1].metric(
            "Physical repetition count",
            "Awaiting edge counter",
        )
        readiness_columns[2].metric(
            "Session JSON and report",
            "Contract ready",
        )
        st.caption(
            "CNN analysis windows are kept separate from completed physical "
            "repetitions. Once the live feed is available, the dashboard can "
            "validate sequence continuity, timing, calibration, and abrupt "
            "signal spikes before accepting a completed session packet."
        )


def render_model_readiness(
    summary: dict,
    input_assessment: dict | None = None,
) -> None:
    st.markdown("### CNN pipeline status")
    input_ready = bool(
        input_assessment is not None and input_assessment["ready"]
    )
    if summary["checkpoint_available"] and input_ready:
        st.success(
            "The dense real-time input contract and both CNN checkpoint "
            "files are ready for an execution check. Device-specific "
            "validation is still required before releasing a final result."
        )
    elif input_ready:
        st.warning(
            "The sensor input passes the real-time contract, but both trained "
            "movement and fatigue checkpoints are not available."
        )
    else:
        st.warning(
            "The real-time Model 1 input is not ready. Review the failed "
            "quality checks below before enabling inference."
        )
    if input_assessment is None:
        readiness = pd.DataFrame(summary["readiness"])
    else:
        readiness = pd.DataFrame(
            [
                {
                    "Requirement": item["check"],
                    "Observed": item["observed"],
                    "Status": "Ready" if item["passed"] else "Not ready",
                }
                for item in input_assessment["checks"]
            ]
            + [
                {
                    "Requirement": "Movement and fatigue checkpoints",
                    "Observed": (
                        "Both checkpoint files installed"
                        if summary["checkpoint_available"]
                        else "One or both checkpoint files are missing"
                    ),
                    "Status": (
                        "Ready"
                        if summary["checkpoint_available"]
                        else "Not ready"
                    ),
                }
            ]
        )
    st.dataframe(
        readiness,
        hide_index=True,
        width="stretch",
        column_config={
            "Requirement": st.column_config.TextColumn(width="medium"),
            "Observed": st.column_config.TextColumn(width="large"),
            "Status": st.column_config.TextColumn(width="small"),
        },
    )
    st.caption(
        "Readiness confirms only that correctly shaped data can reach both "
        "CNNs. A final interpretation is released only after validation on "
        "labelled recordings from the KINESIS device."
    )


def render_signal_quality_summary(input_assessment: dict) -> None:
    """Present sensor artefact checks separately from the CNN result."""

    quality = input_assessment.get("signal_quality", {})
    status = str(quality.get("status", "Unavailable"))
    if status == "Good":
        st.success("Signal quality: Good — the recording passed all input checks.")
    elif status == "Needs review":
        st.warning(
            "Signal quality: Needs review — Model 1 may run, but the warning "
            "below should be checked."
        )
    else:
        st.error(
            "Signal quality: Invalid — Model 1 is blocked until the recording "
            "problem is corrected."
        )

    metrics = quality.get("metrics", {})
    quality_columns = st.columns(4)
    quality_columns[0].metric(
        "Missing samples",
        int(metrics.get("missing_sequence_count", 0)),
    )
    quality_columns[1].metric(
        "Timing outliers",
        f"{float(metrics.get('timing_outlier_fraction', 0.0)):.1%}",
    )
    quality_columns[2].metric(
        "Abrupt sEMG changes",
        f"{float(metrics.get('emg_spike_fraction', 0.0)):.1%}",
    )
    quaternion_error = metrics.get("maximum_quaternion_norm_error")
    quality_columns[3].metric(
        "IMU norm error",
        (
            f"{float(quaternion_error):.4f}"
            if quaternion_error is not None
            else "Unavailable"
        ),
    )

    issues = list(quality.get("blocking_issues", []))
    warnings = list(quality.get("warnings", []))
    if issues or warnings:
        with st.expander("Signal-quality details", expanded=status == "Invalid"):
            for issue in issues:
                st.write(f"❌ {issue}")
            for warning in warnings:
                st.write(f"⚠️ {warning}")
    st.caption(
        "These checks identify transmission and sensor artefacts. They do not "
        "measure clinical movement quality or fatigue."
    )


def render_nexus_model1_prediction(
    sensor_data: pd.DataFrame,
    summary: dict,
    movement_model_file: Path,
    fatigue_model_file: Path,
    muscle_group: str,
    exercise: str,
) -> dict | None:
    st.markdown("### KINESIS-device CNN result")
    if not summary["checkpoint_available"]:
        st.warning(
            f"The trained {muscle_group.lower()} movement and fatigue "
            "checkpoints are not both installed."
        )
        return None

    try:
        prediction = predict_live_model1_session(
            sensor_data,
            muscle_group,
            movement_model_file,
            fatigue_model_file,
        )
    except RealtimeInputNotReady as error:
        st.info(
            "Model 1 has not run because the newest recording failed one or "
            "more input-quality checks. No prediction was guessed."
        )
        failed_checks = pd.DataFrame(
            [
                {
                    "Check": item["check"],
                    "Observed": item["observed"],
                }
                for item in error.assessment["checks"]
                if not item["passed"]
            ]
        )
        st.dataframe(failed_checks, hide_index=True, width="stretch")
        return None
    except Exception as error:
        st.error("The trained CNNs could not analyse the latest window.")
        with st.expander("CNN error details", expanded=False):
            st.exception(error)
        return None

    st.success(
        "Both trained CNN checkpoints completed inference on a dense, "
        "quality-checked sensor window."
    )

    execution_column, window_column, interpretation_column = st.columns(3)
    execution_column.metric("CNN execution", "Both successful")
    window_column.metric(
        "sEMG path",
        f"{prediction['source_rate_hz']} → {prediction['model_rate_hz']} Hz",
    )
    interpretation_column.metric(
        "Result status",
        "Validated" if prediction["prediction_released"] else "Preview only",
    )

    if not prediction["prediction_released"]:
        st.warning(
            "The CNNs ran correctly, but the result remains an engineering "
            "preview because "
            + " and ".join(prediction["release_reasons"])
            + "."
        )

    movement = prediction["movement"]
    fatigue = prediction["fatigue"]
    fatigue_column, movement_column = st.columns(2)
    with fatigue_column:
        st.markdown("#### Fatigue pattern")
        st.metric("CNN result", fatigue["display_label"])
        st.progress(
            fatigue["confidence"],
            text=f"Top model score: {fatigue['confidence']:.1%}",
        )
        st.caption(
            f"Decision margin: {fatigue['decision_margin']:.1%}. "
            "This is a model score, not measured accuracy."
        )
    with movement_column:
        st.markdown("#### Movement pattern")
        st.metric("CNN result", movement["display_label"])
        st.progress(
            movement["confidence"],
            text=f"Top model score: {movement['confidence']:.1%}",
        )
        st.caption(
            f"Decision margin: {movement['decision_margin']:.1%}. "
            "Low-confidence or close decisions are shown as Uncertain."
        )

    st.caption(
        f"Movement model: {movement['model_version']} | Fatigue model: "
        f"{fatigue['model_version']} | {exercise}."
    )
    return prediction


def render_recording_controls(is_live: bool) -> tuple[str, str, str, str]:
    """Render muscle selection and an honest dashboard session indicator."""

    st.markdown("### Session setup")
    status = st.session_state.setdefault(
        "sensor_recording_status",
        "Not started",
    )
    recording = status == "Recording"

    identity_columns = st.columns(2)
    with identity_columns[0]:
        patient_id = st.text_input(
            "Patient ID",
            value="P001",
            disabled=recording,
            key="sensor_patient_id",
            help="Use a de-identified project ID, not a patient name.",
        ).strip()
    with identity_columns[1]:
        session_id = st.text_input(
            "Session ID",
            value="S001",
            disabled=recording,
            key="sensor_session_id",
        ).strip()

    selection_column, exercise_column, status_column = st.columns(3)
    with selection_column:
        muscle_group = st.selectbox(
            "Muscle group",
            list(MUSCLE_GROUPS),
            index=list(MUSCLE_GROUPS).index("Forearm"),
            disabled=recording,
            key="sensor_muscle_group",
        )
    exercise = MUSCLE_GROUPS[muscle_group]["exercise"]
    with exercise_column:
        st.text_input(
            "Exercise",
            value=exercise,
            disabled=True,
            key=f"sensor_exercise_{muscle_group}",
        )
    with status_column:
        st.metric("Dashboard session", status)

    start_column, stop_column, message_column = st.columns([1, 1, 4])
    with start_column:
        if st.button(
            "Start session",
            type="primary",
            disabled=recording or not is_live,
            width="stretch",
        ):
            st.session_state["sensor_recording_status"] = "Recording"
            st.session_state["sensor_recording_started_at"] = str(
                pd.Timestamp.now(tz=SENSOR_TIMEZONE)
            )
            st.rerun(scope="fragment")
    with stop_column:
        if st.button(
            "Stop session",
            disabled=not recording,
            width="stretch",
        ):
            st.session_state["sensor_recording_status"] = "Stopped"
            st.session_state["sensor_recording_stopped_at"] = str(
                pd.Timestamp.now(tz=SENSOR_TIMEZONE)
            )
            st.rerun(scope="fragment")
    with message_column:
        if not is_live:
            st.info(
                "The session cannot start while the dashboard is showing the "
                "bundled workbook snapshot. Connect the live feed first."
            )
        else:
            st.caption(
                "This currently records the dashboard session state. The "
                "firmware still needs a start/stop command connection."
            )

    return patient_id, session_id, muscle_group, exercise


def render_end_session_packet(
    sensor_data: pd.DataFrame,
    input_assessment: dict,
    prediction: dict | None,
    patient_id: str,
    session_id: str,
    muscle_group: str,
    exercise: str,
) -> None:
    """Preview and locally save the packet that Firebase will later receive."""

    with st.expander("End-of-session data packet", expanded=False):
        st.caption(
            "This stable JSON packet is ready for the future Firebase step. "
            "For now it can be downloaded or saved locally for integration testing."
        )
        include_test_count = st.checkbox(
            "Include a test repetition count supplied by the edge layer",
            value=False,
            key="include_test_edge_repetition_count",
            help=(
                "This does not ask the CNN to count repetitions. It imitates "
                "the separate counter that the hardware team will provide."
            ),
        )
        if include_test_count:
            repetition_columns = st.columns(2)
            total_repetitions = repetition_columns[0].number_input(
                "Total repetitions",
                min_value=0,
                value=10,
                step=1,
                key="test_total_repetitions",
            )
            correct_repetitions = repetition_columns[1].number_input(
                "Correct repetitions",
                min_value=0,
                max_value=int(total_repetitions),
                value=min(8, int(total_repetitions)),
                step=1,
                key="test_correct_repetitions",
            )
        else:
            total_repetitions = None
            correct_repetitions = None

        try:
            packet = build_session_packet(
                sensor_data,
                input_assessment,
                prediction,
                patient_id=patient_id,
                session_id=session_id,
                muscle_group=muscle_group,
                exercise=exercise,
                total_repetitions=total_repetitions,
                correct_repetitions=correct_repetitions,
                repetition_source="edge_counter_test",
                completed_at=st.session_state.get(
                    "sensor_recording_stopped_at"
                ),
            )
        except Exception as error:
            st.warning(f"The session packet is not ready: {error}")
            return

        packet_columns = st.columns(3)
        packet_columns[0].metric(
            "Signal quality",
            packet["signal_quality"]["status"],
        )
        packet_columns[1].metric(
            "Repetition count",
            (
                f"{packet['repetition_summary']['correct_repetitions']} / "
                f"{packet['repetition_summary']['total_repetitions']}"
                if packet["repetition_summary"]["available"]
                else "Awaiting edge"
            ),
        )
        packet_columns[2].metric(
            "Model 1",
            packet["model1"]["release_status"].replace("_", " ").title(),
        )
        st.caption(
            "Repetition counts and CNN analysis-window counts are stored as "
            "different fields so they cannot be mistaken for each other."
        )

        packet_json = packet_to_json(packet)
        download_column, save_column = st.columns(2)
        with download_column:
            st.download_button(
                "Download session JSON",
                packet_json.encode("utf-8"),
                f"{patient_id}_{session_id}_session_packet.json",
                "application/json",
                width="stretch",
                key="download_completed_session_packet",
            )
        with save_column:
            if st.button(
                "Save packet locally",
                width="stretch",
                key="save_completed_session_packet",
            ):
                destination = save_session_packet(packet, SESSION_PACKET_ROOT)
                st.success(f"Saved locally: {destination}")

        with st.popover("Preview packet", width="stretch"):
            st.json(packet)


@st.fragment(run_every=f"{SENSOR_REFRESH_SECONDS}s")
def render_live_sensor_monitoring() -> None:
    try:
        sensor_data, source_name, is_live = load_dashboard_sensor_feed(
            SENSOR_FEED_URL,
            str(SENSOR_FEED_FILE),
        )
    except Exception as error:
        show_service_error(
            "The sensor feed could not be loaded. Check its columns and "
            "connection settings.",
            error,
        )
        return

    status_column, refresh_column = st.columns([5, 1])
    with status_column:
        st.caption(
            f"Source: {source_name}. Checked automatically every "
            f"{SENSOR_REFRESH_SECONDS} seconds."
        )
    with refresh_column:
        if st.button("Refresh now", width="stretch"):
            load_dashboard_sensor_feed.clear()
            st.rerun(scope="fragment")

    patient_id, session_id, muscle_group, exercise = render_recording_controls(
        is_live
    )
    movement_model_file = (
        NEXUS_MODEL_DIRECTORY
        / MUSCLE_GROUPS[muscle_group]["movement_model_file"]
    )
    fatigue_model_file = (
        NEXUS_MODEL_DIRECTORY
        / MUSCLE_GROUPS[muscle_group]["fatigue_model_file"]
    )
    both_checkpoints_available = bool(
        movement_model_file.is_file() and fatigue_model_file.is_file()
    )

    summary = summarize_sensor_frame(
        sensor_data,
        movement_model_file if both_checkpoints_available else None,
    )
    input_assessment = assess_realtime_sensor_frame(sensor_data)
    render_feed_status(summary, is_live)

    metric_columns = st.columns(5)
    metric_columns[0].metric(
        "Latest sEMG",
        f"{summary['latest_semg']:.0f}",
        help="Raw device count. The voltage scale has not yet been supplied.",
    )
    metric_columns[1].metric(
        "Latest sequence",
        summary["latest_sequence"],
    )
    metric_columns[2].metric(
        "Calibration code",
        summary["latest_calibration"],
        help="The firmware meaning of this code still needs confirmation.",
    )
    metric_columns[3].metric(
        "Median logged-row interval",
        (
            f"{summary['median_interval_seconds']:.2f} s"
            if summary["median_interval_seconds"] is not None
            else "Unavailable"
        ),
    )
    metric_columns[4].metric(
        "Rows loaded",
        summary["row_count"],
    )

    st.markdown("### Calculated orientation result")
    st.caption(
        "Roll, pitch and yaw are calculated from the quaternion values. "
        "This is a sensor calculation, not an AI prediction."
    )
    orientation_columns = st.columns(3)
    orientation_columns[0].metric(
        "Roll",
        f"{summary['latest_roll']:.2f}°",
    )
    orientation_columns[1].metric(
        "Pitch",
        f"{summary['latest_pitch']:.2f}°",
    )
    orientation_columns[2].metric(
        "Yaw",
        f"{summary['latest_yaw']:.2f}°",
    )

    if muscle_group == "Forearm":
        render_compact_model1_status()

    st.markdown("### Signal-quality result")
    render_signal_quality_summary(input_assessment)

    prediction = render_nexus_model1_prediction(
        sensor_data,
        summary,
        movement_model_file,
        fatigue_model_file,
        muscle_group,
        exercise,
    )
    render_sensor_charts(sensor_data)
    render_model_readiness(summary, input_assessment)
    render_end_session_packet(
        sensor_data,
        input_assessment,
        prediction,
        patient_id,
        session_id,
        muscle_group,
        exercise,
    )

    with st.expander("Latest received rows", expanded=False):
        display_columns = [
            "Timestamp",
            "Session",
            "Seq",
            "STM32 ms",
            "sEMG raw",
            "Quat W",
            "Quat X",
            "Quat Y",
            "Quat Z",
            "Calib",
        ]
        st.dataframe(
            sensor_data[display_columns].tail(20),
            hide_index=True,
            width="stretch",
        )


def render_percentage_bar(
    label: str,
    value: float | str | None,
    color: str,
) -> None:
    if value is None:
        st.markdown(f"**{label}: Not available**")
        return
    if isinstance(value, str) and value.strip().endswith("%"):
        numeric_value = float(value.strip()[:-1]) / 100
    else:
        numeric_value = float(value)
    chart_value = min(max(numeric_value, 0.0), 1.0)
    bar_data = pd.DataFrame(
        {
            "Value": [chart_value],
            "Full scale": [1.0],
        }
    )
    track = (
        alt.Chart(bar_data)
        .mark_bar(
            color="#E5E7EB",
            cornerRadius=8,
            size=18,
        )
        .encode(
            x=alt.X(
                "Full scale:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=None,
            )
        )
    )
    filled = (
        alt.Chart(bar_data)
        .mark_bar(
            color=color,
            cornerRadius=8,
            size=18,
        )
        .encode(
            x=alt.X(
                "Value:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=None,
            ),
            tooltip=[alt.Tooltip("Value:Q", title=label, format=".1%")],
        )
    )
    st.markdown(f"**{label}: {numeric_value:.1%}**")
    st.altair_chart(
        (track + filled).properties(height=28),
        width="stretch",
    )


def render_model2_evidence(summary: dict) -> None:
    """Show the authorised provenance behind a Model 2 response."""

    evidence_items = list(summary.get("evidence_items", []))
    if not evidence_items:
        evidence_items = [
            {
                "title": "Approved knowledge reference",
                "source": source.get("title", "Knowledge note"),
                "finding": source.get("section", "Relevant section"),
                "reference": "Approved KINESIS knowledge library",
                "excerpt": source.get("excerpt", ""),
                "links": source.get("links", []),
            }
            for source in summary.get("knowledge_sources", [])
        ]
    if not evidence_items:
        return

    role = str(summary.get("audience_role", "")).casefold()
    is_general = summary.get("answer_scope") == "general_knowledge"
    if role == "patient":
        label = f"Why am I seeing this answer? ({len(evidence_items)} items)"
        introduction = (
            "This answer uses only approved general rehabilitation references; "
            "no patient or session record was accessed."
            if is_general
            else (
                "This answer uses only the reviewed information released for "
                "your selected session."
            )
        )
    else:
        label = f"Evidence used for this answer ({len(evidence_items)} items)"
        introduction = (
            "This explanation uses only the approved general rehabilitation "
            "references listed below; no patient or session record was accessed."
            if is_general
            else (
                "This explanation was generated only from the authorised records "
                "and approved references listed below."
            )
        )

    with st.expander(label, expanded=False):
        st.caption(introduction)
        for evidence_index, item in enumerate(evidence_items):
            with st.container(border=True):
                st.markdown(f"**{item.get('title', 'Evidence')}**")
                source = str(item.get("source", "Authorised session record"))
                reference = str(item.get("reference", "Selected session"))
                finding = str(
                    item.get("finding", "No additional detail available.")
                )
                excerpt = str(item.get("excerpt", "")).strip()
                links = [
                    str(link).strip()
                    for link in item.get("links", [])
                    if str(link).strip().startswith(("https://", "http://"))
                ]
                if excerpt:
                    st.caption(source)
                    st.markdown(f"**Relevant section:** {finding}")
                    with st.popover(
                        "View reference",
                        key=f"model2_evidence_reference_{evidence_index}",
                        width="stretch",
                    ):
                        st.markdown(f"#### {source}")
                        st.caption(f"Relevant section: {finding}")
                        st.markdown(excerpt)
                        if links:
                            st.markdown("**Original research sources**")
                            for link_index, link in enumerate(links[:6], start=1):
                                st.markdown(
                                    f"{link_index}. [Open original source]({link})"
                                )
                else:
                    st.caption(f"{source} · {reference}")
                    st.write(finding)


def render_standard_report_export(summary: dict) -> None:
    """Expose the same report sections for every completed AI summary."""

    report = build_standard_session_report(summary)
    patient_id = str(summary.get("patient_id", "patient"))
    session_id = str(summary.get("session_id", "session"))
    with st.expander("Standard session report", expanded=False):
        st.caption(
            "Every report uses the same nine sections. Model 2 explains the "
            "evidence but cannot rename sections or change the stored CNN values."
        )
        report_columns = st.columns(3)
        report_columns[0].metric("Format", "KINESIS v1.0")
        report_columns[1].metric("Sections", "9 fixed")
        report_columns[2].metric("Professional review", "Required")
        st.write(
            "Session information · Signal quality · Movement finding · "
            "Fatigue finding · Historical comparison · Physiotherapist review · "
            "Evidence used · Next action · Safety note"
        )
        json_column, markdown_column = st.columns(2)
        with json_column:
            st.download_button(
                "Download report JSON",
                report_to_json(report).encode("utf-8"),
                f"{patient_id}_{session_id}_report.json",
                "application/json",
                width="stretch",
                key=f"download_report_json_{patient_id}_{session_id}",
            )
        with markdown_column:
            st.download_button(
                "Download readable report",
                report_to_markdown(report).encode("utf-8"),
                f"{patient_id}_{session_id}_report.md",
                "text/markdown",
                width="stretch",
                key=f"download_report_markdown_{patient_id}_{session_id}",
            )


def render_automatic_summary(summary: dict) -> None:
    st.markdown("### AI session summary")
    render_model2_delivery_status(summary)
    if summary.get("question"):
        st.markdown(f"**Question:** {summary['question']}")

    with st.container(border=True):
        st.markdown(f"#### {summary['headline']}")
        fatigue_column, movement_column = st.columns(2)

        with fatigue_column:
            st.markdown("##### Fatigue")
            st.write(summary["fatigue"]["result"])
            session_metrics = summary.get("session_metrics", {})
            fatigue_window_fraction = session_metrics.get(
                "fatigue_pattern_window_fraction",
                summary["fatigue"].get(
                    "probability_value",
                    summary["fatigue"]["probability"],
                ),
            )
            render_percentage_bar(
                "Fatigue-pattern windows",
                fatigue_window_fraction,
                "#7C3AED",
            )
            window_count = int(session_metrics.get("window_count", 0))
            fatigue_window_count = int(
                session_metrics.get("fatigue_pattern_window_count", 0)
            )
            if window_count:
                st.caption(
                    f"{fatigue_window_count} of {window_count} analysis windows "
                    "crossed the fatigue-pattern threshold."
                )
            else:
                st.caption("The score describes an AI pattern, not severity.")
            if summary["fatigue"].get("explanation"):
                st.write(summary["fatigue"]["explanation"])
            st.caption(summary["fatigue"]["onset"])

        with movement_column:
            st.markdown("##### Movement")
            st.write(summary["movement"]["result"])
            movement_consistency = session_metrics.get(
                "movement_consistency",
                summary["movement"].get(
                    "confidence_value",
                    summary["movement"]["confidence"],
                ),
            )
            render_percentage_bar(
                "Movement consistency",
                movement_consistency,
                "#0284C7",
            )
            mismatch_count = int(
                session_metrics.get("movement_mismatch_window_count", 0)
            )
            if window_count:
                st.caption(
                    f"{window_count - mismatch_count} of {window_count} analysis "
                    "windows matched the learned movement pattern."
                )
            else:
                st.caption("Consistency summarises the analysed movement windows.")
            if summary["movement"].get("explanation"):
                st.write(summary["movement"]["explanation"])

        st.markdown("##### Compared with earlier sessions")
        st.write(summary["session_comparison"])

        if summary.get("general_explanation"):
            st.markdown("##### General explanation")
            st.write(summary["general_explanation"])

    render_model2_evidence(summary)
    render_standard_report_export(summary)

    review = summary["physiotherapist_review"]
    with st.expander(
        "Latest physiotherapist review included in this summary",
        expanded=False,
    ):
        st.write(f"Status: **{review['status']}**")
        st.write(f"Reviewer: {review['reviewer']}")
        st.write(f"Reviewed at: {review['reviewed_at']}")
        st.write(f"Fatigue decision: {review['fatigue_decision']}")
        st.write(
            "Final fatigue result: "
            f"{review['final_fatigue_result']}"
        )
        st.write(f"Movement decision: {review['movement_decision']}")
        st.write(
            "Final movement result: "
            f"{review['final_movement_result']}"
        )
        st.write(f"Comment: {review['comment']}")

    st.info(summary["important_note"])


def render_role_session_answer(summary: dict) -> None:
    """Show a concise role-appropriate Model 2 answer."""

    st.markdown("### AI answer")
    render_model2_delivery_status(summary)
    st.markdown(f"**Question:** {summary['question']}")
    with st.container(border=True):
        st.write(summary["answer"])

    render_model2_evidence(summary)
    st.info(summary["important_note"])


def render_attention_card(result: dict) -> None:
    """Render a clear traffic-light status without making a diagnosis."""

    level = str(result.get("attention_level", "yellow")).casefold()
    if level not in {"green", "yellow", "red"}:
        level = "yellow"
    icons = {"green": "&#10003;", "yellow": "!", "red": "!"}
    labels = {
        "green": "Green status",
        "yellow": "Yellow status",
        "red": "Red status",
    }
    title = escape(str(result.get("attention_title", "Session status")))
    message = escape(str(result.get("attention_message", "")))
    st.markdown(
        f"""
        <div class="role-status-card role-status-{level}" role="status"
             aria-label="{labels[level]}: {title}">
            <span class="role-status-icon" aria-hidden="true">{icons[level]}</span>
            <span class="role-status-copy">
                <strong>{title}</strong>
                <span>{message}</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_model2_delivery_status(summary: dict) -> None:
    """Distinguish a completed external call from the local safety fallback."""

    is_general = summary.get("answer_scope") == "general_knowledge"
    if summary.get("model2_skipped_reason") == "out_of_scope":
        st.info(
            (
                "This question is outside the general rehabilitation assistant's "
                "scope, so the external AI service was not called."
                if is_general
                else (
                    "This question is outside the session assistant's scope, so "
                    "the external AI service was not called."
                )
            )
        )
        return

    if summary.get("model2_verified"):
        st.success("AI response received")
        st.caption(
            "The AI completed this request using approved rehabilitation references; "
            "no patient record was accessed."
            if is_general
            else "The AI completed this request using the authorised session evidence."
        )
        return

    error_name = str(summary.get("model2_error", "")).strip()
    if error_name:
        st.warning(
            "The AI service could not complete this request. A local safety "
            "summary is shown instead."
        )
        with st.expander("AI connection troubleshooting", expanded=True):
            st.write(f"Error category: `{error_name}`")
            st.write(
                "Check that the API key is valid, service access is enabled, "
                "usage quota remains, and the computer has internet access."
            )
        return

    st.info("No external AI response was requested. Local summary shown.")


def render_probability_timeline(visualisation: dict) -> None:
    window_data = pd.DataFrame(visualisation["window_predictions"])
    timeline = window_data[
        [
            "start_seconds",
            "fatigue_probability",
            "movement_incorrect_probability",
        ]
    ].rename(
        columns={
            "start_seconds": "Time (seconds)",
            "fatigue_probability": "Fatigue probability",
            "movement_incorrect_probability": (
                "Incorrect-movement probability"
            ),
        }
    )
    timeline = timeline.melt(
        id_vars="Time (seconds)",
        var_name="Model output",
        value_name="Probability",
    )

    lines = (
        alt.Chart(timeline)
        .mark_line(point=True)
        .encode(
            x=alt.X("Time (seconds):Q", title="Window start (seconds)"),
            y=alt.Y(
                "Probability:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("Model output:N", title=None),
            tooltip=[
                alt.Tooltip("Time (seconds):Q", format=".1f"),
                "Model output:N",
                alt.Tooltip("Probability:Q", format=".1%"),
            ],
        )
    )
    threshold = (
        alt.Chart(pd.DataFrame({"Threshold": [0.5]}))
        .mark_rule(color="#777777", strokeDash=[6, 4])
        .encode(y="Threshold:Q")
    )
    st.altair_chart(
        (lines + threshold).properties(height=320),
        width="stretch",
    )


def render_signal_overlay(visualisation: dict) -> None:
    signal_data = pd.DataFrame(visualisation["signal_overlay"]).rename(
        columns={
            "time_seconds": "Time (seconds)",
            "emg_activation_normalised": "EMG muscle activation",
            "imu_movement_normalised": "IMU movement",
        }
    )
    signal_data = signal_data.melt(
        id_vars="Time (seconds)",
        var_name="Signal",
        value_name="Normalised amplitude",
    )
    chart = (
        alt.Chart(signal_data)
        .mark_line()
        .encode(
            x=alt.X("Time (seconds):Q", title="Time (seconds)"),
            y=alt.Y(
                "Normalised amplitude:Q",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color("Signal:N", title=None),
            tooltip=[
                alt.Tooltip("Time (seconds):Q", format=".2f"),
                "Signal:N",
                alt.Tooltip("Normalised amplitude:Q", format=".2f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")
    visualisation_note = visualisation["data_note"]
    if any(
        word in visualisation_note.casefold()
        for word in ("synthetic", "demonstration", "demo")
    ):
        visualisation_note = (
            "Signals are normalised for visual comparison and should not be "
            "interpreted as directly comparable amplitudes."
        )
    st.caption(visualisation_note)


def render_session_comparison(comparison: dict) -> None:
    comparison_data = pd.DataFrame(comparison["sessions"])
    comparison_data["fatigue_pattern_windows_fraction"] = (
        comparison_data["fatigue_pattern_windows_percent"] / 100.0
    )
    comparison_data["movement_consistency_fraction"] = (
        comparison_data["movement_consistency_percent"] / 100.0
    )
    chart_data = comparison_data[
        [
            "session_id",
            "fatigue_pattern_windows_fraction",
            "movement_consistency_fraction",
        ]
    ].rename(
        columns={
            "session_id": "Session",
            "fatigue_pattern_windows_fraction": "Fatigue-pattern windows",
            "movement_consistency_fraction": "Movement consistency",
        }
    )
    chart_data = chart_data.melt(
        id_vars="Session",
        var_name="Measure",
        value_name="Score",
    )
    chart = (
        alt.Chart(chart_data)
        .mark_line(point=True)
        .encode(
            x=alt.X("Session:N", sort=None),
            y=alt.Y(
                "Score:Q",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%"),
            ),
            color=alt.Color("Measure:N", title=None),
            tooltip=[
                "Session:N",
                "Measure:N",
                alt.Tooltip("Score:Q", format=".1%"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(chart, width="stretch")
    comparison_note = comparison["interpretation_note"]
    if any(
        word in comparison_note.casefold()
        for word in ("synthetic", "demonstration", "demo")
    ):
        comparison_note = (
            "Changes between session scores require professional "
            "interpretation."
        )
    st.caption(comparison_note)

    table = comparison_data[
        [
            "session_id",
            "fatigue_pattern_windows_percent",
            "movement_quality",
            "movement_consistency_percent",
            "review_status",
        ]
    ].copy()
    table["fatigue_pattern_windows_percent"] = table[
        "fatigue_pattern_windows_percent"
    ].map(
        lambda value: f"{value:.1f}%"
    )
    table["movement_consistency_percent"] = table[
        "movement_consistency_percent"
    ].map(
        lambda value: f"{value:.1f}%"
    )
    table.columns = [
        "Session",
        "Fatigue-pattern windows",
        "Movement result",
        "Movement consistency",
        "Review status",
    ]
    st.dataframe(table, hide_index=True, width="stretch")


def render_physiotherapist(
    result: dict,
    automatic_summary: dict | None = None,
) -> None:
    prediction = result["prediction"]
    measurements = result["supporting_measurements"]
    patient_id = result["patient_id"]
    session_id = result["session_id"]

    predicted_fatigue = (
        "fatigued" if prediction["fatigue_flag"] else "non_fatigued"
    )
    opposite_fatigue = (
        "non_fatigued" if predicted_fatigue == "fatigued" else "fatigued"
    )
    predicted_movement = prediction["movement_quality"]
    opposite_movement = (
        "correct" if predicted_movement == "incorrect" else "incorrect"
    )

    st.subheader(f"Patient {patient_id} - Session {session_id}")
    if prediction["fatigue_flag"] or predicted_movement == "incorrect":
        st.warning("AI result: professional review recommended")
    else:
        st.success("AI result: no pattern was flagged")

    if automatic_summary is not None:
        render_automatic_summary(automatic_summary)

    st.markdown("### Session overview")
    with st.container(border=True):
        st.markdown("#### Fatigue")
        fatigue_overview = st.columns([1.2, 1, 1.2])
        fatigue_overview[0].metric(
            "Fatigue result",
            predicted_fatigue.replace("_", " ").title(),
        )
        with fatigue_overview[1]:
            session_windows = result.get("window_predictions", [])
            fatigue_window_fraction = (
                sum(
                    float(window["fatigue_probability"]) >= 0.5
                    for window in session_windows
                )
                / len(session_windows)
                if session_windows
                else None
            )
            render_percentage_bar(
                "Fatigue-pattern windows",
                fatigue_window_fraction,
                "#7C3AED",
            )
        fatigue_onset = prediction["fatigue_onset_seconds"]
        fatigue_overview[2].metric(
            "Estimated fatigue onset",
            (
                f"{fatigue_onset:.1f} s"
                if fatigue_onset is not None
                else "Not detected"
            ),
        )

    with st.container(border=True):
        st.markdown("#### Movement")
        movement_overview = st.columns(2)
        movement_overview[0].metric(
            "Movement result",
            predicted_movement.title(),
        )
        with movement_overview[1]:
            movement_consistency = (
                sum(
                    float(window["movement_incorrect_probability"]) < 0.5
                    for window in session_windows
                )
                / len(session_windows)
                if session_windows
                else None
            )
            render_percentage_bar(
                "Movement consistency",
                movement_consistency,
                "#0284C7",
            )

    st.info(
        "These are experimental pattern classifications, not a diagnosis. "
        "A physiotherapist must decide whether they are meaningful."
    )

    try:
        visualisation = run_mcp_tool(
            "get_session_visualisation",
            {
                "patient_id": patient_id,
                "session_id": session_id,
                "requesting_role": "physiotherapist",
            },
        )
        st.markdown("### Fatigue and movement timeline")
        render_probability_timeline(visualisation)
        st.markdown("### EMG and IMU activity overlay")
        render_signal_overlay(visualisation)
    except Exception as error:
        st.warning(f"Session visualisation is unavailable: {error}")

    st.markdown("### Supporting measurements")
    supporting = st.columns(5)
    supporting[0].metric("Analysed windows", measurements["number_of_windows"])
    supporting[1].metric(
        "Window duration",
        f"{measurements['window_seconds']:.1f} s",
    )
    supporting[2].metric(
        "Window overlap",
        f"{measurements['window_overlap']:.0%}",
    )
    supporting[3].metric(
        "EMG sampling",
        f"{measurements['emg_sampling_rate_hz']} Hz",
    )
    supporting[4].metric(
        "IMU sampling",
        f"{measurements['imu_sampling_rate_hz']} Hz",
    )

    try:
        comparison = run_mcp_tool(
            "compare_session_trends",
            {
                "patient_id": patient_id,
                "requesting_role": "physiotherapist",
            },
        )
        st.markdown("### Session comparison")
        render_session_comparison(comparison)
    except Exception as error:
        st.warning(f"Session comparison is unavailable: {error}")

    feedback = run_mcp_tool(
        "get_feedback_status",
        {
            "patient_id": patient_id,
            "session_id": session_id,
            "requesting_role": "physiotherapist",
        },
    )
    st.markdown("### Physiotherapist review")
    st.write("Review the AI suggestions below and record your professional decision.")
    st.write(f"Review status: **{feedback['review_status']}**")
    if feedback["review_saved"]:
        if feedback["include_for_retraining"]:
            st.success(
                "The latest review is saved and may help improve future "
                "versions of the AI model."
            )
        else:
            st.info("The latest review is saved and will not be used to improve the AI model.")

    fatigue_options = [
        f"Confirm model label: {predicted_fatigue}",
        f"Correct label to: {opposite_fatigue}",
        "Uncertain - more information required",
    ]
    movement_options = [
        f"Confirm model label: {predicted_movement}",
        f"Correct label to: {opposite_movement}",
        "Uncertain - more information required",
    ]
    # Display friendly sentences while preserving the existing decision values.
    fatigue_sentences = {
        "non_fatigued": "No fatigue was detected",
        "fatigued": "Fatigue was detected",
    }
    movement_sentences = {
        "correct": "The movement was performed correctly",
        "incorrect": "The movement may need correction",
    }
    fatigue_labels = dict(zip(fatigue_options, [
        f"Agree - {fatigue_sentences[predicted_fatigue]}",
        f"Change - {fatigue_sentences[opposite_fatigue]}",
        "Unsure - More information is needed",
    ]))
    movement_labels = dict(zip(movement_options, [
        f"Agree - {movement_sentences[predicted_movement]}",
        f"Change - {movement_sentences[opposite_movement]}",
        "Unsure - More information is needed",
    ]))

    with st.form("physiotherapist_review"):
        reviewer_id = st.text_input("Your staff ID", value="PT001", max_chars=5)
        fatigue_column, movement_column = st.columns(2)
        with fatigue_column:
            st.markdown("#### Fatigue review")
            fatigue_choice = st.selectbox(
                "Your fatigue decision",
                fatigue_options,
                format_func=fatigue_labels.__getitem__,
            )
        with movement_column:
            st.markdown("#### Movement review")
            movement_choice = st.selectbox(
                "Your movement decision",
                movement_options,
                format_func=movement_labels.__getitem__,
            )
        feedback_is_certain = not (
            fatigue_choice.startswith("Uncertain")
            or movement_choice.startswith("Uncertain")
        )
        include_for_retraining = st.checkbox(
            "Allow this reviewed result to help improve future versions of the AI model",
            value=False,
            disabled=not feedback_is_certain,
        )
        st.caption(
            "This permission does not immediately change the current model. "
            "The reviewed result may be used in a future, controlled model update."
        )
        note = st.text_area("Session review notes (optional)", max_chars=500)
        submitted = st.form_submit_button(
            "Save review",
            type="primary",
            disabled=PUBLIC_DEMO_MODE,
        )
        if PUBLIC_DEMO_MODE:
            st.caption(
                "This workspace is read-only. Use the authorised clinical "
                "workspace to record physiotherapist reviews."
            )
        else:
            st.caption(
                "The review is stored only after 'Review saved successfully' "
                "appears. If the connection is lost, reconnect and submit again."
            )

    if submitted and not PUBLIC_DEMO_MODE:
        fatigue_decision, corrected_fatigue = resolve_review_choice(
            fatigue_choice,
            predicted_fatigue,
            opposite_fatigue,
        )
        movement_decision, corrected_movement = resolve_review_choice(
            movement_choice,
            predicted_movement,
            opposite_movement,
        )
        try:
            review = run_mcp_tool(
                "record_review_decision",
                {
                    "patient_id": patient_id,
                    "session_id": session_id,
                    "requesting_role": "physiotherapist",
                    "reviewer_id": reviewer_id.upper(),
                    "fatigue_decision": fatigue_decision,
                    "corrected_fatigue_label": corrected_fatigue,
                    "movement_decision": movement_decision,
                    "corrected_movement_label": corrected_movement,
                    "include_for_retraining": include_for_retraining,
                    "note": note,
                },
            )
            current_question = st.session_state.get("analysis", {}).get(
                "assistant_query",
                "Summarise this rehabilitation session.",
            )
            refreshed_agent = asyncio.run(
                build_session_context(
                    patient_id,
                    session_id,
                    current_question,
                )
            )
            st.session_state["analysis"]["result"] = refreshed_agent[
                "summary_context"
            ]["original_ai_findings"]
            st.session_state["analysis"][
                "automatic_summary"
            ] = refreshed_agent["automatic_summary"]
            st.session_state.pop("review_dashboard", None)
            st.session_state["review_message"] = "Review saved successfully."
            st.session_state["saved_review_trace"] = {
                "patient_id": patient_id,
                "session_id": session_id,
                "review_id": review["review_id"],
            }
            st.rerun()
        except Exception as error:
            show_service_error("Review could not be saved. Please try again.", error)

    show_workflow(
        "physiotherapist",
        patient_id,
        session_id,
        result["review_status"],
    )


def render_clinician(
    result: dict,
    automatic_summary: dict | None = None,
) -> None:
    patient_id = result["patient_id"]
    session_id = result["session_id"]
    st.subheader("Care-team coordination overview")
    st.caption(f"Patient {patient_id} · Session {session_id}")
    render_attention_card(result)

    metric_columns = st.columns(4)
    metric_columns[0].metric(
        "Fatigue-pattern windows",
        f"{result['fatigue_pattern_windows_percent']:.1f}%",
    )
    metric_columns[1].metric(
        "Movement-consistent windows",
        f"{result['movement_consistency_percent']:.1f}%",
    )
    fatigue_onset = result.get("fatigue_onset_seconds")
    metric_columns[2].metric(
        "First fatigue pattern",
        "Not detected" if fatigue_onset is None else f"{fatigue_onset:.1f} s",
    )
    metric_columns[3].metric(
        "Reviews completed",
        f"{result['reviewed_session_count']} / {result['total_session_count']}",
    )
    st.caption(
        "Window percentages summarise the Model 1 session output. The "
        "physiotherapist review determines how the findings are used."
    )

    if automatic_summary:
        render_role_session_answer(automatic_summary)

    findings_column, review_column = st.columns(2)
    with findings_column:
        st.markdown("### Model findings")
        for finding in result["key_findings"]:
            st.write(f"- {finding}")
    with review_column:
        st.markdown("### Physiotherapist review")
        st.info(result["physiotherapist_review_status"])

    try:
        comparison = run_mcp_tool(
            "compare_session_trends",
            {
                "patient_id": patient_id,
                "requesting_role": "clinician",
            },
        )
        st.markdown("### Session comparison")
        render_session_comparison(comparison)
    except Exception as error:
        st.warning(f"Session comparison is unavailable: {error}")

    st.markdown("### Care coordination actions")
    with st.container(border=True):
        for item in result["coordination_guidance"]:
            st.write(f"- {item}")
    with st.expander("Limitations"):
        for limitation in result["limitations"]:
            if any(
                word in limitation.casefold()
                for word in ("synthetic", "demonstration", "demo")
            ):
                continue
            st.write(f"- {limitation}")

    show_workflow(
        "clinician",
        patient_id,
        session_id,
        result["physiotherapist_review_status"],
    )


def render_patient(
    result: dict,
    automatic_summary: dict | None = None,
) -> None:
    st.subheader("Your session at a glance")
    render_attention_card(result)
    st.write(result["summary"])

    fatigue_percent = result.get("fatigue_pattern_windows_percent")
    movement_percent = result.get("movement_consistency_percent")
    review_completed = fatigue_percent is not None and movement_percent is not None
    session_columns = st.columns(2)
    session_columns[0].metric(
        "Session",
        f"{result['session_number']} of {result['total_session_count']}",
    )
    session_columns[1].metric(
        "Review",
        "Completed" if review_completed else "Pending",
    )

    st.markdown("### Session indicators")
    indicator_columns = st.columns(2)
    with indicator_columns[0]:
        with st.container(border=True):
            st.metric(
                "Fatigue-pattern time",
                f"{fatigue_percent:.1f}%" if review_completed else "Pending",
            )
            if review_completed:
                st.progress(float(fatigue_percent) / 100)
                st.caption(
                    "Share of short session periods that showed a fatigue-like "
                    "pattern. A lower percentage means fewer such periods."
                )
            else:
                st.caption("Available after physiotherapist review.")

    with indicator_columns[1]:
        with st.container(border=True):
            st.metric(
                "Steady movement",
                f"{movement_percent:.1f}%" if review_completed else "Pending",
            )
            if review_completed:
                st.progress(float(movement_percent) / 100)
                st.caption(
                    "Share of short session periods with consistent movement. "
                    "A higher percentage means movement stayed steady more often."
                )
            else:
                st.caption("Available after physiotherapist review.")

    if review_completed:
        st.caption(
            "These indicators summarise short periods across this reviewed session."
        )
    else:
        st.caption(
            "Session percentages will appear after your physiotherapist "
            "completes the review."
        )

    if automatic_summary:
        render_role_session_answer(automatic_summary)

    st.markdown("### What to do next")
    with st.container(border=True):
        st.write(result["next_step"])

    st.markdown("### Session history")
    for item in result["history"]:
        symbol = {
            "green": "✅",
            "yellow": "🟡",
            "red": "🔴",
        }.get(item.get("attention_level"), "🟡")
        st.write(
            f"{symbol} **{item['session_id']}** · "
            f"{item.get('attention_title', item['status'])} · "
            f"{item['analysis_date']}"
        )
    st.caption(result["safety_note"])


def friendly_review_result(value: str | None) -> str:
    labels = {
        "non_fatigued": "No fatigue detected",
        "fatigued": "Fatigue detected",
        "correct": "Movement performed correctly",
        "incorrect": "Movement may need correction",
    }
    return labels.get(value, "More information needed")


def friendly_review_decision(value: str) -> str:
    labels = {
        "confirmed": "Agreed with the AI result",
        "corrected": "Changed the AI result",
        "uncertain": "More information is needed",
    }
    return labels[value]


def render_review_dashboard(session_catalog: list[dict]) -> None:
    st.subheader("Physiotherapist review records")
    st.write(
        "See what the AI suggested, what the physiotherapist decided, "
        "and any comments saved for each session."
    )
    refresh = st.button("Refresh review records")
    catalog_key = tuple(sorted(
        (item["patient_id"], item["session_id"]) for item in session_catalog
    ))
    snapshot = st.session_state.get("review_dashboard")
    if refresh or snapshot is None or snapshot["catalog"] != catalog_key:
        records, failures, reviews = [], [], []
        with st.spinner("Loading review records"):
            for patient_id, session_id in catalog_key:
                try:
                    record = run_mcp_tool("get_feedback_status", {
                        "patient_id": patient_id,
                        "session_id": session_id,
                        "requesting_role": "physiotherapist",
                    })
                    if (record["patient_id"], record["session_id"]) != (patient_id, session_id):
                        raise ValueError("Returned session identifiers did not match the request")
                    records.append(record)
                except Exception as error:
                    failures.append({
                        "Patient": patient_id, "Session": session_id,
                        "Error": str(error),
                    })
            try:
                reviews = run_mcp_tool(
                    "get_review_history",
                    {"requesting_role": "physiotherapist"},
                )["reviews"]
            except Exception as error:
                failures.append({
                    "Patient": "All", "Session": "Review history",
                    "Error": str(error),
                })
        snapshot = {
            "catalog": catalog_key,
            "records": records,
            "reviews": reviews,
            "failures": failures,
            "loaded_at": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC"),
        }
        st.session_state["review_dashboard"] = snapshot

    st.caption(f"Last refreshed: {snapshot['loaded_at']}")
    if snapshot["failures"]:
        st.warning(
            "Some review information could not be loaded. Refresh after the "
            "MCP backend has been updated."
        )
        with st.expander("Technical error details", expanded=False):
            st.json(snapshot["failures"])
    if not snapshot["records"]:
        st.info("No review records are available to display.")
        return

    patient_column, status_column = st.columns(2)
    patient_filter = patient_column.selectbox(
        "Patient", ["All patients"] + sorted({r["patient_id"] for r in snapshot["records"]}),
    )
    status_filter = status_column.selectbox(
        "Review status", ["All sessions", "Reviewed", "Awaiting review"],
    )
    records = [r for r in snapshot["records"] if (
        (patient_filter == "All patients" or r["patient_id"] == patient_filter)
        and (status_filter == "All sessions"
             or (status_filter == "Reviewed" and r["review_saved"])
             or (status_filter == "Awaiting review" and not r["review_saved"]))
    )]
    reviews = [r for r in snapshot["reviews"] if (
        patient_filter == "All patients" or r["patient_id"] == patient_filter
    )]
    latest_reviews = {
        (review["patient_id"], review["session_id"]): review
        for review in reviews
        if review["is_latest"]
    }

    totals = st.columns(4)
    reviewed = sum(bool(r["review_saved"]) for r in records)
    totals[0].metric("Sessions shown", len(records))
    totals[1].metric("Reviewed", reviewed)
    totals[2].metric("Awaiting review", len(records) - reviewed)
    totals[3].metric("Saved review entries", len(reviews))
    if not records:
        st.info("No sessions match these filters.")
        return

    rows = []
    for record in records:
        review = latest_reviews.get((record["patient_id"], record["session_id"]))
        reviewed_at = pd.to_datetime(record["reviewed_at"], utc=True, errors="coerce")
        rows.append({
            "Patient": record["patient_id"],
            "Session": record["session_id"],
            "Review status": record["review_status"],
            "Fatigue review": (
                friendly_review_decision(review["fatigue_decision"])
                if review else "Not reviewed"
            ),
            "Movement review": (
                friendly_review_decision(review["movement_decision"])
                if review else "Not reviewed"
            ),
            "Comment": (
                review["note"] or "No comment provided"
                if review else "Not reviewed"
            ),
            "Reviewed at (UTC)": (
                reviewed_at.strftime("%Y-%m-%d %H:%M:%S")
                if pd.notna(reviewed_at) else "Not available"
            ),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("### Latest review details")
    reviewed_records = [
        record for record in records
        if (record["patient_id"], record["session_id"]) in latest_reviews
    ]
    if not reviewed_records:
        st.info("No reviewed sessions match these filters.")
    for record in reviewed_records:
        review = latest_reviews[(record["patient_id"], record["session_id"])]
        title = (
            f"Patient {record['patient_id']} - Session {record['session_id']} - "
            f"{record['review_status']}"
        )
        with st.expander(title, expanded=len(reviewed_records) == 1):
            fatigue_column, movement_column = st.columns(2)
            with fatigue_column:
                st.markdown("#### Fatigue review")
                st.write(f"**AI result:** {friendly_review_result(review['ai_fatigue_label'])}")
                st.write(
                    "**Physiotherapist decision:** "
                    f"{friendly_review_decision(review['fatigue_decision'])}"
                )
                st.write(
                    "**Final reviewed result:** "
                    f"{friendly_review_result(review['final_fatigue_label'])}"
                )
            with movement_column:
                st.markdown("#### Movement review")
                st.write(f"**AI result:** {friendly_review_result(review['ai_movement_label'])}")
                st.write(
                    "**Physiotherapist decision:** "
                    f"{friendly_review_decision(review['movement_decision'])}"
                )
                st.write(
                    "**Final reviewed result:** "
                    f"{friendly_review_result(review['final_movement_label'])}"
                )
            st.markdown("#### Review information")
            details = st.columns(3)
            details[0].write(f"**Staff ID:** {review['reviewer_id']}")
            reviewed_at = pd.to_datetime(review["reviewed_at"], utc=True)
            details[1].write(
                f"**Reviewed at:** {reviewed_at.strftime('%Y-%m-%d %H:%M:%S UTC')}"
            )
            details[2].write(
                "**Future AI improvement:** "
                f"{'Allowed' if review['include_for_retraining'] else 'Not allowed'}"
            )
            st.info(
                f"Comment: {review['note']}"
                if review["note"] else "Comment: No comment was provided."
            )

    with st.expander("View complete review history", expanded=False):
        st.caption(
            "Each saved review appears here. Earlier decisions and comments "
            "remain visible even after a newer review is saved."
        )
        if not reviews:
            st.info("No saved reviews match this patient selection.")
        else:
            history_rows = []
            for review in reviews:
                reviewed_at = pd.to_datetime(review["reviewed_at"], utc=True)
                history_rows.append({
                    "Patient": review["patient_id"],
                    "Session": review["session_id"],
                    "Reviewed at (UTC)": reviewed_at.strftime("%Y-%m-%d %H:%M:%S"),
                    "Staff ID": review["reviewer_id"],
                    "Fatigue - AI result": friendly_review_result(review["ai_fatigue_label"]),
                    "Fatigue - decision": friendly_review_decision(review["fatigue_decision"]),
                    "Fatigue - final result": friendly_review_result(review["final_fatigue_label"]),
                    "Movement - AI result": friendly_review_result(review["ai_movement_label"]),
                    "Movement - decision": friendly_review_decision(review["movement_decision"]),
                    "Movement - final result": friendly_review_result(review["final_movement_label"]),
                    "Comment": review["note"] or "No comment provided",
                    "Future AI improvement": (
                        "Allowed" if review["include_for_retraining"] else "Not allowed"
                    ),
                    "Latest review": "Yes" if review["is_latest"] else "No",
                    "Review ID": review["review_id"],
                })
            history_frame = pd.DataFrame(history_rows)
            st.dataframe(history_frame, hide_index=True, width="stretch")
            st.download_button(
                "Download review history for Excel or Google Sheets",
                history_frame.to_csv(index=False).encode("utf-8-sig"),
                "kinesis_review_history.csv",
                "text/csv",
            )

    st.caption(
        "The AI result is kept unchanged for traceability. The final reviewed "
        "result records the physiotherapist's decision."
    )


st.title("KINESIS Rehabilitation Session")
with st.sidebar:
    st.header("Dashboard")
    dashboard_workspace = st.radio(
        "Workspace",
        ["Live session analysis", "AI session assistant"],
    )

if dashboard_workspace == "Live session analysis":
    if SENSOR_FEED_URL:
        render_live_sensor_monitoring()
    else:
        if render_compact_model1_status(show_muscle_selector=True):
            render_synthetic_model1_demo()
    st.stop()

st.caption("Session evidence, trends, and professional review in one workspace.")
model2 = model2_status()
if model2["configured"]:
    st.success("AI session assistant ready")
    st.caption(
        "Ask about a selected session. Patients can also choose approved "
        "general rehabilitation questions. A confirmation appears only "
        "after the AI answers."
    )
else:
    st.warning(
        "AI session assistant is not connected. Contact the system "
        "administrator."
    )
render_browser_connection_indicator()
if PUBLIC_DEMO_MODE:
    st.info(
        "Read-only training records are active. Session information can be "
        "viewed, but physiotherapist reviews cannot be changed."
    )

try:
    backend_health = run_mcp_tool("health_check", {})
except Exception as error:
    show_service_error(
        "KINESIS cannot reach the data service. Check your internet "
        "connection and try again.",
        error,
    )
    st.info(
        "Previously saved information remains on the backend. Any review "
        "that did not show a success message has not been saved."
    )
    if st.button("Retry connection", type="primary"):
        st.rerun()
    if not PUBLIC_DEMO_MODE:
        with st.expander("Technical connection details", expanded=False):
            st.write("Start the MCP backend in a VS Code terminal:")
            st.code("python src\\mcp_server\\server.py", language=None)
    st.stop()

with st.sidebar:
    st.header("Session request")
    role_label = st.selectbox("I am a", list(ROLE_NAMES))
    role = ROLE_NAMES[role_label]
    available_question_modes = question_modes_for_role(role)
    question_mode = available_question_modes[0]
    if len(available_question_modes) > 1:
        question_mode = st.radio(
            "Ask about",
            available_question_modes,
        )
    st.success("KINESIS data service online")
    if role != "patient" and not PUBLIC_DEMO_MODE:
        with st.expander("Technical connection details", expanded=False):
            st.write(f"MCP backend: {backend_health['status']}")
            st.caption(MCP_SERVER_URL)

    requesting_patient_id = None
    if role == "patient" and question_mode == "A selected session":
        requesting_patient_id = st.text_input(
            "Your patient ID",
            value="P006",
            max_chars=4,
        ).upper()

if role == "patient" and question_mode == "General rehabilitation":
    general_suggestions = {
        "physiotherapist": [
            "What can surface EMG show during rehabilitation?",
            "How should fatigue-pattern scores be interpreted?",
            "What are the limitations of IMU movement analysis?",
        ],
        "clinician": [
            "How can sensor data support rehabilitation monitoring?",
            "Why is professional review needed for AI findings?",
            "What should be considered when comparing sessions?",
        ],
        "patient": [
            "What does surface EMG measure?",
            "What does a fatigue pattern mean?",
            "What can a movement sensor tell me?",
        ],
    }
    general_query_key = f"general_assistant_query_{role}"
    if general_query_key not in st.session_state:
        st.session_state[general_query_key] = general_suggestions[role][0]

    with st.sidebar:
        st.markdown("**Suggested general questions**")
        for index, suggestion in enumerate(general_suggestions[role]):
            if st.button(
                suggestion,
                key=f"general_question_suggestion_{role}_{index}",
                width="stretch",
            ):
                st.session_state[general_query_key] = suggestion

        general_query = st.text_area(
            "Ask a general rehabilitation question, or type your own",
            max_chars=500,
            key=general_query_key,
            help=(
                "This mode uses approved rehabilitation references only. It "
                "does not access a patient or session record."
            ),
        )
        with st.expander("How general questions work", expanded=False):
            st.markdown(
                "1. Ask about rehabilitation monitoring, sEMG, IMU, movement, "
                "muscle activation, or fatigue patterns.\n"
                "2. Select **Ask AI**.\n"
                "3. Read the role-appropriate explanation.\n"
                "4. Open the evidence section to inspect the approved "
                "references used.\n\n"
                "No patient or session information is loaded in this mode."
            )
        general_load_button = st.button(
            "Ask AI",
            type="primary",
            width="stretch",
            key=f"ask_general_ai_{role}",
        )

    st.subheader("General rehabilitation assistant")
    st.caption(
        "Ask an educational question without selecting or opening a patient "
        "session. Answers remain limited to the approved KINESIS references."
    )
    if general_load_button:
        try:
            with st.spinner("Retrieving approved rehabilitation references"):
                general_result = build_general_question_context(
                    role,
                    general_query,
                )
                st.session_state["general_analysis"] = {
                    "role": role,
                    "automatic_summary": general_result["automatic_summary"],
                }
            st.success("General rehabilitation information loaded successfully.")
        except Exception as error:
            show_service_error(
                "Unable to answer the general question. Please try again.",
                error,
            )

    general_analysis = st.session_state.get("general_analysis")
    if general_analysis and general_analysis.get("role") == role:
        render_role_session_answer(general_analysis["automatic_summary"])
    else:
        st.info("Choose a suggested question or type a general question to begin.")
    st.stop()

try:
    session_catalog = load_session_catalog(role, requesting_patient_id)
except Exception as error:
    show_service_error("Unable to load available sessions. Please try again.", error)
    st.stop()

if not session_catalog:
    st.info("No authorised verified sessions are available.")
    st.stop()

if role == "physiotherapist":
    with st.sidebar:
        physiotherapist_view = st.radio(
            "Workspace", ["Session review", "Review records"],
        )
    if physiotherapist_view == "Review records":
        render_review_dashboard(session_catalog)
        st.stop()

patient_options = sorted({item["patient_id"] for item in session_catalog})
with st.sidebar:
    if role == "patient":
        patient_id = requesting_patient_id
        st.caption(f"Patient record: {patient_id}")
    else:
        default_patient_index = (
            patient_options.index("P003") if "P003" in patient_options else 0
        )
        patient_id = st.selectbox(
            "Patient record",
            patient_options,
            index=default_patient_index,
        )

    session_options = [
        item["session_id"]
        for item in session_catalog
        if item["patient_id"] == patient_id
    ]
    session_id = st.selectbox(
        "Session",
        session_options,
        index=max(len(session_options) - 1, 0),
    )
    default_questions = {
        "physiotherapist": (
            "Summarise the fatigue, movement, and progress for this session."
        ),
        "clinician": "Summarise the key findings for care coordination.",
        "patient": "What is the status of this session?",
    }
    question_suggestions = {
        "physiotherapist": [
            "Summarise the fatigue, movement, and progress for this session.",
            "When did the fatigue pattern begin, and what evidence supports it?",
            "Compare this session with the patient's earlier sessions.",
        ],
        "clinician": [
            "Summarise the key findings for care coordination.",
            "Has this session been reviewed, and what follow-up is needed?",
            "Compare the fatigue and movement trends across sessions.",
        ],
        "patient": [
            "What is the status of this session, and what should I do next?",
            "What do my fatigue and movement indicators mean?",
            "How has my progress changed across my sessions?",
        ],
    }
    query_key = f"assistant_query_{role}"
    if query_key not in st.session_state:
        st.session_state[query_key] = default_questions[role]

    st.markdown("**Suggested questions**")
    for index, suggestion in enumerate(question_suggestions[role]):
        if st.button(
            suggestion,
            key=f"question_suggestion_{role}_{index}",
            width="stretch",
        ):
            st.session_state[query_key] = suggestion

    assistant_query = st.text_area(
        "Ask about this session, or type your own question",
        max_chars=500,
        key=query_key,
        help=(
            "The assistant retrieves only the session evidence permitted for "
            "your role and needed to answer this question."
        ),
    )
    with st.expander("How to use the AI assistant", expanded=False):
        st.markdown(
            "1. Select the patient and session.\n"
            "2. Enter a question about fatigue, movement, or progress.\n"
            "3. Select **Ask AI**.\n"
            "4. Look for **AI response received** above the answer.\n"
            "5. Open **Evidence used for this answer** (or **Why am I "
            "seeing this answer?** in the patient view) to inspect the "
            "authorised sources behind the response."
        )
    load_button = st.button(
        "Ask AI",
        type="primary",
        width="stretch",
    )

if "review_message" in st.session_state:
    st.success(st.session_state.pop("review_message"))

if load_button:
    try:
        with st.spinner("Loading session information"):
            automatic_summary = None
            if role == "physiotherapist":
                agent_result = asyncio.run(
                    build_session_context(
                        patient_id,
                        session_id,
                        assistant_query,
                    )
                )
                tool_name = "get_model1_outputs"
                result = agent_result["summary_context"][
                    "original_ai_findings"
                ]
                automatic_summary = agent_result["automatic_summary"]
            else:
                agent_result = asyncio.run(
                    build_role_session_context(
                        patient_id,
                        session_id,
                        role,
                        requesting_patient_id,
                        assistant_query,
                    )
                )
                tool_name = agent_result["primary_tool"]
                result = agent_result["role_evidence"]
                automatic_summary = agent_result["automatic_summary"]
            st.session_state["analysis"] = {
                "role": role,
                "patient_id": patient_id,
                "session_id": session_id,
                "tool_name": tool_name,
                "result": result,
                "automatic_summary": automatic_summary,
                "assistant_query": assistant_query,
            }
        st.success("Session information loaded successfully.")
    except Exception as error:
        show_service_error("Unable to load the session. Please try again.", error)

analysis = st.session_state.get("analysis")
if analysis and (
    analysis["role"] == role
    and analysis["patient_id"] == patient_id
    and analysis["session_id"] == session_id
):
    if role == "physiotherapist":
        render_physiotherapist(
            analysis["result"],
            analysis.get("automatic_summary"),
        )
    elif role == "clinician":
        render_clinician(
            analysis["result"],
            analysis.get("automatic_summary"),
        )
    else:
        render_patient(
            analysis["result"],
            analysis.get("automatic_summary"),
        )
else:
    st.info("Select a role and session, then view the session information.")

