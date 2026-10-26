import os
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from forearm_fatigue_runtime import (
    load_forearm_fatigue_model,
    predict_normalized_fatigue_window,
)
from forearm_movement_runtime import (
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


def show_service_error(message: str, error: Exception) -> None:
    st.error(message)
    with st.expander("Technical error details", expanded=False):
        st.code(str(error), language=None)


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


st.title("KINESIS Rehabilitation Session")
with st.sidebar:
    st.header("Dashboard")
    dashboard_workspace = st.radio(
        "Workspace",
        ["Specialist review (KINESIS app)", "Live session analysis"],
    )

if dashboard_workspace == "Specialist review (KINESIS app)":
    # Sessions handed over by the patient app under report/; needs no MCP backend.
    from specialist_dashboard import render_specialist_dashboard

    render_specialist_dashboard()
    st.stop()

if dashboard_workspace == "Live session analysis":
    if SENSOR_FEED_URL:
        render_live_sensor_monitoring()
    else:
        if render_compact_model1_status(show_muscle_selector=True):
            render_synthetic_model1_demo()
    st.stop()
