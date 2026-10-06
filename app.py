import io
import os

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go

from Hydromatic_Simulator.model.model import GeneratorModel
from Hydromatic_Simulator.utils.config import config


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hydromatic Simulator",
    page_icon="💧",
    layout="wide",
)

S = config["structure-dim"]
NUM_COORD = config["num-coord"]
NUM_TIMESTEPS = config["num-timesteps"]

TIMESTEPS = ["0 min"] + config["timesteps"]

AHC_PATTERNS = {
    "AHC-1": "10101010101",
    "AHC-2": "1100110011",
    "AHC-6": "111111",
}

AHC_VALUES = {
    "AHC-1": 1,
    "AHC-2": 2,
    "AHC-6": 3,
}


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading trained Hydromatic models...")
def load_web_models():
    """
    Load all 16 trained GeneratorModel instances.

    This intentionally avoids loading test_data.pkl.
    The web application only needs the trained weights and
    the model architecture.
    """

    models = []

    weights_dir = os.path.join(
        "Hydromatic_Simulator",
        "model",
        "weights",
    )

    for i in range(NUM_COORD):

        model = GeneratorModel()

        # Build the model before loading the submodel weights.
        dummy_design = tf.zeros(
            (1, config["structure-dim"]),
            dtype=tf.float32,
        )

        dummy_position = tf.zeros(
            (1, config["nodal-dim"]),
            dtype=tf.float32,
        )

        _ = model.recursive_generate(
            dummy_design,
            dummy_position,
            training=False,
        )

        encoder_path = os.path.join(
            weights_dir,
            f"encoder_{NUM_COORD}coords_{i}.weights.h5",
        )

        decoder_path = os.path.join(
            weights_dir,
            f"decoder_{NUM_COORD}coords_{i}.weights.h5",
        )

        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Missing encoder weight file:\n{encoder_path}"
            )

        if not os.path.exists(decoder_path):
            raise FileNotFoundError(
                f"Missing decoder weight file:\n{decoder_path}"
            )

        model.encoder.load_weights(encoder_path)
        model.decoder.load_weights(decoder_path)

        models.append(model)

    return models


# ============================================================
# INFERENCE
# ============================================================

def run_web_inference(models, binary_string):
    """
    Run the trained 16-coordinate generator models.

    Returns
    -------
    prediction : np.ndarray
        Shape:
            (16, 9, 2)

        16 = predicted nodal coordinates
        9  = deformation timesteps
        2  = x/y coordinates
    """

    design_input = np.asarray(
        [int(x) for x in binary_string],
        dtype=np.float32,
    )

    if design_input.shape[0] != S:
        raise ValueError(
            f"Expected {S} binary values, "
            f"received {design_input.shape[0]}."
        )

    predictions = []

    initial_positions = config["init-pos"]

    for i, model in enumerate(models):

        design_tensor = tf.convert_to_tensor(
            design_input[None, :],
            dtype=tf.float32,
        )

        initial_position = tf.convert_to_tensor(
            initial_positions[i][None, :],
            dtype=tf.float32,
        )

        prediction = model.recursive_generate(
            design_tensor,
            initial_position,
            training=False,
        )

        prediction = prediction.numpy()

        # Expected shape: (1, 9, 2)
        prediction = np.squeeze(prediction, axis=0)

        predictions.append(prediction)

    return np.stack(predictions, axis=0)


# ============================================================
# DESIGN VALIDATION
# ============================================================

def validate_binary_code(binary_string):
    if len(binary_string) != S:
        return False, f"Binary code must contain exactly {S} bits."

    if any(char not in "01" for char in binary_string):
        return False, "Binary code may contain only 0 and 1."

    if set(binary_string) == {"0"}:
        return False, "The design cannot be an all-zero structure."

    return True, ""


# ============================================================
# DESIGN EDITOR
# ============================================================

def initialize_editor():
    if "block_values" not in st.session_state:
        st.session_state.block_values = [0] * S

    if "placed_blocks" not in st.session_state:
        st.session_state.placed_blocks = []


def reset_editor():
    st.session_state.block_values = [0] * S
    st.session_state.placed_blocks = []


def can_place_block(start, pattern_length):
    end = start + pattern_length - 1

    # Must remain within the 65 mm structure.
    if start < 0 or end >= S:
        return False, "The structure would extend beyond the 65 mm domain."

    # Maximum of 3 structures.
    if len(st.session_state.placed_blocks) >= 3:
        return False, "A maximum of three structures can be placed."

    # Prevent overlap.
    new_range = set(range(start, end + 1))

    for old_start, old_end, _ in st.session_state.placed_blocks:

        old_range = set(range(old_start, old_end + 1))

        if new_range.intersection(old_range):
            return False, "The new structure overlaps an existing structure."

        # Minimum 5 mm gap.
        if abs(start - old_end) < 5:
            return False, "Structures must be separated by at least 5 mm."

        if abs(old_start - end) < 5:
            return False, "Structures must be separated by at least 5 mm."

    return True, ""


def place_block(ahc_name, start):
    pattern = AHC_PATTERNS[ahc_name]
    value = AHC_VALUES[ahc_name]

    valid, message = can_place_block(
        start,
        len(pattern),
    )

    if not valid:
        return False, message

    end = start + len(pattern) - 1

    # Write pattern into 65-bit design.
    for j, bit in enumerate(pattern):
        st.session_state.block_values[start + j] = (
            value if bit == "1" else 0
        )

    st.session_state.placed_blocks.append(
        (start, end, value)
    )

    return True, ""


def draw_design():
    """
    Display the current 65-position design as a colored grid.
    """

    values = st.session_state.block_values

    html = """
    <div style="
        width: 100%;
        overflow-x: auto;
        padding: 10px 0;
    ">
        <div style="
            display: flex;
            min-width: 780px;
            border: 1px solid #888;
        ">
    """

    colors = {
        0: "#eeeeee",
        1: "#00bfff",
        2: "#00bfff",
        3: "#00bfff",
    }

    for i, value in enumerate(values):

        html += f"""
        <div title="{i} mm: {value}" style="
            width: 12px;
            height: 35px;
            background: {colors[value]};
            border-right: 1px solid #ffffff;
            position: relative;
        ">
        """

        if i % 5 == 0:
            html += f"""
            <span style="
                position: absolute;
                top: 38px;
                left: -2px;
                font-size: 9px;
            ">
                {i}
            </span>
            """

        html += "</div>"

    html += """
        </div>
    </div>
    """

    st.markdown(html, unsafe_allow_html=True)


def binary_code():
    return "".join(
        str(x)
        for x in st.session_state.block_values
    )


# ============================================================
# PREDICTION VISUALIZATION
# ============================================================

def draw_contour(centerline, width=2.0):
    """
    Reproduce the rectangular contour construction used by
    the original visualization.py.
    """

    centerline = np.asarray(centerline)

    x = centerline[:, 0]
    y = centerline[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)

    tangent_norm = np.sqrt(dx**2 + dy**2)

    tangent_norm[tangent_norm == 0] = 1.0

    nx = -dy / tangent_norm
    ny = dx / tangent_norm

    x1 = x + nx * width / 2
    y1 = y + ny * width / 2

    x2 = x - nx * width / 2
    y2 = y - ny * width / 2

    contour_x = np.concatenate(
        [x1, x2[::-1]]
    )

    contour_y = np.concatenate(
        [y1, y2[::-1]]
    )

    return contour_x, contour_y


def make_frame(prediction, timestep):
    """
    Construct one visualization frame.

    timestep = 0:
        undeformed initial configuration

    timestep > 0:
        predicted configuration
    """

    if timestep == 0:

        x = np.linspace(
            4.0625,
            65.0,
            16,
        )

        y = np.zeros(16)

        centerline = np.column_stack(
            [
                np.concatenate([[0.0], x]),
                np.concatenate([[0.0], y]),
            ]
        )

        condition = "50°C (as-prepared)"

    else:

        xy = prediction[:, timestep - 1, :]

        centerline = np.vstack(
            [
                np.array([[0.0, 0.0]]),
                xy,
            ]
        )

        if timestep == NUM_TIMESTEPS:
            condition = "20°C (equilibrium)"
        else:
            condition = "20°C"

    contour_x, contour_y = draw_contour(
        centerline,
        width=2,
    )

    return centerline, contour_x, contour_y, condition


def create_animation(prediction):

    centerline, contour_x, contour_y, condition = make_frame(
        prediction,
        0,
    )

    fig = go.Figure()

    # Centerline
    fig.add_trace(
        go.Scatter(
            x=centerline[:, 0],
            y=centerline[:, 1],
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=6),
            name="Predicted centerline",
        )
    )

    # Contour
    fig.add_trace(
        go.Scatter(
            x=contour_x,
            y=contour_y,
            mode="lines",
            fill="toself",
            name="Actuator contour",
        )
    )

    frames = []

    for t in range(NUM_TIMESTEPS + 1):

        centerline, contour_x, contour_y, condition = make_frame(
            prediction,
            t,
        )

        frames.append(
            go.Frame(
                name=str(t),
                data=[
                    go.Scatter(
                        x=centerline[:, 0],
                        y=centerline[:, 1],
                        mode="lines+markers",
                        line=dict(width=3),
                        marker=dict(size=6),
                    ),
                    go.Scatter(
                        x=contour_x,
                        y=contour_y,
                        mode="lines",
                        fill="toself",
                    ),
                ],
            )
        )

    fig.frames = frames

    fig.update_layout(
        title=dict(
            text=f"Deformation: {TIMESTEPS[0]} — {condition}",
        ),
        xaxis=dict(
            title="x (mm)",
            range=[-70, 80],
            zeroline=True,
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title="y (mm)",
            range=[-90, 60],
            zeroline=True,
        ),
        height=650,
        margin=dict(
            l=50,
            r=30,
            t=80,
            b=50,
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.12,
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": 200,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 0,
                                },
                                "fromcurrent": True,
                            },
                        ],
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False,
                                },
                                "mode": "immediate",
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": 0,
                "x": 0.15,
                "y": 1.05,
                "len": 0.75,
                "steps": [
                    {
                        "label": TIMESTEPS[t],
                        "method": "animate",
                        "args": [
                            [str(t)],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 0,
                                },
                            },
                        ],
                    }
                    for t in range(NUM_TIMESTEPS + 1)
                ],
            }
        ],
    )

    return fig


# ============================================================
# RESULT TABLE
# ============================================================

def prediction_dataframe(prediction):

    rows = []

    for node in range(NUM_COORD):

        for t in range(NUM_TIMESTEPS):

            rows.append(
                {
                    "node": node + 1,
                    "time": config["timesteps"][t],
                    "x_mm": prediction[node, t, 0],
                    "y_mm": prediction[node, t, 1],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN APPLICATION
# ============================================================

initialize_editor()

st.title("💧 Hydromatic Simulator")

st.markdown(
    """
Predict the time-dependent deformation of a designed
hydrogel actuator using the trained Hydromatic Simulator model.
"""
)

# ------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------

with st.sidebar:

    st.header("Design")

    st.markdown(
        f"""
        **Structure length:** {S} mm

        **Number of predicted nodes:** {NUM_COORD}

        **Predicted time points:** {NUM_TIMESTEPS}
        """
    )

    st.divider()

    st.subheader("AHC structures")

    for name, pattern in AHC_PATTERNS.items():

        st.code(pattern)

    st.caption(
        "The design editor supports up to three AHC structures "
        "with a minimum 5 mm separation."
    )


# ------------------------------------------------------------
# DESIGN EDITOR
# ------------------------------------------------------------

st.header("1. Configure the actuator")

draw_design()

col1, col2 = st.columns(2)

with col1:

    ahc_name = st.selectbox(
        "Structure type",
        list(AHC_PATTERNS.keys()),
    )

with col2:

    max_start = S - len(AHC_PATTERNS[ahc_name])

    start = st.number_input(
        "Starting position (mm)",
        min_value=0,
        max_value=max_start,
        value=0,
        step=1,
    )

if st.button(
    "Place structure",
    type="secondary",
):

    success, message = place_block(
        ahc_name,
        int(start),
    )

    if success:
        st.success(
            f"{ahc_name} placed at {start} mm."
        )
        st.rerun()

    else:
        st.error(message)


col1, col2 = st.columns(2)

with col1:

    if st.button("Reset design"):

        reset_editor()
        st.rerun()

with col2:

    if st.button("Print binary code"):

        st.code(binary_code())


st.markdown("### Current 65-bit design")

draw_design()

current_code = binary_code()

st.code(current_code)


# ------------------------------------------------------------
# ADVANCED INPUT
# ------------------------------------------------------------

with st.expander("Advanced: enter a 65-bit binary code"):

    manual_code = st.text_input(
        "65-bit binary code",
        value=current_code,
        max_chars=S,
    )

    if st.button("Use manual code"):

        valid, message = validate_binary_code(
            manual_code
        )

        if not valid:
            st.error(message)

        else:

            st.session_state.block_values = [
                int(x)
                for x in manual_code
            ]

            st.success("Manual design accepted.")
            st.rerun()


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

st.header("2. Predict deformation")

valid, message = validate_binary_code(
    current_code
)

if not valid:
    st.warning(message)

if st.button(
    "Run prediction",
    type="primary",
    disabled=not valid,
):

    with st.spinner(
        "Running the trained Hydromatic Simulator..."
    ):

        try:

            models = load_web_models()

            prediction = run_web_inference(
                models,
                current_code,
            )

            st.session_state.prediction = prediction
            st.session_state.prediction_code = current_code

        except Exception as e:

            st.error(
                "The prediction could not be completed."
            )

            st.exception(e)


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

if "prediction" in st.session_state:

    prediction = st.session_state.prediction

    st.header("3. Predicted deformation")

    st.plotly_chart(
        create_animation(prediction),
        use_container_width=True,
    )

    st.caption(
        "The animation follows the original model visualization: "
        "0 min at 50°C, followed by the predicted 20°C deformation "
        "sequence through equilibrium."
    )

    st.subheader("Predicted nodal coordinates")

    df = prediction_dataframe(prediction)

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    csv_data = df.to_csv(index=False).encode(
        "utf-8"
    )

    st.download_button(
        label="Download prediction as CSV",
        data=csv_data,
        file_name="hydromatic_prediction.csv",
        mime="text/csv",
    )

    st.subheader("Input design")

    st.code(
        st.session_state.prediction_code
    )
