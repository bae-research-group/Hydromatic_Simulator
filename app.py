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
    page_icon="🔵",
    layout="wide",
)

S = config["structure-dim"]
NUM_COORD = config["num-coord"]
NUM_TIMESTEPS = config["num-timesteps"]

TIMESTEPS = ["0 min"] + config["timesteps"]


# ============================================================
# AHC PATTERNS
# ============================================================

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
# SESSION STATE
# ============================================================

def initialize_editor():

    if "block_values" not in st.session_state:
        st.session_state.block_values = [0] * S

    if "placed_blocks" not in st.session_state:
        st.session_state.placed_blocks = []

    if "prediction" not in st.session_state:
        st.session_state.prediction = None

    if "prediction_code" not in st.session_state:
        st.session_state.prediction_code = None


def reset_editor():

    st.session_state.block_values = [0] * S
    st.session_state.placed_blocks = []

    st.session_state.prediction = None
    st.session_state.prediction_code = None


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading trained Hydromatic models...")
def load_web_models():
    """
    Load all 16 trained GeneratorModel instances.

    This intentionally avoids test_data.pkl.
    The web application only needs the trained weights
    and model architecture.
    """

    models = []

    weights_dir = os.path.join(
        "Hydromatic_Simulator",
        "model",
        "weights",
    )

    for i in range(NUM_COORD):

        model = GeneratorModel()

        # Build the model before loading weights.
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
        prediction = np.squeeze(
            prediction,
            axis=0,
        )

        predictions.append(prediction)

    return np.stack(
        predictions,
        axis=0,
    )


# ============================================================
# DESIGN VALIDATION
# ============================================================

def validate_binary_code(binary_string):

    if len(binary_string) != S:

        return (
            False,
            f"Binary code must contain exactly {S} bits.",
        )

    if any(char not in "01" for char in binary_string):

        return (
            False,
            "Binary code may contain only 0 and 1.",
        )

    if set(binary_string) == {"0"}:

        return (
            False,
            "The design cannot be an all-zero structure.",
        )

    return True, ""


# ============================================================
# AHC PLACEMENT
# ============================================================

def can_place_block(start, pattern_length):

    end = start + pattern_length - 1

    # Maximum of three structures.
    if len(st.session_state.placed_blocks) >= 3:

        return (
            False,
            "A maximum of three AHC structures can be placed.",
        )

    # Must remain inside the 65-mm actuator.
    if start < 0 or end >= S:

        return (
            False,
            "The AHC structure would exceed the 65-mm actuator.",
        )

    new_range = set(
        range(
            start,
            end + 1,
        )
    )

    # Check against existing structures.
    for old_start, old_end, _ in st.session_state.placed_blocks:

        old_range = set(
            range(
                old_start,
                old_end + 1,
            )
        )

        # No overlap.
        if new_range.intersection(old_range):

            return (
                False,
                "The new structure overlaps an existing AHC structure.",
            )

        # Minimum 5-mm separation.
        if abs(start - old_end) < 5:

            return (
                False,
                "AHC structures must have at least a 5-mm gap.",
            )

        if abs(old_start - end) < 5:

            return (
                False,
                "AHC structures must have at least a 5-mm gap.",
            )

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

    # Write the pattern into the 65-bit design.
    for j, bit in enumerate(pattern):

        st.session_state.block_values[
            start + j
        ] = value if bit == "1" else 0

    st.session_state.placed_blocks.append(
        (
            start,
            end,
            value,
        )
    )

    return True, ""


# ============================================================
# GRAPHICAL AHC PATTERN DISPLAY
# ============================================================

def draw_ahc_patterns():
    """
    Display AHC-1, AHC-2, and AHC-6 as graphical patterns,
    matching the visual language of the original Tkinter GUI.
    """

    st.markdown("### AHC structures")

    for name, pattern in AHC_PATTERNS.items():

        cells = ""

        for bit in pattern:

            if bit == "1":

                background = "#00BFFF"

            else:

                background = "#808080"

            cells += f"""
<div style="
    width:16px;
    height:28px;
    background:{background};
    border-right:1px solid white;
    box-sizing:border-box;
"></div>
"""

        html = f"""
<div style="
    margin-bottom:20px;
">

<div style="
    font-weight:bold;
    font-size:16px;
    margin-bottom:5px;
">
{name}
</div>

<div style="
    display:flex;
    width:max-content;
    border:1px solid #777;
">
{cells}
</div>

<div style="
    font-family:monospace;
    font-size:12px;
    color:#555;
    margin-top:4px;
">
{pattern}
</div>

</div>
"""

        st.markdown(
            html,
            unsafe_allow_html=True,
        )


# ============================================================
# GRAPHICAL 65-MM ACTUATOR DESIGN
# ============================================================

def draw_design():

    """
    Display the current 65-mm actuator as a graphical strip.

    Gray:
        Empty actuator

    Blue:
        AHC structure
    """

    values = st.session_state.block_values

    cells = ""

    for i, value in enumerate(values):

        if value == 0:

            background = "#D3D3D3"

        else:

            background = "#00BFFF"

        cells += f"""
<div
    title="{i} mm: {value}"
    style="
        width:10px;
        height:30px;
        background:{background};
        border-right:1px solid white;
        box-sizing:border-box;
    ">
</div>
"""

    html = f"""
<div style="
    width:100%;
    overflow-x:auto;
    padding:15px 0 45px 0;
">

<div style="
    min-width:700px;
">

<div style="
    font-size:20px;
    font-weight:bold;
    text-align:center;
    margin-bottom:10px;
">
Hydromatic Actuator
</div>

<div style="
    display:flex;
    width:650px;
    height:30px;
    border:1px solid #777;
">
{cells}
</div>

<div style="
    width:650px;
    display:flex;
    justify-content:space-between;
    margin-top:8px;
    font-size:13px;
">
<span>0</span>
<span>10</span>
<span>20</span>
<span>30</span>
<span>40</span>
<span>50</span>
<span>60</span>
<span>65 mm</span>
</div>

<div style="
    width:650px;
    margin-top:10px;
    border-top:2px solid #222;
    position:relative;
">
<span style="
    position:absolute;
    right:-25px;
    top:-13px;
    font-size:18px;
">
x →
</span>
</div>

</div>

</div>
"""

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# BINARY CODE
# ============================================================

def binary_code():

    return "".join(
        str(x)
        for x in st.session_state.block_values
    )


# ============================================================
# PREDICTION VISUALIZATION
# ============================================================

def draw_contour(
    centerline,
    width=2.0,
):

    centerline = np.asarray(
        centerline
    )

    x = centerline[:, 0]
    y = centerline[:, 1]

    dx = np.gradient(x)
    dy = np.gradient(y)

    tangent_norm = np.sqrt(
        dx**2 + dy**2
    )

    tangent_norm[
        tangent_norm == 0
    ] = 1.0

    nx = -dy / tangent_norm
    ny = dx / tangent_norm

    x1 = x + nx * width / 2
    y1 = y + ny * width / 2

    x2 = x - nx * width / 2
    y2 = y - ny * width / 2

    contour_x = np.concatenate(
        [
            x1,
            x2[::-1],
        ]
    )

    contour_y = np.concatenate(
        [
            y1,
            y2[::-1],
        ]
    )

    return (
        contour_x,
        contour_y,
    )


def make_frame(
    prediction,
    timestep,
):

    if timestep == 0:

        x = np.linspace(
            4.0625,
            65.0,
            16,
        )

        y = np.zeros(16)

        centerline = np.column_stack(
            [
                np.concatenate(
                    [
                        [0.0],
                        x,
                    ]
                ),
                np.concatenate(
                    [
                        [0.0],
                        y,
                    ]
                ),
            ]
        )

        condition = "50°C (as-prepared)"

    else:

        xy = prediction[
            :,
            timestep - 1,
            :,
        ]

        centerline = np.vstack(
            [
                np.array(
                    [
                        [0.0, 0.0]
                    ]
                ),
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

    return (
        centerline,
        contour_x,
        contour_y,
        condition,
    )


# ============================================================
# PLOTLY ANIMATION
# ============================================================

def create_animation(prediction):

    (
        centerline,
        contour_x,
        contour_y,
        condition,
    ) = make_frame(
        prediction,
        0,
    )

    fig = go.Figure()

    # --------------------------------------------------------
    # Initial centerline
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Initial contour
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=contour_x,
            y=contour_y,
            mode="lines",
            fill="toself",
            name="Actuator contour",
        )
    )

    # --------------------------------------------------------
    # Animation frames
    # --------------------------------------------------------

    frames = []

    for t in range(
        NUM_TIMESTEPS + 1
    ):

        (
            centerline,
            contour_x,
            contour_y,
            frame_condition,
        ) = make_frame(
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

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(

        title=dict(
            text=(
                f"Deformation: "
                f"{TIMESTEPS[0]} — "
                f"{condition}"
            )
        ),

        xaxis=dict(
            title="x (mm)",
            range=[
                -70,
                80,
            ],
            zeroline=True,
            scaleanchor="y",
            scaleratio=1,
        ),

        yaxis=dict(
            title="y (mm)",
            range=[
                -90,
                60,
            ],
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

                    for t in range(
                        NUM_TIMESTEPS + 1
                    )
                ],
            }
        ],
    )

    return fig


# ============================================================
# RESULT DATAFRAME
# ============================================================

def prediction_dataframe(prediction):

    rows = []

    for node in range(
        NUM_COORD
    ):

        for t in range(
            NUM_TIMESTEPS
        ):

            rows.append(
                {
                    "node": node + 1,

                    "time": config[
                        "timesteps"
                    ][t],

                    "x_mm": prediction[
                        node,
                        t,
                        0,
                    ],

                    "y_mm": prediction[
                        node,
                        t,
                        1,
                    ],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MAIN APPLICATION
# ============================================================

initialize_editor()


# ============================================================
# TITLE
# ============================================================

st.title(
    "🔵 Hydromatic Simulator"
)

st.markdown(
    """
Predict the time-dependent deformation of a designed
hydrogel actuator using the trained Hydromatic Simulator model.
"""
)


# ============================================================
# SIDEBAR
# ============================================================

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

    draw_ahc_patterns()

    st.caption(
        "The design editor supports up to three AHC structures "
        "with a minimum 5 mm separation."
    )


# ============================================================
# DESIGN EDITOR
# ============================================================

st.header(
    "1. Configure the actuator"
)

st.markdown(
    """
Place two or three AHC structures along the
65-mm Hydromatic actuator.
"""
)

draw_design()


# ============================================================
# PLACEMENT CONTROLS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    ahc_name = st.selectbox(
        "Structure type",
        list(
            AHC_PATTERNS.keys()
        ),
    )


with col2:

    max_start = (
        S
        - len(
            AHC_PATTERNS[
                ahc_name
            ]
        )
    )

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
            f"{ahc_name} placed at "
            f"{start} mm."
        )

        st.rerun()

    else:

        st.error(
            message
        )


# ============================================================
# DESIGN CONTROLS
# ============================================================

col1, col2 = st.columns(2)


with col1:

    if st.button(
        "Reset design"
    ):

        reset_editor()

        st.rerun()


with col2:

    if st.button(
        "Print Code"
    ):

        if len(
            st.session_state.placed_blocks
        ) <= 1:

            st.error(
                "Double/triple structure requires "
                "at least two AHC structures."
            )

        else:

            st.success(
                "Last valid actuator design:"
            )

            st.code(
                binary_code()
            )


# ============================================================
# CURRENT DESIGN
# ============================================================

st.markdown(
    "### Current 65-bit design"
)

draw_design()

current_code = binary_code()

st.code(
    current_code
)


# ============================================================
# ADVANCED BINARY INPUT
# ============================================================

with st.expander(
    "Advanced: enter a 65-bit binary code"
):

    manual_code = st.text_input(
        "65-bit binary code",
        value=current_code,
        max_chars=S,
    )

    if st.button(
        "Use manual code"
    ):

        valid, message = (
            validate_binary_code(
                manual_code
            )
        )

        if not valid:

            st.error(
                message
            )

        else:

            st.session_state.block_values = [
                int(x)
                for x in manual_code
            ]

            # Manual input represents a valid design,
            # but the exact AHC block history is unknown.
            st.session_state.placed_blocks = []

            st.success(
                "Manual design accepted."
            )

            st.rerun()


# ============================================================
# PREDICTION
# ============================================================

st.header(
    "2. Predict deformation"
)


valid, message = (
    validate_binary_code(
        current_code
    )
)


if len(
    st.session_state.placed_blocks
) < 2:

    st.info(
        "Place at least two AHC structures "
        "before running the prediction."
    )

    valid = False


if not valid:

    st.warning(
        message
        if message
        else
        "A valid actuator design is required."
    )


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

            st.session_state.prediction = (
                prediction
            )

            st.session_state.prediction_code = (
                current_code
            )

            st.success(
                "Prediction completed."
            )

        except Exception as e:

            st.error(
                "The prediction could not be completed."
            )

            st.exception(e)


# ============================================================
# RESULTS
# ============================================================

if (
    st.session_state.prediction
    is not None
):

    prediction = (
        st.session_state.prediction
    )

    st.header(
        "3. Predicted deformation"
    )

    st.plotly_chart(
        create_animation(
            prediction
        ),
        use_container_width=True,
    )

    st.caption(
        "The animation follows the original model "
        "visualization: 0 min at 50°C (as-prepared), "
        "followed by the predicted 20°C deformation "
        "sequence through equilibrium."
    )

    # --------------------------------------------------------
    # DATA TABLE
    # --------------------------------------------------------

    st.subheader(
        "Predicted nodal coordinates"
    )

    df = prediction_dataframe(
        prediction
    )

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # CSV DOWNLOAD
    # --------------------------------------------------------

    csv_data = (
        df
        .to_csv(index=False)
        .encode("utf-8")
    )

    st.download_button(
        label="Download prediction as CSV",
        data=csv_data,
        file_name="hydromatic_prediction.csv",
        mime="text/csv",
    )

    # --------------------------------------------------------
    # INPUT DESIGN
    # --------------------------------------------------------

    st.subheader(
        "Input design"
    )

    st.code(
        st.session_state.prediction_code
    )
