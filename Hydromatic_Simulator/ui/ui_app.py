import os
import textwrap

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go


from Hydromatic_Simulator.model.model import GeneratorModel
from Hydromatic_Simulator.utils.config import config


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hydromatic Simulator",
    page_icon="🔵",
    layout="centered",
)

S = config["structure-dim"]
NUM_COORD = config["num-coord"]
NUM_TIMESTEPS = config["num-timesteps"]

TIMESTEPS = ["0 min"] + config["timesteps"]


# ============================================================
# AHC STRUCTURES
# ============================================================

AHC_PATTERNS = {
    1: "10101010101",
    2: "1100110011",
    3: "111111",
}

AHC_LABELS = {
    1: "AHC-1",
    2: "AHC-2",
    3: "AHC-6",
}

AHC_COLORS = {
    1: "#00BFFF",
    2: "#00BFFF",
    3: "#00BFFF",
}


# ============================================================
# SESSION STATE
# ============================================================

def initialize_state():

    defaults = {
        "models_loaded": False,
        "models": None,

        "binary_code": "",
        "prediction": None,
        "prediction_code": "",

        "block_values": [0] * S,
        "placed_blocks": [],

        "animation_started": False,

        "editor_message": "",
        "editor_message_type": "",
    }

    for key, value in defaults.items():

        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_web_models():

    """
    Load all 16 trained GeneratorModel instances.

    This web-specific loader does not depend on test_data.pkl.
    It constructs each model using dummy inputs and then loads
    the trained encoder/decoder weights.
    """

    models = []

    weights_dir = os.path.join(
        "Hydromatic_Simulator",
        "model",
        "weights",
    )

    for i in range(NUM_COORD):

        model = GeneratorModel()

        # ----------------------------------------------------
        # Build model
        # ----------------------------------------------------

        dummy_design = tf.zeros(
            (
                1,
                config["structure-dim"],
            ),
            dtype=tf.float32,
        )

        dummy_position = tf.zeros(
            (
                1,
                config["nodal-dim"],
            ),
            dtype=tf.float32,
        )

        _ = model.recursive_generate(
            dummy_design,
            dummy_position,
            training=False,
        )

        # ----------------------------------------------------
        # Weight paths
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Load trained weights
        # ----------------------------------------------------

        model.encoder.load_weights(
            encoder_path
        )

        model.decoder.load_weights(
            decoder_path
        )

        models.append(model)

    return models


# ============================================================
# WEB INFERENCE
# ============================================================

def run_web_inference(
    models,
    binary_string,
):

    design_input = np.asarray(
        [
            int(x)
            for x in binary_string
        ],
        dtype=np.float32,
    )

    if len(design_input) != S:

        raise ValueError(
            f"Expected {S} binary values, "
            f"received {len(design_input)}."
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

        # (1, 9, 2) -> (9, 2)
        prediction = np.squeeze(
            prediction,
            axis=0,
        )

        predictions.append(
            prediction
        )

    # Final shape:
    # (16, 9, 2)

    return np.stack(
        predictions,
        axis=0,
    )


# ============================================================
# DESIGN VALIDATION
# ============================================================

def validate_binary_code(
    binary_string,
):

    if len(binary_string) != S:

        return (
            False,
            f"Enter exactly {S} binary digits (0 or 1).",
        )

    if not all(
        c in "01"
        for c in binary_string
    ):

        return (
            False,
            f"Enter exactly {S} binary digits (0 or 1).",
        )

    if binary_string == "0" * S:

        return (
            False,
            "Enter valid design.",
        )

    return True, ""


# ============================================================
# DESIGN EDITOR
# ============================================================

def reset_editor():

    st.session_state.block_values = [0] * S

    st.session_state.placed_blocks = []

    st.session_state.editor_message = ""

    st.session_state.editor_message_type = ""


def can_place_block(
    new_start,
    block_type,
):

    pattern = AHC_PATTERNS[block_type]

    new_end = (
        new_start
        + len(pattern)
        - 1
    )

    # --------------------------------------------------------
    # Maximum 3 structures
    # --------------------------------------------------------

    if len(
        st.session_state.placed_blocks
    ) >= 3:

        return (
            False,
            "Error: Max 3 AHC structures to place.",
        )

    # --------------------------------------------------------
    # Actuator length
    # --------------------------------------------------------

    if new_start < 0:

        return (
            False,
            "Error: Invalid starting position.",
        )

    if new_end >= S:

        remaining = S - new_start

        required = len(pattern)

        return (
            False,
            (
                f"Error: AHC structure too long for "
                f"starting position {new_start} mm "
                f"(needs {required - remaining} mm more)."
            ),
        )

    # --------------------------------------------------------
    # Check separation
    # --------------------------------------------------------

    for (
        old_start,
        old_end,
        _,
    ) in st.session_state.placed_blocks:

        # Overlap
        if not (
            new_end < old_start
            or new_start > old_end
        ):

            return (
                False,
                "Error: AHC structures overlap.",
            )

        # New block is to the right
        if new_start > old_end:

            gap = (
                new_start
                - old_end
                - 1
            )

            if gap < 5:

                return (
                    False,
                    (
                        "Error: Too close to another "
                        "AHC structure (less than 5 mm gap)."
                    ),
                )

        # New block is to the left
        if new_end < old_start:

            gap = (
                old_start
                - new_end
                - 1
            )

            if gap < 5:

                return (
                    False,
                    (
                        "Error: Too close to another "
                        "AHC structure (less than 5 mm gap)."
                    ),
                )

    return True, ""


def place_block(
    block_type,
    start,
):

    valid, message = can_place_block(
        start,
        block_type,
    )

    if not valid:

        st.session_state.editor_message = message

        st.session_state.editor_message_type = "error"

        return False

    pattern = AHC_PATTERNS[block_type]

    end = (
        start
        + len(pattern)
        - 1
    )

    # --------------------------------------------------------
    # Write pattern into actuator
    # --------------------------------------------------------

    for i, bit in enumerate(pattern):

        st.session_state.block_values[
            start + i
        ] = int(bit)

    # --------------------------------------------------------
    # Store placed structure
    # --------------------------------------------------------

    st.session_state.placed_blocks.append(
        (
            start,
            end,
            block_type,
        )
    )

    st.session_state.editor_message = (
        f"{AHC_LABELS[block_type]} placed at "
        f"{start} mm."
    )

    st.session_state.editor_message_type = "success"

    return True


# ============================================================
# DESIGN STRIP
# ============================================================

def draw_design_strip():

    values = (
        st.session_state.block_values
    )

    html = """
    <div style="
        width: 100%;
        overflow-x: auto;
        padding: 10px 0 42px 0;
    ">

        <div style="
            position: relative;
            min-width: 660px;
        ">

            <div style="
                display: flex;
                height: 32px;
                border: 1px solid #777777;
                background: #d3d3d3;
            ">
    """

    for i, value in enumerate(values):

        if value == 0:

            background = "#D3D3D3"

        else:

            background = AHC_COLORS[
                value
            ]

        html += f"""
                <div
                    title="{i} mm"
                    style="
                        width: 10px;
                        height: 30px;
                        background: {background};
                        border-right: 1px solid white;
                        flex-shrink: 0;
                    "
                ></div>
        """

    html += """
            </div>

            <div style="
                position: relative;
                height: 35px;
            ">
    """

    # x-axis labels every 10 mm

    for x in range(
        0,
        S + 1,
        10,
    ):

        left = (
            x / S
        ) * 100

        html += f"""
                <span style="
                    position: absolute;
                    left: {left}%;
                    transform: translateX(-50%);
                    top: 4px;
                    font-size: 12px;
                    color: #222222;
                ">
                    {x}
                </span>
        """

    html += """
            </div>

            <div style="
                text-align: right;
                font-size: 13px;
                margin-top: -8px;
                margin-right: 5px;
            ">
                x (mm)
            </div>

        </div>

    </div>
    """

    html = textwrap.dedent(
        html
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# ============================================================
# AHC SAMPLE VISUALIZATION
# ============================================================

def draw_ahc_sample(
    block_type,
):

    pattern = AHC_PATTERNS[
        block_type
    ]

    cells = ""

    for bit in pattern:

        if bit == "1":

            background = "#00BFFF"

        else:

            background = "#808080"

        cells += f"""
        <div style="
            width: 12px;
            height: 25px;
            background: {background};
            margin-right: 1px;
        "></div>
        """

    html = f"""
    <div style="
        text-align: center;
        margin-bottom: 5px;
    ">

        <div style="
            font-weight: bold;
            font-size: 16px;
            margin-bottom: 5px;
        ">
            {AHC_LABELS[block_type]}
        </div>

        <div style="
            display: flex;
            justify-content: center;
            align-items: center;
        ">
            {cells}
        </div>

        <div style="
            font-family: monospace;
            font-size: 12px;
            margin-top: 4px;
        ">
            {pattern}
        </div>

    </div>
    """

    st.markdown(
        textwrap.dedent(html),
        unsafe_allow_html=True,
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

    nx = (
        -dy
        / tangent_norm
    )

    ny = (
        dx
        / tangent_norm
    )

    x1 = (
        x
        + nx * width / 2
    )

    y1 = (
        y
        + ny * width / 2
    )

    x2 = (
        x
        - nx * width / 2
    )

    y2 = (
        y
        - ny * width / 2
    )

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

        condition = (
            "50°C (as-prepared)"
        )

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

            condition = (
                "20°C (equilibrium)"
            )

        else:

            condition = "20°C"

    contour_x, contour_y = (
        draw_contour(
            centerline,
            width=2,
        )
    )

    return (
        centerline,
        contour_x,
        contour_y,
        condition,
    )


def create_animation(
    prediction,
):

    initial_centerline, initial_contour_x, initial_contour_y, _ = (
        make_frame(
            prediction,
            0,
        )
    )

    fig = go.Figure()

    # --------------------------------------------------------
    # Centerline
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=initial_centerline[:, 0],
            y=initial_centerline[:, 1],
            mode="lines+markers",
            line=dict(
                width=3,
            ),
            marker=dict(
                size=6,
            ),
            name="Predicted centerline",
        )
    )

    # --------------------------------------------------------
    # Contour
    # --------------------------------------------------------

    fig.add_trace(
        go.Scatter(
            x=initial_contour_x,
            y=initial_contour_y,
            mode="lines",
            fill="toself",
            name="Actuator contour",
        )
    )

    # --------------------------------------------------------
    # Frames
    # --------------------------------------------------------

    frames = []

    for t in range(
        NUM_TIMESTEPS + 1
    ):

        (
            centerline,
            contour_x,
            contour_y,
            condition,
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
                        line=dict(
                            width=3,
                        ),
                        marker=dict(
                            size=6,
                        ),
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
                "Deformation: "
                "0 min — "
                "50°C (as-prepared)"
            ),
            x=0.5,
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

        height=520,

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
                        "label": "▶ Start Deformation",
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
                        "label": "⏸ Stop Deformation",
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
# PREDICTION DATAFRAME
# ============================================================

def prediction_dataframe(
    prediction,
):

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

    return pd.DataFrame(
        rows
    )


# ============================================================
# RESET APPLICATION
# ============================================================

def reset_application():

    st.session_state.binary_code = ""

    st.session_state.prediction = None

    st.session_state.prediction_code = ""

    st.session_state.animation_started = False

    reset_editor()


# ============================================================
# MAIN GUI
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
# LOAD MODELS
# ============================================================

if not st.session_state.models_loaded:

    if st.button(
        "Load Models",
        type="secondary",
        use_container_width=True,
    ):

        with st.spinner(
            "Loading models..."
        ):

            try:

                models = load_web_models()

                st.session_state.models = models

                st.session_state.models_loaded = True

                st.success(
                    "Models loaded ✅"
                )

                st.rerun()

            except Exception as e:

                st.error(
                    "Model loading failed."
                )

                st.exception(e)

else:

    st.success(
        "Models loaded ✅"
    )


# ============================================================
# RESET
# ============================================================

if st.session_state.models_loaded:

    if st.button(
        "Reset",
        use_container_width=True,
    ):

        reset_application()

        st.rerun()


# ============================================================
# MAIN INPUT
# ============================================================

st.divider()

st.subheader(
    f"Enter {S}-bit Binary Code:"
)


binary_input = st.text_input(
    f"Enter {S}-bit Binary Code",
    value=st.session_state.binary_code,
    max_chars=S,
    disabled=not st.session_state.models_loaded,
    label_visibility="collapsed",
)


if binary_input != st.session_state.binary_code:

    st.session_state.binary_code = binary_input


# ============================================================
# DESIGN EDITOR
# ============================================================

st.subheader(
    "Design Editor"
)

editor_enabled = (
    st.session_state.models_loaded
)


with st.expander(
    "Open Design Editor",
    expanded=False,
):

    if not editor_enabled:

        st.info(
            "Load the models before opening the design editor."
        )

    else:

        st.markdown(
            """
            <div style="
                text-align: center;
                font-size: 22px;
                font-weight: bold;
                margin-bottom: 12px;
            ">
                Hydromatic Actuator
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Actuator strip
        # ----------------------------------------------------

        draw_design_strip()

        # ----------------------------------------------------
        # AHC sample structures
        # ----------------------------------------------------

        st.markdown(
            "### AHC structures"
        )

        sample_cols = st.columns(3)

        for col, block_type in zip(
            sample_cols,
            [1, 2, 3],
        ):

            with col:

                draw_ahc_sample(
                    block_type
                )

        # ----------------------------------------------------
        # Placement controls
        # ----------------------------------------------------

        st.markdown(
            "### Place AHC structure"
        )

        col1, col2 = st.columns(2)

        with col1:

            selected_type = st.selectbox(
                "AHC structure",
                options=[1, 2, 3],
                format_func=lambda x: AHC_LABELS[x],
            )

        with col2:

            max_start = (
                S
                - len(
                    AHC_PATTERNS[
                        selected_type
                    ]
                )
            )

            selected_start = st.number_input(
                "Starting Position (mm)",
                min_value=0,
                max_value=max_start,
                value=0,
                step=1,
            )

        if st.button(
            "Place AHC Structure",
            use_container_width=True,
        ):

            place_block(
                selected_type,
                int(selected_start),
            )

            st.rerun()

        # ----------------------------------------------------
        # Editor message
        # ----------------------------------------------------

        if (
            st.session_state.editor_message
        ):

            if (
                st.session_state.editor_message_type
                == "error"
            ):

                st.error(
                    st.session_state.editor_message
                )

            else:

                st.success(
                    st.session_state.editor_message
                )

        # ----------------------------------------------------
        # Current design
        # ----------------------------------------------------

        st.markdown(
            "### Current actuator"
        )

        draw_design_strip()

        # ----------------------------------------------------
        # Print Code
        # ----------------------------------------------------

        if st.button(
            "Print Code",
            use_container_width=True,
        ):

            if len(
                st.session_state.placed_blocks
            ) <= 1:

                st.error(
                    "Error: Double/triple structure requires more input."
                )

            else:

                code = "".join(
                    str(v)
                    for v in st.session_state.block_values
                )

                st.session_state.binary_code = code

                st.success(
                    f"Last valid actuator design: {code}"
                )

        # ----------------------------------------------------
        # Reset editor
        # ----------------------------------------------------

        if st.button(
            "Reset Editor",
            use_container_width=True,
        ):

            reset_editor()

            st.rerun()

        # ----------------------------------------------------
        # Submit Code
        # ----------------------------------------------------

        if st.button(
            "Submit Code",
            type="primary",
            use_container_width=True,
        ):

            if len(
                st.session_state.placed_blocks
            ) <= 1:

                st.error(
                    "Error: Double/triple structure requires more input."
                )

            else:

                code = "".join(
                    str(v)
                    for v in st.session_state.block_values
                )

                st.session_state.binary_code = code

                st.success(
                    "Design submitted."
                )

                st.rerun()


# ============================================================
# PREDICT
# ============================================================

st.divider()

predict_enabled = (
    st.session_state.models_loaded
)

valid, validation_message = (
    validate_binary_code(
        st.session_state.binary_code
    )
)


if st.button(
    "Predict",
    type="primary",
    disabled=(
        not predict_enabled
        or not valid
    ),
    use_container_width=True,
):

    st.session_state.animation_started = False

    binary_str = (
        st.session_state.binary_code
        .strip()
    )

    valid, message = (
        validate_binary_code(
            binary_str
        )
    )

    if not valid:

        st.error(
            message
        )

    else:

        with st.spinner(
            "Running the trained Hydromatic Simulator..."
        ):

            try:

                prediction = (
                    run_web_inference(
                        st.session_state.models,
                        binary_str,
                    )
                )

                st.session_state.prediction = (
                    prediction
                )

                st.session_state.prediction_code = (
                    binary_str
                )

                st.session_state.animation_started = True

                st.success(
                    "Prediction completed."
                )

            except Exception as e:

                st.error(
                    "The prediction could not be completed."
                )

                st.exception(e)


if (
    not valid
    and st.session_state.models_loaded
):

    st.warning(
        validation_message
    )


# ============================================================
# PREDICTION RESULTS
# ============================================================

if (
    st.session_state.prediction
    is not None
):

    prediction = (
        st.session_state.prediction
    )

    st.divider()

    st.subheader(
        "Predicted deformation"
    )

    # --------------------------------------------------------
    # Original-style start / stop controls
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        if st.button(
            "▶ Start Deformation",
            use_container_width=True,
        ):

            st.session_state.animation_started = True

    with col2:

        if st.button(
            "⏸ Stop Deformation",
            use_container_width=True,
        ):

            st.session_state.animation_started = False

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    fig = create_animation(
        prediction
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    st.caption(
        "0 min: 50°C (as-prepared). "
        "The subsequent predicted deformation sequence is at 20°C, "
        "ending at 20°C equilibrium."
    )

    # --------------------------------------------------------
    # Input code
    # --------------------------------------------------------

    st.subheader(
        "Input binary code"
    )

    st.code(
        st.session_state.prediction_code
    )

    # --------------------------------------------------------
    # Numerical results
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
    # CSV
    # --------------------------------------------------------

    csv_data = (
        df.to_csv(
            index=False
        ).encode("utf-8")
    )

    st.download_button(
        label="Download prediction as CSV",
        data=csv_data,
        file_name="hydromatic_prediction.csv",
        mime="text/csv",
        use_container_width=True,
    )
