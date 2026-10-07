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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Hydromatic Simulator",
    page_icon="",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

S = config["structure-dim"]                 # 65
NUM_COORD = config["num-coord"]             # 16
NUM_TIMESTEPS = config["num-timesteps"]     # 9

TIMESTEPS = ["0 min"] + config["timesteps"]

AHC_PATTERNS = {
    "AHC-1": "10101010101",
    "AHC-2": "1100110011",
    "AHC-6": "111111",
}


# ============================================================
# SESSION STATE
# ============================================================

if "block_values" not in st.session_state:
    # IMPORTANT:
    # This list contains ONLY 0 and 1.
    st.session_state.block_values = [0] * S

if "placed_blocks" not in st.session_state:
    # AHC identity is stored separately from binary values.
    st.session_state.placed_blocks = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "prediction_code" not in st.session_state:
    st.session_state.prediction_code = None

if "manual_code_active" not in st.session_state:
    st.session_state.manual_code_active = False


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():

    models = []

    weights_path = os.path.join(
        "Hydromatic_Simulator",
        "model",
        "weights"
    )

    # Dummy inputs used to initialize the model architecture.
    dummy_design = tf.zeros(
        (1, config["structure-dim"]),
        dtype=tf.float32
    )

    dummy_position = tf.zeros(
        (1, 1, config["nodal-dim"]),
        dtype=tf.float32
    )

    for i in range(NUM_COORD):

        model = GeneratorModel()

        # Build the model.
        model.recursive_generate(
            dummy_design,
            dummy_position,
            training=False
        )

        encoder_file = os.path.join(
            weights_path,
            f"encoder_{NUM_COORD}coords_{i}.weights.h5"
        )

        decoder_file = os.path.join(
            weights_path,
            f"decoder_{NUM_COORD}coords_{i}.weights.h5"
        )

        model.encoder.load_weights(encoder_file)
        model.decoder.load_weights(decoder_file)

        models.append(model)

    return models


# ============================================================
# INFERENCE
# ============================================================

def run_inference(models, binary_string):

    # --------------------------------------------------------
    # Convert the 65-bit binary code to a float tensor.
    # --------------------------------------------------------

    design = np.array(
        [float(x) for x in binary_string],
        dtype=np.float32
    )

    design = design.reshape(1, -1)

    # --------------------------------------------------------
    # Initial nodal positions.
    # --------------------------------------------------------

    initial_pos = np.array(
        config["init-pos"],
        dtype=np.float32
    )

    initial_pos = initial_pos.reshape(
        1,
        NUM_COORD,
        config["nodal-dim"]
    )

    predictions = []

    # --------------------------------------------------------
    # Run all 16 trained models.
    # --------------------------------------------------------

    for model in models:

        design_tensor = tf.convert_to_tensor(
            design,
            dtype=tf.float32
        )

        position_tensor = tf.convert_to_tensor(
            initial_pos,
            dtype=tf.float32
        )

        pred = model.recursive_generate(
            design_tensor,
            position_tensor,
            training=False
        )

        pred = pred.numpy()

        predictions.append(pred)

    # Expected shape:
    # (16, 9, 2)

    return np.stack(predictions, axis=0)


# ============================================================
# BINARY CODE
# ============================================================

def binary_code():

    # --------------------------------------------------------
    # STRICTLY binary.
    #
    # Even if AHC metadata exists, only the actual bit value
    # is returned here.
    # --------------------------------------------------------

    return "".join(
        "1" if int(x) == 1 else "0"
        for x in st.session_state.block_values
    )


def validate_binary_code(code):

    if len(code) != S:
        return False, f"Code must contain exactly {S} bits."

    if any(char not in "01" for char in code):
        return False, "Code must contain only 0 and 1."

    if set(code) == {"0"}:
        return False, "The design cannot be completely empty."

    return True, ""


# ============================================================
# DESIGN PLACEMENT
# ============================================================

def can_place_block(start, length):

    end = start + length

    # Must fit inside the 65-mm design.
    if start < 0 or end > S:
        return False, "The structure does not fit within 65 mm."

    # Maximum of 3 structures.
    if len(st.session_state.placed_blocks) >= 3:
        return False, "A maximum of 3 structures is allowed."

    # Check overlap and minimum 5-mm gap.
    for block in st.session_state.placed_blocks:

        old_start = block["start"]
        old_end = old_start + block["length"]

        # Overlap
        if start < old_end and end > old_start:
            return False, "Structures cannot overlap."

        # Gap
        if end <= old_start:
            gap = old_start - end
        else:
            gap = start - old_end

        if gap < 5:
            return False, "A minimum 5 mm gap is required between structures."

    return True, ""


def place_block(structure_name, start):

    pattern = AHC_PATTERNS[structure_name]
    length = len(pattern)

    valid, message = can_place_block(
        start,
        length
    )

    if not valid:
        st.error(message)
        return

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Store ONLY the actual 0/1 pattern.
    #
    # AHC-1 -> 10101010101
    # AHC-2 -> 1100110011
    # AHC-6 -> 111111
    #
    # Do NOT store the AHC number (1, 2, or 3) as a bit.
    # --------------------------------------------------------

    for j, bit in enumerate(pattern):
        st.session_state.block_values[start + j] = int(bit)

    # --------------------------------------------------------
    # Store AHC identity separately.
    # This is visual metadata only.
    # --------------------------------------------------------

    st.session_state.placed_blocks.append(
        {
            "name": structure_name,
            "start": start,
            "length": length,
        }
    )

    st.session_state.manual_code_active = False
    st.session_state.prediction = None
    st.session_state.prediction_code = None


def reset_design():

    st.session_state.block_values = [0] * S
    st.session_state.placed_blocks = []

    st.session_state.prediction = None
    st.session_state.prediction_code = None

    st.session_state.manual_code_active = False


# ============================================================
# AHC PATTERN DISPLAY
# ============================================================

def draw_ahc_patterns():

    st.markdown(
        """
        <div style="
            font-size: 22px;
            font-weight: 700;
            margin-bottom: 12px;
            color: #30323d;
        ">
            AHC Structures
        </div>
        """,
        unsafe_allow_html=True
    )

    for name, pattern in AHC_PATTERNS.items():

        cells = ""

        for bit in pattern:

            if bit == "1":
                cells += """
                <div style="
                    width: 18px;
                    height: 38px;
                    background-color: #19b5e6;
                    border-right: 1px solid white;
                    box-sizing: border-box;
                "></div>
                """
            else:
                cells += """
                <div style="
                    width: 18px;
                    height: 38px;
                    background-color: #d0d0d0;
                    border-right: 1px solid white;
                    box-sizing: border-box;
                "></div>
                """

        html = f"""
        <div style="
            margin-bottom: 18px;
        ">

            <div style="
                font-size: 17px;
                font-weight: 600;
                margin-bottom: 6px;
                color: #30323d;
            ">
                {name}
            </div>

            <div style="
                display: flex;
                height: 38px;
                width: fit-content;
                border: 1px solid #777;
            ">
                {cells}
            </div>

        </div>
        """

        st.markdown(
            textwrap.dedent(html),
            unsafe_allow_html=True
        )


# ============================================================
# MAIN ACTUATOR DESIGN DISPLAY
# ============================================================

def draw_design():

    bits = st.session_state.block_values

    # --------------------------------------------------------
    # Create actuator cells.
    # --------------------------------------------------------

    cells = ""

    for bit in bits:

        if int(bit) == 1:

            cells += """
            <div style="
                width: 16px;
                height: 48px;
                background-color: #19b5e6;
                border-right: 1px solid white;
                box-sizing: border-box;
                flex-shrink: 0;
            "></div>
            """

        else:

            cells += """
            <div style="
                width: 16px;
                height: 48px;
                background-color: #d0d0d0;
                border-right: 1px solid white;
                box-sizing: border-box;
                flex-shrink: 0;
            "></div>
            """

    # --------------------------------------------------------
    # Create AHC labels.
    #
    # These labels are NOT part of the binary code.
    # --------------------------------------------------------

    labels_html = ""

    for block in st.session_state.placed_blocks:

        name = block["name"]
        start = block["start"]
        length = block["length"]

        center = start + length / 2

        left_percent = (center / S) * 100

        labels_html += f"""
        <div style="
            position: absolute;
            left: {left_percent}%;
            transform: translateX(-50%);
            top: 0;
            font-size: 15px;
            font-weight: 600;
            color: #30323d;
            white-space: nowrap;
        ">
            {name}
        </div>
        """

    # --------------------------------------------------------
    # Full actuator graphic.
    # --------------------------------------------------------

    html = f"""
    <div style="
        margin-top: 10px;
        width: 100%;
    ">

        <!-- Left-aligned Hydromatic Actuator label -->
        <div style="
            font-size: 26px;
            font-weight: 700;
            color: #30323d;
            margin-bottom: 42px;
            text-align: left;
        ">
            Hydromatic Actuator
        </div>


        <!-- Actuator and labels -->
        <div style="
            position: relative;
            width: 1040px;
            max-width: 100%;
            overflow-x: auto;
            padding-top: 0;
        ">

            <!-- AHC labels -->
            <div style="
                position: relative;
                width: 1040px;
                height: 25px;
            ">
                {labels_html}
            </div>


            <!-- Actuator bar -->
            <div style="
                display: flex;
                width: 1040px;
                height: 48px;
                border: 1px solid #777;
                box-sizing: border-box;
                overflow: hidden;
            ">
                {cells}
            </div>


            <!-- Scale -->
            <div style="
                width: 1040px;
                display: flex;
                justify-content: space-between;
                margin-top: 14px;
                font-size: 18px;
                color: #454752;
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


            <!-- X axis -->
            <div style="
                width: 1040px;
                margin-top: 18px;
                border-top: 3px solid #303030;
                position: relative;
                height: 35px;
            ">

                <span style="
                    position: absolute;
                    right: -5px;
                    top: -15px;
                    font-size: 26px;
                    color: #303030;
                ">
                    X →
                </span>

            </div>

        </div>

    </div>
    """

    st.markdown(
        textwrap.dedent(html),
        unsafe_allow_html=True
    )


# ============================================================
# PLOTLY DEFORMATION VISUALIZATION
# ============================================================

def make_prediction_figure(pred_sequence):

    fig = go.Figure()

    # --------------------------------------------------------
    # Initial geometry
    # --------------------------------------------------------

    initial_x = np.linspace(
        0,
        65,
        NUM_COORD + 1
    )

    initial_y = np.zeros(
        NUM_COORD + 1
    )

    fig.add_trace(
        go.Scatter(
            x=initial_x,
            y=initial_y,
            mode="lines",
            line=dict(width=4),
            name="Actuator"
        )
    )

    # --------------------------------------------------------
    # Animation frames
    # --------------------------------------------------------

    frames = []

    # Initial frame
    frames.append(
        go.Frame(
            name="0",
            data=[
                go.Scatter(
                    x=initial_x,
                    y=initial_y,
                    mode="lines",
                    line=dict(width=4)
                )
            ]
        )
    )

    # Prediction frames
    for t in range(NUM_TIMESTEPS):

        x_pred = np.concatenate(
            (
                [0.0],
                pred_sequence[:, t, 0]
            )
        )

        y_pred = np.concatenate(
            (
                [0.0],
                pred_sequence[:, t, 1]
            )
        )

        frames.append(
            go.Frame(
                name=str(t + 1),
                data=[
                    go.Scatter(
                        x=x_pred,
                        y=y_pred,
                        mode="lines",
                        line=dict(width=4)
                    )
                ]
            )
        )

    fig.frames = frames

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    fig.update_layout(

        xaxis=dict(
            range=[-70, 80],
            title="x"
        ),

        yaxis=dict(
            range=[-90, 60],
            title="y",
            scaleanchor="x",
            scaleratio=1
        ),

        height=650,

        margin=dict(
            l=60,
            r=40,
            t=60,
            b=60
        ),

        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.0,
                "y": 1.12,
                "buttons": [

                    {
                        "label": "▶ Start Deformation",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": 500,
                                    "redraw": True
                                },
                                "transition": {
                                    "duration": 200
                                },
                                "fromcurrent": True
                            }
                        ]
                    },

                    {
                        "label": "⏸ Stop Deformation",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {
                                    "duration": 0,
                                    "redraw": False
                                },
                                "mode": "immediate"
                            }
                        ]
                    }

                ]
            }
        ],

        sliders=[
            {
                "active": 0,
                "x": 0.0,
                "y": -0.08,
                "len": 0.9,

                "steps": [

                    {
                        "label": TIMESTEPS[i],
                        "method": "animate",
                        "args": [
                            [str(i)],
                            {
                                "mode": "immediate",
                                "frame": {
                                    "duration": 0,
                                    "redraw": True
                                },
                                "transition": {
                                    "duration": 0
                                }
                            }
                        ]
                    }

                    for i in range(len(TIMESTEPS))
                ]
            }
        ]
    )

    return fig


# ============================================================
# PREDICTION DATAFRAME
# ============================================================

def prediction_dataframe(pred_sequence):

    rows = []

    for t in range(NUM_TIMESTEPS):

        for node in range(NUM_COORD):

            rows.append(
                {
                    "Time": config["timesteps"][t],
                    "Node": node + 1,
                    "X": pred_sequence[node, t, 0],
                    "Y": pred_sequence[node, t, 1],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# PAGE HEADER
# ============================================================

st.title("Hydromatic Simulator")

st.markdown(
    "Design a 65-bit Hydromatic actuator and predict its deformation over time."
)


# ============================================================
# DESIGN AREA
# ============================================================

left_col, right_col = st.columns(
    [2.2, 1]
)


# ============================================================
# LEFT COLUMN
# ============================================================

with left_col:

    st.subheader("Current 65-bit design")

    draw_design()


# ============================================================
# RIGHT COLUMN
# ============================================================

with right_col:

    draw_ahc_patterns()


# ============================================================
# ADD AHC STRUCTURE
# ============================================================

st.markdown("---")

st.subheader("Add AHC structure")

control_col1, control_col2, control_col3 = st.columns(
    [1.2, 1, 1]
)

with control_col1:

    selected_ahc = st.selectbox(
        "Structure",
        list(AHC_PATTERNS.keys())
    )


with control_col2:

    start_position = st.number_input(
        "Starting position (mm)",
        min_value=0,
        max_value=64,
        value=0,
        step=1
    )


with control_col3:

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "Place structure",
        use_container_width=True
    ):

        place_block(
            selected_ahc,
            int(start_position)
        )

        st.rerun()


# ============================================================
# RESET / PRINT CODE
# ============================================================

button_col1, button_col2 = st.columns(2)


with button_col1:

    if st.button(
        "Reset",
        use_container_width=True
    ):

        reset_design()
        st.rerun()


with button_col2:

    if st.button(
        "Print Code",
        use_container_width=True
    ):

        code = binary_code()

        valid, message = validate_binary_code(code)

        if valid:

            st.code(
                code,
                language=None
            )

        else:

            st.error(message)


# ============================================================
# CURRENT BINARY CODE
# ============================================================

st.markdown("")

current_code = binary_code()

st.code(
    current_code,
    language=None
)


# ============================================================
# ADVANCED MANUAL INPUT
# ============================================================

with st.expander(
    "Advanced: enter a 65-bit binary code"
):

    st.write("65-bit binary code")

    manual_code = st.text_input(
        "Enter binary code",
        value=current_code,
        max_chars=S,
        label_visibility="collapsed"
    )

    if st.button(
        "Use manual code"
    ):

        manual_code = manual_code.strip()

        valid, message = validate_binary_code(
            manual_code
        )

        if valid:

            # Store ONLY 0/1.
            st.session_state.block_values = [
                int(bit)
                for bit in manual_code
            ]

            # Manual designs do not receive AHC labels.
            st.session_state.placed_blocks = []

            st.session_state.manual_code_active = True

            st.session_state.prediction = None
            st.session_state.prediction_code = None

            st.success(
                "Manual 65-bit binary code applied."
            )

            st.rerun()

        else:

            st.error(message)


# ============================================================
# PREDICTION
# ============================================================

st.markdown("---")

st.subheader("Prediction")

current_code = binary_code()

valid_code, validation_message = validate_binary_code(
    current_code
)

can_predict = (
    valid_code
    and (
        len(st.session_state.placed_blocks) >= 2
        or st.session_state.manual_code_active
    )
)


if not valid_code:

    st.warning(
        validation_message
    )

elif (
    len(st.session_state.placed_blocks) < 2
    and not st.session_state.manual_code_active
):

    st.info(
        "Place at least 2 AHC structures before prediction, "
        "or use the advanced manual binary-code input."
    )


# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button(
    "Predict deformation",
    type="primary",
    disabled=not can_predict,
    use_container_width=True
):

    with st.spinner(
        "Loading trained models and running prediction..."
    ):

        models = load_models()

        prediction = run_inference(
            models,
            current_code
        )

        st.session_state.prediction = prediction
        st.session_state.prediction_code = current_code


# ============================================================
# RESULTS
# ============================================================

if st.session_state.prediction is not None:

    st.markdown("---")

    st.subheader("Predicted deformation")

    prediction = st.session_state.prediction

    pred_sequence = prediction

    fig = make_prediction_figure(
        pred_sequence
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------------------------------
    # Input code
    # --------------------------------------------------------

    st.subheader("Input binary code")

    st.code(
        st.session_state.prediction_code,
        language=None
    )

    # --------------------------------------------------------
    # Numerical results
    # --------------------------------------------------------

    st.subheader("Prediction data")

    df = prediction_dataframe(
        pred_sequence
    )

    st.dataframe(
        df,
        use_container_width=True
    )

    # --------------------------------------------------------
    # CSV download
    # --------------------------------------------------------

    csv_data = df.to_csv(
        index=False
    )

    st.download_button(
        label="Download prediction CSV",
        data=csv_data,
        file_name="hydromatic_prediction.csv",
        mime="text/csv",
        use_container_width=True
    )
