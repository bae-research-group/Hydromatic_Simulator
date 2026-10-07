import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go

from Hydromatic_Simulator.model.model import GeneratorModel
from Hydromatic_Simulator.utils.config import config


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hydromatic Simulator",
    page_icon="🔵",
    layout="wide",
)


# ============================================================
# CONSTANTS
# ============================================================

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
# SESSION STATE
# ============================================================

if "block_values" not in st.session_state:
    st.session_state.block_values = [0] * S

if "placed_blocks" not in st.session_state:
    st.session_state.placed_blocks = []

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "prediction_code" not in st.session_state:
    st.session_state.prediction_code = None


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource
def load_models():
    models = []

    dummy_design = tf.zeros((1, S))
    dummy_position = tf.zeros((1, NUM_COORD, 2))

    for i in range(NUM_COORD):
        model = GeneratorModel()

        # Build the model
        _ = model.recursive_generate(
            dummy_design,
            dummy_position[:, 0, :],
            training=False,
        )

        encoder_path = (
            f"./Hydromatic_Simulator/model/weights/"
            f"encoder_{NUM_COORD}coords_{i}.weights.h5"
        )

        decoder_path = (
            f"./Hydromatic_Simulator/model/weights/"
            f"decoder_{NUM_COORD}coords_{i}.weights.h5"
        )

        model.encoder.load_weights(encoder_path)
        model.decoder.load_weights(decoder_path)

        models.append(model)

    return models


# ============================================================
# INFERENCE
# ============================================================

def run_inference(models, binary_code):
    design = np.array(
        [int(x) for x in binary_code],
        dtype=np.float32,
    )

    design = tf.convert_to_tensor(
        design.reshape(1, -1),
        dtype=tf.float32,
    )

    initial_position = tf.convert_to_tensor(
        config["init-pos"],
        dtype=tf.float32,
    )

    initial_position = tf.expand_dims(
        initial_position,
        axis=0,
    )

    predictions = []

    for model in models:
        pred = model.recursive_generate(
            design,
            initial_position[:, 0, :],
            training=False,
        )

        pred = pred.numpy()

        predictions.append(pred[0])

    return np.stack(predictions, axis=0)


# ============================================================
# VALIDATION
# ============================================================

def validate_binary_code(binary_code):
    if len(binary_code) != S:
        return False, f"Code must contain exactly {S} bits."

    if any(char not in "01" for char in binary_code):
        return False, "Code must contain only 0 and 1."

    if "1" not in binary_code:
        return False, "Code cannot be all zeros."

    return True, ""


# ============================================================
# PLACEMENT VALIDATION
# ============================================================

def can_place_block(start, pattern):
    end = start + len(pattern)

    if start < 0 or end > S:
        return False, "Structure does not fit within the 65 mm actuator."

    if len(st.session_state.placed_blocks) >= 3:
        return False, "Maximum of 3 structures allowed."

    new_range = set(range(start, end))

    for block in st.session_state.placed_blocks:
        old_start = block["start"]
        old_end = old_start + len(block["pattern"])
        old_range = set(range(old_start, old_end))

        if new_range.intersection(old_range):
            return False, "Structures cannot overlap."

        gap = min(
            abs(start - old_end),
            abs(old_start - end),
        )

        if gap < 5:
            return False, "Structures must have at least a 5 mm gap."

    return True, ""


# ============================================================
# PLACE AHC STRUCTURE
# ============================================================

def place_block():
    ahc_type = st.session_state.selected_ahc
    start = st.session_state.start_position
    pattern = AHC_PATTERNS[ahc_type]
    value = AHC_VALUES[ahc_type]

    valid, message = can_place_block(start, pattern)

    if not valid:
        st.error(message)
        return

    # IMPORTANT:
    # block_values must ALWAYS contain only 0/1.
    # The AHC value (1/2/3) is only metadata for the placed block.
    for j, bit in enumerate(pattern):
        st.session_state.block_values[start + j] = int(bit)

    st.session_state.placed_blocks.append(
        {
            "type": ahc_type,
            "start": start,
            "pattern": pattern,
            "value": value,
        }
    )

    st.rerun()


# ============================================================
# RESET
# ============================================================

def reset_design():
    st.session_state.block_values = [0] * S
    st.session_state.placed_blocks = []
    st.session_state.prediction = None
    st.session_state.prediction_code = None
    st.rerun()


# ============================================================
# BINARY CODE
# ============================================================

def binary_code():
    return "".join(
        str(int(v))
        for v in st.session_state.block_values
    )


# ============================================================
# DRAW AHC PATTERN SAMPLES
# ============================================================

def draw_ahc_patterns():
    st.markdown(
        "### AHC Structures"
    )

    for name, pattern in AHC_PATTERNS.items():

        cells = ""

        for bit in pattern:
            if bit == "1":
                cells += """
                    <div style="
                        width: 12px;
                        height: 28px;
                        background-color: #20B8E8;
                        border-right: 1px solid white;
                        box-sizing: border-box;
                    "></div>
                """
            else:
                cells += """
                    <div style="
                        width: 12px;
                        height: 28px;
                        background-color: #D3D3D3;
                        border-right: 1px solid white;
                        box-sizing: border-box;
                    "></div>
                """

        st.markdown(
            f"""
            <div style="
                margin-bottom: 4px;
                font-weight: 600;
                color: #343746;
            ">
                {name}
            </div>

            <div style="
                display: flex;
                height: 28px;
                margin-bottom: 14px;
                border: 1px solid #888;
                width: fit-content;
            ">
                {cells}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# DRAW CURRENT DESIGN
# ============================================================

def draw_design():

    bits = "".join(
        str(int(v))
        for v in st.session_state.block_values
    )

    # --------------------------------------------------------
    # Hydromatic Actuator label
    # Left aligned with the beginning of the bar
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            width: 100%;
            margin-top: 10px;
            margin-bottom: 6px;
            font-size: 20px;
            font-weight: 700;
            color: #343746;
            text-align: left;
        ">
            Hydromatic Actuator
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # 65-bit actuator bar
    # --------------------------------------------------------

    cells = ""

    for bit in bits:

        if bit == "1":
            cells += """
                <div style="
                    flex: 1;
                    height: 48px;
                    background-color: #20B8E8;
                    border-right: 1px solid white;
                    box-sizing: border-box;
                "></div>
            """
        else:
            cells += """
                <div style="
                    flex: 1;
                    height: 48px;
                    background-color: #D3D3D3;
                    border-right: 1px solid white;
                    box-sizing: border-box;
                "></div>
            """

    st.markdown(
        f"""
        <div style="
            display: flex;
            width: 100%;
            height: 48px;
            border: 1px solid #888;
            box-sizing: border-box;
            overflow: hidden;
        ">
            {cells}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # Scale
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            display: flex;
            justify-content: space-between;
            width: 100%;
            margin-top: 12px;
            font-size: 16px;
            color: #343746;
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
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # X-axis
    # --------------------------------------------------------

    st.markdown(
        """
        <div style="
            width: 100%;
            margin-top: 18px;
            border-top: 3px solid #222;
            position: relative;
            height: 28px;
        ">
            <span style="
                position: absolute;
                right: -4px;
                top: -14px;
                font-size: 25px;
                color: #222;
            ">
                X →
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PLOT PREDICTION
# ============================================================

def make_prediction_figure(pred_sequence):

    initial_x = np.linspace(
        4.0625,
        65.0,
        NUM_COORD,
    )

    initial_y = np.zeros(NUM_COORD)

    frames = []

    # --------------------------------------------------------
    # Initial frame
    # --------------------------------------------------------

    x0 = np.concatenate(
        [[0.0], initial_x]
    )

    y0 = np.concatenate(
        [[0.0], initial_y]
    )

    frames.append(
        go.Frame(
            name="0 min",
            data=[
                go.Scatter(
                    x=x0,
                    y=y0,
                    mode="lines",
                    line=dict(
                        color="red",
                        width=3,
                    ),
                )
            ],
        )
    )

    # --------------------------------------------------------
    # Predicted frames
    # --------------------------------------------------------

    for t in range(NUM_TIMESTEPS):

        x = np.concatenate(
            [[0.0], pred_sequence[:, t, 0]]
        )

        y = np.concatenate(
            [[0.0], pred_sequence[:, t, 1]]
        )

        frames.append(
            go.Frame(
                name=TIMESTEPS[t + 1],
                data=[
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines",
                        line=dict(
                            color="red",
                            width=3,
                        ),
                    )
                ],
            )
        )

    # --------------------------------------------------------
    # Initial figure
    # --------------------------------------------------------

    fig = go.Figure(
        data=[
            go.Scatter(
                x=x0,
                y=y0,
                mode="lines",
                line=dict(
                    color="red",
                    width=3,
                ),
            )
        ],
        frames=frames,
    )

    fig.update_layout(
        xaxis=dict(
            range=[-70, 80],
            title="X",
        ),
        yaxis=dict(
            range=[-90, 60],
            title="Y",
            scaleanchor="x",
            scaleratio=1,
        ),
        height=600,
        margin=dict(
            l=40,
            r=40,
            t=60,
            b=40,
        ),
        title="50°C (as-prepared)",
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {
                        "label": "▶ Start Deformation",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {
                                    "duration": 700,
                                    "redraw": True,
                                },
                                "transition": {
                                    "duration": 200,
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
                "currentvalue": {
                    "prefix": "Time: ",
                },
                "steps": [
                    {
                        "label": TIMESTEPS[i],
                        "method": "animate",
                        "args": [
                            [TIMESTEPS[i]],
                            {
                                "mode": "immediate",
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
                    for i in range(len(TIMESTEPS))
                ],
            }
        ],
    )

    return fig


# ============================================================
# PAGE HEADER
# ============================================================

st.title("Hydromatic Simulator")

st.markdown(
    "Configure a 65-bit Hydromatic Actuator and predict its deformation."
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("Design Editor")

    draw_ahc_patterns()

    st.markdown("---")

    st.selectbox(
        "AHC structure",
        list(AHC_PATTERNS.keys()),
        key="selected_ahc",
    )

    st.number_input(
        "Starting position (mm)",
        min_value=0,
        max_value=S - 1,
        value=0,
        step=1,
        key="start_position",
    )

    st.button(
        "Place structure",
        on_click=place_block,
        use_container_width=True,
    )

    st.button(
        "Reset",
        on_click=reset_design,
        use_container_width=True,
    )

    st.markdown("---")

    if st.session_state.placed_blocks:

        st.markdown("### Placed structures")

        for block in st.session_state.placed_blocks:
            st.write(
                f"{block['type']} — "
                f"position {block['start']}–"
                f"{block['start'] + len(block['pattern'])} mm"
            )


# ============================================================
# CURRENT DESIGN
# ============================================================

st.subheader("Current 65-bit design")

draw_design()

st.markdown("")

current_code = binary_code()

st.code(
    current_code,
    language="text",
)


# ============================================================
# ADVANCED MANUAL CODE
# ============================================================

with st.expander(
    "Advanced: enter a 65-bit binary code"
):

    st.write("65-bit binary code")

    manual_code = st.text_input(
        "65-bit binary code",
        value=current_code,
        label_visibility="collapsed",
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

            st.session_state.placed_blocks = []

            st.session_state.prediction = None
            st.session_state.prediction_code = None

            st.rerun()


# ============================================================
# PREDICTION
# ============================================================

st.markdown("---")

valid_code, validation_message = validate_binary_code(
    current_code
)

predict_disabled = (
    not valid_code
    or len(st.session_state.placed_blocks) < 2
)

if len(st.session_state.placed_blocks) < 2:
    st.info(
        "Place at least 2 AHC structures before prediction."
    )

if st.button(
    "Predict",
    type="primary",
    disabled=predict_disabled,
    use_container_width=True,
):

    with st.spinner(
        "Loading models and generating prediction..."
    ):

        models = load_models()

        prediction = run_inference(
            models,
            current_code,
        )

        st.session_state.prediction = prediction
        st.session_state.prediction_code = current_code


# ============================================================
# PREDICTION RESULTS
# ============================================================

if st.session_state.prediction is not None:

    st.markdown("---")

    st.subheader("Predicted deformation")

    fig = make_prediction_figure(
        st.session_state.prediction
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Numerical results
    # --------------------------------------------------------

    prediction = st.session_state.prediction

    rows = []

    for t in range(NUM_TIMESTEPS):

        for node in range(NUM_COORD):

            rows.append(
                {
                    "Time": config["timesteps"][t],
                    "Node": node + 1,
                    "X": prediction[node, t, 0],
                    "Y": prediction[node, t, 1],
                }
            )

    df = pd.DataFrame(rows)

    st.subheader("Prediction data")

    st.dataframe(
        df,
        use_container_width=True,
    )

    csv = df.to_csv(index=False)

    st.download_button(
        "Download prediction CSV",
        data=csv,
        file_name="hydromatic_prediction.csv",
        mime="text/csv",
    )
