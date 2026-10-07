import os
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import plotly.graph_objects as go

from Hydromatic_Simulator.model.main import load_trained_model
from Hydromatic_Simulator.utils.config import config


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Hydromatic Simulator",
    page_icon="",
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
    return load_trained_model(
        data_path="./Hydromatic_Simulator/dataset",
        weights_path="./Hydromatic_Simulator/model/weights",
    )


# ============================================================
# INFERENCE
# ============================================================

# ============================================================
# INFERENCE
# ============================================================

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

    # Initial nodal positions
    init_pos = np.array(
        config["init-pos"],
        dtype=np.float32,
    )

    init_pos = tf.convert_to_tensor(
        init_pos,
        dtype=tf.float32,
    )

    predictions = []

    # Each trained model corresponds to one initial nodal position.
    for i, model in enumerate(models):

        # Shape: (1, 2)
        node_init_pos = tf.reshape(
            init_pos[i],
            (1, 2),
        )

        pred = model.recursive_generate(
            design,
            node_init_pos,
            training=False,
        )

        pred = pred.numpy()

        # Remove batch dimension
        predictions.append(pred[0])

    predictions = np.stack(
        predictions,
        axis=0,
    )

    # Expected shape:
    # (16, 9, 2)
    return predictions


# ============================================================
# VALIDATION
# ============================================================

def validate_binary_code(code):

    code = code.strip()

    if len(code) != S:
        return False, f"Code must contain exactly {S} bits."

    if any(char not in "01" for char in code):
        return False, "Code must contain only 0 and 1."

    if set(code) == {"0"}:
        return False, "The design cannot be completely empty."

    return True, ""


# ============================================================
# BINARY CODE
# ============================================================

def binary_code():

    # IMPORTANT:
    # AHC type numbers (1, 2, 3) are only identifiers.
    # The actual model input is strictly binary:
    #
    # 0 -> 0
    # any occupied cell -> 1
    #
    # Therefore AHC-2 and AHC-6 cells are converted to 1.

    return "".join(
        "1" if value != 0 else "0"
        for value in st.session_state.block_values
    )


# ============================================================
# PLACEMENT VALIDATION
# ============================================================

def can_place_block(start, pattern):

    length = len(pattern)

    # Maximum of 3 structures
    if len(st.session_state.placed_blocks) >= 3:
        return False, "Maximum of 3 structures is allowed."

    # Must fit inside 65 mm
    if start < 0 or start + length > S:
        return False, "The structure must fit within the 65-mm actuator."

    occupied = [
        i
        for i, value in enumerate(st.session_state.block_values)
        if value != 0
    ]

    new_cells = set(range(start, start + length))

    # No overlap
    if any(i in new_cells for i in occupied):
        return False, "Structures cannot overlap."

    # Minimum 5-mm gap
    for i in occupied:
        for j in new_cells:
            if abs(i - j) < 5:
                return False, "Structures must have at least a 5-mm gap."

    return True, ""


# ============================================================
# PLACE AHC STRUCTURE
# ============================================================

def place_block(name, start):

    pattern = AHC_PATTERNS[name]
    value = AHC_VALUES[name]

    valid, message = can_place_block(start, pattern)

    if not valid:
        st.error(message)
        return

    # --------------------------------------------------------
    # IMPORTANT:
    # Store the actual binary pattern, NOT the AHC number.
    #
    # AHC-1 -> 10101010101
    # AHC-2 -> 1100110011
    # AHC-6 -> 111111
    #
    # The AHC number is metadata only.
    # --------------------------------------------------------

    for j, bit in enumerate(pattern):
        st.session_state.block_values[start + j] = int(bit)

    st.session_state.placed_blocks.append(
        {
            "name": name,
            "start": start,
            "length": len(pattern),
            "value": value,
        }
    )


# ============================================================
# RESET DESIGN
# ============================================================

def reset_design():

    st.session_state.block_values = [0] * S
    st.session_state.placed_blocks = []
    st.session_state.prediction = None
    st.session_state.prediction_code = None


# ============================================================
# AHC PATTERN DISPLAY
# ============================================================

def draw_ahc_patterns():

    st.markdown("### AHC structures")

    for name, pattern in AHC_PATTERNS.items():

        cells = ""

        for bit in pattern:

            if bit == "1":
                bg = "#18b7e8"
            else:
                bg = "#d0d0d0"

            cells += f"""
            <div style="
                width:16px;
                height:28px;
                background:{bg};
                border-right:1px solid white;
                display:inline-block;
            "></div>
            """

        st.markdown(
            f"""
            <div style="
                margin-bottom:12px;
                font-weight:600;
            ">
                {name}
            </div>

            <div style="
                display:flex;
                align-items:center;
                margin-bottom:18px;
            ">
                {cells}
            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# CURRENT DESIGN DISPLAY
# ============================================================

def draw_design():

    # Display is based on binary occupancy only.
    # Every nonzero value is visually represented as occupied.

    cells = ""

    for value in st.session_state.block_values:

        if value != 0:
            bg = "#18b7e8"
        else:
            bg = "#d0d0d0"

        cells += f"""
        <div style="
            width:16px;
            height:48px;
            background:{bg};
            border-right:1px solid white;
            display:inline-block;
            box-sizing:border-box;
        "></div>
        """

    st.markdown(
        """
        <div style="
            text-align:center;
            font-size:28px;
            font-weight:700;
            margin-top:10px;
            margin-bottom:20px;
            color:#30313d;
        ">
            Hydromatic Actuator
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div style="
            display:flex;
            align-items:center;
            width:100%;
            overflow:hidden;
            border:1px solid #777;
        ">
            {cells}
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scale labels
    st.markdown(
        """
        <div style="
            display:flex;
            justify-content:space-between;
            margin-top:14px;
            font-size:16px;
            color:#444;
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
            margin-top:18px;
            font-size:25px;
            color:#222;
        ">
            ─────────────────────────────────────────── X →
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# PLOTLY PREDICTION GRAPH
# ============================================================

def make_prediction_figure(pred_sequence):

    frames = []

    # Initial coordinates
    initial_x = np.linspace(
        4.0625,
        65.0,
        NUM_COORD,
    )

    initial_y = np.zeros(NUM_COORD)

    # --------------------------------------------------------
    # Initial frame
    # --------------------------------------------------------

    frames.append(
        go.Frame(
            name="0 min",
            data=[
                go.Scatter(
                    x=np.concatenate([[0.0], initial_x]),
                    y=np.concatenate([[0.0], initial_y]),
                    mode="lines+markers",
                    line=dict(
                        width=4,
                    ),
                    marker=dict(
                        size=5,
                    ),
                    fill="tozeroy",
                )
            ],
        )
    )

    # --------------------------------------------------------
    # Predicted frames
    # --------------------------------------------------------

    for t in range(NUM_TIMESTEPS):

        x = np.concatenate(
            [
                [0.0],
                pred_sequence[:, t, 0],
            ]
        )

        y = np.concatenate(
            [
                [0.0],
                pred_sequence[:, t, 1],
            ]
        )

        frames.append(
            go.Frame(
                name=TIMESTEPS[t + 1],
                data=[
                    go.Scatter(
                        x=x,
                        y=y,
                        mode="lines+markers",
                        line=dict(
                            width=4,
                        ),
                        marker=dict(
                            size=5,
                        ),
                        fill="tozeroy",
                    )
                ],
            )
        )

    # Initial graph
    fig = go.Figure(
        data=[
            go.Scatter(
                x=np.concatenate([[0.0], initial_x]),
                y=np.concatenate([[0.0], initial_y]),
                mode="lines+markers",
                line=dict(
                    width=4,
                ),
                marker=dict(
                    size=5,
                ),
                fill="tozeroy",
            )
        ],
        frames=frames,
    )

    fig.update_layout(
        xaxis=dict(
            range=[-70, 80],
            title="X position",
            zeroline=True,
        ),
        yaxis=dict(
            range=[-90, 60],
            title="Y position",
            zeroline=True,
            scaleanchor="x",
            scaleratio=1,
        ),
        height=650,
        margin=dict(
            l=50,
            r=30,
            t=50,
            b=50,
        ),
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.1,
                "y": 1.15,
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
                "x": 0.1,
                "y": 0,
                "len": 0.85,
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
# PAGE TITLE
# ============================================================

st.title("Hydromatic Simulator")

st.markdown(
    "Configure a 65-bit Hydromatic Actuator design and predict its deformation."
)


# ============================================================
# DESIGN SECTION
# ============================================================

st.header("Current 65-bit design")

draw_design()


# ============================================================
# CURRENT BINARY CODE
# ============================================================

# IMPORTANT:
# This now ALWAYS displays a true binary string.
# AHC-2 and AHC-6 occupied cells are displayed as 1,
# never as 2 or 3.

current_code = binary_code()

st.code(
    current_code,
    language=None,
)


# ============================================================
# DESIGN CONTROLS
# ============================================================

st.subheader("Add AHC structure")

col1, col2, col3 = st.columns([2, 1, 1])

with col1:

    selected_ahc = st.selectbox(
        "AHC type",
        list(AHC_PATTERNS.keys()),
    )

with col2:

    start_position = st.number_input(
        "Starting position (mm)",
        min_value=0,
        max_value=S - 1,
        value=0,
        step=1,
    )

with col3:

    st.write("")
    st.write("")

    if st.button(
        "Place structure",
        use_container_width=True,
    ):

        place_block(
            selected_ahc,
            int(start_position),
        )

        st.rerun()


col_reset, col_print = st.columns(2)

with col_reset:

    if st.button(
        "Reset",
        use_container_width=True,
    ):

        reset_design()
        st.rerun()

with col_print:

    if st.button(
        "Print Code",
        use_container_width=True,
    ):

        st.code(
            binary_code(),
            language=None,
        )


# ============================================================
# AHC PATTERN REFERENCE
# ============================================================

with st.expander("AHC pattern reference"):

    draw_ahc_patterns()


# ============================================================
# ADVANCED MANUAL BINARY INPUT
# ============================================================

with st.expander("Advanced: enter a 65-bit binary code"):

    st.markdown("**65-bit binary code**")

    manual_code = st.text_input(
        "65-bit binary code",
        value=current_code,
        label_visibility="collapsed",
    )

    if st.button("Use manual code"):

        valid, message = validate_binary_code(
            manual_code
        )

        if valid:

            # Convert the manual binary string into
            # the internal occupancy representation.
            st.session_state.block_values = [
                int(bit)
                for bit in manual_code
            ]

            # Manual code is not associated with
            # specific AHC metadata.
            st.session_state.placed_blocks = []

            st.success(
                "Manual 65-bit binary code applied."
            )

            st.rerun()

        else:

            st.error(message)


# ============================================================
# PREDICTION
# ============================================================

st.header("Prediction")

valid_code, validation_message = validate_binary_code(
    binary_code()
)

if not valid_code:

    st.warning(validation_message)

else:

    if st.button(
        "Predict deformation",
        type="primary",
        use_container_width=True,
    ):

        with st.spinner(
            "Loading trained models and predicting deformation..."
        ):

            models = load_models()

            prediction = run_inference(
                models,
                binary_code(),
            )

            st.session_state.prediction = prediction
            st.session_state.prediction_code = binary_code()


# ============================================================
# DISPLAY PREDICTION
# ============================================================

if st.session_state.prediction is not None:

    st.subheader("Predicted deformation")

    fig = make_prediction_figure(
        st.session_state.prediction
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # Prediction data
    # --------------------------------------------------------

    st.subheader("Prediction data")

    pred = st.session_state.prediction

    rows = []

    for t in range(NUM_TIMESTEPS):

        for node in range(NUM_COORD):

            rows.append(
                {
                    "Time": config["timesteps"][t],
                    "Node": node + 1,
                    "X": pred[node, t, 0],
                    "Y": pred[node, t, 1],
                }
            )

    prediction_df = pd.DataFrame(rows)

    st.dataframe(
        prediction_df,
        use_container_width=True,
    )

    csv = prediction_df.to_csv(
        index=False
    )

    st.download_button(
        "Download prediction CSV",
        data=csv,
        file_name="hydromatic_prediction.csv",
        mime="text/csv",
    )
