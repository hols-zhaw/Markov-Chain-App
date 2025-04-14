import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from pyvis.network import Network
import streamlit.components.v1 as components
import os
import plotly.graph_objects as go

# --- Constants ---
MAX_STATES = 6
MAX_TIMESTEPS = 100
MAX_REPLICATIONS = 1000
DEFAULT_TIMESTEPS = 20
DEFAULT_REPS = 100
NODE_FONT_SIZE = 24
EDGE_FONT_SIZE = 18

INIT_STATES = ["Hotdog", "Burger", "Pizza"]
INIT_MATRIX = [
    [0.5, 0.5, 0.0],
    [0.2, 0.2, 0.6],
    [0.7, 0.3, 0.0],
]

VISUALIZATION_OPTIONS = {
    "trajectory": "Single trajectory",
    "distribution": "Distribution",
    "statistics": "Mean & Std over time"
}

# --- Streamlit Page Config ---
st.set_page_config(page_title="Markov Chain Simulator", layout="wide")

# --- Session State Setup ---
if "states" not in st.session_state:
    st.session_state.states = INIT_STATES

if "matrix" not in st.session_state:
    st.session_state.matrix = pd.DataFrame(
        np.array(INIT_MATRIX),
        columns=st.session_state.states,
        index=st.session_state.states
    )

for key in ["simulation_result", "heatmap_data", "mean_plot_data"]:
    st.session_state.setdefault(key, None)

# --- Sidebar: State Management ---
st.sidebar.header("🧩 States")
updated_states = [
    st.sidebar.text_input(f"State {i + 1}", value=s, key=f"state_{i}")
    for i, s in enumerate(st.session_state.states)
]

if len(set(updated_states)) != len(updated_states):
    st.sidebar.error("State names must be unique.")
else:
    if updated_states != st.session_state.states:
        st.session_state.matrix.columns = updated_states
        st.session_state.matrix.index = updated_states
        st.session_state.states = updated_states

col1, col2 = st.sidebar.columns(2)
if col1.button("➕ Add State") and len(st.session_state.states) < MAX_STATES:
    new_state = f"S{len(st.session_state.states) + 1}"
    while new_state in st.session_state.states:
        new_state += "_"
    st.session_state.states.append(new_state)
    st.session_state.matrix[new_state] = 0
    st.session_state.matrix.loc[new_state] = 0
    st.rerun()

if col2.button("➖ Remove Last") and len(st.session_state.states) > 2:
    st.session_state.states.pop()
    st.session_state.matrix = st.session_state.matrix.loc[
        st.session_state.states, st.session_state.states
    ]
    st.rerun()

# --- Transition Matrix Editor ---
st.subheader("🔁 Transition Matrix")
matrix = st.session_state.matrix.copy()
states = st.session_state.states

for i, from_state in enumerate(states):
    with st.expander(f"Row: {from_state}", expanded=True):
        values = []
        row_sum = 0
        for j, to_state in enumerate(states):
            key = f"prob_{i}_{j}"
            if i == j:
                values.append(0.0)
            else:
                val = st.number_input(
                    f"P({from_state}→{to_state})", min_value=0.0, max_value=1.0,
                    step=0.01, value=float(matrix.iloc[i, j]), key=key
                )
                values.append(val)
                row_sum += val
        values[i] = max(0.0, 1.0 - row_sum)
        st.write(f"→ Diagonal auto-adjusted to {values[i]:.4f}")
        matrix.iloc[i] = values

st.session_state.matrix = matrix

# --- Graph Rendering ---
def render_markov_graph(matrix, states):
    net = Network(height="600px", width="100%", directed=True)
    net.barnes_hut()
    for state in states:
        net.add_node(state, label=state, font={"size": NODE_FONT_SIZE}, labelHighlightBold=True, physics=True)
    for i, from_state in enumerate(states):
        for j, to_state in enumerate(states):
            prob = matrix[i, j]
            label = f"{prob:.2f}"
            if prob > 0:# and from_state != to_state:
                net.add_edge(from_state, to_state, value=prob, label=label, arrows="to", font={"size": EDGE_FONT_SIZE}, physics=True)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        tmp_path = tmp_file.name
        net.save_graph(tmp_path)
    with open(tmp_path, "r", encoding="utf-8") as f:
        html_content = f.read()
    components.html(html_content, height=650, scrolling=True)
    os.remove(tmp_path)

st.subheader("🕸 Markov Chain Graph")
render_markov_graph(st.session_state.matrix.values, st.session_state.states)

# --- Simulation Setup ---
st.subheader("▶️ Simulation")
num_steps = st.slider("Number of timesteps", 1, MAX_TIMESTEPS, DEFAULT_TIMESTEPS)
num_reps = st.slider("Number of replications", 1, MAX_REPLICATIONS, DEFAULT_REPS)
visual_type = st.radio("Visualization type", list(VISUALIZATION_OPTIONS.values()), horizontal=True)

state_to_idx = {s: i for i, s in enumerate(states)}
start_state = st.selectbox("Start state", states)
start_idx = state_to_idx[start_state]
P = matrix.values

# --- Simulation Functions ---
def simulate_chain(P, steps, start_idx):
    n_states = P.shape[0]
    idx = start_idx if 0 <= start_idx < n_states else 0
    seq = [idx]
    for _ in range(steps):
        idx = np.random.choice(n_states, p=P[idx])
        seq.append(idx)
    return np.array(seq)

def simulate_many_chains(P, n_reps, steps, start_idx):
    return np.array([simulate_chain(P, steps, start_idx) for _ in range(n_reps)])

# --- Run Simulation ---
if st.button("Run Simulation"):
    if visual_type == VISUALIZATION_OPTIONS["trajectory"]:
        st.session_state.simulation_result = simulate_chain(P, num_steps, start_idx)
        st.session_state.heatmap_data = None
        st.session_state.mean_plot_data = None
    elif visual_type == VISUALIZATION_OPTIONS["distribution"]:
        chains = simulate_many_chains(P, num_reps, num_steps, start_idx)
        heat = np.zeros((num_steps + 1, len(states)))
        for chain in chains:
            for t, s in enumerate(chain):
                heat[t, s] += 1
        st.session_state.heatmap_data = heat / num_reps
        st.session_state.simulation_result = None
        st.session_state.mean_plot_data = None
    elif visual_type == VISUALIZATION_OPTIONS["statistics"]:
        st.session_state.mean_plot_data = simulate_many_chains(P, num_reps, num_steps, start_idx)
        st.session_state.simulation_result = None
        st.session_state.heatmap_data = None

# --- Visualization ---
if visual_type == VISUALIZATION_OPTIONS["trajectory"] and st.session_state.simulation_result is not None:
    traj = st.session_state.simulation_result
    st.write("Trajectory:")
    st.write(" → ".join([states[i] for i in traj]))
    st.line_chart(pd.Series(traj).astype("category").cat.codes)

elif visual_type == VISUALIZATION_OPTIONS["distribution"] and st.session_state.heatmap_data is not None:
    timestep = st.slider("Timestep", 0, num_steps, num_steps)
    plot_type = st.radio("Plot type", ["Histogram", "Heatmap"])
    heat = st.session_state.heatmap_data
    if plot_type == "Histogram":
        st.bar_chart(pd.Series(heat[timestep], index=states))
    else:
        st.plotly_chart(go.Figure(
            data=[go.Heatmap(z=heat.T, x=np.arange(num_steps + 1), y=states, colorscale="Blues")]
        ), use_container_width=True)

elif visual_type == VISUALIZATION_OPTIONS["statistics"] and st.session_state.mean_plot_data is not None:
    chains = st.session_state.mean_plot_data
    mean = chains.mean(axis=0)
    std = chains.std(axis=0)
    ci = 2.576 * std / np.sqrt(chains.shape[0])  # 99% CI

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(mean, label="mean", color="blue")
    ax.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.2, label="±1σ", color="blue")
    ax.fill_between(range(len(mean)), mean - ci, mean + ci, alpha=0.3, label="99% CI", color="gray")
    ax.set_xlim(0, len(mean) - 1)
    ax.set_xticks(range(0, len(mean), max(1, len(mean) // 10)))
    ax.set_ylim(0, len(states) - 1)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Mean state index")
    ax.set_title(f"Mean and confidence intervals of {chains.shape[0]} chains")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)
