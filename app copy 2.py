import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import random

# Constants
MAX_STATES = 6
MAX_TIMESTEPS = 100
MAX_REPLICATIONS = 500
DEFAULT_TIMESTEPS = 50
DEFAULT_REPS = 50

INI_STATES = ["Hotdog", "Burger", "Pizza"]
INI_MATRIX = [
    [0.5, 0.5, 0.0],
    [0.2, 0.2, 0.6],
    [0.7, 0.3, 0.0],
]

st.set_page_config(page_title="Markov Chain Simulator", layout="wide")

st.title("📊 Discrete-Time Markov Chain Simulator")
st.markdown("Define your states and transition probabilities, then run simulations and visualize results.")

# Session state initialization
if "states" not in st.session_state:
    st.session_state.states = INI_STATES
if "matrix" not in st.session_state:
    st.session_state.matrix = pd.DataFrame(
        np.array(INI_MATRIX),
        columns=st.session_state.states,
        index=st.session_state.states,
    )
if "simulation_result" not in st.session_state:
    st.session_state.simulation_result = None
if "heatmap_data" not in st.session_state:
    st.session_state.heatmap_data = None
if "added_state" not in st.session_state:
    st.session_state.added_state = False

# --- Sidebar: State Management ---
st.sidebar.header("🧩 States")

# Rename States
new_states = [
    st.sidebar.text_input(f"State {i+1}", value=s, key=f"name_{i+1}")
    for i, s in enumerate(st.session_state.states)
]

if len(set(new_states)) != len(new_states):
    st.sidebar.error("State names must be unique.")
else:
    if new_states != st.session_state.states:
        st.session_state.states = new_states
        st.session_state.matrix.columns = new_states
        st.session_state.matrix.index = new_states

# Add or Remove State
col1, col2 = st.sidebar.columns(2)
if col1.button("➕ Add State") and len(st.session_state.states) < MAX_STATES:
    new_state = f"S{len(st.session_state.states)+1}"
    while new_state in st.session_state.states:
        new_state += "_"
    st.session_state.states.append(new_state)
    st.session_state.matrix[new_state] = 0
    st.session_state.matrix.loc[new_state] = 0
    st.session_state.added_state = True
    st.rerun()

if col2.button("➖ Remove Last") and len(st.session_state.states) > 2:
    st.session_state.states.pop()
    st.session_state.matrix = st.session_state.matrix.loc[
        st.session_state.states, st.session_state.states
    ]

# --- Transition Matrix Editor ---
st.subheader("🔁 Transition Matrix")
edited_matrix = st.data_editor(
    st.session_state.matrix, use_container_width=True, key="matrix_editor"
)

def normalize_rows(df):
    return df.div(df.sum(axis=1), axis=0).fillna(0)

normalized_matrix = normalize_rows(edited_matrix)
if not np.allclose(normalized_matrix.sum(axis=1).values, 1.0):
    st.error("Each row must sum to 1. Matrix has been auto-normalized.")
st.session_state.matrix = normalized_matrix

# --- Graph Visualization ---
st.subheader("📉 Markov Chain Graph")

def plot_markov_graph(matrix):
    G = nx.DiGraph()
    for state in matrix.index:
        G.add_node(state)
    for from_state in matrix.index:
        for to_state in matrix.columns:
            prob = matrix.loc[from_state, to_state]
            if prob > 0:
                G.add_edge(from_state, to_state, weight=prob)
    pos = nx.circular_layout(G)
    edge_x, edge_y, edge_text = [], [], []
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{edge[0]} → {edge[1]}: {edge[2]['weight']:.2f}")
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"), hoverinfo="text", mode="lines")
    node_x, node_y = zip(*[pos[k] for k in G.nodes()])
    node_trace = go.Scatter(
        x=node_x, y=node_y, text=list(G.nodes()),
        mode="markers+text", hoverinfo="text",
        marker=dict(size=30, color="lightblue", line=dict(width=2)),
        textposition="bottom center",
    )
    return go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(showlegend=False, hovermode="closest", margin=dict(b=20, l=5, r=5, t=40)))

st.plotly_chart(plot_markov_graph(st.session_state.matrix), use_container_width=True)

# --- Simulation ---
st.subheader("▶️ Simulation")

num_steps = st.slider("Number of timesteps", 1, MAX_TIMESTEPS, DEFAULT_TIMESTEPS)
num_reps = st.slider("Number of replications", 1, MAX_REPLICATIONS, DEFAULT_REPS)
visual_type = st.radio("Visualization type", ["Single trajectory", "Distribution"], horizontal=True)

matrix = st.session_state.matrix.values
states = st.session_state.states
state_to_idx = {s: i for i, s in enumerate(states)}

def simulate_chain(matrix, steps, start_state_idx):
    state_seq = [start_state_idx]
    for _ in range(steps):
        next_state = np.random.choice(len(states), p=matrix[state_seq[-1]])
        state_seq.append(next_state)
    return state_seq

start_state = st.selectbox("Start state", states)
start_idx = state_to_idx[start_state]

if st.button("Run Simulation"):
    if visual_type == "Single trajectory":
        traj = simulate_chain(matrix, num_steps, start_idx)
        st.session_state.simulation_result = traj
        st.session_state.heatmap_data = None
    else:
        heat = np.zeros((num_steps + 1, len(states)))
        for _ in range(num_reps):
            traj = simulate_chain(matrix, num_steps, start_idx)
            for t, s in enumerate(traj):
                heat[t, s] += 1
        heat = heat / num_reps
        st.session_state.heatmap_data = heat
        st.session_state.simulation_result = None

# --- Show Results ---
if visual_type == "Single trajectory" and st.session_state.simulation_result:
    traj_states = [states[i] for i in st.session_state.simulation_result]
    st.write("Trajectory:")
    st.write(" → ".join(traj_states))
    st.line_chart(pd.Series(traj_states).astype("category").cat.codes)

elif visual_type == "Distribution" and st.session_state.heatmap_data is not None:
    timestep = st.slider("Select timestep for histogram", 0, num_steps, num_steps)
    plot_option = st.radio("Plot type", ["Histogram", "Heatmap"])
    heat = st.session_state.heatmap_data
    if plot_option == "Histogram":
        st.bar_chart(pd.Series(heat[timestep], index=states))
    else:
        st.write("State distribution heatmap (Time vs State)")
        st.plotly_chart(
            go.Figure(
                data=go.Heatmap(
                    z=heat.T,
                    x=np.arange(num_steps + 1),
                    y=states,
                    colorscale="Blues",
                )
            ),
            use_container_width=True,
        )
