import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt

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

# --- State Init ---
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
if "mean_plot_data" not in st.session_state:
    st.session_state.mean_plot_data = None

# --- Sidebar ---
st.sidebar.header("🧩 States")

# Rename states
new_states = [
    st.sidebar.text_input(f"State {i+1}", value=s, key=f"name_{i+1}")
    for i, s in enumerate(st.session_state.states)
]
if len(set(new_states)) != len(new_states):
    st.sidebar.error("State names must be unique.")
else:
    if new_states != st.session_state.states:
        st.session_state.matrix.columns = new_states
        st.session_state.matrix.index = new_states
        st.session_state.states = new_states

# Add/remove state
col1, col2 = st.sidebar.columns(2)
if col1.button("➕ Add State") and len(st.session_state.states) < MAX_STATES:
    new_state = f"S{len(st.session_state.states)+1}"
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
for i, row in enumerate(states):
    editable = [True] * len(states)
    editable[i] = False  # lock diagonal
    with st.expander(f"Row: {row}", expanded=True):
        values = []
        row_sum = 0
        for j, col in enumerate(states):
            key = f"prob_{i}_{j}"
            if i == j:
                values.append(0.0)  # placeholder for diagonal
            else:
                val = st.number_input(f"P({row}→{col})", min_value=0.0, max_value=1.0, step=0.01,
                                      value=float(matrix.iloc[i, j]), key=key)
                values.append(val)
                row_sum += val
        values[i] = max(0.0, 1.0 - row_sum)  # set diagonal
        st.write(f"→ Diagonal adjusted to {values[i]:.4f}")
        matrix.iloc[i] = values

st.session_state.matrix = matrix

# --- Graph Plot ---
st.subheader("📉 Markov Chain Graph")

def plot_markov_graph(matrix):
    G = nx.DiGraph()
    for s in matrix.index:
        G.add_node(s)
    for i in matrix.index:
        for j in matrix.columns:
            G.add_edge(i, j, weight=matrix.loc[i, j])
    pos = nx.circular_layout(G)
    edge_x, edge_y, texts = [], [], []
    for u, v, d in G.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
        texts.append(f"{u} → {v}: {d['weight']:.2f}")
    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color="#888"),
                            hoverinfo="text", mode="lines")
    node_x, node_y = zip(*[pos[n] for n in G.nodes()])
    node_trace = go.Scatter(
        x=node_x, y=node_y, text=list(G.nodes()),
        mode="markers+text", hoverinfo="text",
        marker=dict(size=30, color="lightblue", line=dict(width=2)),
        textposition="bottom center",
    )
    return go.Figure(data=[edge_trace, node_trace],
                     layout=go.Layout(hovermode="closest", margin=dict(b=20,l=5,r=5,t=40)))

st.plotly_chart(plot_markov_graph(matrix), use_container_width=True)

# --- Simulation ---
st.subheader("▶️ Simulation")

num_steps = st.slider("Number of timesteps", 1, MAX_TIMESTEPS, DEFAULT_TIMESTEPS)
num_reps = st.slider("Number of replications", 1, MAX_REPLICATIONS, DEFAULT_REPS)
visual_type = st.radio("Visualization type", ["Single trajectory", "Distribution", "Mean over time"], horizontal=True)

state_to_idx = {s: i for i, s in enumerate(states)}
start_state = st.selectbox("Start state", states)
start_idx = state_to_idx[start_state]
P = matrix.values

def simulate_chain(P, steps, start_idx):
    seq = [start_idx]
    for _ in range(steps):
        seq.append(np.random.choice(len(states), p=P[seq[-1]]))
    return np.array(seq)

def simulate_many_chains(P, n_reps, steps, start_idx):
    return np.array([simulate_chain(P, steps, start_idx) for _ in range(n_reps)])

if st.button("Run Simulation"):
    if visual_type == "Single trajectory":
        traj = simulate_chain(P, num_steps, start_idx)
        st.session_state.simulation_result = traj
        st.session_state.heatmap_data = None
        st.session_state.mean_plot_data = None
    elif visual_type == "Distribution":
        chains = simulate_many_chains(P, num_reps, num_steps, start_idx)
        heat = np.zeros((num_steps + 1, len(states)))
        for chain in chains:
            for t, s in enumerate(chain):
                heat[t, s] += 1
        st.session_state.heatmap_data = heat / num_reps
        st.session_state.simulation_result = None
        st.session_state.mean_plot_data = None
    elif visual_type == "Mean over time":
        chains = simulate_many_chains(P, num_reps, num_steps, start_idx)
        st.session_state.mean_plot_data = chains
        st.session_state.simulation_result = None
        st.session_state.heatmap_data = None

# --- Visual Output ---
if visual_type == "Single trajectory" and st.session_state.simulation_result is not None:
    traj = st.session_state.simulation_result
    st.write("Trajectory:")
    st.write(" → ".join([states[i] for i in traj]))
    st.line_chart(pd.Series(traj).astype("category").cat.codes)

elif visual_type == "Distribution" and st.session_state.heatmap_data is not None:
    timestep = st.slider("Timestep", 0, num_steps, num_steps)
    plot_option = st.radio("Plot type", ["Histogram", "Heatmap"])
    heat = st.session_state.heatmap_data
    if plot_option == "Histogram":
        st.bar_chart(pd.Series(heat[timestep], index=states))
    else:
        st.plotly_chart(
            go.Figure(data=go.Heatmap(
                z=heat.T,
                x=np.arange(num_steps + 1),
                y=states,
                colorscale='Blues'
            )),
            use_container_width=True
        )

elif visual_type == "Mean over time" and st.session_state.mean_plot_data is not None:
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
