# 🎲 Discrete-Time Markov Chain Simulator

An interactive [Streamlit](https://streamlit.io) web app to define, visualize, and simulate discrete-time Markov chains. Perfect for teaching or exploring stochastic processes.

## 🚀 Features

- Define custom states (up to 6)
- Interactive transition probability matrix
- Live-updating graph visualization
- Run simulations:
  - View single trajectory
  - Analyze distribution over multiple replications
- Visualize results with histograms and heatmaps

## 🛠 Setup

### Using Conda (Recommended)

```bash
conda env create -f environment.yaml
conda activate markov_simulator
```

## 🧑‍💻 Running Locally

After activating the environment:

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501` (Streamlit should launch it automatically).

> Tip: You can edit the `app.py` and refresh the browser to see changes immediately.

## ☁️ Deploying to Streamlit Cloud

1. Upload `app.py` and `environment.yaml` to a GitHub repository.
2. Go to [Streamlit Cloud](https://streamlit.io/cloud) and connect your repo.
3. Deploy your app in seconds — it runs right from the browser!


## License

This project is licensed under the MIT License.