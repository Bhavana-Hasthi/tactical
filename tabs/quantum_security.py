import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from qiskit import QuantumCircuit
from config import background_color, primary_color, secondary_color

def render(view_mode: str, quantum_data: pd.DataFrame):
    st.markdown(f'<div class="subheader"><h2>Quantum Security Layer - {view_mode}</h2></div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Quantum Entanglement Circuit")
        qc = QuantumCircuit(3)
        qc.h(0); qc.cx(0, 1); qc.cx(0, 2); qc.barrier()
        qc.ry(0.8, 0); qc.rz(0.5, 1); qc.rx(0.3, 2); qc.barrier()
        qc.cx(1, 2); qc.h(1); qc.measure_all()

        fig = qc.draw(output='mpl', style='clifford', plot_barriers=True, fold=25)
        fig.patch.set_facecolor(background_color)
        import io
        from PIL import Image
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor=background_color)
        buf.seek(0)
        st.image(Image.open(buf), caption="Quantum Circuit Diagram")

    with col2:
        st.subheader("Quantum Link Performance")
        stability_data = pd.DataFrame({
            'Round': quantum_data["Round"],
            'Fidelity': quantum_data["Qubit Fidelity"],
            'Error Rate': quantum_data["Quantum Bit Error Rate"],
            'Decoherence Time': quantum_data["Decoherence Time (μs)"],
            'Status': np.where(quantum_data["Entanglement Stability"], 'Stable', 'Unstable')
        })
        fig = make_subplots(rows=2, cols=1, subplot_titles=('Qubit Fidelity Over Time', 'Quantum Bit Error Rate'))
        fig.add_trace(go.Scatter(x=stability_data['Round'], y=stability_data['Fidelity'],
                                 mode='lines+markers', name='Fidelity',
                                 line=dict(color=primary_color)), row=1, col=1)
        fig.add_trace(go.Scatter(x=stability_data['Round'], y=stability_data['Error Rate'],
                                 mode='lines', name='Error Rate',
                                 line=dict(color=secondary_color)), row=2, col=1)
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", annotation_text="Threshold", row=1, col=1)
        fig.update_layout(height=600, showlegend=True, plot_bgcolor=background_color, paper_bgcolor=background_color)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Quantum Network Metrics")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Average Fidelity",
                      f"{quantum_data['Qubit Fidelity'].mean()*100:.2f}%",
                      delta=f"{quantum_data['Qubit Fidelity'].mean()*100 - 90:.2f}%")
            st.metric("Avg Error Rate",
                      f"{quantum_data['Quantum Bit Error Rate'].mean()*100:.3f}%")
        with metric_col2:
            st.metric("Avg Decoherence Time",
                      f"{quantum_data['Decoherence Time (μs)'].mean():.1f}μs")
            st.metric("Stable Links",
                      f"{int(quantum_data['Entanglement Stability'].sum())}/{len(quantum_data)}")

def api_render(view_mode: str, quantum_data: pd.DataFrame):
    # Example metrics for API response
    return {
        "average_fidelity": quantum_data["Qubit Fidelity"].mean(),
        "average_error_rate": quantum_data["Quantum Bit Error Rate"].mean(),
        "average_decoherence_time": quantum_data["Decoherence Time (μs)"].mean(),
        "stable_links": int(quantum_data["Entanglement Stability"].sum()),
        "total_links": len(quantum_data),
    }