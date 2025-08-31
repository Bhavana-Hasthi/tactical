import numpy as np
import streamlit as st
from  config import app_css, primary_color, background_color
from authentication import military_authentication
from data_generation import (
    generate_realtime_data, generate_historical_data, generate_simulation_data,
    generate_quantum_data, generate_performance_metrics
)
from tabs import overview, threat_analysis, quantum_security, qkd_encryption, performance, battlefield, chat_assistant

# Page config
st.set_page_config(page_title="Quantum Military Cybersecurity Command", page_icon="🛡", layout="wide", initial_sidebar_state="expanded")
st.markdown(app_css(), unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🛡 Quantum Military Cybersecurity Command")
    st.markdown("""
    ### Tactical Network Defense System
    Combining:
    - Quantum-secured communication (QKD)
    - Swarm intelligence
    - AI-powered threat detection
    - Military-grade encryption
    - Real-time response
    """)

    auth_level = military_authentication()
    st.markdown("---")
    st.subheader("Tactical Configuration")

    view_mode = st.selectbox("Operational Mode", ["Real-time", "Historical", "Simulation"], key="view_mode_select")
    swarm_size = st.slider("Tactical Unit Count", 3, 20, 8, key="swarm_size_slider")
    quantum_strength = st.slider("Quantum Link Integrity", 0, 10, 9, key="quantum_strength_slider")
    quantum_enabled = st.checkbox("Enable Quantum Entanglement", True, key="quantum_enabled_check")
    qkd_enabled = st.checkbox("Enable QKD Encryption", True, key="qkd_enabled_check")
    learning_enabled = st.checkbox("Enable Adaptive Learning", True, key="learning_enabled_check")

    groq_api_key = st.text_input("Enter Command API Key", type="password", key="api_key_input",
                                 value=st.session_state.get("groq_api_key", ""))
    if groq_api_key != st.session_state.get("groq_api_key", ""):
        st.session_state.groq_api_key = groq_api_key
        st.session_state.chat_history = []
        if groq_api_key:
            st.success("API key updated successfully!")

    if st.button("🚨 EMERGENCY SHUTDOWN", help="Initiate emergency protocol", key="emergency_button"):
        shutdown_confirmed = st.checkbox("CONFIRM SHUTDOWN", key="shutdown_confirm")
        if shutdown_confirmed:
            st.error("🛑 EMERGENCY SHUTDOWN INITIATED!")
            st.stop()
        else:
            st.warning("Shutdown not confirmed")

# Data generation based on view mode
if view_mode == "Real-time":
    swarm_data = generate_realtime_data(swarm_size)
elif view_mode == "Historical":
    swarm_data = generate_historical_data(swarm_size)
else:
    swarm_data = generate_simulation_data(swarm_size)

quantum_data = generate_quantum_data(view_mode, swarm_size)
metrics = generate_performance_metrics(view_mode, swarm_size)

# Header
st.markdown(f'<div class="header"><h1>Quantum Military Cybersecurity Command - {view_mode} Mode</h1></div>', unsafe_allow_html=True)
st.markdown(f"### {view_mode} Tactical Network Threat Operations\nThis command dashboard visualizes the quantum-enhanced cybersecurity tactical system in {view_mode.lower()} mode.")

# Top metrics row
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    active_threats = int((swarm_data["Action"] != "Monitoring").sum())
    st.markdown(f"""
    <div class="metric-button" onclick="alert('Viewing threat details')">
        <h3>Active Threats</h3>
        <p>{active_threats:,}</p>
        <small>{np.random.randint(50, 200)} new</small>
    </div>
    """, unsafe_allow_html=True)
with col2:
    st.markdown(f"""
    <div class="metric-button" onclick="alert('Viewing tactical units')">
        <h3>Tactical Units</h3>
        <p>{swarm_size}</p>
        <small>All operational</small>
    </div>
    """, unsafe_allow_html=True)
with col3:
    unstable_links = int((~quantum_data["Entanglement Stability"]).sum())
    st.markdown(f"""
    <div class="metric-button" onclick="alert('Viewing quantum links')">
        <h3>Quantum Links</h3>
        <p>{quantum_strength}/10</p>
        <small>{unstable_links} unstable</small>
    </div>
    """, unsafe_allow_html=True)
with col4:
    accuracy_row = metrics[metrics["Metric"] == "Accuracy"]
    accuracy = float(accuracy_row["Value"].values[0]) if not accuracy_row.empty else 0.95
    st.metric("Detection Accuracy", f"{accuracy*100:.0f}%",
              f"{'+' if accuracy > 0.9 else ''}{(accuracy-0.9)*100:.0f}% from baseline" if view_mode != "Historical" else None)
with col5:
    avg_confidence = float(np.random.uniform(0.85, 0.99))
    st.metric("AI Confidence", f"{avg_confidence*100:.0f}%",
              f"{'+' if avg_confidence > 0.9 else ''}{(avg_confidence-0.9)*100:.0f}% from baseline")

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Command Overview", "Threat Analysis", "Quantum Security",
    "QKD Encryption", "System Performance", "Battlefield Visualization", "Command Chat Assistant"
])

with tab1:
    overview.render(view_mode, swarm_data, swarm_size, quantum_data, metrics, quantum_enabled, qkd_enabled)
with tab2:
    threat_analysis.render(view_mode, swarm_data, swarm_size)
with tab3:
    quantum_security.render(view_mode, quantum_data)
with tab4:
    qkd_encryption.render(view_mode, qkd_enabled)
with tab5:
    performance.render(view_mode, metrics)
with tab6:
    battlefield.render(swarm_data)
with tab7:
    chat_assistant.render(view_mode, swarm_size)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {primary_color};">
    <h4>Quantum Military Cybersecurity Command | {view_mode} Operation Mode | Security Clearance: {auth_level}</h4>
    <p>Developed for military applications with Streamlit, Qiskit, and tactical AI |
    Real-time threat neutralization using quantum-enhanced swarm intelligence</p>
    <p>🛡️ Classified: This system contains information protected under military cybersecurity protocols</p>
</div>
""", unsafe_allow_html=True)