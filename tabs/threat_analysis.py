import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

def render(view_mode: str, swarm_data: pd.DataFrame, swarm_size: int):
    st.markdown(f'<div class="subheader"><h2>Threat Analysis Dashboard - {view_mode}</h2></div>', unsafe_allow_html=True)
    agents = sorted([f"Tactical-Unit-{i+1}" for i in range(swarm_size)], key=lambda x: int(x.split('-')[-1]))
    selected_agent = st.radio("Select Tactical Unit", agents, horizontal=True, key="agent_radio_1")

    col1, col2 = st.columns([2, 1])
    agent_data = swarm_data[swarm_data["Agent"] == selected_agent]

    with col1:
        st.subheader(f"Threat Action Analysis - {selected_agent}")
        threat_action_data = agent_data.groupby(["Threat Type", "Action"]).size().reset_index(name="Count")
        fig = px.sunburst(threat_action_data, path=["Threat Type", "Action"], values="Count",
                          title=f"Threat Action Distribution - {selected_agent}",
                          color="Count", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"Threat Timeline - {selected_agent}")
        if view_mode == "Real-time":
            timeline_data = agent_data.groupby([agent_data["Timestamp"].dt.floor("min"), "Threat Type"]).size().reset_index(name="Count")
            fig = px.line(timeline_data, x="Timestamp", y="Count", color="Threat Type",
                          title=f"Real-time Threat Activity - {selected_agent}",
                          color_discrete_sequence=px.colors.qualitative.Dark2)
        elif view_mode == "Historical":
            timeline_data = agent_data.groupby([agent_data["Timestamp"].dt.floor("H"), "Threat Type"]).size().reset_index(name="Count")
            fig = px.line(timeline_data, x="Timestamp", y="Count", color="Threat Type",
                          title=f"Historical Threat Activity - {selected_agent}",
                          color_discrete_sequence=px.colors.qualitative.Dark2)
        else:
            timeline_data = agent_data.groupby(["Simulation ID", "Threat Type"]).size().reset_index(name="Count")
            fig = px.line(timeline_data, x="Simulation ID", y="Count", color="Threat Type",
                          title=f"Simulation Threat Activity - {selected_agent}",
                          color_discrete_sequence=px.colors.qualitative.Dark2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader(f"Threat Statistics - {selected_agent}")
        st.markdown("**Threat Severity Distribution**")
        severity_counts = agent_data["Severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        fig = px.pie(severity_counts, values="Count", names="Severity",
                     title=f"Threat Severity - {selected_agent}",
                     color="Severity",
                     color_discrete_map={"Critical": "darkred", "High": "red", "Medium": "orange", "Low": "green"})
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("**Unit Performance Metrics**")
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            st.metric("Threats Detected", len(agent_data))
            st.metric("Neutralization Rate", f"{(agent_data['Action'] == 'Neutralized').mean()*100:.1f}%")
        with col_perf2:
            st.metric("Critical Threats", f"{(agent_data['Severity'] == 'Critical').sum()}")
            st.metric("False Positives", f"{(agent_data['Action'] == 'Monitoring').mean()*100:.1f}%")

        st.markdown("**Top Threat Actors**")
        threat_actors = agent_data["Threat Actor"].value_counts().head(5).reset_index()
        threat_actors.columns = ["Threat Actor", "Count"]
        st.dataframe(threat_actors, use_container_width=True)

def api_render(view_mode: str, swarm_data: pd.DataFrame, swarm_size: int):
    # Example: return summary stats for all agents
    result = []
    agents = sorted([f"Tactical-Unit-{i+1}" for i in range(swarm_size)], key=lambda x: int(x.split('-')[-1]))
    for agent in agents:
        agent_data = swarm_data[swarm_data["Agent"] == agent]
        summary = {
            "agent": agent,
            "threats_detected": int(len(agent_data)),
            "neutralization_rate": float((agent_data["Action"] == "Neutralized").mean()),
            "critical_threats": int((agent_data["Severity"] == "Critical").sum()),
            "false_positives": float((agent_data["Action"] == "Monitoring").mean()),
            "top_threat_actors": agent_data["Threat Actor"].value_counts().head(3).index.tolist()
        }
        result.append(summary)
    return result