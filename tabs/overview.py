import time
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from config import background_color, primary_color

def render(view_mode: str, swarm_data: pd.DataFrame, swarm_size: int, quantum_data: pd.DataFrame,
           metrics: pd.DataFrame, quantum_enabled: bool, qkd_enabled: bool):
    st.markdown(f'<div class="subheader"><h2>Command Overview - {view_mode} Operations</h2></div>', unsafe_allow_html=True)
    st.markdown('<div class="filter-container"><h3>Active Threat Details</h3></div>', unsafe_allow_html=True)

    col_filter1, col_filter2, col_filter3 = st.columns(3)
    with col_filter1:
        threat_types = ["All"] + sorted(swarm_data["Threat Type"].unique())
        selected_threat_type = st.selectbox("Filter by Threat Type", threat_types, key="threat_type_filter_1")
    with col_filter2:
        severities = ["All"] + sorted(swarm_data["Severity"].unique())
        selected_severity = st.selectbox("Filter by Severity", severities, key="severity_filter_1")
    with col_filter3:
        agents = ["All"] + sorted(swarm_data["Agent"].unique(),
                                  key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)
        selected_agent = st.selectbox("Filter by Agent", agents, key="agent_filter_1")

    filtered_data = swarm_data.copy()
    if selected_threat_type != "All":
        filtered_data = filtered_data[filtered_data["Threat Type"] == selected_threat_type]
    if selected_severity != "All":
        filtered_data = filtered_data[filtered_data["Severity"] == selected_severity]
    if selected_agent != "All":
        filtered_data = filtered_data[filtered_data["Agent"] == selected_agent]

    st.markdown('<div class="threat-table"><h4>Filtered Threat Data</h4></div>', unsafe_allow_html=True)
    display_columns = ["Timestamp", "Agent", "Threat Type", "Severity", "Source IP",
                       "Source Country", "Destination IP", "Protocol", "Port", "Action"]
    display_data = filtered_data[display_columns].copy()
    display_data.loc[:, "Timestamp"] = display_data["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(display_data, use_container_width=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Tactical Unit Network")
        G = nx.Graph()
        for i in range(swarm_size):
            G.add_node(f"TU-{i+1}", type="unit", status="active")
        for i in range(swarm_size):
            for j in range(i+1, swarm_size):
                latency = float(np.random.uniform(0.5, 5.0))
                G.add_edge(f"TU-{i+1}", f"TU-{j+1}", weight=latency,
                           label=f"{latency:.1f}ms", quantum=bool(np.random.random() > 0.3))
        pos = nx.spring_layout(G, seed=42)
        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_color = "#9370DB" if edge[2]['quantum'] else "#4682B4"
            edge_width = 3 if edge[2]['quantum'] else 2
            edge_trace = go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                line=dict(width=edge_width, color=edge_color),
                hoverinfo='text',
                text=f"Latency: {edge[2]['weight']:.1f}ms<br>Type: {'Quantum' if edge[2]['quantum'] else 'Classical'}",
                mode='lines')
            edge_traces.append(edge_trace)
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            marker=dict(size=30, color=primary_color),
            text=[node for node in G.nodes()],
            textposition="middle center",
            hoverinfo='text',
            textfont=dict(color='white'))
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(showlegend=False, hovermode='closest',
                                         margin=dict(b=0, l=0, r=0, t=0),
                                         xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                         height=500, plot_bgcolor=background_color, paper_bgcolor=background_color))
        st.plotly_chart(fig, use_container_width=True)

        if view_mode == "Real-time":
            st.subheader("Live Threat Feed")
            live_feed = st.empty()
            for _ in range(5):
                threat = swarm_data.sample(1).iloc[0]
                live_feed.info(f"🚨 {threat['Timestamp']} - {threat['Threat Type']} (Severity: {threat['Severity']}) "
                               f"detected by {threat['Agent']} → Action: {threat['Action']}")
                time.sleep(1.5)
        elif view_mode == "Historical":
            st.subheader("Threat Patterns Timeline")
            timeline_data = swarm_data.groupby([swarm_data["Timestamp"].dt.floor("D"), "Threat Type"]).size().reset_index(name="Count")
            fig = px.area(timeline_data, x="Timestamp", y="Count", color="Threat Type",
                          title="Historical Threat Patterns", color_discrete_sequence=px.colors.qualitative.Dark2)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Simulation Results")
            sim_data = swarm_data.groupby(["Simulation ID", "Scenario"]).agg({
                "Packet Size": "mean",
                "Severity": lambda x: (x == "Critical").sum()
            }).reset_index()
            fig = px.scatter(sim_data, x="Simulation ID", y="Packet Size", size="Severity", color="Scenario",
                             title="Simulation Results by Scenario", hover_name="Scenario", size_max=30)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Tactical Unit Status")
        agent_summary = swarm_data.groupby("Agent")["Action"].value_counts().unstack().fillna(0)
        st.dataframe(agent_summary.style.background_gradient(cmap='RdYlGn_r'), use_container_width=True)
        with st.expander("Latest Command Decision"):
            latest = swarm_data.iloc[-1] if view_mode == "Real-time" else swarm_data.sample(1).iloc[0]
            consensus = np.random.choice([f"{swarm_size-1}/{swarm_size}", f"{swarm_size}/{swarm_size}"], p=[0.2, 0.8])
            st.write(f"""
            **Threat Type:** {latest['Threat Type']}  
            **Threat Actor:** {latest['Threat Actor']}  
            **Severity:** {latest['Severity']}  
            **Detected By:** {latest['Agent']}  
            **Consensus:** {consensus} units agree  
            **Action Taken:** {latest['Action']}  
            **Quantum Verification:** {"✅ Stable" if quantum_enabled else "⚠ Disabled"}  
            **QKD Secured:** {"✅ Enabled" if qkd_enabled else "⚠ Disabled"}  
            **AI Confidence:** {int(np.random.randint(90, 100))}%  
            **Classification:** {latest['Classification']}
            """)