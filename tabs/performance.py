import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

def render(view_mode: str, metrics: pd.DataFrame):
    st.markdown(f'<div class="subheader"><h2>System Performance Metrics - {view_mode}</h2></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detection Performance")
        fig = px.bar(metrics, x="Metric", y="Value", color="Status",
                     title=f"{view_mode} Performance Metrics",
                     color_discrete_map={"✅ Exceeds": "green", "✅ Optimal": "blue",
                                         "✅ Secure": "purple", "✅ Robust": "orange", "✅ High": "darkgreen"},
                     text="Value")
        fig.update_traces(texttemplate='%{text}', textposition='outside')
        fig.add_hline(y=0.9, line_dash="dash", line_color="red", opacity=0.5)
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Adaptive Learning Progress")
        if view_mode == "Real-time":
            learning_data = pd.DataFrame({
                "Time": pd.date_range(end=pd.Timestamp.now(), periods=30, freq="min"),
                "Reward": np.cumsum(np.random.normal(1.2, 0.4, 30)),
                "Exploration": np.random.uniform(0.1, 0.3, 30)
            })
            fig = px.line(learning_data, x="Time", y="Reward", title="Live Learning Progress",
                          color_discrete_sequence=["#FFD700"])
            fig.add_scatter(x=learning_data["Time"], y=learning_data["Exploration"],
                            mode='lines', name='Exploration Rate', line=dict(dash='dot'))
        elif view_mode == "Historical":
            learning_data = pd.DataFrame({
                "Date": pd.date_range(end=pd.Timestamp.now() - pd.Timedelta(days=30), periods=30, freq="D"),
                "Reward": np.cumsum(np.random.normal(1.0, 0.3, 30)),
                "Success Rate": np.random.uniform(0.85, 0.99, 30)
            })
            fig = px.line(learning_data, x="Date", y=["Reward", "Success Rate"], title="Historical Learning Progress",
                          color_discrete_sequence=["#FFD700", "#FFA500"])
        else:
            learning_data = pd.DataFrame({
                "Run": range(1, 31),
                "Reward": np.cumsum(np.random.normal(1.5, 0.2, 30)),
                "Efficiency": np.random.uniform(0.9, 1.0, 30)
            })
            fig = px.line(learning_data, x="Run", y=["Reward", "Efficiency"], title="Simulation Learning Results",
                          color_discrete_sequence=["#FFD700", "#8B4513"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Resource Utilization")
        if view_mode == "Real-time":
            resources = ["CPU", "Memory", "Network", "Quantum", "AI", "Storage", "I/O"]
            usage = np.random.randint(20, 85, len(resources))
            efficiency = np.random.randint(75, 99, len(resources))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=resources, y=usage, name='Usage (%)', marker_color="#FFD700"))
            fig.add_trace(go.Scatter(x=resources, y=efficiency, name='Efficiency (%)', line=dict(color="#8B4513", width=3)))
            fig.update_layout(title="Current Resource Utilization", yaxis_title="Percentage")
        elif view_mode == "Historical":
            resources = ["CPU", "Memory", "Network", "Quantum", "AI"]
            data = {"Resource": resources * 5, "Usage": np.random.randint(30, 90, len(resources) * 5),
                    "Day": np.repeat(["Mon", "Tue", "Wed", "Thu", "Fri"], len(resources))}
            df = pd.DataFrame(data)
            fig = px.box(df, x="Resource", y="Usage", color="Day", title="Weekly Resource Utilization")
        else:
            resources = ["CPU", "Memory", "Network", "Quantum", "AI"]
            usage = np.random.randint(40, 95, len(resources))
            fig = px.bar_polar(pd.DataFrame({"Resource": resources, "Usage": usage}),
                               r="Usage", theta="Resource", title="Simulated Resource Load",
                               color="Usage", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("System Health Status")
        health_data = pd.DataFrame({
            "Component": ["Network", "Quantum", "AI", "Storage", "Security"],
            "Status": ["Healthy", "Degraded", "Healthy", "Healthy", "Optimal"],
            "Value": [98, 82, 95, 99, 100]
        })
        fig = px.scatter(health_data, x="Component", y="Value", size="Value", color="Status",
                         color_discrete_map={"Healthy": "green", "Degraded": "orange", "Optimal": "blue"},
                         title="System Component Health", size_max=30)
        st.plotly_chart(fig, use_container_width=True)

def api_render(view_mode: str, metrics: pd.DataFrame):
    # Detection Performance
    detection_metrics = [
        {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
         for k, v in row.items()}
        for row in metrics.to_dict(orient="records")
    ]

    # Adaptive Learning Progress
    if view_mode == "Real-time":
        learning_data = pd.DataFrame({
            "Time": pd.date_range(end=pd.Timestamp.now(), periods=30, freq="min"),
            "Reward": np.cumsum(np.random.normal(1.2, 0.4, 30)),
            "Exploration": np.random.uniform(0.1, 0.3, 30)
        })
        learning = [
            {
                "Time": str(row["Time"]),
                "Reward": float(row["Reward"]),
                "Exploration": float(row["Exploration"])
            }
            for row in learning_data.to_dict(orient="records")
        ]
    elif view_mode == "Historical":
        learning_data = pd.DataFrame({
            "Date": pd.date_range(end=pd.Timestamp.now() - pd.Timedelta(days=30), periods=30, freq="D"),
            "Reward": np.cumsum(np.random.normal(1.0, 0.3, 30)),
            "Success Rate": np.random.uniform(0.85, 0.99, 30)
        })
        learning = [
            {
                "Date": str(row["Date"]),
                "Reward": float(row["Reward"]),
                "Success Rate": float(row["Success Rate"])
            }
            for row in learning_data.to_dict(orient="records")
        ]
    else:
        learning_data = pd.DataFrame({
            "Run": range(1, 31),
            "Reward": np.cumsum(np.random.normal(1.5, 0.2, 30)),
            "Efficiency": np.random.uniform(0.9, 1.0, 30)
        })
        learning = [
            {
                "Run": int(row["Run"]),
                "Reward": float(row["Reward"]),
                "Efficiency": float(row["Efficiency"])
            }
            for row in learning_data.to_dict(orient="records")
        ]

    # Resource Utilization
    if view_mode == "Real-time":
        resources = ["CPU", "Memory", "Network", "Quantum", "AI", "Storage", "I/O"]
        usage = np.random.randint(20, 85, len(resources))
        efficiency = np.random.randint(75, 99, len(resources))
        resource_util = [
            {
                "Resource": r,
                "Usage": int(u),
                "Efficiency": int(e)
            }
            for r, u, e in zip(resources, usage, efficiency)
        ]
    elif view_mode == "Historical":
        resources = ["CPU", "Memory", "Network", "Quantum", "AI"]
        data = {"Resource": resources * 5, "Usage": np.random.randint(30, 90, len(resources) * 5),
                "Day": np.repeat(["Mon", "Tue", "Wed", "Thu", "Fri"], len(resources))}
        df = pd.DataFrame(data)
        resource_util = [
            {
                "Resource": row["Resource"],
                "Usage": int(row["Usage"]),
                "Day": row["Day"]
            }
            for row in df.to_dict(orient="records")
        ]
    else:
        resources = ["CPU", "Memory", "Network", "Quantum", "AI"]
        usage = np.random.randint(40, 95, len(resources))
        resource_util = [
            {
                "Resource": r,
                "Usage": int(u)
            }
            for r, u in zip(resources, usage)
        ]

    # System Health Status
    health_data = pd.DataFrame({
        "Component": ["Network", "Quantum", "AI", "Storage", "Security"],
        "Status": ["Healthy", "Degraded", "Healthy", "Healthy", "Optimal"],
        "Value": [98, 82, 95, 99, 100]
    })
    health_status = [
        {
            "Component": row["Component"],
            "Status": row["Status"],
            "Value": int(row["Value"])
        }
        for row in health_data.to_dict(orient="records")
    ]

    return {
        "detection_performance": detection_metrics,
        "adaptive_learning": learning,
        "resource_utilization": resource_util,
        "system_health": health_status
    }