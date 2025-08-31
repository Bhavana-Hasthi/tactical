import streamlit as st
from visualization import create_battlefield_map, folium_static
import plotly.express as px
import numpy as np
import pandas as pd
import random

def get_country_coords(country):
    coords = {
        "India": (20.5937, 78.9629),
        "Pakistan": (30.3753, 69.3451),
        "Bangladesh": (23.6850, 90.3563),
        "Nepal": (28.3949, 84.1240),
        "Bhutan": (27.5142, 90.4336),
        "Sri Lanka": (7.8731, 80.7718),
        "Myanmar": (21.9162, 95.9560),
        "China": (35.8617, 104.1954),
        "Afghanistan": (33.9391, 67.7100)
    }
    if country in coords:
        lat, lon = coords[country]
        lat += random.uniform(-5, 5)
        lon += random.uniform(-5, 5)
    else:
        lat = random.uniform(-55, 70)
        lon = random.uniform(-180, 180)
    return lat, lon

def render(swarm_data: pd.DataFrame):
    st.markdown(f'<div class="subheader"><h2>Global Threat Visualization</h2></div>', unsafe_allow_html=True)
    st.info("Simulated threat locations on a global tactical map. Red: critical, orange: high, yellow: medium.")

    threat_map = create_battlefield_map(swarm_data)
    folium_static(threat_map, width=1000, height=600)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Threat Concentration by Region")
        regions = ["NORTHCOM", "SOUTHCOM", "EUCOM", "INDOPACOM", "CENTCOM", "AFRICOM"]
        threat_counts = {region: int(np.random.randint(50, 500)) for region in regions}
        fig = px.bar(x=list(threat_counts.keys()), y=list(threat_counts.values()),
                     title="Threats by Combatant Command",
                     labels={"x": "Region", "y": "Threat Count"},
                     color=list(threat_counts.values()), color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.subheader("Tactical Network Status")
        networks = ["Tactical LAN", "SATCOM", "Radio Network", "Drone Command", "Weapons System", "Surveillance Feed"]
        status_data = []
        for network in networks:
            status = np.random.choice(["Operational", "Degraded", "Compromised", "Offline"], p=[0.7, 0.2, 0.05, 0.05])
            latency = float(np.random.uniform(1, 50))
            status_data.append({"Network": network, "Status": status, "Latency": latency})
        status_df = pd.DataFrame(status_data)
        fig = px.scatter(status_df, x="Network", y="Latency", color="Status",
                         color_discrete_map={"Operational": "green", "Degraded": "orange", "Compromised": "red", "Offline": "gray"},
                         size="Latency", size_max=30, title="Tactical Network Status")
        st.plotly_chart(fig, use_container_width=True)

def api_render(swarm_data: pd.DataFrame):
    # Map data for top 50 threats
    map_data = []
    for _, threat in swarm_data.head(50).iterrows():
        lat, lon = get_country_coords(threat['Source Country'])
        map_data.append({
            "threat_type": threat["Threat Type"],
            "severity": threat["Severity"],
            "action": threat["Action"],
            "source_ip": threat["Source IP"],
            "destination_ip": threat["Destination IP"],
            "source_country": threat["Source Country"],
            "timestamp": str(threat["Timestamp"]),
            "latitude": lat,
            "longitude": lon
        })

    # Threat summary
    threat_summary = swarm_data.groupby("Severity")["Threat Type"].count().reset_index()
    threat_summary.columns = ["Severity", "Count"]

    # Top threat types
    top_threats = swarm_data["Threat Type"].value_counts().head(5).reset_index()
    top_threats.columns = ["Threat Type", "Count"]

    # Threats by region (simulate)
    regions = ["NORTHCOM", "SOUTHCOM", "EUCOM", "INDOPACOM", "CENTCOM", "AFRICOM"]
    threat_counts = {region: int(np.random.randint(50, 500)) for region in regions}

    # Tactical network status (simulate)
    networks = ["Tactical LAN", "SATCOM", "Radio Network", "Drone Command", "Weapons System", "Surveillance Feed"]
    status_data = []
    for network in networks:
        status = np.random.choice(["Operational", "Degraded", "Compromised", "Offline"], p=[0.7, 0.2, 0.05, 0.05])
        latency = float(np.random.uniform(1, 50))
        status_data.append({"Network": network, "Status": status, "Latency": latency})

    return {
        "map_data": map_data,
        "threats_by_region": threat_counts,
        "tactical_network_status": status_data
    }