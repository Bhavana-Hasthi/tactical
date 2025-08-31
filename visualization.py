import plotly.graph_objects as go
from plotly.subplots import make_subplots
from qiskit import QuantumCircuit
from PIL import Image
from io import BytesIO
import streamlit as st
import folium
from streamlit_folium import folium_static
import random

# Color theme (should match your main app)
PRIMARY_COLOR = "#FFD700"
SECONDARY_COLOR = "#FFA500"
BACKGROUND_COLOR = "#FFFFE0"
TEXT_COLOR = "#000000"

def visualize_bb84_protocol(qkd_data, show_bits=25):
    """Create visualization of BB84 protocol steps"""
    length = min(show_bits, len(qkd_data["alice_bits"]))
    positions = list(range(1, length+1))
    alice_bits = qkd_data["alice_bits"][:length]
    alice_bases = ["Z" if b == 0 else "X" for b in qkd_data["alice_bases"][:length]]
    bob_bases = ["Z" if b == 0 else "X" for b in qkd_data["bob_bases"][:length]]
    bob_bits = qkd_data["bob_bits"][:length] if len(qkd_data["bob_bits"]) >= length else [""]*length

    basis_match = []
    errors = []
    for i in range(length):
        if alice_bases[i] == bob_bases[i]:
            basis_match.append("✓")
            if qkd_data.get("eve_present") and i in qkd_data.get("error_positions", []):
                errors.append("E")
            else:
                errors.append("")
        else:
            basis_match.append("✗")
            errors.append("")

    columns = [
        positions,
        alice_bits,
        alice_bases,
        bob_bases,
        bob_bits,
        basis_match,
        errors
    ]
    column_names = [
        "Qubit",
        "Alice Bits",
        "Alice Bases",
        "Bob Bases",
        "Bob Bits",
        "Match",
        "Errors"
    ]

    # Add Eve's data if present
    if qkd_data.get("eve_present"):
        eve_bits = qkd_data.get("eve_bits", [""]*length)[:length]
        eve_bases = ["Z" if b == 0 else "X" for b in qkd_data.get("eve_bases", [0]*length)[:length]]
        columns.insert(3, eve_bases)
        columns.insert(3, eve_bits)
        column_names.insert(3, "Eve Bases")
        column_names.insert(3, "Eve Bits")

    fig = go.Figure()
    fig.add_trace(go.Table(
        header=dict(
            values=column_names,
            fill_color=SECONDARY_COLOR,
            font=dict(color="white", size=12),
            align="center"
        ),
        cells=dict(
            values=columns,
            fill_color=[BACKGROUND_COLOR],
            font=dict(color=TEXT_COLOR),
            height=30,
            align="center"
        )
    ))

    title = "BB84 QKD Protocol Execution"
    if qkd_data.get("eve_present"):
        title += " (Eavesdropper Detected)"

    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, b=10, t=40),
        title=dict(text=title, font=dict(size=16))
    )
    return fig

def visualize_qkd_summary(qkd_data):
    """Create a summary visualization of the QKD process"""
    steps = ["Raw Key", "Sifted Key", "Corrected Key", "Final Key"]
    values = [
        len(qkd_data["alice_bits"]),
        len(qkd_data["sifted_key"]),
        len(qkd_data["corrected_key"]),
        len(qkd_data["final_key"])
    ]
    colors = [PRIMARY_COLOR, "#9370DB", "#4682B4", "#32CD32"]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=steps,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto',
        textfont=dict(color='white')
    ))

    if qkd_data.get("eve_present"):
        fig.add_annotation(
            x=2, y=max(values)*0.9,
            text=f"Eavesdropper detected {len(qkd_data.get('error_positions', []))} errors",
            showarrow=True,
            arrowhead=1,
            ax=0,
            ay=-40,
            bgcolor="#FF6347",
            font=dict(color="white")
        )

    fig.update_layout(
        title=dict(text="QKD Key Generation Summary", font=dict(size=16)),
        yaxis_title="Number of Bits",
        height=450,
        plot_bgcolor=BACKGROUND_COLOR,
        paper_bgcolor=BACKGROUND_COLOR
    )
    return fig

def create_qkd_circuit(eve_present=False):
    """Create a visualization of QKD circuit with optional eavesdropper"""
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.cx(0, 1)
    if eve_present:
        qc.barrier()
        qc.cx(1, 2)
        qc.barrier()
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    return qc

def save_figure_to_buffer(fig):
    """Save matplotlib figure to a BytesIO buffer"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor=BACKGROUND_COLOR)
    buf.seek(0)
    return buf

def display_circuit(qc):
    """Display quantum circuit with proper handling in Streamlit"""
    fig = qc.draw(output='mpl', style='clifford', plot_barriers=True, fold=25)
    fig.patch.set_facecolor(BACKGROUND_COLOR)
    buf = save_figure_to_buffer(fig)
    img = Image.open(buf)
    st.image(img, caption="Quantum Circuit Diagram", use_column_width=True, clamp=True)

def create_battlefield_map(threat_data):
    """Create a global battlefield visualization of threats"""
    # Create a base world map
    m = folium.Map(location=[20, 0], zoom_start=2, tiles='OpenStreetMap')

    # Get country coordinates (simplified approach)
    country_coords = {
        "India": {"lat": 20.5937, "lon": 78.9629},
        "Pakistan": {"lat": 30.3753, "lon": 69.3451},
        "Bangladesh": {"lat": 23.6850, "lon": 90.3563},
        "Nepal": {"lat": 28.3949, "lon": 84.1240},
        "Bhutan": {"lat": 27.5142, "lon": 90.4336},
        "Sri Lanka": {"lat": 7.8731, "lon": 80.7718},
        "Myanmar": {"lat": 21.9162, "lon": 95.9560},
        "China": {"lat": 35.8617, "lon": 104.1954},
        "Afghanistan": {"lat": 33.9391, "lon": 67.7100}
    }

    # Add threats to the map
    for _, threat in threat_data.head(50).iterrows():
        country = threat['Source Country']
        if country in country_coords:
            lat, lon = float(country_coords[country]["lat"]), float(country_coords[country]["lon"])
            lat += random.uniform(-5, 5)
            lon += random.uniform(-5, 5)
        else:
            lat = random.uniform(-55, 70)
            lon = random.uniform(-180, 180)

        color = "green" if threat["Severity"] == "Low" else \
                "orange" if threat["Severity"] == "Medium" else \
                "red" if threat["Severity"] == "High" else \
                "darkred"  # Critical

        popup_text = f"""
        <b>Threat:</b> {threat['Threat Type']}<br>
        <b>Source:</b> {threat['Source IP']}<br>
        <b>Target:</b> {threat['Destination IP']}<br>
        <b>Severity:</b> {threat['Severity']}<br>
        <b>Status:</b> {threat['Action']}<br>
        <b>Country:</b> {threat['Source Country']}
        """

        folium.Marker(
            [lat, lon],
            popup=popup_text,
            tooltip=f"{threat['Threat Type']} - {threat['Source Country']}",
            icon=folium.Icon(color=color, icon='warning-sign')
        ).add_to(m)

    return m