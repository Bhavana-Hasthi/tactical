import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import time
import qiskit
from qiskit.visualization import plot_bloch_multivector
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, Aer, execute
import hashlib
from io import BytesIO
from PIL import Image
from faker import Faker
import requests
import json
import qiskit_algorithms
from qiskit.circuit.library import PhaseOracle
from qiskit.visualization.bloch import Bloch 
import pycountry
import datetime
import random
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
import networkx as nx
from datetime import datetime
import folium
from streamlit_folium import folium_static
from geopy.geocoders import Nominatim
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from groq import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
except Exception as e:
    GROQ_AVAILABLE = False
    st.warning(f"Groq initialization issue: {e}")

def query_leave_policy(user_message):
    url = "http://34.47.251.161:5007/chat"   # API
    payload = {"message": user_message}
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed with status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

# Initialize Faker for realistic data generation
fake = Faker()

# Military-specific configuration
MILITARY_BRANCHES = ["Army", "Navy", "Air Force", "Marines", "Coast Guard", "Space Force"]
MILITARY_RANKS = ["General", "Colonel", "Major", "Captain", "Lieutenant", "Sergeant", "Corporal"]
TACTICAL_NETWORKS = ["Tactical LAN", "SATCOM", "Radio Network", "Drone Command", "Weapons System", "Surveillance Feed"]
MILITARY_INSTALLATIONS = ["Fort Bragg", "Naval Base San Diego", "MacDill AFB", "Camp Lejeune", 
                         "Pearl Harbor", "Buckley SFB", "NORAD", "Pentagon Network"]

# Updated color theme - light yellow range with black text
primary_color = "#FFD700"  # Gold
secondary_color = "#FFA500"  # Orange
background_color = "#FFFFE0"  # Light Yellow
text_color = "#000000"  # Black
accent_color = "#8B4513"  # SaddleBrown

# Initialize Groq client
def get_groq_client():
    """Initialize and return Groq client if API key is available"""
    if "groq_api_key" in st.session_state and st.session_state.groq_api_key:
        try:
            return Groq(api_key=st.session_state.groq_api_key)
        except Exception as e:
            st.error(f"Error initializing Groq client: {str(e)}")
            return None
    return None

# Function to get random country
def get_random_country():
    indian_region_countries = [
        "India", "Pakistan", "Bangladesh", "Nepal", "Bhutan",
        "Sri Lanka", "Myanmar", "China", "Afghanistan"
    ]
    return np.random.choice(indian_region_countries)

# Function to generate military-specific IP addresses
def generate_military_ip():
    base_ips = [
        "6.", "7.", "11.", "21.", "22.", "26.", "28.", "29.", "30.", "33.",
        "55.", "56.", "214.", "215."
    ]
    return f"{random.choice(base_ips)}{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

# Function to generate military-specific threat data
def generate_military_threats():
    return np.random.choice([
        "APT-28 (Fancy Bear)", "APT-29 (Cozy Bear)", "Lazarus Group", "Equation Group",
        "Sandworm Team", "Turla", "SOGHUM", "TEMP.Veles", "Night Dragon", "Eenergetic Bear",
        "CyberBerkut", "GhostNet", "Titan Rain", "Moonlight Maze", "Operation Aurora",
        "Stuxnet", "Flame", "Gauss", "Duqu", "Regin"
    ])

# Function to generate military installation data
def generate_military_installation():
    return np.random.choice(MILITARY_INSTALLATIONS)

# Function to generate tactical network data
def generate_tactical_network():
    return np.random.choice(TACTICAL_NETWORKS)

# Enhanced mock data generation functions with military context
def generate_realtime_data(agent_count=5):
    agents = [f"Tactical-Unit-{i+1}" for i in range(agent_count)]
    threats = ["Cyber Espionage", "Infrastructure Attack", "Weapons System Compromise", 
              "Command & Control Breach", "SATCOM Jamming", "GPS Spoofing", "Drone Hijacking",
              "Supply Chain Attack", "Zero-Day Exploit", "Insider Threat"]
    data_size = 75000  # 75k records
    
    data = {
        "Agent": np.random.choice(agents, data_size),
        "Threat Type": np.random.choice(threats, data_size),
        "Threat Actor": [generate_military_threats() for _ in range(data_size)],
        "Packet Size": np.random.randint(40, 1500, data_size),
        "Action": np.random.choice(["Neutralized", "Contained", "Monitoring", "Escalated"], data_size, p=[0.4, 0.3, 0.2, 0.1]),
        "Timestamp": pd.date_range(end=pd.Timestamp.now(), periods=data_size, freq="s"),
        "Source IP": [generate_military_ip() for _ in range(data_size)],
        "Destination IP": [generate_military_ip() for _ in range(data_size)],
        "Source Country": [get_random_country() for _ in range(data_size)],
        "Destination Country": [get_random_country() for _ in range(data_size)],
        "Military Installation": [generate_military_installation() for _ in range(data_size)],
        "Tactical Network": [generate_tactical_network() for _ in range(data_size)],
        "Severity": np.random.choice(["Low", "Medium", "High", "Critical"], data_size, p=[0.1, 0.3, 0.4, 0.2]),
        "Port": np.random.choice([21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 993, 995, 1723, 3306, 3389, 5900, 8080], data_size),
        "Protocol": np.random.choice(["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "FTP", "SSH", "DNS", "SMTP"], data_size),
        "Classification": np.random.choice(["UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP SECRET"], data_size, p=[0.3, 0.4, 0.2, 0.1])
    }
    return pd.DataFrame(data)

def generate_historical_data(agent_count=5):
    agents = [f"Tactical-Unit-{i+1}" for i in range(agent_count)]
    threats = ["Cyber Espionage", "Infrastructure Attack", "Weapons System Compromise", 
              "Command & Control Breach", "SATCOM Jamming", "GPS Spoofing", "Drone Hijacking",
              "Supply Chain Attack", "Zero-Day Exploit", "Insider Threat"]
    data_size = 150000  # 150k records
    
    # Create date range from Jan 2024 to now
    end_date = pd.Timestamp.now()
    start_date = pd.Timestamp('2024-01-01')
    date_range = pd.date_range(start=start_date, end=end_date, periods=data_size)
    
    data = {
        "Agent": np.random.choice(agents, data_size),
        "Threat Type": np.random.choice(threats, data_size),
        "Threat Actor": [generate_military_threats() for _ in range(data_size)],
        "Packet Size": np.random.randint(40, 1500, data_size),
        "Action": np.random.choice(["Neutralized", "Contained", "Monitoring", "Escalated"], data_size, p=[0.4, 0.3, 0.2, 0.1]),
        "Timestamp": date_range,
        "Source IP": [generate_military_ip() for _ in range(data_size)],
        "Destination IP": [generate_military_ip() for _ in range(data_size)],
        "Source Country": [get_random_country() for _ in range(data_size)],
        "Destination Country": [get_random_country() for _ in range(data_size)],
        "Military Installation": [generate_military_installation() for _ in range(data_size)],
        "Tactical Network": [generate_tactical_network() for _ in range(data_size)],
        "Severity": np.random.choice(["Low", "Medium", "High", "Critical"], data_size, p=[0.1, 0.3, 0.4, 0.2]),
        "Port": np.random.choice([21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 993, 995, 1723, 3306, 3389, 5900, 8080], data_size),
        "Protocol": np.random.choice(["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "FTP", "SSH", "DNS", "SMTP"], data_size),
        "Classification": np.random.choice(["UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP SECRET"], data_size, p=[0.3, 0.4, 0.2, 0.1])
    }
    return pd.DataFrame(data)

def generate_simulation_data(agent_count=5):
    agents = [f"Tactical-Unit-{i+1}" for i in range(agent_count)]
    threats = ["Cyber Espionage", "Infrastructure Attack", "Weapons System Compromise", 
              "Command & Control Breach", "SATCOM Jamming", "GPS Spoofing", "Drone Hijacking",
              "Supply Chain Attack", "Zero-Day Exploit", "Insider Threat"]
    data_size = 100000  # 100k records
    
    data = {
        "Agent": np.random.choice(agents, data_size),
        "Threat Type": np.random.choice(threats, data_size),
        "Threat Actor": [generate_military_threats() for _ in range(data_size)],
        "Packet Size": np.random.randint(40, 1500, data_size),
        "Action": np.random.choice(["Neutralized", "Contained", "Monitoring", "Escalated"], data_size, p=[0.4, 0.3, 0.2, 0.1]),
        "Timestamp": pd.date_range(end=pd.Timestamp.now(), periods=data_size, freq="5min"),
        "Simulation ID": np.random.randint(1, 21, data_size),
        "Scenario": np.random.choice(["Red Team Exercise", "Blue Team Defense", "Adversary Simulation", "Contingency Plan Test"], data_size),
        "Source IP": [generate_military_ip() for _ in range(data_size)],
        "Destination IP": [generate_military_ip() for _ in range(data_size)],
        "Source Country": [get_random_country() for _ in range(data_size)],
        "Destination Country": [get_random_country() for _ in range(data_size)],
        "Military Installation": [generate_military_installation() for _ in range(data_size)],
        "Tactical Network": [generate_tactical_network() for _ in range(data_size)],
        "Severity": np.random.choice(["Low", "Medium", "High", "Critical"], data_size, p=[0.1, 0.3, 0.4, 0.2]),
        "Port": np.random.choice([21, 22, 23, 25, 53, 80, 110, 135, 139, 143, 443, 993, 995, 1723, 3306, 3389, 5900, 8080], data_size),
        "Protocol": np.random.choice(["TCP", "UDP", "ICMP", "HTTP", "HTTPS", "FTP", "SSH", "DNS", "SMTP"], data_size),
        "Classification": np.random.choice(["UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP SECRET"], data_size, p=[0.3, 0.4, 0.2, 0.1])
    }
    return pd.DataFrame(data)

def generate_quantum_data(view_mode, agent_count=5):
    if view_mode == "Real-time":
        rounds = 200
    elif view_mode == "Historical":
        rounds = 1000
    else:  # Simulation
        rounds = 500
        
    return pd.DataFrame({
        "Round": range(1, rounds + 1),
        "Entanglement Stability": np.random.choice([True, False], rounds, p=[0.92, 0.08]),
        "Qubit Fidelity": np.random.uniform(0.85, 0.99, rounds),
        "Agent": np.random.choice([f"Tactical-Unit-{i+1}" for i in range(agent_count)], rounds),
        "Latency (ms)": np.random.uniform(0.1, 2.0, rounds),
        "Throughput (Gbps)": np.random.uniform(5, 50, rounds),
        "Quantum Bit Error Rate": np.random.uniform(0.001, 0.05, rounds),
        "Decoherence Time (μs)": np.random.uniform(50, 200, rounds)
    })

def generate_performance_metrics(view_mode, agent_count=5):
    if view_mode == "Real-time":
        return pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "False Positive Rate", "Detection Rate", "Response Time", "Threat Intel Accuracy"],
            "Value": [0.96, 0.94, 0.97, 0.95, 0.03, 0.98, "85ms", 0.92],
            "Threshold": [0.9, 0.85, 0.9, 0.85, 0.1, 0.9, "100ms", 0.85],
            "Status": ["✅ Exceeds", "✅ Exceeds", "✅ Exceeds", "✅ Exceeds", "✅ Below", "✅ Exceeds", "✅ Below", "✅ Exceeds"],
            "Agent": np.random.choice([f"Tactical-Unit-{i+1}" for i in range(agent_count)], 8)
        })
    elif view_mode == "Historical":
        return pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "False Positive Rate", "Detection Rate", "Response Time", "Threat Intel Accuracy"],
            "Value": [0.92, 0.89, 0.94, 0.91, 0.06, 0.95, "110ms", 0.88],
            "Threshold": [0.85, 0.8, 0.85, 0.8, 0.12, 0.85, "150ms", 0.8],
            "Status": ["✅ Exceeds", "✅ Exceeds", "✅ Exceeds", "✅ Exceeds", "✅ Below", "✅ Exceeds", "✅ Below", "✅ Exceeds"],
            "Agent": np.random.choice([f"Tactical-Unit-{i+1}" for i in range(agent_count)], 8)
        })
    else:  # Simulation
        return pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "False Positive Rate", "Detection Rate", "Response Time", "Threat Intel Accuracy"],
            "Value": [0.98, 0.97, 0.99, 0.98, 0.01, 0.99, "65ms", 0.95],
            "Threshold": [0.9, 0.85, 0.9, 0.85, 0.1, 0.9, "100ms", 0.85],
            "Status": ["✅ Exceeds", "✅ Exceeds", "✅ Exceeds", "✅ Exceeds", "✅ Below", "✅ Exceeds", "✅ Below", "✅ Exceeds"],
            "Agent": np.random.choice([f"Tactical-Unit-{i+1}" for i in range(agent_count)], 8)
        })

def generate_agentic_ai_data(agent_count=5):
    agents = [f"Tactical-Unit-{i+1}" for i in range(agent_count)]
    data_size = 30000  # 30k records
    
    return pd.DataFrame({
        "Agent": np.random.choice(agents, data_size),
        "Decision Type": np.random.choice(["Threat Analysis", "Network Optimization", "Incident Response", 
                                         "Policy Update", "Countermeasure Deployment", "Forensic Analysis"], data_size),
        "Confidence Level": np.random.uniform(0.8, 0.99, data_size),
        "Execution Time (ms)": np.random.uniform(1, 100, data_size),
        "Timestamp": pd.date_range(end=pd.Timestamp.now(), periods=data_size, freq="min"),
        "Resource Usage (%)": np.random.uniform(5, 60, data_size),
        "Collaboration Score": np.random.uniform(0.7, 1.0, data_size),
        "Military Branch": np.random.choice(MILITARY_BRANCHES, data_size)
    })

# Enhanced QKD Functions with military-grade security
def generate_bb84_key(length=1024, eve_present=False):
    """Simulate BB84 QKD protocol to generate a key with optional eavesdropping"""
    # Alice's random bits and bases
    alice_bits = np.random.randint(2, size=length)
    alice_bases = np.random.randint(2, size=length)
    
    # Eve's interception if present
    eve_bits = None
    eve_bases = None
    if eve_present:
        eve_bases = np.random.randint(2, size=length)
        eve_bits = []
        for i in range(length):
            if eve_bases[i] == alice_bases[i]:
                eve_bits.append(alice_bits[i])  # Correct measurement
            else:
                eve_bits.append(np.random.randint(2))  # Random measurement
        
        # Eve re-sends qubits in her basis
        alice_bases = eve_bases.copy()
        alice_bits = eve_bits.copy()
    
    # Bob's random bases
    bob_bases = np.random.randint(2, size=length)
    
    # Bob's measurements
    bob_bits = []
    for i in range(length):
        if bob_bases[i] == alice_bases[i]:  # Same basis - perfect measurement
            bob_bits.append(alice_bits[i])
        else:  # Different basis - random result
            bob_bits.append(np.random.randint(2))
    
    # Sifted key (only bits where bases match)
    sifted_key = []
    for i in range(length):
        if alice_bases[i] == bob_bases[i]:
            sifted_key.append(alice_bits[i])
    
    # Error correction (simplified)
    corrected_key = sifted_key.copy()
    error_positions = []
    if eve_present:
        # Introduce errors where Eve measured incorrectly
        for i in range(len(sifted_key)):
            if np.random.random() < 0.25:  # 25% error rate from Eve
                corrected_key[i] = 1 - corrected_key[i]
                error_positions.append(i)
    
    # Privacy amplification (simplified) - ensure we have enough bits
    final_key_length = min(256, len(corrected_key))
    final_key = corrected_key[:final_key_length]  # Reduce to 256 bits for AES-256
    
    # Convert to hexadecimal for display
    if len(final_key) >= 256:
        final_str = ''.join(map(str, final_key))
        hex_key = hex(int(final_str, 2))[2:].upper().zfill(64)
    else:
        hex_key = "Not enough bits for full key"
    
    return {
        "alice_bits": alice_bits,
        "alice_bases": alice_bases,
        "bob_bases": bob_bases,
        "bob_bits": bob_bits,
        "sifted_key": sifted_key,
        "corrected_key": corrected_key,
        "final_key": final_key,
        "hex_key": hex_key,
        "matching_bases": (alice_bases == bob_bases).sum(),
        "error_rate": np.mean([a != b for a, b in zip(alice_bits[:len(bob_bits)], bob_bits)]) if len(bob_bits) > 0 else 0,
        "eve_present": eve_present,
        "eve_bits": eve_bits,
        "eve_bases": eve_bases,
        "error_positions": error_positions
    }

def visualize_bb84_protocol(qkd_data, show_bits=25):
    """Create visualization of BB84 protocol steps"""
    length = min(show_bits, len(qkd_data["alice_bits"]))
    
    # Prepare data for visualization
    positions = list(range(1, length+1))
    alice_bits = qkd_data["alice_bits"][:length]
    alice_bases = ["Z" if b == 0 else "X" for b in qkd_data["alice_bases"][:length]]
    bob_bases = ["Z" if b == 0 else "X" for b in qkd_data["bob_bases"][:length]]
    bob_bits = qkd_data["bob_bits"][:length] if len(qkd_data["bob_bits"]) >= length else [""]*length
    
    # Determine basis matching and errors
    basis_match = []
    errors = []
    for i in range(length):
        if alice_bases[i] == bob_bases[i]:
            basis_match.append("✓")
            if qkd_data["eve_present"] and i in qkd_data["error_positions"]:
                errors.append("E")
            else:
                errors.append("")
        else:
            basis_match.append("✗")
            errors.append("")
    
    # Create table
    fig = go.Figure()
    
    # Add columns to table
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
    if qkd_data["eve_present"]:
        eve_bits = qkd_data["eve_bits"][:length] if qkd_data["eve_bits"] is not None else [""]*length
        eve_bases = ["Z" if b == 0 else "X" for b in qkd_data["eve_bases"][:length]] if qkd_data["eve_bases"] is not None else [""]*length
        columns.insert(3, eve_bases)
        columns.insert(3, eve_bits)
        column_names.insert(3, "Eve Bases")
        column_names.insert(3, "Eve Bits")
    
    fig.add_trace(go.Table(
        header=dict(
            values=column_names,
            fill_color=secondary_color,
            font=dict(color="white", size=12),
            align="center"
        ),
        cells=dict(
            values=columns,
            fill_color=[background_color],
            font=dict(color=text_color),
            height=30,
            align="center"
        )
    ))
    
    title = "BB84 QKD Protocol Execution - Military Grade"
    if qkd_data["eve_present"]:
        title += " (Eavesdropper Detected)"
    
    fig.update_layout(
        height=450,
        margin=dict(l=10, r=10, b=10, t=40),
        title=dict(text=title, font=dict(size=16))
    )
    
    return fig

def visualize_qkd_summary(qkd_data):
    """Create a summary visualization of the QKD process"""
    fig = make_subplots(rows=1, cols=1)
    
    # Create summary data
    steps = ["Raw Key", "Sifted Key", "Corrected Key", "Final Key"]
    values = [
        len(qkd_data["alice_bits"]),
        len(qkd_data["sifted_key"]),
        len(qkd_data["corrected_key"]),
        len(qkd_data["final_key"])
    ]
    
    colors = [primary_color, "#9370DB", "#4682B4", "#32CD32"]
    
    fig.add_trace(go.Bar(
        x=steps,
        y=values,
        marker_color=colors,
        text=values,
        textposition='auto',
        textfont=dict(color='white')
    ))
    
    if qkd_data["eve_present"]:
        fig.add_annotation(
            x=2, y=max(values)*0.9,
            text=f"Eavesdropper detected {len(qkd_data['error_positions'])} errors",
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
        plot_bgcolor=background_color,
        paper_bgcolor=background_color
    )
    
    return fig

def create_qkd_circuit(eve_present=False):
    """Create a visualization of QKD circuit with optional eavesdropper"""
    qc = QuantumCircuit(3, 3)
    
    # Alice prepares qubits
    qc.h(0)
    qc.cx(0, 1)
    
    # Eve's interference if present
    if eve_present:
        qc.barrier()
        qc.cx(1, 2)  # Eve's measurement
        qc.barrier()
    
    # Bob measures
    qc.cx(0, 1)
    qc.h(0)
    qc.measure([0, 1], [0, 1])
    
    return qc

def save_figure_to_buffer(fig):
    """Save matplotlib figure to a BytesIO buffer"""
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor=background_color)
    buf.seek(0)
    return buf

def display_circuit(qc):
    """Display quantum circuit with proper handling"""
    fig = qc.draw(output='mpl', style='clifford', plot_barriers=True, fold=25)
    fig.patch.set_facecolor(background_color)
    buf = save_figure_to_buffer(fig)
    img = Image.open(buf)
    st.image(img, caption="Quantum Circuit Diagram", width=800)

# Enhanced Quantum-AI Verification Functions
def quantum_validate_threat(threat_type, ai_confidence):
    """Simulate quantum verification of AI threat assessment"""
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    
    # Encode threat type as rotation
    threat_levels = {
        "Cyber Espionage": 0.7,
        "Infrastructure Attack": 1.2,
        "Weapons System Compromise": 1.5,
        "Command & Control Breach": 1.4,
        "SATCOM Jamming": 1.1,
        "GPS Spoofing": 1.3,
        "Drone Hijacking": 1.6,
        "Supply Chain Attack": 0.9,
        "Zero-Day Exploit": 1.7,
        "Insider Threat": 1.0
    }
    
    rotation_angle = threat_levels.get(threat_type, 0.8)
    qc.ry(rotation_angle, 0)
        
    qc.measure_all()
    
    # Calculate quantum confidence (more sophisticated simulation)
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1024).result()
    counts = result.get_counts(qc)
    
    # Calculate confidence based on measurement distribution
    total_shots = sum(counts.values())
    confidence_0 = counts.get('000', 0) / total_shots if '000' in counts else 0
    confidence_1 = counts.get('111', 0) / total_shots if '111' in counts else 0
    
    quantum_confidence = max(confidence_0, confidence_1)
    quantum_confidence = min(0.99, max(0.7, ai_confidence * 0.9 + quantum_confidence * 0.1))
    
    return quantum_confidence, qc

def quantum_consensus(votes):
    """Quantum voting system for swarm decisions"""
    num_agents = len(votes)
    
    # Create quantum circuit
    qc = QuantumCircuit(num_agents)
    qc.h(range(num_agents))
    
    # Encode votes into quantum states
    for i, vote in enumerate(votes):
        if vote == "Neutralized":
            qc.ry(0.8, i)  # Strong agreement
        elif vote == "Contained":
            qc.ry(0.5, i)  # Moderate agreement
        elif vote == "Monitoring":
            qc.ry(0.2, i)  # Weak agreement
        else:  # Escalated
            qc.ry(1.2, i)  # Strong disagreement
    
    # Apply Grover-like amplification for consensus finding
    qc.h(range(num_agents))
    qc.x(range(num_agents))
    qc.h(num_agents-1)
    qc.mct(list(range(num_agents-1)), num_agents-1)
    qc.h(num_agents-1)
    qc.x(range(num_agents))
    qc.h(range(num_agents))
    
    qc.measure_all()
    
    # Simulate and determine consensus
    simulator = Aer.get_backend('qasm_simulator')
    result = execute(qc, simulator, shots=1024).result()
    counts = result.get_counts(qc)
    
    # Find the most common measurement
    most_common = max(counts, key=counts.get)
    
    # Convert measurement to decision (simplified)
    if most_common.count('1') > num_agents / 2:
        return "Neutralized", qc
    elif most_common.count('1') > num_agents / 3:
        return "Contained", qc
    else:
        return "Escalated", qc

def generate_quantum_metrics():
    """Generate quantum-AI performance metrics"""
    return pd.DataFrame({
        "Metric": ["Threat Accuracy", "False Positives", "Response Time", "Quantum Match", "Decoherence Resistance", "Entanglement Quality"],
        "AI Value": [0.96, 0.03, "85ms", "N/A", "N/A", "N/A"],
        "Quantum Value": [0.92, 0.05, "95ms", "94%", "98%", "96%"],
        "Status": ["✅ Verified", "✅ Optimal", "✅ Optimal", "✅ Secure", "✅ Robust", "✅ High"],
        "Threshold": [0.9, 0.1, "100ms", "85%", "90%", "90%"]
    })

# Enhanced predefined responses for military context
def get_predefined_response(question):
    """Return predefined responses for common questions"""
    question = question.lower()
    
    if any(word in question for word in ["quantum", "qkd", "encryption"]) and any(word in question for word in ["security", "secure"]):
        return "Quantum security in this system uses military-grade QKD (Quantum Key Distribution) to establish unhackable encryption keys between tactical units. The enhanced BB84 protocol is implemented with 512-bit raw keys distilled to 256-bit AES keys, achieving 99.8% qubit fidelity with real-time eavesdropper detection."
    
    elif any(word in question for word in ["threat", "attack", "intrusion"]) and any(word in question for word in ["detection", "prevention"]):
        return "The tactical swarm currently detects advanced threats with 96% accuracy. Top threat types are: Cyber Espionage (28%), Infrastructure Attack (22%), Weapons System Compromise (18%), and Command & Control Breach (15%). Critical threats are automatically neutralized within 85ms."
    
    elif any(word in question for word in ["swarm", "agent", "tactical"]) or "unit" in question:
        return f"The swarm currently has {swarm_size} active tactical units. Each unit specializes in different military network domains and collaborates using quantum-secured channels with 2ms latency. Unit roles include detection, analysis, response, forensics, and command coordination."
    
    elif any(word in question for word in ["performance", "metrics", "stats", "kpi"]):
        return "Current system metrics:\n- Threat detection accuracy: 96%\n- False positive rate: 3%\n- Quantum verification match: 94%\n- Average response time: 85ms\n- Qubit fidelity: 99.8%\nAll metrics exceed military operational thresholds."
    
    elif any(word in question for word in ["dashboard", "view", "display", "interface"]):
        return f"You're currently viewing the command dashboard in {view_mode} mode. This shows {'live tactical threat data' if view_mode=='Real-time' else 'historical patterns and trends' if view_mode=='Historical' else 'simulation and exercise results'}. Use the sidebar to switch modes and configure tactical parameters."
    
    elif any(word in question for word in ["military", "defense", "command"]):
        return "This system is designed to meet military cybersecurity requirements with quantum-resistant encryption, real-time threat neutralization, and seamless integration with existing command and control infrastructure. All communications are secured with AES-256 encryption using quantum-generated keys."
    
    elif any(word in question for word in ["help", "support", "guide", "manual"]):
        return "I can help with:\n- Explaining quantum security features\n- Showing threat detection metrics\n- Describing tactical unit functions\n- Explaining dashboard views and controls\n- Providing military cybersecurity context\nJust ask about any of these topics!"
    
    else:
        return None

# Military authentication simulation
def military_authentication():
    """Simulate military-grade authentication"""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Command Authentication")
    
    auth_status = st.sidebar.selectbox("Authentication Level", 
                                      ["UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP SECRET"],
                                      key="auth_level_select")
    
    if auth_status == "TOP SECRET":
        st.sidebar.success("✅ TOP SECRET clearance granted")
        st.sidebar.info("Full system access enabled")
        return 4
    elif auth_status == "SECRET":
        st.sidebar.warning("⚠️ SECRET clearance granted")
        st.sidebar.info("Limited system access")
        return 3
    elif auth_status == "CONFIDENTIAL":
        st.sidebar.warning("⚠️ CONFIDENTIAL clearance granted")
        st.sidebar.info("Restricted system access")
        return 2
    else:
        st.sidebar.error("🔒 UNCLASSIFIED access only")
        st.sidebar.info("Basic dashboard view only")
        return 1

# Battlefield visualization function with global threat data
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
        # Get country coordinates or use random if not in our dictionary
        country = threat['Source Country']
        if country in country_coords:
            lat, lon = float(country_coords[country]["lat"]), float(country_coords[country]["lon"])

            # Add small random offset to avoid overlapping markers
            lat += random.uniform(-5, 5)
            lon += random.uniform(-5, 5)
        else:
            # Random global coordinates for other countries
            lat = random.uniform(-55, 70)
            lon = random.uniform(-180, 180)

    
        
        # Determine color based on severity
        color = "green" if threat["Severity"] == "Low" else \
                "orange" if threat["Severity"] == "Medium" else \
                "red" if threat["Severity"] == "High" else \
                "darkred"  # Critical
        
        # Create popup content
        popup_text = f"""
        <b>Threat:</b> {threat['Threat Type']}<br>
        <b>Source:</b> {threat['Source IP']}<br>
        <b>Target:</b> {threat['Destination IP']}<br>
        <b>Severity:</b> {threat['Severity']}<br>
        <b>Status:</b> {threat['Action']}<br>
        <b>Country:</b> {threat['Source Country']}
        """
        
        # Add marker to map
        folium.Marker(
            [lat, lon],
            popup=popup_text,
            tooltip=f"{threat['Threat Type']} - {threat['Source Country']}",
            icon=folium.Icon(color=color, icon='warning-sign')
        ).add_to(m)
    
    return m

# Initialize session state
if 'groq_api_key' not in st.session_state:
    st.session_state.groq_api_key = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'view_mode' not in st.session_state:
    st.session_state.view_mode = "Real-time"
if 'agent_count' not in st.session_state:
    st.session_state.agent_count = 5
if 'qkd_eve_present' not in st.session_state:
    st.session_state.qkd_eve_present = False

# Page configuration
st.set_page_config(
    page_title="Quantum Military Cybersecurity Command",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for updated color theme
st.markdown(f"""
<style>
    .main {{
        background-color: {background_color};
        color: {text_color};
    }}
    .sidebar .sidebar-content {{
        background-color: {primary_color};
        color: {text_color};
    }}
    .metric-box {{
        background-color: {secondary_color};
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid {accent_color};
        color: {text_color};
    }}
    .threat-card {{
        background-color: #FFA07A;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #FF6347;
        color: {text_color};
    }}
    .agent-card {{
        background-color: #F0E68C;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #DAA520;
        color: {text_color};
    }}
    .quantum-card {{
        background-color: #E6E6FA;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #9370DB;
        color: {text_color};
    }}
    .qkd-card {{
        background-color: #F5DEB3;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #D2B48C;
        color: {text_color};
    }}
    .chat-card {{
        background-color: #FAFAD2;
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #DAA520;
        color: {text_color};
    }}
    .stAlert {{
        background-color: {secondary_color} !important;
        color: {text_color};
    }}
    .st-bb {{
        background-color: {secondary_color};
    }}
    .st-at {{
        background-color: {primary_color};
    }}
    .css-1aumxhk {{
        background-color: {primary_color};
        background-image: none;
        color: {text_color};
    }}
    .scanning-animation {{
        border: 2px solid {accent_color};
        border-radius: 50%;
        animation: scanning 2s linear infinite;
        position: relative;
    }}
    @keyframes scanning {{
        0% {{ transform: scale(0.8); opacity: 0.7; }}
        50% {{ transform: scale(1.1); opacity: 1; }}
        100% {{ transform: scale(0.8); opacity: 0.7; }}
    }}
    .agent-button {{
        background-color: {primary_color};
        border: 2px solid {accent_color};
        border-radius: 3px;
        padding: 5px 10px;
        margin: 5px;
        cursor: pointer;
        color: {text_color};
    }}
    .agent-button.active {{
        background-color: {accent_color};
        color: white;
        font-weight: bold;
    }}
    .chat-message {{
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        max-width: 80%;
    }}
    .user-message {{
        background-color: {primary_color};
        color: {text_color};
        margin-left: auto;
    }}
    .assistant-message {{
        background-color: {accent_color};
        color: white;
        margin-right: auto;
    }}
    .metric-button {{
        background-color: {primary_color};
        border: 2px solid {accent_color};
        border-radius: 3px;
        padding: 10px;
        margin: 5px;
        cursor: pointer;
        text-align: center;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        color: {text_color};
    }}
    .metric-button:hover {{
        background-color: {accent_color};
        color: white;
    }}
    .metric-button h3 {{
        font-size: 1.1rem;
        margin-bottom: 5px;
        color: {text_color};
    }}
    .metric-button p {{
        font-size: 1.4rem;
        font-weight: bold;
        margin: 5px 0;
        color: {text_color};
    }}
    .metric-button small {{
        font-size: 0.8rem;
        color: #666;
    }}
    .header {{
        background-color: {primary_color};
        color: {text_color};
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }}
    .subheader {{
        background-color: {secondary_color};
        color: {text_color};
        padding: 10px;
        border-radius: 3px;
        margin: 10px 0;
    }}
    .filter-container {{
        background-color: {primary_color};
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 15px;
    }}
    .threat-table {{
        background-color: {background_color};
        border-radius: 5px;
        padding: 15px;
        margin: 10px 0;
        border: 1px solid {accent_color};
    }}
</style>
""", unsafe_allow_html=True)

# Sidebar with military authentication
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
    
    # Military authentication
    auth_level = military_authentication()
    
    st.markdown("---")
    st.subheader("Tactical Configuration")
    
    view_mode = st.selectbox("Operational Mode", ["Real-time", "Historical", "Simulation"], key="view_mode_select")
    swarm_size = st.slider("Tactical Unit Count", 3, 20, 8, key="swarm_size_slider")
    quantum_strength = st.slider("Quantum Link Integrity", 0, 10, 9, key="quantum_strength_slider")
    quantum_enabled = st.checkbox("Enable Quantum Entanglement", True, key="quantum_enabled_check")
    qkd_enabled = st.checkbox("Enable QKD Encryption", True, key="qkd_enabled_check")
    learning_enabled = st.checkbox("Enable Adaptive Learning", True, key="learning_enabled_check")
    
    # Groq API Key input
    groq_api_key = st.text_input("Enter Command API Key", type="password", key="api_key_input", 
                                value=st.session_state.groq_api_key)
    
    if groq_api_key != st.session_state.groq_api_key:
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

# Generate appropriate data based on view mode
if view_mode == "Real-time":
    swarm_data = generate_realtime_data(swarm_size)
elif view_mode == "Historical":
    swarm_data = generate_historical_data(swarm_size)
else:  # Simulation
    swarm_data = generate_simulation_data(swarm_size)

quantum_data = generate_quantum_data(view_mode, swarm_size)
metrics = generate_performance_metrics(view_mode, swarm_size)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Welcome to Quantum Military Cybersecurity Command. I'm your tactical assistant. How can I help with threat analysis, quantum security, or mission operations today?"}
    ]

# Main content
st.markdown(f'<div class="header"><h1>Quantum Military Cybersecurity Command - {view_mode} Mode</h1></div>', unsafe_allow_html=True)
st.markdown(f"""
### {view_mode} Tactical Network Threat Operations
This command dashboard visualizes the quantum-enhanced cybersecurity tactical system in {view_mode.lower()} mode.
""")

# Top metrics row - make them clickable
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    active_threats = len(swarm_data[swarm_data["Action"] != "Monitoring"])
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
    unstable_links = len(quantum_data[quantum_data["Entanglement Stability"] == False])
    st.markdown(f"""
    <div class="metric-button" onclick="alert('Viewing quantum links')">
        <h3>Quantum Links</h3>
        <p>{quantum_strength}/10</p>
        <small>{unstable_links} unstable</small>
    </div>
    """, unsafe_allow_html=True)

with col4:
    accuracy = float(metrics[metrics["Metric"] == "Accuracy"]["Value"].values[0])
    st.metric("Detection Accuracy", f"{accuracy*100:.0f}%", 
             f"{'+' if accuracy > 0.9 else ''}{(accuracy-0.9)*100:.0f}% from baseline" if view_mode != "Historical" else None)

with col5:
    avg_confidence = np.random.uniform(0.85, 0.99)
    st.metric("AI Confidence", f"{avg_confidence*100:.0f}%", 
             f"{'+' if avg_confidence > 0.9 else ''}{(avg_confidence-0.9)*100:.0f}% from baseline")

# Initialize session state for showing details
if 'show_threats' not in st.session_state:
    st.session_state.show_threats = True
    st.session_state.show_agents = False
    st.session_state.show_quantum = False

# Main content tabs - adding tab7 for Command Chat Assistant
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Command Overview", 
    "Threat Analysis", 
    "Quantum Security", 
    "QKD Encryption", 
    "System Performance", 
    "Battlefield Visualization",
    "Command Chat Assistant"
])

# Command Overview Tab
with tab1:
    st.markdown(f'<div class="subheader"><h2>Command Overview - {view_mode} Operations</h2></div>', unsafe_allow_html=True)
    
    # Add filters and threat table
    st.markdown('<div class="filter-container"><h3>Active Threat Details</h3></div>', unsafe_allow_html=True)
    
    col_filter1, col_filter2, col_filter3 = st.columns(3)
    
    with col_filter1:
        threat_types = ["All"] + sorted(swarm_data["Threat Type"].unique())
        selected_threat_type = st.selectbox("Filter by Threat Type", threat_types, key="threat_type_filter_1")
    
    with col_filter2:
        severities = ["All"] + sorted(swarm_data["Severity"].unique())
        selected_severity = st.selectbox("Filter by Severity", severities, key="severity_filter_1")
    
    with col_filter3:
        # Sort agents numerically for proper ordering
        agents = ["All"] + sorted(swarm_data["Agent"].unique(), 
                                 key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isdigit() else 0)
        selected_agent = st.selectbox("Filter by Agent", agents, key="agent_filter_1")
    
    # Apply filters
    filtered_data = swarm_data.copy()
    if selected_threat_type != "All":
        filtered_data = filtered_data[filtered_data["Threat Type"] == selected_threat_type]
    if selected_severity != "All":
        filtered_data = filtered_data[filtered_data["Severity"] == selected_severity]
    if selected_agent != "All":
        filtered_data = filtered_data[filtered_data["Agent"] == selected_agent]
    
    # Display filtered threat table
    st.markdown('<div class="threat-table"><h4>Filtered Threat Data</h4></div>', unsafe_allow_html=True)
    
    # Prepare data for display
    display_columns = ["Timestamp", "Agent", "Threat Type", "Severity", "Source IP", 
                  "Source Country", "Destination IP", "Protocol", "Port", "Action"]

    # Create a copy to avoid the warning
    display_data = filtered_data[display_columns].copy()

    # Format timestamp for better readability - use .loc to avoid the warning
    display_data.loc[:, "Timestamp"] = display_data["Timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Display the table
    st.dataframe(display_data, use_container_width=True)
   
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader("Tactical Unit Network")
        
        # Create a fully connected network graph
        G = nx.Graph()
        
        # Add nodes (tactical units)
        for i in range(swarm_size):
            G.add_node(f"TU-{i+1}", type="unit", status="active")
        
        # Add edges (communication links) - ensure all units are connected
        for i in range(swarm_size):
            for j in range(i+1, swarm_size):
                latency = np.random.uniform(0.5, 5.0)
                G.add_edge(f"TU-{i+1}", f"TU-{j+1}", weight=latency, 
                          label=f"{latency:.1f}ms", quantum=np.random.random() > 0.3)
        
        # Create Plotly figure
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
                       layout=go.Layout(
                           showlegend=False,
                           hovermode='closest',
                           margin=dict(b=0, l=0, r=0, t=0),
                           xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                           height=500,
                           plot_bgcolor=background_color,
                           paper_bgcolor=background_color
                       ))
        
        st.plotly_chart(fig, use_container_width=True)
        
        if view_mode == "Real-time":
            st.subheader("Live Threat Feed")
            live_feed = st.empty()
            for i in range(5):
                threat = swarm_data.sample(1).iloc[0]
                live_feed.info(f"🚨 {threat['Timestamp']} - {threat['Threat Type']} (Severity: {threat['Severity']}) detected by {threat['Agent']} → Action: {threat['Action']}")
                time.sleep(1.5)
        elif view_mode == "Historical":
            st.subheader("Threat Patterns Timeline")
            timeline_data = swarm_data.groupby([swarm_data["Timestamp"].dt.floor("D"), "Threat Type"]).size().reset_index(name="Count")
            fig = px.area(timeline_data, x="Timestamp", y="Count", color="Threat Type",
                         title="Historical Threat Patterns",
                         color_discrete_sequence=px.colors.qualitative.Dark2)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.subheader("Simulation Results")
            sim_data = swarm_data.groupby(["Simulation ID", "Scenario"]).agg({
                "Packet Size": "mean",
                "Severity": lambda x: (x == "Critical").sum()
            }).reset_index()
            fig = px.scatter(sim_data, x="Simulation ID", y="Packet Size", size="Severity", color="Scenario",
                           title="Simulation Results by Scenario",
                           hover_name="Scenario",
                           size_max=30)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tactical Unit Status")
        agent_summary = swarm_data.groupby("Agent")["Action"].value_counts().unstack().fillna(0)
        st.dataframe(agent_summary.style.background_gradient(cmap='RdYlGn_r'), 
                    use_container_width=True)
        
        with st.expander("Latest Command Decision"):
            latest = swarm_data.iloc[-1] if view_mode == "Real-time" else swarm_data.sample(1).iloc[0]
            consensus = np.random.choice([f"{swarm_size-1}/{swarm_size}", f"{swarm_size}/{swarm_size}"], 
                                        p=[0.2, 0.8])
            st.write(f"""
            **Threat Type:** {latest['Threat Type']}  
            **Threat Actor:** {latest['Threat Actor']}  
            **Severity:** {latest['Severity']}  
            **Detected By:** {latest['Agent']}  
            **Consensus:** {consensus} units agree  
            **Action Taken:** {latest['Action']}  
            **Quantum Verification:** {"✅ Stable" if quantum_enabled else "⚠ Disabled"}  
            **QKD Secured:** {"✅ Enabled" if qkd_enabled else "⚠ Disabled"}  
            **AI Confidence:** {np.random.randint(90, 100)}%  
            **Classification:** {latest['Classification']}
            """)

# Threat Analysis Tab
with tab2:
    st.markdown(f'<div class="subheader"><h2>Threat Analysis Dashboard - {view_mode}</h2></div>', unsafe_allow_html=True)
    
    # Sort agents numerically for proper ordering
    agents = sorted([f"Tactical-Unit-{i+1}" for i in range(swarm_size)], 
                   key=lambda x: int(x.split('-')[-1]))
    selected_agent = st.radio("Select Tactical Unit", agents, horizontal=True, key="agent_radio_1")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Threat Action Analysis - {selected_agent}")
        agent_data = swarm_data[swarm_data["Agent"] == selected_agent]
        
        # Create sunburst chart for threat analysis
        threat_action_data = agent_data.groupby(["Threat Type", "Action"]).size().reset_index(name="Count")
        fig = px.sunburst(threat_action_data, path=["Threat Type", "Action"], values="Count",
                         title=f"Threat Action Distribution - {selected_agent}",
                         color="Count", color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader(f"Threat Timeline - {selected_agent}")
        if view_mode == "Real-time":
            timeline_data = (agent_data.groupby([agent_data["Timestamp"].dt.floor("min"), "Threat Type"])
                           .size().reset_index(name="Count"))
            fig = px.line(timeline_data, x="Timestamp", y="Count", color="Threat Type",
                         title=f"Real-time Threat Activity - {selected_agent}",
                         color_discrete_sequence=px.colors.qualitative.Dark2)
        elif view_mode == "Historical":
            timeline_data = (agent_data.groupby([agent_data["Timestamp"].dt.floor("H"), "Threat Type"])
                           .size().reset_index(name="Count"))
            fig = px.line(timeline_data, x="Timestamp", y="Count", color="Threat Type",
                         title=f"Historical Threat Activity - {selected_agent}",
                         color_discrete_sequence=px.colors.qualitative.Dark2)
        else:
            timeline_data = (agent_data.groupby(["Simulation ID", "Threat Type"])
                           .size().reset_index(name="Count"))
            fig = px.line(timeline_data, x="Simulation ID", y="Count", color="Threat Type",
                         title=f"Simulation Threat Activity - {selected_agent}",
                         color_discrete_sequence=px.colors.qualitative.Dark2)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader(f"Threat Statistics - {selected_agent}")
        
        # Threat severity indicators
        st.markdown("**Threat Severity Distribution**")
        severity_counts = agent_data["Severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]
        
        fig = px.pie(severity_counts, values="Count", names="Severity",
                     title=f"Threat Severity - {selected_agent}",
                     color="Severity", 
                     color_discrete_map={"Critical": "darkred", "High": "red", 
                                       "Medium": "orange", "Low": "green"})
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance metrics
        st.markdown("**Unit Performance Metrics**")
        col_perf1, col_perf2 = st.columns(2)
        with col_perf1:
            st.metric("Threats Detected", len(agent_data))
            st.metric("Neutralization Rate", 
                     f"{(agent_data['Action'] == 'Neutralized').sum()/len(agent_data)*100:.1f}%")
        with col_perf2:
            st.metric("Critical Threats", 
                     f"{(agent_data['Severity'] == 'Critical').sum()}")
            st.metric("False Positives", 
                     f"{(agent_data['Action'] == 'Monitoring').sum()/len(agent_data)*100:.1f}%")
        
        # Threat actor analysis
        st.markdown("**Top Threat Actors**")
        threat_actors = agent_data["Threat Actor"].value_counts().head(5).reset_index()
        threat_actors.columns = ["Threat Actor", "Count"]
        st.dataframe(threat_actors, use_container_width=True)

# Quantum Security Tab
with tab3:
    st.markdown(f'<div class="subheader"><h2>Quantum Security Layer - {view_mode}</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced quantum circuit visualization
        st.subheader("Quantum Entanglement Circuit")
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)
        qc.cx(0, 2)
        qc.barrier()
        
        # Add some quantum operations for visualization
        qc.ry(0.8, 0)
        qc.rz(0.5, 1)
        qc.rx(0.3, 2)
        qc.barrier()
        
        qc.cx(1, 2)
        qc.h(1)
        qc.measure_all()
        
        # Display circuit using matplotlib with enhanced styling
        fig = qc.draw(output='mpl', style='clifford', plot_barriers=True, fold=25)
        fig.patch.set_facecolor(background_color)
        fig.set_size_inches(10, 4)
        buf = BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=300, facecolor=background_color)
        buf.seek(0)
        img = Image.open(buf)
        st.image(img, caption="Quantum Circuit Diagram")
    
    with col2:
        # Quantum link stability with enhanced visualization
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
        
        fig.add_hline(y=0.9, line_dash="dash", line_color="red",
                     annotation_text="Threshold", row=1, col=1)
        
        fig.update_layout(height=600, showlegend=True,
                         plot_bgcolor=background_color,
                         paper_bgcolor=background_color)
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quantum metrics
        st.subheader("Quantum Network Metrics")
        
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Average Fidelity", 
                     f"{quantum_data['Qubit Fidelity'].mean()*100:.2f}%",
                     delta=f"{quantum_data['Qubit Fidelity'].mean()*100 - 90:.2f}%",
                     delta_color="normal")
            
            st.metric("Avg Error Rate", 
                     f"{quantum_data['Quantum Bit Error Rate'].mean()*100:.3f}%",
                     delta_color="inverse")
        
        with metric_col2:
            st.metric("Avg Decoherence Time", 
                     f"{quantum_data['Decoherence Time (μs)'].mean():.1f}μs",
                     delta_color="off")
            
            st.metric("Stable Links", 
                     f"{quantum_data['Entanglement Stability'].sum()}/{len(quantum_data)}",
                     delta_color="off")

# QKD Encryption Tab
with tab4:
    st.markdown(f'<div class="subheader"><h2>Quantum Key Distribution (QKD) - {view_mode}</h2></div>', unsafe_allow_html=True)
    
    if not qkd_enabled:
        st.warning("QKD Encryption is currently disabled. Enable it in the sidebar to see QKD operations.")
    else:
        eve_present = st.checkbox("Simulate Eavesdropper (Adversary)", False, key="eve_present_check")
        qkd_data = generate_bb84_key(1024, eve_present)  # Increased length to ensure enough bits
        st.success("🔒 QKD Encryption is active and securing tactical communications")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("BB84 Protocol Execution")
            st.plotly_chart(visualize_bb84_protocol(qkd_data), use_container_width=True)
            
            # Key statistics
            st.subheader("Key Generation Metrics")
            col_stat1, col_stat2 = st.columns(2)
            with col_stat1:
                st.metric("Raw Key Length", f"{len(qkd_data['alice_bits'])} bits")
                st.metric("Sifted Key Length", f"{len(qkd_data['sifted_key'])} bits")
            with col_stat2:
                st.metric("Final Key Length", f"{len(qkd_data['final_key'])} bits")
                st.metric("Key Rate", f"{len(qkd_data['final_key'])/len(qkd_data['alice_bits'])*100:.1f}%")
            
            if qkd_data["eve_present"]:
                st.error(f"🕵️ Adversary detected! {len(qkd_data['error_positions'])} errors found")
            else:
                st.success("✅ No eavesdropper detected")
            
            if len(qkd_data['final_key']) >= 256:
                st.success(f"Final 256-bit Key: {qkd_data['hex_key']}")
            else:
                st.warning("Insufficient matching bits for full key - continuing protocol")
        
        with col2:
            st.subheader("Key Generation Process")
            st.plotly_chart(visualize_qkd_summary(qkd_data), use_container_width=True)
            
            st.subheader("QKD Circuit Diagram")
            qc = create_qkd_circuit(eve_present)
            display_circuit(qc)
            
            st.subheader("Message Authentication")
            message = st.text_input("Enter message to authenticate:", "Tactical alert: perimeter breach detected", key="message_auth_input")
            
            if st.button("Generate Quantum HMAC", key="hmac_button"):
                if qkd_data and len(qkd_data['final_key']) >= 256:
                    key_bytes = bytes(''.join(map(str, qkd_data['final_key'])), 'utf-8')
                    message_bytes = bytes(message, 'utf-8')
                    hmac = hashlib.sha256(key_bytes + message_bytes).hexdigest()
                    st.success("Message authenticated with quantum key:")
                    st.code(hmac)
                else:
                    st.error("Not enough key material for authentication")

# System Performance Tab
with tab5:
    st.markdown(f'<div class="subheader"><h2>System Performance Metrics - {view_mode}</h2></div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Detection Performance")
        
        # Enhanced performance visualization
        fig = px.bar(metrics, x="Metric", y="Value", color="Status",
                     title=f"{view_mode} Performance Metrics",
                     color_discrete_map={"✅ Exceeds": "green", "✅ Optimal": "blue", "✅ Secure": "purple", "✅ Robust": "orange", "✅ High": "darkgreen"},
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
            fig = px.line(learning_data, x="Time", y="Reward", 
                          title="Live Learning Progress",
                          color_discrete_sequence=[primary_color])
            fig.add_scatter(x=learning_data["Time"], y=learning_data["Exploration"], 
                           mode='lines', name='Exploration Rate', line=dict(dash='dot'))
        elif view_mode == "Historical":
            learning_data = pd.DataFrame({
                "Date": pd.date_range(end=pd.Timestamp.now() - pd.Timedelta(days=30), periods=30, freq="D"),
                "Reward": np.cumsum(np.random.normal(1.0, 0.3, 30)),
                "Success Rate": np.random.uniform(0.85, 0.99, 30)
            })
            fig = px.line(learning_data, x="Date", y=["Reward", "Success Rate"], 
                          title="Historical Learning Progress",
                          color_discrete_sequence=[primary_color, secondary_color])
        else:
            learning_data = pd.DataFrame({
                "Run": range(1, 31),
                "Reward": np.cumsum(np.random.normal(1.5, 0.2, 30)),
                "Efficiency": np.random.uniform(0.9, 1.0, 30)
            })
            fig = px.line(learning_data, x="Run", y=["Reward", "Efficiency"], 
                          title="Simulation Learning Results",
                          color_discrete_sequence=[primary_color, accent_color])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Resource Utilization")
        
        # Enhanced resource utilization visualization
        if view_mode == "Real-time":
            resources = ["CPU", "Memory", "Network", "Quantum", "AI", "Storage", "I/O"]
            usage = np.random.randint(20, 85, len(resources))
            efficiency = np.random.randint(75, 99, len(resources))
            
            fig = go.Figure()
            fig.add_trace(go.Bar(x=resources, y=usage, name='Usage (%)',
                               marker_color=primary_color))
            fig.add_trace(go.Scatter(x=resources, y=efficiency, name='Efficiency (%)',
                                  line=dict(color=accent_color, width=3)))
            fig.update_layout(title="Current Resource Utilization",
                            yaxis_title="Percentage")
        elif view_mode == "Historical":
            resources = ["CPU", "Memory", "Network", "Quantum", "AI"]
            data = {
                "Resource": resources * 5,
                "Usage": np.random.randint(30, 90, len(resources) * 5),
                "Day": np.repeat(["Mon", "Tue", "Wed", "Thu", "Fri"], len(resources))
            }
            df = pd.DataFrame(data)
            fig = px.box(df, x="Resource", y="Usage", color="Day",
                        title="Weekly Resource Utilization")
        else:
            resources = ["CPU", "Memory", "Network", "Quantum", "AI"]
            usage = np.random.randint(40, 95, len(resources))
            fig = px.bar_polar(pd.DataFrame({"Resource": resources, "Usage": usage}), 
                             r="Usage", theta="Resource",
                             title="Simulated Resource Load",
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
                       title="System Component Health",
                       size_max=30)
        st.plotly_chart(fig, use_container_width=True)

# Battlefield Visualization Tab
with tab6:
    st.markdown(f'<div class="subheader"><h2>Global Threat Visualization</h2></div>', unsafe_allow_html=True)
    
    st.info("This visualization shows simulated threat locations on a global tactical map. Red markers indicate critical threats, orange indicates high severity, and yellow indicates medium severity.")
    
    # Create battlefield map with global threats
    threat_map = create_battlefield_map(swarm_data)
    folium_static(threat_map, width=1000, height=600)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Threat Concentration by Region")
        
        # Generate regional threat data
        regions = ["NORTHCOM", "SOUTHCOM", "EUCOM", "INDOPACOM", "CENTCOM", "AFRICOM"]
        threat_counts = {region: np.random.randint(50, 500) for region in regions}
        
        fig = px.bar(x=list(threat_counts.keys()), y=list(threat_counts.values()),
                    title="Threats by Combatant Command",
                    labels={"x": "Region", "y": "Threat Count"},
                    color=list(threat_counts.values()),
                    color_continuous_scale="RdYlGn_r")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Tactical Network Status")
        
        # Network status indicators
        networks = TACTICAL_NETWORKS
        status_data = []
        for network in networks:
            status = np.random.choice(["Operational", "Degraded", "Compromised", "Offline"], p=[0.7, 0.2, 0.05, 0.05])
            latency = np.random.uniform(1, 50)
            status_data.append({"Network": network, "Status": status, "Latency": latency})
        
        status_df = pd.DataFrame(status_data)
        
        fig = px.scatter(status_df, x="Network", y="Latency", color="Status",
                        color_discrete_map={"Operational": "green", "Degraded": "orange", 
                                          "Compromised": "red", "Offline": "gray"},
                        size="Latency", size_max=30,
                        title="Tactical Network Status")
        st.plotly_chart(fig, use_container_width=True)

# Command Chat Assistant Tab
with tab7:
    st.markdown('<div class="subheader"><h2>Command Chat Assistant</h2></div>', unsafe_allow_html=True)

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! How can I assist you with defence-related information today?"}
        ]

    # Show chat history
    for chat in st.session_state.chat_history:
        with st.chat_message(chat["role"]):
            st.write(chat["content"])

    # User input
    user_input = st.chat_input("Write a message", key="chat_input_tab7")

    if user_input:
        # Append user message immediately
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        # Add temporary "typing…" placeholder
        placeholder_idx = len(st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": "💭 Assistant is typing..."})
        with st.chat_message("assistant"):
            st.write("💭 Assistant is typing...")

        # Query backend / predefined
        predefined = get_predefined_response(user_input)
        if predefined:
            response_text = predefined
        else:
            response = query_leave_policy(user_input)
            if "error" in response:
                response_text = f"⚠️ {response['error']}"
            else:
                response_text = response.get("data", "No data received")

        # Replace placeholder with actual response
        st.session_state.chat_history[placeholder_idx] = {
            "role": "assistant",
            "content": response_text,
        }

        # Force rerun to refresh chat with new message
        st.rerun()

# Footer with enhanced military branding
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: {primary_color};">
    <h4>Quantum Military Cybersecurity Command | {view_mode} Operation Mode | Security Clearance: {auth_level}</h4>
    <p>Developed for military applications with Streamlit, Qiskit, and tactical AI | 
    Real-time threat neutralization using quantum-enhanced swarm intelligence</p>
    <p>🛡️ Classified: This system contains information protected under military cybersecurity protocols</p>
</div>
""", unsafe_allow_html=True)