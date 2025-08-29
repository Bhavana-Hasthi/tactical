import numpy as np
import pandas as pd
import random
from datetime import datetime

# Define fixed lists
MILITARY_BRANCHES = ["Army", "Navy", "Air Force", "Marines", "Coast Guard", "Space Force"]
MILITARY_INSTALLATIONS = ["Fort Bragg", "Naval Base San Diego", "MacDill AFB", "Camp Lejeune", 
                          "Pearl Harbor", "Buckley SFB", "NORAD", "Pentagon Network"]
TACTICAL_NETWORKS = ["Tactical LAN", "SATCOM", "Radio Network", "Drone Command", "Weapons System", "Surveillance Feed"]
INDIAN_REGION_COUNTRIES = [
    "India", "Pakistan", "Bangladesh", "Nepal", "Bhutan",
    "Sri Lanka", "Myanmar", "China", "Afghanistan"
]
THREATS = [
    "Cyber Espionage", "Infrastructure Attack", "Weapons System Compromise", 
    "Command & Control Breach", "SATCOM Jamming", "GPS Spoofing", "Drone Hijacking",
    "Supply Chain Attack", "Zero-Day Exploit", "Insider Threat"
]

def get_random_country():
    return np.random.choice(INDIAN_REGION_COUNTRIES)

def generate_military_ip():
    base_ips = [
        "6.", "7.", "11.", "21.", "22.", "26.", "28.", "29.", "30.", "33.",
        "55.", "56.", "214.", "215."
    ]
    return f"{random.choice(base_ips)}{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}"

def generate_military_threats():
    return np.random.choice([
        "APT-28 (Fancy Bear)", "APT-29 (Cozy Bear)", "Lazarus Group", "Equation Group",
        "Sandworm Team", "Turla", "SOGHUM", "TEMP.Veles", "Night Dragon", "Energetic Bear",
        "CyberBerkut", "GhostNet", "Titan Rain", "Moonlight Maze", "Operation Aurora",
        "Stuxnet", "Flame", "Gauss", "Duqu", "Regin"
    ])

def generate_military_installation():
    return np.random.choice(MILITARY_INSTALLATIONS)

def generate_tactical_network():
    return np.random.choice(TACTICAL_NETWORKS)

# ========== DATA FUNCTIONS ========== #

def generate_realtime_data(agent_count=5):
    agents = [f"Tactical-Unit-{i+1}" for i in range(agent_count)]
    data_size = 1000
    data = {
        "Agent": np.random.choice(agents, data_size),
        "Threat Type": np.random.choice(THREATS, data_size),
        "Threat Actor": [generate_military_threats() for _ in range(data_size)],
        "Packet Size": np.random.randint(40, 1500, data_size),
        "Action": np.random.choice(["Neutralized", "Contained", "Monitoring", "Escalated"], data_size),
        "Timestamp": pd.date_range(end=pd.Timestamp.now(), periods=data_size, freq="s"),
        "Source IP": [generate_military_ip() for _ in range(data_size)],
        "Destination IP": [generate_military_ip() for _ in range(data_size)],
        "Source Country": [get_random_country() for _ in range(data_size)],
        "Destination Country": [get_random_country() for _ in range(data_size)],
        "Military Installation": [generate_military_installation() for _ in range(data_size)],
        "Tactical Network": [generate_tactical_network() for _ in range(data_size)],
        "Severity": np.random.choice(["Low", "Medium", "High", "Critical"], data_size),
        "Port": np.random.choice([80, 443, 21, 22, 25, 3306, 8080], data_size),
        "Protocol": np.random.choice(["TCP", "UDP", "HTTP", "HTTPS", "DNS"], data_size),
        "Classification": np.random.choice(["UNCLASSIFIED", "CONFIDENTIAL", "SECRET", "TOP SECRET"], data_size)
    }
    return pd.DataFrame(data)

def generate_historical_data(agent_count=5):
    df = generate_realtime_data(agent_count)
    df["Timestamp"] = pd.date_range(start="2024-01-01", periods=len(df), freq="min")
    return df

def generate_simulation_data(agent_count=5):
    df = generate_realtime_data(agent_count)
    df["Simulation ID"] = np.random.randint(1, 21, len(df))
    df["Scenario"] = np.random.choice(["Red Team Exercise", "Blue Team Defense"], len(df))
    return df

def generate_quantum_data(view_mode="Real-time", agent_count=5):
    rounds = 200 if view_mode == "Real-time" else 500
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

def generate_performance_metrics(view_mode="Real-time", agent_count=5):
    return pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "False Positive Rate"],
        "Value": [0.96, 0.94, 0.97, 0.95, 0.03],
        "Threshold": [0.9, 0.85, 0.9, 0.85, 0.1],
        "Status": ["✅ Exceeds"]*4 + ["✅ Below"],
        "Agent": np.random.choice([f"Tactical-Unit-{i+1}" for i in range(agent_count)], 5)
    })

def generate_agentic_ai_data(agent_count=5):
    agents = [f"Tactical-Unit-{i+1}" for i in range(agent_count)]
    data_size = 500
    return pd.DataFrame({
        "Agent": np.random.choice(agents, data_size),
        "Decision Type": np.random.choice(["Threat Analysis", "Response", "Optimization"], data_size),
        "Confidence Level": np.random.uniform(0.8, 0.99, data_size),
        "Execution Time (ms)": np.random.uniform(1, 100, data_size),
        "Timestamp": pd.date_range(end=pd.Timestamp.now(), periods=data_size, freq="min"),
        "Resource Usage (%)": np.random.uniform(5, 60, data_size),
        "Collaboration Score": np.random.uniform(0.7, 1.0, data_size),
        "Military Branch": np.random.choice(MILITARY_BRANCHES, data_size)
    })
