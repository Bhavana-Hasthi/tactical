import numpy as np
import pandas as pd
import random
from faker import Faker
from config import MILITARY_INSTALLATIONS, TACTICAL_NETWORKS, MILITARY_BRANCHES

fake = Faker()

def get_random_country():
    indian_region_countries = [
        "India", "Pakistan", "Bangladesh", "Nepal", "Bhutan",
        "Sri Lanka", "Myanmar", "China", "Afghanistan"
    ]
    return np.random.choice(indian_region_countries)

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

def generate_realtime_data(agent_count=5):
    agents = [f"Tactical-Unit-{i+1}" for i in range(agent_count)]
    threats = ["Cyber Espionage", "Infrastructure Attack", "Weapons System Compromise", 
               "Command & Control Breach", "SATCOM Jamming", "GPS Spoofing", "Drone Hijacking",
               "Supply Chain Attack", "Zero-Day Exploit", "Insider Threat"]
    # data_size = 75000
    data_size = 100
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
        "Port": np.random.choice([21,22,23,25,53,80,110,135,139,143,443,993,995,1723,3306,3389,5900,8080], data_size),
        "Protocol": np.random.choice(["TCP","UDP","ICMP","HTTP","HTTPS","FTP","SSH","DNS","SMTP"], data_size),
        "Classification": np.random.choice(["UNCLASSIFIED","CONFIDENTIAL","SECRET","TOP SECRET"], data_size, p=[0.3,0.4,0.2,0.1])
    }
    return pd.DataFrame(data)

def generate_historical_data(agent_count=5):
    agents = [f"Tactical-Unit-{i+1}" for i in range(agent_count)]
    threats = ["Cyber Espionage", "Infrastructure Attack", "Weapons System Compromise", 
               "Command & Control Breach", "SATCOM Jamming", "GPS Spoofing", "Drone Hijacking",
               "Supply Chain Attack", "Zero-Day Exploit", "Insider Threat"]
    data_size = 100
    # data_size = 150000
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
        "Port": np.random.choice([21,22,23,25,53,80,110,135,139,143,443,993,995,1723,3306,3389,5900,8080], data_size),
        "Protocol": np.random.choice(["TCP","UDP","ICMP","HTTP","HTTPS","FTP","SSH","DNS","SMTP"], data_size),
        "Classification": np.random.choice(["UNCLASSIFIED","CONFIDENTIAL","SECRET","TOP SECRET"], data_size, p=[0.3,0.4,0.2,0.1])
    }
    return pd.DataFrame(data)

def generate_simulation_data(agent_count=5):
    agents = [f"Tactical-Unit-{i+1}" for i in range(agent_count)]
    threats = ["Cyber Espionage", "Infrastructure Attack", "Weapons System Compromise", 
               "Command & Control Breach", "SATCOM Jamming", "GPS Spoofing", "Drone Hijacking",
               "Supply Chain Attack", "Zero-Day Exploit", "Insider Threat"]
    # data_size = 100000
    data_size = 1000
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
        "Port": np.random.choice([21,22,23,25,53,80,110,135,139,143,443,993,995,1723,3306,3389,5900,8080], data_size),
        "Protocol": np.random.choice(["TCP","UDP","ICMP","HTTP","HTTPS","FTP","SSH","DNS","SMTP"], data_size),
        "Classification": np.random.choice(["UNCLASSIFIED","CONFIDENTIAL","SECRET","TOP SECRET"], data_size, p=[0.3,0.4,0.2,0.1])
    }
    return pd.DataFrame(data)

def generate_quantum_data(view_mode, agent_count=5):
    rounds = 200 if view_mode == "Real-time" else 1000 if view_mode == "Historical" else 500
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
        values = [0.96, 0.94, 0.97, 0.95, 0.03, 0.98, "85ms", 0.92]
        thresholds = [0.9, 0.85, 0.9, 0.85, 0.1, 0.9, "100ms", 0.85]
    elif view_mode == "Historical":
        values = [0.92, 0.89, 0.94, 0.91, 0.06, 0.95, "110ms", 0.88]
        thresholds = [0.85, 0.8, 0.85, 0.8, 0.12, 0.85, "150ms", 0.8]
    else:
        values = [0.98, 0.97, 0.99, 0.98, 0.01, 0.99, "65ms", 0.95]
        thresholds = [0.9, 0.85, 0.9, 0.85, 0.1, 0.9, "100ms", 0.85]

    return pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "False Positive Rate", "Detection Rate", "Response Time", "Threat Intel Accuracy"],
        "Value": values,
        "Threshold": thresholds,
        "Status": ["✅ Exceeds", "✅ Exceeds", "✅ Exceeds", "✅ Exceeds", "✅ Below", "✅ Exceeds", "✅ Below", "✅ Exceeds"],
        "Agent": np.random.choice([f"Tactical-Unit-{i+1}" for i in range(agent_count)], 8)
    })

def generate_agentic_ai_data(agent_count=5):
    agents = [f"Tactical-Unit-{i+1}" for i in range(agent_count)]
    data_size = 30000
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

def generate_bb84_key(length=1024, eve_present=False):
    # Simulate BB84 QKD protocol
    alice_bits = np.random.randint(0, 2, length)
    alice_bases = np.random.choice(['X', 'Z'], length)
    bob_bases = np.random.choice(['X', 'Z'], length)
    eve_bases = np.random.choice(['X', 'Z'], length) if eve_present else None

    # Bob measures
    if eve_present:
        # Eve intercepts and measures
        eve_bits = np.where(eve_bases == alice_bases, alice_bits, np.random.randint(0, 2, length))
        bob_bits = np.where(bob_bases == eve_bases, eve_bits, np.random.randint(0, 2, length))
    else:
        bob_bits = np.where(bob_bases == alice_bases, alice_bits, np.random.randint(0, 2, length))

    # Sifting
    sift_mask = alice_bases == bob_bases
    sifted_key = alice_bits[sift_mask]
    bob_sifted = bob_bits[sift_mask]

    # Error detection
    error_positions = np.where(sifted_key != bob_sifted)[0]
    error_rate = len(error_positions) / len(sifted_key) if len(sifted_key) else 0

    # Final key (after error correction, simplified)
    final_key = sifted_key if len(error_positions) == 0 else np.delete(sifted_key, error_positions)
    hex_key = ''.join(map(str, final_key[:256]))
    hex_key = hex(int(hex_key, 2))[2:].zfill(64) if len(hex_key) == 256 else ""

    return {
        "alice_bits": alice_bits,
        "alice_bases": alice_bases,
        "bob_bases": bob_bases,
        "bob_bits": bob_bits,
        "sifted_key": sifted_key,
        "final_key": final_key,
        "error_positions": error_positions,
        "error_rate": error_rate,
        "eve_present": eve_present,
        "hex_key": hex_key
    }