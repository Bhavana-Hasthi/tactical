import json
import requests
import streamlit as st

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except Exception:
    GROQ_AVAILABLE = False

def get_groq_client():
    api_key = st.session_state.get("groq_api_key", "")
    if not api_key:
        return None
    try:
        return Groq(api_key=api_key) if GROQ_AVAILABLE else None
    except Exception:
        return None

def query_leave_policy(user_message: str):
    url = "http://34.47.251.161:5007/chat"
    payload = {"message": user_message}
    headers = {"Content-Type": "application/json"}
    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=20)
        if response.status_code == 200:
            return response.json()
        return {"error": f"Failed with status {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def get_predefined_response(question: str, swarm_size: int, view_mode: str):
    q = (question or "").lower()
    if any(w in q for w in ["quantum", "qkd", "encryption"]) and any(w in q for w in ["security", "secure"]):
        return ("Quantum security uses QKD to establish encryption keys between tactical units. "
                "The BB84 protocol distills 512-bit raw keys to 256-bit AES keys with real-time eavesdropper detection.")
    if any(w in q for w in ["threat", "attack", "intrusion"]) and any(w in q for w in ["detection", "prevention"]):
        return ("The swarm detects advanced threats with ~96% accuracy. Top types: Cyber Espionage, Infrastructure Attack, "
                "Weapons System Compromise, and Command & Control Breach. Critical threats are auto-neutralized within ~85ms.")
    if any(w in q for w in ["swarm", "agent", "tactical"]) or "unit" in q:
        return (f"The swarm has {swarm_size} active tactical units. Each specializes in different network domains "
                "and collaborates over quantum-secured channels with low latency.")
    if any(w in q for w in ["performance", "metrics", "stats", "kpi"]):
        return ("Current metrics:\n- Detection accuracy: ~96%\n- False positive rate: ~3%\n"
                "- Quantum verification match: ~94%\n- Avg response time: ~85ms\n- Qubit fidelity: ~99%")
    if any(w in q for w in ["dashboard", "view", "display", "interface"]):
        mode_text = "live tactical threat data" if view_mode == "Real-time" else \
                    "historical patterns and trends" if view_mode == "Historical" else \
                    "simulation and exercise results"
        return f"You're viewing the command dashboard in {view_mode} mode. This shows {mode_text}."
    if any(w in q for w in ["military", "defense", "command"]):
        return ("Designed for military cybersecurity with quantum-resistant encryption, real-time threat neutralization, "
                "and C2 integration. Comms are secured with AES-256 using quantum-generated keys.")
    if any(w in q for w in ["help", "support", "guide", "manual"]):
        return ("I can help explain quantum security, show detection metrics, describe unit functions, "
                "and explain dashboard views and controls.")
    return None