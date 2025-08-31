from fastapi import FastAPI, APIRouter, Depends, Body
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from fastapi.middleware.cors import CORSMiddleware
from authentication import military_authentication
from data_generation import (
    generate_realtime_data, generate_historical_data, generate_simulation_data,
    generate_quantum_data, generate_performance_metrics
)
from tabs import overview, threat_analysis, quantum_security, qkd_encryption, performance, battlefield, chat_assistant

app = FastAPI(title="Quantum Military Cybersecurity Command API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_methods=["*"],
    allow_headers=["*"]
)


# --- Models ---
class AuthRequest(BaseModel):
    username: str
    password: str

class ChatRequest(BaseModel):
    message: str
    api_key: Optional[str] = None

# --- Routers ---
auth_router = APIRouter(prefix="/auth", tags=["Authentication"])
data_router = APIRouter(prefix="/data", tags=["Data"])
tabs_router = APIRouter(prefix="/tabs", tags=["Tabs"])
chat_router = APIRouter(prefix="/chat", tags=["Chat"])
system_router = APIRouter(prefix="/system", tags=["System"])

# --- Authentication ---
@app.get("/")
def home_page():
    return {"message": "Welcome to the Quantum Military Cybersecurity Command API"}

@auth_router.post("/authenticate")
def authenticate(req: AuthRequest):
    level = military_authentication(req.username, req.password)
    return {"auth_level": level}

# --- Data Endpoints ---
@data_router.get("/realtime")
def realtime(swarm_size: int = 8):
    df = generate_realtime_data(swarm_size)
    return df.to_dict(orient="records")

@data_router.get("/historical")
def historical(swarm_size: int = 8):
    df = generate_historical_data(swarm_size)
    return df.to_dict(orient="records")

@data_router.get("/simulation")
def simulation(swarm_size: int = 8):
    df = generate_simulation_data(swarm_size)
    return df.to_dict(orient="records")

@data_router.get("/quantum")
def quantum(view_mode: str = "Real-time", swarm_size: int = 8):
    df = generate_quantum_data(view_mode, swarm_size)
    return df.to_dict(orient="records")

@data_router.get("/metrics")
def metrics(view_mode: str = "Real-time", swarm_size: int = 8):
    df = generate_performance_metrics(view_mode, swarm_size)
    return df.to_dict(orient="records")

@data_router.get("/kpi")
def kpi(
    view_mode: str = "Real-time",
    swarm_size: int = 8,
    quantum_strength: int = 9
):
    import numpy as np
    # Swarm data
    if view_mode == "Real-time":
        swarm_data = generate_realtime_data(swarm_size)
    elif view_mode == "Historical":
        swarm_data = generate_historical_data(swarm_size)
    else:
        swarm_data = generate_simulation_data(swarm_size)

    # Quantum data
    quantum_data = generate_quantum_data(view_mode, swarm_size)
    unstable_links = int((~quantum_data["Entanglement Stability"]).sum())

    # Metrics
    metrics = generate_performance_metrics(view_mode, swarm_size)
    accuracy_row = metrics[metrics["Metric"] == "Accuracy"]
    accuracy = float(accuracy_row["Value"].values[0]) if not accuracy_row.empty else 0.95

    # AI Confidence (simulated, as in main.py)
    avg_confidence = float(np.random.uniform(0.85, 0.99))

    # KPIs
    active_threats = int((swarm_data["Action"] != "Monitoring").sum())
    tactical_units = len(set(swarm_data["Agent"]))

    # Random trend values for each KPI
    def random_trend():
        value = np.random.uniform(-2, 2)
        return f"{value:+.2f}%"

    return {
        "active_threats": {
            "value": active_threats,
            "trend": random_trend()
        },
        "tactical_units": {
            "value": tactical_units,
            "trend": random_trend()
        },
        "quantum_links": {
            "value": f"{quantum_strength}/10",
            "trend": random_trend()
        },
        "detection_accuracy": {
            "value": f"{accuracy*100:.0f}%",
            "trend": random_trend()
        },
        "ai_confidence": {
            "value": f"{avg_confidence*100:.0f}%",
            "trend": random_trend()
        }
    }

@data_router.get("/swarm")
def swarm(view_mode: str = "Real-time", swarm_size: int = 8):
    if view_mode == "Real-time":
        df = generate_realtime_data(swarm_size)
    elif view_mode == "Historical":
        df = generate_historical_data(swarm_size)
    elif view_mode == "Simulation":
        df = generate_simulation_data(swarm_size)
    else:
        return {"error": "Invalid view_mode"}
    return df.to_dict(orient="records")

# --- Tabs Endpoints ---
@tabs_router.get("/overview")
def overview_tab(view_mode: str = "Real-time", swarm_size: int = 8, quantum_enabled: bool = True, qkd_enabled: bool = True):
    swarm_data = generate_realtime_data(swarm_size) if view_mode == "Real-time" else (
        generate_historical_data(swarm_size) if view_mode == "Historical" else generate_simulation_data(swarm_size)
    )
    quantum_data = generate_quantum_data(view_mode, swarm_size)
    metrics = generate_performance_metrics(view_mode, swarm_size)
    # Assume overview.render returns a dict for API
    return overview.render(view_mode, swarm_data, swarm_size, quantum_data, metrics, quantum_enabled, qkd_enabled)

@tabs_router.get("/threat-analysis")
def threat_tab(view_mode: str = "Real-time", swarm_size: int = 8):
    swarm_data = generate_realtime_data(swarm_size) if view_mode == "Real-time" else (
        generate_historical_data(swarm_size) if view_mode == "Historical" else generate_simulation_data(swarm_size)
    )
    return threat_analysis.api_render(view_mode, swarm_data, swarm_size)

@tabs_router.get("/quantum-security")
def quantum_tab(view_mode: str = "Real-time", swarm_size: int = 8):
    quantum_data = generate_quantum_data(view_mode, swarm_size)
    
    # Prepare fidelity and error rate data for chart
    fidelity = quantum_data["Qubit Fidelity"].tolist() if "Qubit Fidelity" in quantum_data else []
    error_rate = quantum_data["Quantum Bit Error Rate"].tolist() if "Quantum Bit Error Rate" in quantum_data else []
    rounds = quantum_data["Round"].tolist() if "Round" in quantum_data else list(range(len(fidelity)))
    threshold = 0.9

    # Existing tab data
    tab_data = quantum_security.api_render(view_mode, quantum_data)
    
    # Add chart data to response
    tab_data["link_performance"] = {
        "qubit_fidelity": {
            "rounds": rounds,
            "values": fidelity,
            "threshold": threshold
        },
        "error_rate": {
            "rounds": rounds,
            "values": error_rate
        }
    }
    return tab_data

@tabs_router.get("/qkd-encryption")
def qkd_tab(view_mode: str = "Real-time", qkd_enabled: bool = True):
    return qkd_encryption.api_render(view_mode, qkd_enabled)

@tabs_router.get("/performance")
def performance_tab(view_mode: str = "Real-time", swarm_size: int = 8):
    metrics = generate_performance_metrics(view_mode, swarm_size)
    return performance.api_render(view_mode, metrics)

@tabs_router.get("/battlefield")
def battlefield_tab(view_mode: str = "Real-time", swarm_size: int = 8):
    swarm_data = generate_realtime_data(swarm_size) if view_mode == "Real-time" else (
        generate_historical_data(swarm_size) if view_mode == "Historical" else generate_simulation_data(swarm_size)
    )
    return battlefield.api_render(swarm_data)

@tabs_router.get("/threat-analysis/details")
def threat_analysis_details(
    agent: str,
    view_mode: str = "Real-time",
    swarm_size: int = 8
):
    swarm_data = generate_realtime_data(swarm_size) if view_mode == "Real-time" else (
        generate_historical_data(swarm_size) if view_mode == "Historical" else generate_simulation_data(swarm_size)
    )
    agent_data = swarm_data[swarm_data["Agent"] == agent]

    # Threat timeline
    if view_mode == "Real-time":
        timeline_data = agent_data.groupby([agent_data["Timestamp"].dt.floor("min"), "Threat Type"]).size().reset_index(name="Count")
    elif view_mode == "Historical":
        timeline_data = agent_data.groupby([agent_data["Timestamp"].dt.floor("H"), "Threat Type"]).size().reset_index(name="Count")
    else:
        timeline_data = agent_data.groupby(["Simulation ID", "Threat Type"]).size().reset_index(name="Count")

    # Threat action distribution
    threat_action_data = agent_data.groupby(["Threat Type", "Action"]).size().reset_index(name="Count")

    # Severity distribution
    severity_counts = agent_data["Severity"].value_counts().reset_index()
    severity_counts.columns = ["Severity", "Count"]

    # Top threat actors with counts
    threat_actors = agent_data["Threat Actor"].value_counts().head(5).reset_index()
    threat_actors.columns = ["Threat Actor", "Count"]

    # --- Unit Performance Metrics ---
    threats_detected = int(threat_action_data["Count"].sum())
    critical_threats = int(severity_counts[severity_counts["Severity"] == "Critical"]["Count"].sum()) if "Critical" in severity_counts["Severity"].values else 0
    neutralized = int(threat_action_data[threat_action_data["Action"] == "Neutralized"]["Count"].sum())
    monitoring = int(threat_action_data[threat_action_data["Action"] == "Monitoring"]["Count"].sum())
    neutralization_rate = round(neutralized / threats_detected * 100, 1) if threats_detected else 0
    false_positive_rate = round(monitoring / threats_detected * 100, 1) if threats_detected else 0

    return {
        "timeline": timeline_data.to_dict(orient="records"),
        "action_distribution": threat_action_data.to_dict(orient="records"),
        "severity_distribution": severity_counts.to_dict(orient="records"),
        "top_threat_actors": threat_actors.to_dict(orient="records"),
        "unit_performance_metrics": {
            "threats_detected": threats_detected,
            "critical_threats": critical_threats,
            "neutralization_rate": f"{neutralization_rate}%",
            "false_positives": f"{false_positive_rate}%"
        }
    }

# --- Chat Assistant ---
@chat_router.post("/")
def chat(req: ChatRequest, view_mode: str = "Real-time", swarm_size: int = 8):
    # You may want to pass more context if needed
    return chat_assistant.render(view_mode, swarm_size, req.message, req.api_key)

# --- Emergency Protocol ---
@system_router.post("/emergency-shutdown")
def emergency_shutdown(confirm: bool = Body(...)):
    if confirm:
        # Add your shutdown logic here
        return {"status": "Shutdown initiated"}
    return {"status": "Shutdown not confirmed"}

# --- Register Routers ---
app.include_router(auth_router)
app.include_router(data_router)
app.include_router(tabs_router)
app.include_router(chat_router)
app.include_router(system_router)