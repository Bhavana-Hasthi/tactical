from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from data_generator import (
    generate_realtime_data,
    generate_historical_data,
    generate_simulation_data,
    generate_quantum_data,
    generate_performance_metrics,
    generate_agentic_ai_data
)

app = FastAPI(title="Quantum Swarm API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust this in production
    allow_methods=["*"],
    allow_headers=["*"]
)

# ------------------- DATA ENDPOINTS ------------------- #

@app.get("/data/realtime")
def get_realtime(agent_count: int = 5):
    return generate_realtime_data(agent_count).to_dict(orient="records")

@app.get("/data/historical")
def get_historical(agent_count: int = 5):
    return generate_historical_data(agent_count).to_dict(orient="records")

@app.get("/data/simulation")
def get_simulation(agent_count: int = 5):
    return generate_simulation_data(agent_count).to_dict(orient="records")

@app.get("/data/quantum")
def get_quantum(view_mode: str = "Real-time", agent_count: int = 5):
    return generate_quantum_data(view_mode, agent_count).to_dict(orient="records")

@app.get("/data/performance")
def get_performance(view_mode: str = "Real-time", agent_count: int = 5):
    return generate_performance_metrics(view_mode, agent_count).to_dict(orient="records")

@app.get("/data/agentic")
def get_agentic(agent_count: int = 5):
    return generate_agentic_ai_data(agent_count).to_dict(orient="records")


# ------------------- ROOT ENDPOINT ------------------- #

@app.get("/")
def homepage():
    return {"message": "Welcome to Quantum Swarm API"}
