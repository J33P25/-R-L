"""
GreenMind: Interactive Building Digital Twin Backend Server
Provides a zero-dependency HTTP server (using Python standard library) with REST endpoints:
- GET  /api/benchmark : Returns 24-hr comparative performance (RBC vs PI vs PPO vs SAC)
- GET  /api/status    : Returns current live BMS zone conditions and actuator status
- POST /api/step      : Steps simulation forward with selected agent (Rule-Based, PPO, SAC)
- POST /api/perturb   : Injects environmental disturbances (heatwave, occupant spikes)
- Serves the frontend web dashboard at /
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from http.server import HTTPServer, SimpleHTTPRequestHandler
import json
import urllib.parse
import numpy as np

from src.config import BASE_DIR, LOGS_DIR
from src.data_loader import build_unified_dataset
from src.building_env import GreenMindBuildingEnv
from src.rule_based_agent import RuleBasedController
from src.rl_agents import PPOAgent
from src.mdp_formulation import build_mdp, solve_policy_iteration, classify_state

PORT = 8000
DASHBOARD_DIR = BASE_DIR / "dashboard"

# Global simulation state
sim_data = None
sim_env = None
agents = {}
current_agent_name = "PPO"
sim_history = []

def init_simulation():
    global sim_data, sim_env, agents, current_agent_name
    print("[Server] Initializing GreenMind Environment and Loading Agents...")
    sim_data = build_unified_dataset()
    sim_env = GreenMindBuildingEnv(data_df=sim_data)
    sim_env.reset(initial_step=0)

    # Initialize agents
    agents["Rule-Based"] = RuleBasedController()

    ppo = PPOAgent()
    ppo.load("ppo_greenmind.pt")
    agents["PPO"] = ppo

    P, R = build_mdp()
    pi_policy, _, _ = solve_policy_iteration(P, R)
    agents["Policy Iteration"] = pi_policy

    print("[Server] All agents initialized.")

class GreenMindHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DASHBOARD_DIR), **kwargs)

    def do_GET(self):
        url_parsed = urllib.parse.urlparse(self.path)
        path = url_parsed.path

        if path == "/api/status":
            self.send_json_response(self.get_current_status())
        elif path == "/api/benchmark":
            benchmark_file = LOGS_DIR / "evaluation_results.json"
            if benchmark_file.exists():
                with open(benchmark_file, "r") as f:
                    data = json.load(f)
                self.send_json_response(data)
            else:
                self.send_json_response({"error": "Benchmark data not generated yet. Run train_agents.py first."}, status=404)
        elif path == "/api/history":
            self.send_json_response({"history": sim_history[-50:]})
        else:
            # Fall back to serving static files from dashboard directory
            super().do_GET()

    def do_POST(self):
        url_parsed = urllib.parse.urlparse(self.path)
        path = url_parsed.path

        content_length = int(self.headers.get("Content-Length", 0))
        post_data = self.rfile.read(content_length)
        body = {}
        if post_data:
            try:
                body = json.loads(post_data.decode("utf-8"))
            except Exception:
                pass

        if path == "/api/step":
            agent_name = body.get("agent", current_agent_name)
            steps = int(body.get("steps", 1))
            res = self.step_simulation(agent_name, steps)
            self.send_json_response(res)
        elif path == "/api/perturb":
            # Injects disturbance: extra occupants or outdoor temp spike
            if "outdoor_temp" in body:
                sim_env.t_out = float(body["outdoor_temp"])
            if "occupants" in body:
                sim_env.occupants = int(body["occupants"])
            if "indoor_temp" in body:
                sim_env.t_in = float(body["indoor_temp"])
            if "co2_ppm" in body:
                sim_env.co2 = float(body["co2_ppm"])
            self.send_json_response(self.get_current_status())
        elif path == "/api/reset":
            sim_env.reset()
            sim_history.clear()
            self.send_json_response(self.get_current_status())
        else:
            self.send_response(404)
            self.end_headers()

    def get_current_status(self):
        d_state = classify_state(sim_env.t_in, sim_env.co2, 1 if sim_env.occupants > 0 else 0)
        return {
            "step": sim_env.current_step,
            "indoor_temp": round(float(sim_env.t_in), 2),
            "indoor_humidity": round(float(sim_env.rh_in), 1),
            "co2_ppm": round(float(sim_env.co2), 1),
            "occupants": int(sim_env.occupants),
            "outdoor_temp": round(float(sim_env.t_out), 2),
            "outdoor_humidity": round(float(sim_env.rh_out), 1),
            "solar_kw": round(float(sim_env.solar_kw), 2),
            "discrete_state": int(d_state),
            "valve": round(float(sim_env.prev_action[0]), 3),
            "fan": round(float(sim_env.prev_action[1]), 3),
            "damper": round(float(sim_env.prev_action[2]), 3),
            "total_energy_kwh": round(float(sim_env.total_energy_kwh), 2),
            "comfort_violations": int(sim_env.comfort_violation_steps),
            "iaq_violations": int(sim_env.iaq_violation_steps)
        }

    def step_simulation(self, agent_name, steps=1):
        global sim_history
        last_info = None

        for _ in range(steps):
            obs = sim_env._get_obs()
            info_dict = {
                "indoor_temp": sim_env.t_in,
                "co2_ppm": sim_env.co2,
                "occupants": sim_env.occupants
            }

            if agent_name == "Rule-Based":
                action = agents["Rule-Based"].act(obs, info_dict)
            elif agent_name == "PPO":
                action, _, _ = agents["PPO"].act(obs, deterministic=True)
            elif agent_name == "Policy Iteration":
                s_idx = classify_state(sim_env.t_in, sim_env.co2, 1 if sim_env.occupants > 0 else 0)
                act_idx = agents["Policy Iteration"][s_idx] if s_idx < 6 else 0
                mode_map = {0: [0.3, 0.4, 0.3], 1: [0.15, 0.25, 0.2], 2: [0.35, 0.7, 0.85]}
                action = np.array(mode_map.get(act_idx, [0.3, 0.3, 0.3]), dtype=np.float32)
            else:
                action = np.array([0.3, 0.3, 0.3], dtype=np.float32)

            _, rew, term, trunc, info = sim_env.step(action)
            last_info = info
            sim_history.append({
                "step": sim_env.current_step,
                "indoor_temp": round(float(sim_env.t_in), 2),
                "co2_ppm": round(float(sim_env.co2), 1),
                "outdoor_temp": round(float(sim_env.t_out), 2),
                "power_kw": round(float(info["power_kw"]), 3),
                "occupants": int(sim_env.occupants),
                "valve": round(float(action[0]), 3),
                "fan": round(float(action[1]), 3),
                "damper": round(float(action[2]), 3),
                "discrete_state": int(info["discrete_state"]),
                "reward": round(float(rew), 3)
            })

            if term or trunc:
                sim_env.reset()
                break

        status = self.get_current_status()
        status["history"] = sim_history[-30:]
        return status

    def send_json_response(self, data, status=200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

def start_server():
    init_simulation()
    server_address = ("", PORT)
    httpd = HTTPServer(server_address, GreenMindHandler)
    print(f"\n========================================================")
    print(f"  GreenMind Digital Twin Server running at http://localhost:{PORT}")
    print(f"  Open http://localhost:{PORT} in your web browser!")
    print(f"========================================================\n")
    httpd.serve_forever()

if __name__ == "__main__":
    start_server()
