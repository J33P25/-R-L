"""
GreenMind: 24-Hour Comparative Evaluation Benchmark
Evaluates:
1. Rule-Based Control (RBC Baseline)
2. Policy Iteration (Analytical MDP Benchmark)
3. PPO (Proximal Policy Optimization)
Outputs full metrics to logs/evaluation_results.json
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import json
import time
from src.config import LOGS_DIR
from src.data_loader import build_unified_dataset
from src.mdp_formulation import build_mdp, solve_policy_iteration, classify_state
from src.building_env import GreenMindBuildingEnv
from src.rule_based_agent import RuleBasedController
from src.rl_agents import PPOAgent

def run_evaluation():
    print("[GreenMind Evaluation] Loading unified dataset and trained models...")
    df = build_unified_dataset()
    env = GreenMindBuildingEnv(data_df=df)

    # 1. Analytical Policy Iteration
    P, R = build_mdp()
    pi_policy, V, pi_dict = solve_policy_iteration(P, R)

    # 2. Rule-Based Controller
    rbc = RuleBasedController()

    # 3. Load Trained PPO Agent
    ppo = PPOAgent()
    ppo.load("ppo_greenmind.pt")

    agents = {
        "Rule-Based": rbc,
        "Policy Iteration": pi_policy,
        "PPO": ppo
    }

    results = {}
    eval_steps = 96 # 24 hours of 15-minute intervals

    for name, agent in agents.items():
        obs, _ = env.reset(initial_step=0)
        history = {
            "time_step": [],
            "indoor_temp": [],
            "co2_ppm": [],
            "occupants": [],
            "outdoor_temp": [],
            "power_kw": [],
            "valve": [],
            "fan": [],
            "damper": [],
            "discrete_state": []
        }

        action_diffs = []
        prev_act = np.zeros(3)

        for s in range(eval_steps):
            if name == "Rule-Based":
                info_dict = {"indoor_temp": env.t_in, "co2_ppm": env.co2, "occupants": env.occupants}
                action = agent.act(obs, info_dict)
            elif name == "PPO":
                action, _, _ = agent.act(obs, deterministic=True)
            elif name == "Policy Iteration":
                s_idx = classify_state(env.t_in, env.co2, 1 if env.occupants > 0 else 0)
                act_idx = int(agent[s_idx]) if s_idx < 6 else 0
                mode_map = {0: [0.30, 0.40, 0.30], 1: [0.15, 0.25, 0.20], 2: [0.35, 0.70, 0.85]}
                action = np.array(mode_map.get(act_idx, [0.3, 0.3, 0.3]), dtype=np.float32)

            next_obs, rew, term, trunc, info = env.step(action)
            action_diffs.append(float(np.sum((action - prev_act) ** 2)))
            prev_act = action.copy()

            history["time_step"].append(int(s))
            history["indoor_temp"].append(round(float(info["indoor_temp"]), 2))
            history["co2_ppm"].append(round(float(info["co2_ppm"]), 1))
            history["occupants"].append(int(info["occupants"]))
            history["outdoor_temp"].append(round(float(info["outdoor_temp"]), 2))
            history["power_kw"].append(round(float(info["power_kw"]), 3))
            history["valve"].append(round(float(info["valve"]), 3))
            history["fan"].append(round(float(info["fan"]), 3))
            history["damper"].append(round(float(info["damper"]), 3))
            history["discrete_state"].append(int(info["discrete_state"]))

            obs = next_obs
            if term:
                break

        discomfort_hours = (env.comfort_violation_steps * 15.0) / 60.0
        iaq_hours = (env.iaq_violation_steps * 15.0) / 60.0

        results[name] = {
            "agent": name,
            "total_energy_kwh": round(float(env.total_energy_kwh), 2),
            "discomfort_hours": round(float(discomfort_hours), 2),
            "iaq_violation_hours": round(float(iaq_hours), 2),
            "actuator_jitter": round(float(np.mean(action_diffs)), 4),
            "history": history
        }

    # Relative Energy Savings
    rbc_energy = results["Rule-Based"]["total_energy_kwh"]
    for k in results:
        saved = max(0.0, rbc_energy - results[k]["total_energy_kwh"])
        pct = (saved / rbc_energy * 100.0) if rbc_energy > 0 else 0.0
        results[k]["energy_saved_pct"] = round(float(pct), 1)

    benchmark_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy_iteration_theory": pi_dict,
        "results": results
    }

    def np_serializer(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        return str(obj)

    out_file = LOGS_DIR / "evaluation_results.json"
    with open(out_file, "w") as f:
        json.dump(benchmark_data, f, indent=2, default=np_serializer)

    print("\n================ EVALUATION BENCHMARK SUMMARY ================")
    print(f"{'Agent':<18} | {'Energy (kWh)':<12} | {'Energy Saved':<12} | {'Discomfort (hrs)':<16} | {'IAQ Viol (hrs)':<14} | {'Jitter':<8}")
    print("-" * 92)
    for k in ["Rule-Based", "Policy Iteration", "PPO"]:
        r = results[k]
        print(f"{r['agent']:<18} | {r['total_energy_kwh']:<12.2f} | {r['energy_saved_pct']:>9.1f}% | {r['discomfort_hours']:<16.2f} | {r['iaq_violation_hours']:<14.2f} | {r['actuator_jitter']:<8.4f}")
    print("==============================================================")
    print(f"Results successfully saved to {out_file}")
    return benchmark_data

if __name__ == "__main__":
    run_evaluation()
