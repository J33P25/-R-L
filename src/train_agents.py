"""
GreenMind: Model Training & Evaluation Pipeline
Trains:
1. Policy Iteration (Analytical MDP Benchmark)
2. PPO (Proximal Policy Optimization)
And evaluates against Rule-Based Control (RBC)
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import json
import time
from src.config import LOGS_DIR, MODELS_DIR
from src.data_loader import build_unified_dataset
from src.mdp_formulation import build_mdp, solve_policy_iteration, classify_state
from src.building_env import GreenMindBuildingEnv
from src.rule_based_agent import RuleBasedController
from src.rl_agents import PPOAgent

def train_ppo(env, episodes=120, steps_per_episode=96):
    print("\n[GreenMind Training] >>> Training PPO Agent (Continuous Control)...")
    agent = PPOAgent()
    episode_rewards = []

    for ep in range(episodes):
        obs, _ = env.reset()
        ep_states, ep_actions, ep_log_probs, ep_rewards, ep_values = [], [], [], [], []
        total_ep_reward = 0.0

        for step in range(steps_per_episode):
            action, log_prob, val = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)

            ep_states.append(obs)
            ep_actions.append(action)
            ep_log_probs.append(log_prob)
            ep_rewards.append(reward)
            ep_values.append(val)
            total_ep_reward += reward

            obs = next_obs
            if terminated or truncated:
                break

        # Compute discounted returns and Generalized Advantage Estimation (GAE)
        returns = []
        discounted_sum = 0
        for r in reversed(ep_rewards):
            discounted_sum = r + agent.gamma * discounted_sum
            returns.insert(0, discounted_sum)

        # Advantage calculation
        advantages = [ret - val for ret, val in zip(returns, ep_values)]
        agent.train_step(ep_states, ep_actions, ep_log_probs, returns, advantages)
        episode_rewards.append(total_ep_reward)

        if (ep + 1) % 15 == 0 or ep == episodes - 1:
            avg_rew = np.mean(episode_rewards[-15:])
            print(f"  --> PPO Episode [{ep+1:3d}/{episodes}] | Avg Return (last 15): {avg_rew:6.2f} | Comfort Violations: {env.comfort_violation_steps:2d} | Zone Temp: {env.t_in:4.1f}°C")

    agent.save("ppo_greenmind.pt")
    print("  [PPO] Saved trained model weights to saved_models/ppo_greenmind.pt")
    return agent, episode_rewards

def evaluate_agent(agent_name, agent_obj, env, eval_steps=96):
    """
    Evaluates an agent over a full continuous 24-hour cycle (96 15-minute steps).
    """
    obs, info = env.reset(initial_step=0)
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
        "discrete_state": [],
        "step_reward": []
    }

    cumulative_reward = 0.0
    action_changes = []
    prev_act = np.zeros(3)

    for s in range(eval_steps):
        if agent_name == "Rule-Based":
            action = agent_obj.act(obs, info)
        elif agent_name == "PPO":
            action, _, _ = agent_obj.act(obs, deterministic=True)
        elif agent_name == "Policy Iteration":
            # Map state to action and translate discrete mode to continuous setpoint
            s_idx = classify_state(env.t_in, env.co2, 1 if env.occupants > 0 else 0)
            act_idx = agent_obj[s_idx] if s_idx < 6 else 0
            # Heuristic action mapping for discrete modes
            mode_map = {
                0: [0.30, 0.40, 0.30], # Maintain / Low
                1: [0.15, 0.25, 0.20], # Eco
                2: [0.35, 0.70, 0.85]  # Ventilation / Full
            }
            action = np.array(mode_map.get(act_idx, [0.3, 0.3, 0.3]), dtype=np.float32)

        next_obs, rew, terminated, truncated, info = env.step(action)
        cumulative_reward += rew
        action_changes.append(np.sum((action - prev_act) ** 2))
        prev_act = action.copy()

        # Record telemetry
        history["time_step"].append(s)
        history["indoor_temp"].append(round(float(info["indoor_temp"]), 2))
        history["co2_ppm"].append(round(float(info["co2_ppm"]), 1))
        history["occupants"].append(int(info["occupants"]))
        history["outdoor_temp"].append(round(float(info["outdoor_temp"]), 2))
        history["power_kw"].append(round(float(info["power_kw"]), 3))
        history["valve"].append(round(float(info["valve"]), 3))
        history["fan"].append(round(float(info["fan"]), 3))
        history["damper"].append(round(float(info["damper"]), 3))
        history["discrete_state"].append(int(info["discrete_state"]))
        history["step_reward"].append(round(float(rew), 3))

        obs = next_obs
        if terminated:
            break

    # Calculate Summary Metrics
    discomfort_hours = (env.comfort_violation_steps * 15.0) / 60.0
    iaq_violation_hours = (env.iaq_violation_steps * 15.0) / 60.0
    actuator_smoothness = float(np.mean(action_changes))

    summary = {
        "agent": agent_name,
        "total_energy_kwh": round(float(env.total_energy_kwh), 2),
        "discomfort_hours": round(discomfort_hours, 2),
        "iaq_violation_hours": round(iaq_violation_hours, 2),
        "actuator_jitter": round(actuator_smoothness, 4),
        "cumulative_reward": round(cumulative_reward, 2),
        "green_days_achieved": 1 if env.green_day_counter >= int(0.75 * eval_steps) else 0,
        "history": history
    }
    return summary

def run_full_pipeline():
    print("=================================================================")
    print(" GREENMIND: UNIFIED REINFORCEMENT LEARNING TRAINING & BENCHMARK ")
    print("=================================================================")

    # 1. Prepare Data
    df = build_unified_dataset()
    env = GreenMindBuildingEnv(data_df=df)

    # 2. Analytical Policy Iteration (from PDF)
    print("\n[Step 1] Solving Exact Discrete Policy Iteration (pages 5-8 of PDF)...")
    P, R = build_mdp()
    pi_policy, V, pi_dict = solve_policy_iteration(P, R)
    print("Policy Iteration Optimal Policy:")
    for s_name, res in pi_dict.items():
        print(f"  {s_name[:24]} -> {res['optimal_action_name']} (V*={res['state_value_V']})")

    # 3. Train PPO
    print("\n[Step 2] Training PPO Agent...")
    ppo_agent, _ = train_ppo(env, episodes=100)

    # 4. Comparative Evaluation
    print("\n[Step 3] Running Comparative 24-Hour Evaluation Benchmark...")
    eval_env = GreenMindBuildingEnv(data_df=df)
    rbc = RuleBasedController()

    eval_agents = [
        ("Rule-Based", rbc),
        ("Policy Iteration", pi_policy),
        ("PPO", ppo_agent)
    ]

    results = {}
    for name, agent in eval_agents:
        results[name] = evaluate_agent(name, agent, eval_env)

    # Calculate Energy Savings relative to Rule-Based
    rbc_energy = results["Rule-Based"]["total_energy_kwh"]
    for name, res in results.items():
        saved_kwh = max(0.0, rbc_energy - res["total_energy_kwh"])
        pct_saved = (saved_kwh / rbc_energy * 100.0) if rbc_energy > 0 else 0.0
        res["energy_saved_pct"] = round(pct_saved, 1)

    # Export results JSON
    benchmark_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "policy_iteration_theory": pi_dict,
        "results": results
    }

    def np_json_serializer(obj):
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    out_json = LOGS_DIR / "evaluation_results.json"
    with open(out_json, "w") as f:
        json.dump(benchmark_data, f, indent=2, default=np_json_serializer)

    print("\n================ BENCHMARK SUMMARY ================")
    print(f"{'Agent':<18} | {'Energy (kWh)':<12} | {'Energy Saved':<12} | {'Discomfort (hrs)':<16} | {'IAQ Viol (hrs)':<14} | {'Jitter':<8}")
    print("-" * 92)
    for k in results:
        r = benchmark_data["results"][k]
        print(f"{r['agent']:<18} | {r['total_energy_kwh']:<12.2f} | {r['energy_saved_pct']:>9.1f}% | {r['discomfort_hours']:<16.2f} | {r['iaq_violation_hours']:<14.2f} | {r['actuator_jitter']:<8.4f}")
    print("====================================================")
    print(f"Results exported to {out_json}")
    return benchmark_data

if __name__ == "__main__":
    run_full_pipeline()
