"""
GreenMind: Model-Based vs. Model-Free Comparative Analysis & Visualization
Generates publication-quality comparison figures:
1. 24-Hour Power Demand Curve (kW)
2. Total Energy Consumption & % Savings Bar Chart
3. Thermal Comfort Trajectory (21°C - 24°C Comfort Envelope)
4. Indoor Air Quality CO2 Trajectory (1000 ppm Safety Limit)
5. Actuator Stability / Smoothness (Mechanical Hunting & Jitter)
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from src.config import LOGS_DIR

def generate_comparison_plots():
    results_file = LOGS_DIR / "evaluation_results.json"
    if not results_file.exists():
        print(f"Error: {results_file} not found. Run train_agents.py first.")
        return

    with open(results_file, "r") as f:
        data = json.load(f)

    results = data["results"]
    rbc = results["Rule-Based"]
    pi = results["Policy Iteration"]  # Model-Based
    ppo = results["PPO"]              # Model-Free (On-Policy)
    sac = results["SAC"]              # Model-Free (Off-Policy Max Entropy)

    time_steps = np.array(rbc["history"]["time_step"]) * 0.25  # Convert 15-min steps to hours (0..24h)

    # Set aesthetic plotting style
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig = plt.figure(figsize=(16, 12), dpi=300)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)

    colors = {
        "Rule-Based": "#64748b",         # Slate Gray (Baseline)
        "Model-Based (PI)": "#8b5cf6",   # Purple (Analytical MDP)
        "Model-Free (PPO)": "#0284c7",   # Sky Blue
        "Model-Free (SAC)": "#10b981"    # Emerald Green (Top Performer)
    }

    # -------------------------------------------------------------
    # 1. 24-Hour Power Demand Curve (kW)
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_steps, rbc["history"]["power_kw"], label="Baseline: Rule-Based Control (RBC)", color=colors["Rule-Based"], linestyle="--", alpha=0.7, linewidth=1.8)
    ax1.plot(time_steps, pi["history"]["power_kw"], label="Model-Based: Analytical Policy Iteration", color=colors["Model-Based (PI)"], linestyle="-.", linewidth=2.0)
    ax1.plot(time_steps, ppo["history"]["power_kw"], label="Model-Free: PPO Continuous", color=colors["Model-Free (PPO)"], linewidth=2.2)
    ax1.plot(time_steps, sac["history"]["power_kw"], label="Model-Free: SAC Continuous (Max Entropy ⭐)", color=colors["Model-Free (SAC)"], linewidth=2.5)

    ax1.set_title("1. Dynamic 24-Hour Electrical Power Demand Profile (kW)", fontsize=13, fontweight='bold', pad=10)
    ax1.set_xlabel("Time of Day (Hours)", fontsize=11)
    ax1.set_ylabel("HVAC Power Draw (kW)", fontsize=11)
    ax1.set_xlim(0, 24)
    ax1.set_xticks(range(0, 25, 2))
    ax1.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # 2. Total Energy Consumed & % Savings
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    agents = ["Rule-Based\n(Baseline)", "Model-Based\n(Policy Iteration)", "Model-Free\n(PPO)", "Model-Free\n(SAC ⭐)"]
    energy_vals = [rbc["total_energy_kwh"], pi["total_energy_kwh"], ppo["total_energy_kwh"], sac["total_energy_kwh"]]
    bar_colors = [colors["Rule-Based"], colors["Model-Based (PI)"], colors["Model-Free (PPO)"], colors["Model-Free (SAC)"]]

    bars = ax2.bar(agents, energy_vals, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)
    ax2.set_title("2. Total Daily Energy Consumption (kWh)", fontsize=12, fontweight='bold', pad=10)
    ax2.set_ylabel("Total Energy (kWh)", fontsize=11)
    ax2.set_ylim(0, max(energy_vals) * 1.25)

    # Add text labels on bars showing % savings
    for bar, val, agent_key in zip(bars, energy_vals, ["Rule-Based", "Policy Iteration", "PPO", "SAC"]):
        saved_pct = results[agent_key].get("energy_saved_pct", 0.0)
        h = bar.get_height()
        label_text = f"{val:.2f} kWh\n(Baseline)" if agent_key == "Rule-Based" else f"{val:.2f} kWh\n(-{saved_pct:.1f}%)"
        ax2.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax2.grid(axis="y", linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # 3. Thermal Comfort Trajectory (°C)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.axhspan(21.0, 24.0, color="#d1fae5", alpha=0.5, label="ASHRAE Comfort Zone (21°C - 24°C)")
    ax3.axhline(21.0, color="#10b981", linestyle="--", linewidth=1.2)
    ax3.axhline(24.0, color="#10b981", linestyle="--", linewidth=1.2)

    ax3.plot(time_steps, rbc["history"]["indoor_temp"], color=colors["Rule-Based"], linestyle="--", label="Rule-Based (RBC)")
    ax3.plot(time_steps, pi["history"]["indoor_temp"], color=colors["Model-Based (PI)"], linestyle="-.", label="Model-Based (PI)")
    ax3.plot(time_steps, ppo["history"]["indoor_temp"], color=colors["Model-Free (PPO)"], label="Model-Free (PPO)")
    ax3.plot(time_steps, sac["history"]["indoor_temp"], color=colors["Model-Free (SAC)"], linewidth=2.2, label="Model-Free (SAC ⭐)")

    ax3.set_title("3. Indoor Temperature Trajectory (°C)", fontsize=12, fontweight='bold', pad=10)
    ax3.set_xlabel("Time of Day (Hours)", fontsize=11)
    ax3.set_ylabel("Temperature (°C)", fontsize=11)
    all_temps = (rbc["history"]["indoor_temp"] + pi["history"]["indoor_temp"] +
                 ppo["history"]["indoor_temp"] + sac["history"]["indoor_temp"])
    ax3.set_ylim(min(18.0, min(all_temps) - 1.0), max(28.0, max(all_temps) + 1.0))
    ax3.set_xticks(range(0, 25, 4))
    ax3.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax3.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # 4. Indoor Air Quality CO2 Trajectory (ppm)
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axhline(1000.0, color="#ef4444", linestyle="--", linewidth=1.5, label="ASHRAE 62.1 Limit (1000 ppm)")

    ax4.plot(time_steps, rbc["history"]["co2_ppm"], color=colors["Rule-Based"], linestyle="--", label="Rule-Based")
    ax4.plot(time_steps, pi["history"]["co2_ppm"], color=colors["Model-Based (PI)"], linestyle="-.", label="Model-Based (PI)")
    ax4.plot(time_steps, ppo["history"]["co2_ppm"], color=colors["Model-Free (PPO)"], label="Model-Free (PPO)")
    ax4.plot(time_steps, sac["history"]["co2_ppm"], color=colors["Model-Free (SAC)"], linewidth=2.2, label="Model-Free (SAC ⭐)")

    ax4.set_title("4. Indoor Air Quality (CO₂ ppm)", fontsize=12, fontweight='bold', pad=10)
    ax4.set_xlabel("Time of Day (Hours)", fontsize=11)
    ax4.set_ylabel("CO₂ Concentration (ppm)", fontsize=11)
    ax4.set_xlim(0, 24)
    ax4.set_xticks(range(0, 25, 4))
    ax4.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax4.grid(True, linestyle=":", alpha=0.6)

    # -------------------------------------------------------------
    # 5. Actuator Smoothness & Mechanical Jitter (Section 11 Metric)
    # -------------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 1])
    jitter_vals = [rbc["actuator_jitter"], pi["actuator_jitter"], ppo["actuator_jitter"], sac["actuator_jitter"]]
    jitter_bars = ax5.bar(agents, jitter_vals, color=bar_colors, width=0.55, edgecolor="black", linewidth=0.8)

    ax5.set_title("5. Actuator Jitter / Hunting (Lower = Less Wear)", fontsize=12, fontweight='bold', pad=10)
    ax5.set_ylabel("Action Variance ||Δa||²", fontsize=11)
    ax5.set_ylim(0, max(jitter_vals) * 1.3)

    for bar, val in zip(jitter_bars, jitter_vals):
        h = bar.get_height()
        ax5.annotate(f"{val:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 4), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9.5, fontweight='bold')
    ax5.grid(axis="y", linestyle=":", alpha=0.6)

    # Super Title
    plt.suptitle("GreenMind Benchmark: Model-Based (Analytical MDP) vs. Model-Free Deep RL (SAC + PPO)", fontsize=16, fontweight='bold', y=0.99)

    out_png = LOGS_DIR / "model_based_vs_model_free_comparison.png"
    plt.savefig(out_png, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n[Visualization] Successfully generated comparison graph saved to:\n  {out_png}\n")

if __name__ == "__main__":
    generate_comparison_plots()
