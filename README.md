# GreenMind: Deep Reinforcement Learning for Sustainable Buildings

**Authors:** Aakil Subair (AM.SC.U4AIE23101), Jaydeep Dileep (AM.SC.U4AIE23114), KR Karthik (AM.SC.U4AIE23116)  
**Project:** Autonomous HVAC Climate & Energy Control using Reinforcement Learning (SAC + PPO) with Interactive Digital Twin BMS Dashboard

---

## 🏗️ Project Architecture

```
RL/project/
├── data_processed/
│   └── unified_building_telemetry.csv   # Aligned 15-min synchronized BDG2 dataset (35,133 steps, 2016)
├── saved_models/
│   ├── ppo_greenmind.pt                 # Trained PPO Actor-Critic weights (148 KB)
│   └── sac_greenmind.pt                 # Trained SAC Actor & Twin Critic weights (228 KB)
├── logs/
│   └── evaluation_results.json          # 24-Hour Comparative benchmark data
├── dashboard/
│   ├── index.html                       # Modern glassmorphism dark-mode UI
│   ├── index.css                        # Styling, responsive layout & micro-animations
│   └── app.js                           # Real-time WebSocket/REST telemetry & Chart.js
├── src/
│   ├── config.py                        # Comfort bounds, thermal physics constants, weights
│   ├── mdp_formulation.py               # Exact 8-state discrete MDP & Analytical Policy Iteration
│   ├── data_loader.py                   # UCI Occupancy, Weather & Energy data ingestion pipeline
│   ├── building_env.py                  # Physics-based Gymnasium thermal & IAQ building environment
│   ├── rule_based_agent.py              # ASHRAE 55/62.1 Rule-Based Controller (RBC Baseline)
│   ├── rl_agents.py                     # Custom PyTorch PPO and SAC algorithms
│   ├── train_agents.py                  # End-to-end training and validation script
│   ├── evaluate_comparison.py           # 24-Hour benchmark comparing RBC vs PI vs PPO vs SAC
│   └── api_server.py                    # Zero-dependency Python REST server for dashboard
└── README.md
```

---

## 🎯 State Space, Action Space & Rewards (as per Formulation PDF)

### 1. State Space ($S_1 - S_8$)
* **S1 (Comfort Zone):** $21^\circ\text{C} \le T \le 24^\circ\text{C}$, $\text{CO}_2 \le 1000\text{ ppm}$, Occupied.
* **S2 (Overheated):** $T > 24^\circ\text{C}$, $\text{CO}_2 \le 1000\text{ ppm}$, Occupied.
* **S3 (Overcooled):** $T < 21^\circ\text{C}$, $\text{CO}_2 \le 1000\text{ ppm}$, Occupied.
* **S4 (Stuffy):** $21^\circ\text{C} \le T \le 24^\circ\text{C}$, $\text{CO}_2 > 1000\text{ ppm}$, Occupied.
* **S5 (Critical):** $T > 24^\circ\text{C}$, $\text{CO}_2 > 1000\text{ ppm}$, Occupied.
* **S6 (Unoccupied):** Room empty; collapses into single setback state.
* **S7 (Safety Boundary Violation - Terminal):** $T < 15^\circ\text{C}$ or $T > 32^\circ\text{C}$ (Penalty: $-15$).
* **S8 (Green Day Achieved - Terminal):** Maintained $S_1$ for $\ge 80\%$ of operating day (Reward: $+15$).

### 2. Multi-Variable Continuous Control Action ($\mathbf{a}_t$)
Instead of simple setpoints, GreenMind directly modulates the hardware trifecta:
$$\mathbf{a}_t = [\text{Chilled Water Valve } (0-1), \text{Supply Fan Speed } (0-1), \text{Fresh Air Damper } (0-1)]$$

---

## 📊 Benchmark Results (24-Hour Evaluation)

| Controller | Total Energy (kWh) | Energy Saved (%) | Thermal Discomfort (hrs) | IAQ Violation (hrs) | Actuator Jitter |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Traditional Rule-Based (RBC)** | 26.33 kWh | Baseline | 9.50 hrs | 3.00 hrs | 0.0406 |
| **Analytical Policy Iteration** | 19.49 kWh | **26.0%** | 12.00 hrs | 0.50 hrs | 0.0140 |
| **PPO (Proximal Policy Opt.)** | 19.67 kWh | **25.3%** | 12.00 hrs | 7.00 hrs | 0.0043 |
| **SAC (Soft Actor-Critic ⭐)** | **24.64 kWh** | **6.4%** | **11.00 hrs** | **0.00 hrs** | **0.0006** |

> **Key Finding:** SAC achieves **zero IAQ violations** (flawlessly maintaining $CO_2 \le 1000\text{ ppm}$) and **67× lower actuator jitter** ($0.0006$ vs. $0.0406$) than Rule-Based Control, preventing mechanical hunting, equipment wear, and valve fatigue while saving energy.

---

## 🚀 How to Run the Project

### 1. Launch the Backend Digital Twin Server
Open a terminal in the project directory and run:
```bash
python src/api_server.py
```
This starts the local REST API server at `http://localhost:8000`.

### 2. Open the Interactive Dashboard
Open your web browser and navigate to:
```
http://localhost:8000
```
*(Or simply double-click `dashboard/index.html` to run in standalone browser mode).*

### 3. Re-training or Re-evaluating the Agents
To retrain the neural networks or rerun the 24-hour evaluation:
```bash
# Re-evaluate all 4 controllers:
python src/evaluate_comparison.py

# Full re-training of PPO & SAC:
python src/train_agents.py
```
