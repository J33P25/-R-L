"""
GreenMind: Analytical Markov Decision Process Formulation & Policy Iteration Solver
Faithfully implements the exact states, actions, transition probabilities, and rewards
specified in GreenMind RL Formulation PDF (pages 2-8).
"""
import numpy as np

# State IDs
STATE_NAMES = {
    0: "S1: Comfort Zone (21-24°C, CO2<=1000, Occupied)",
    1: "S2: Overheated (>24°C, CO2<=1000, Occupied)",
    2: "S3: Overcooled (<21°C, CO2<=1000, Occupied)",
    3: "S4: Stuffy (21-24°C, CO2>1000, Occupied)",
    4: "S5: Critical (>24°C, CO2>1000, Occupied)",
    5: "S6: Unoccupied (Empty Zone)",
    6: "S7: Safety Violation (Terminal, Failure)",
    7: "S8: Green Day Achieved (Terminal, Success)"
}

NUM_STATES = 8
DISCOUNT_FACTOR = 0.95  # γ = 0.95 from PDF Section 5

# Actions available per state (3 actions per non-terminal state)
ACTIONS_PER_STATE = {
    0: ["A1: Maintain Mode", "A2: Eco Mode", "A3: Boost Ventilation"],
    1: ["B1: Full Cooling", "B2: Moderate Cooling & Vent", "B3: Vent Boost Only"],
    2: ["C1: Reduce Cooling", "C2: Activate Heating", "C3: Wait"],
    3: ["D1: Boost Fresh Air", "D2: Full Ventilation", "D3: Do Nothing"],
    4: ["E1: Full Cooling + Vent", "E2: Full Cooling Only", "E3: Full Vent Only"],
    5: ["F1: Standby Mode", "F2: Full Shutdown", "F3: Pre-Condition"],
    6: ["None (Terminal)"],
    7: ["None (Terminal)"]
}

def classify_state(temperature: float, co2: float, occupancy: int) -> int:
    """
    Maps continuous sensor readings to discrete GreenMind states S1-S8.
    """
    # Safety Boundary S7 Check first
    if temperature < 15.0 or temperature > 32.0:
        return 6 # S7

    # If unoccupied, all conditions collapse to S6
    if occupancy == 0:
        return 5 # S6

    # Occupied conditions
    is_temp_comfort = 21.0 <= temperature <= 24.0
    is_temp_hot = temperature > 24.0
    is_temp_cold = temperature < 21.0
    is_co2_safe = co2 <= 1000.0

    if is_temp_comfort and is_co2_safe:
        return 0 # S1
    elif is_temp_hot and is_co2_safe:
        return 1 # S2
    elif is_temp_cold and is_co2_safe:
        return 2 # S3
    elif is_temp_comfort and not is_co2_safe:
        return 3 # S4
    elif is_temp_hot and not is_co2_safe:
        return 4 # S5
    else: # Cold and High CO2 (rare edge case, classify as S3 or S4 based on priority)
        return 3 # S4

def build_mdp():
    """
    Constructs the transition probability tensor P[s, a, s'] and reward matrix R[s, a, s']
    strictly using pages 5-8 of the GreenMind RL Formulation PDF.
    """
    # Max 3 actions per state
    P = np.zeros((NUM_STATES, 3, NUM_STATES), dtype=np.float64)
    R = np.zeros((NUM_STATES, 3, NUM_STATES), dtype=np.float64)

    # --- S1: Comfort Zone ---
    # A1 Hold Steady (Action 0)
    transitions_s1_a1 = [(7, 0.15, +15), (0, 0.55, -1), (1, 0.15, -3), (3, 0.10, -3), (5, 0.05, -1)]
    for next_s, prob, r in transitions_s1_a1:
        P[0, 0, next_s] = prob
        R[0, 0, next_s] = r

    # A2 Eco Setting (Action 1)
    transitions_s1_a2 = [(7, 0.10, +15), (0, 0.35, 0), (1, 0.30, -2), (3, 0.15, -2), (5, 0.10, 0)]
    for next_s, prob, r in transitions_s1_a2:
        P[0, 1, next_s] = prob
        R[0, 1, next_s] = r

    # A3 Ventilation Boost (Action 2)
    transitions_s1_a3 = [(7, 0.10, +15), (0, 0.65, -2), (2, 0.15, -4), (5, 0.10, -2)]
    for next_s, prob, r in transitions_s1_a3:
        P[0, 2, next_s] = prob
        R[0, 2, next_s] = r

    # --- S2: Overheated ---
    # B1 Set Cooling (Action 0)
    transitions_s2_b1 = [(0, 0.75, -3), (1, 0.15, -5), (2, 0.05, -5), (4, 0.05, -7)]
    for next_s, prob, r in transitions_s2_b1:
        P[1, 0, next_s] = prob
        R[1, 0, next_s] = r

    # B2 Mode Cooling and Ventilation (Action 1)
    transitions_s2_b2 = [(0, 0.60, -2), (1, 0.30, -4), (4, 0.05, -4), (5, 0.05, -2)]
    for next_s, prob, r in transitions_s2_b2:
        P[1, 1, next_s] = prob
        R[1, 1, next_s] = r

    # B3 Vent Only (Action 2)
    transitions_s2_b3 = [(0, 0.30, -2), (1, 0.60, -4), (6, 0.10, -15)]
    for next_s, prob, r in transitions_s2_b3:
        P[1, 2, next_s] = prob
        R[1, 2, next_s] = r

    # --- S3: Overcooled ---
    # C1 Less Cooling (Action 0)
    transitions_s3_c1 = [(0, 0.70, -1), (2, 0.20, -3), (1, 0.05, -3), (5, 0.05, -1)]
    for next_s, prob, r in transitions_s3_c1:
        P[2, 0, next_s] = prob
        R[2, 0, next_s] = r

    # C2 Set Heating (Action 1)
    transitions_s3_c2 = [(0, 0.80, -2), (2, 0.15, -4), (1, 0.05, -4)]
    for next_s, prob, r in transitions_s3_c2:
        P[2, 1, next_s] = prob
        R[2, 1, next_s] = r

    # C3 Wait (Action 2)
    transitions_s3_c3 = [(0, 0.40, 0), (2, 0.45, -2), (6, 0.10, -15), (5, 0.05, 0)]
    for next_s, prob, r in transitions_s3_c3:
        P[2, 2, next_s] = prob
        R[2, 2, next_s] = r

    # --- S4: Stuffy (Comfort Zone, High CO2) ---
    # D1 Boost Air (Action 0)
    transitions_s4_d1 = [(0, 0.70, -2), (3, 0.20, -4), (2, 0.05, -4), (5, 0.05, -2)]
    for next_s, prob, r in transitions_s4_d1:
        P[3, 0, next_s] = prob
        R[3, 0, next_s] = r

    # D2 Set Ventilation (Action 1)
    transitions_s4_d2 = [(0, 0.80, -3), (3, 0.10, -5), (4, 0.05, -5), (5, 0.05, -3)]
    for next_s, prob, r in transitions_s4_d2:
        P[3, 1, next_s] = prob
        R[3, 1, next_s] = r

    # D3 Do Nothing (Action 2)
    transitions_s4_d3 = [(0, 0.30, -1), (3, 0.50, -3), (4, 0.15, -5), (5, 0.05, -1)]
    for next_s, prob, r in transitions_s4_d3:
        P[3, 2, next_s] = prob
        R[3, 2, next_s] = r

    # --- S5: Critical (Overheated + High CO2) ---
    # E1 Set Cooling and Ventilation (Action 0)
    transitions_s5_e1 = [(0, 0.60, -4), (1, 0.20, -6), (3, 0.10, -6), (4, 0.05, -8), (5, 0.05, -4)]
    for next_s, prob, r in transitions_s5_e1:
        P[5 if False else 4, 0, next_s] = prob
        R[4, 0, next_s] = r

    # E2 Set Cooling (Action 1)
    transitions_s5_e2 = [(3, 0.55, -5), (4, 0.20, -7), (6, 0.15, -15), (5, 0.10, -3)]
    for next_s, prob, r in transitions_s5_e2:
        P[4, 1, next_s] = prob
        R[4, 1, next_s] = r

    # E3 Full Vent Only (Action 2)
    transitions_s5_e3 = [(1, 0.40, -4), (4, 0.30, -6), (6, 0.20, -15), (5, 0.10, -2)]
    for next_s, prob, r in transitions_s5_e3:
        P[4, 2, next_s] = prob
        R[4, 2, next_s] = r

    # --- S6: Empty Room (Unoccupied) ---
    # F1 On Hold / Standby (Action 0)
    transitions_s6_f1 = [(5, 0.80, -1), (0, 0.15, -1), (1, 0.05, -3)]
    for next_s, prob, r in transitions_s6_f1:
        P[5, 0, next_s] = prob
        R[5, 0, next_s] = r

    # F2 Switch Off (Action 1)
    transitions_s6_f2 = [(5, 0.70, 0), (1, 0.15, -2), (2, 0.10, -2), (6, 0.05, -15)]
    for next_s, prob, r in transitions_s6_f2:
        P[5, 1, next_s] = prob
        R[5, 1, next_s] = r

    # F3 Pre-Condition / Set Conditioned Air (Action 2)
    transitions_s6_f3 = [(5, 0.75, -2), (0, 0.15, -2), (2, 0.05, -4), (1, 0.05, -4)]
    for next_s, prob, r in transitions_s6_f3:
        P[5, 2, next_s] = prob
        R[5, 2, next_s] = r

    # --- S7 & S8: Terminal States ---
    for a in range(3):
        P[6, a, 6] = 1.0  # S7 absorbs
        R[6, a, 6] = 0.0
        P[7, a, 7] = 1.0  # S8 absorbs
        R[7, a, 7] = 0.0

    return P, R

def solve_policy_iteration(P, R, gamma=DISCOUNT_FACTOR, tol=1e-7, max_iter=1000):
    """
    Implements exact Policy Iteration:
    1. Policy Evaluation: Solves (I - γ * P_π) V_π = R_π analytically
    2. Policy Improvement: Greedy action selection w.r.t Q_π(s, a)
    Repeated until policy stability (convergence).
    """
    num_non_terminal = 6
    # Initialize policy arbitrarily (action 0 for each state)
    policy = np.zeros(num_non_terminal, dtype=int)
    V = np.zeros(NUM_STATES, dtype=np.float64)

    for iteration in range(max_iter):
        # 1. Policy Evaluation (solving linear system)
        P_pi = np.zeros((num_non_terminal, num_non_terminal), dtype=np.float64)
        R_pi = np.zeros(num_non_terminal, dtype=np.float64)

        for s in range(num_non_terminal):
            a = policy[s]
            for next_s in range(NUM_STATES):
                prob = P[s, a, next_s]
                rew = R[s, a, next_s]
                R_pi[s] += prob * rew
                if next_s < num_non_terminal:
                    P_pi[s, next_s] += prob * gamma
                else:
                    # Terminal state value is 0
                    pass

        # V_non_terminal = (I - P_pi)^(-1) * R_pi
        I = np.eye(num_non_terminal)
        try:
            V_non_terminal = np.linalg.solve(I - P_pi, R_pi)
        except np.linalg.LinAlgError:
            V_non_terminal = np.linalg.pinv(I - P_pi) @ R_pi

        V[:num_non_terminal] = V_non_terminal
        V[6] = 0.0 # Terminal S7
        V[7] = 0.0 # Terminal S8

        # 2. Policy Improvement
        policy_stable = True
        for s in range(num_non_terminal):
            old_action = policy[s]
            q_values = np.zeros(3)
            for a in range(3):
                for next_s in range(NUM_STATES):
                    q_values[a] += P[s, a, next_s] * (R[s, a, next_s] + gamma * V[next_s])

            best_action = int(np.argmax(q_values))
            if best_action != old_action:
                policy_stable = False
                policy[s] = best_action

        if policy_stable:
            break

    # Build human-readable output
    optimal_policy_dict = {}
    for s in range(num_non_terminal):
        chosen_action_idx = policy[s]
        optimal_policy_dict[STATE_NAMES[s]] = {
            "optimal_action_index": chosen_action_idx,
            "optimal_action_name": ACTIONS_PER_STATE[s][chosen_action_idx],
            "state_value_V": round(float(V[s]), 4)
        }

    return policy, V, optimal_policy_dict

if __name__ == "__main__":
    P, R = build_mdp()
    policy, V, policy_dict = solve_policy_iteration(P, R)
    print("=== GreenMind Analytical Policy Iteration Results ===")
    for state_desc, res in policy_dict.items():
        print(f"{state_desc[:25]} -> {res['optimal_action_name']} (V*={res['state_value_V']})")
