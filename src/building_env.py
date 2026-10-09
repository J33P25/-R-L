"""
GreenMind: Physics-Informed Building Simulation Environment (Gymnasium-Compatible)
Implements:
1. Zone thermal dynamics (sensible heat gain, envelope conduction, HVAC heat extraction)
2. Indoor Air Quality (IAQ) CO2 mass balance with occupancy generation & fresh-air dilution
3. Fan affinity power laws and chiller electrical consumption
4. GreenMind State Mapping (S1-S8) and Multi-Objective Rewards from project PDFs
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
from src.config import (
    T_COMFORT_MIN, T_COMFORT_MAX, CO2_SAFE_LIMIT, T_SAFETY_MIN, T_SAFETY_MAX,
    HEAT_CAPACITANCE, THERMAL_RESISTANCE, ZONE_AIR_VOLUME, AIR_DENSITY, AIR_SPECIFIC_HEAT,
    HUMAN_SENSIBLE_HEAT, HUMAN_CO2_GEN_RATE, OUTDOOR_CO2_PPM,
    MAX_VENT_FLOW, MAX_FAN_FLOW, MAX_COOLING_CAPACITY, MAX_HEATING_CAPACITY,
    WEIGHT_ENERGY, WEIGHT_COMFORT, WEIGHT_IAQ, WEIGHT_SMOOTHNESS
)
from src.mdp_formulation import classify_state

class GreenMindBuildingEnv:
    """
    Building environment designed for RL agents (PPO, SAC, Rule-Based, Policy Iteration).
    Observation: [T_in, RH_in, CO2, Occupancy_count, T_out, RH_out, Hour_sin, Hour_cos]
    Action (Continuous 3D): [ChilledWaterValve (0..1), FanSpeed (0..1), OutdoorDamper (0..1)]
    """
    def __init__(self, data_df=None, time_step_sec=900): # 15-minute time steps
        self.dt = time_step_sec
        self.data_df = data_df
        self.current_step = 0
        self.max_steps = 96 if data_df is None else min(len(data_df) - 1, 96) # 24 hours (96 x 15min)

        # Internal State Variables
        self.t_in = 22.5
        self.rh_in = 50.0
        self.co2 = 650.0
        self.occupants = 0
        self.t_out = 25.0
        self.rh_out = 60.0
        self.solar_kw = 0.0
        self.prev_action = np.array([0.2, 0.2, 0.1], dtype=np.float32)

        # Metrics Tracking
        self.total_energy_kwh = 0.0
        self.comfort_violation_steps = 0
        self.iaq_violation_steps = 0
        self.green_day_counter = 0

    def reset(self, initial_step=None):
        if initial_step is not None:
            self.current_step = initial_step
        elif self.data_df is not None:
            max_start = max(0, len(self.data_df) - self.max_steps - 2)
            self.current_step = np.random.randint(0, max_start) if max_start > 0 else 0
        else:
            self.current_step = 0

        # Load starting environmental condition
        if self.data_df is not None and len(self.data_df) > self.current_step:
            row = self.data_df.iloc[self.current_step]
            self.t_in = float(np.clip(row.get("indoor_temp", 22.5), 21.0, 25.0))
            self.rh_in = float(row.get("indoor_humidity", 50.0))
            self.co2 = float(np.clip(row.get("co2_ppm", 650.0), 450.0, 950.0))
            self.occupants = int(row.get("occupants_count", 2))
            self.t_out = float(row.get("outdoor_temp", 28.0))
            self.rh_out = float(row.get("outdoor_humidity", 60.0))
            self.solar_kw = float(row.get("solar_renewable_kwh", 0.0))
        else:
            self.t_in = 22.5
            self.rh_in = 50.0
            self.co2 = 650.0
            self.occupants = 4
            self.t_out = 30.0
            self.rh_out = 60.0
            self.solar_kw = 2.0

        self.prev_action = np.array([0.2, 0.2, 0.1], dtype=np.float32)
        self.total_energy_kwh = 0.0
        self.comfort_violation_steps = 0
        self.iaq_violation_steps = 0
        self.green_day_counter = 0

        return self._get_obs(), {}

    def _get_obs(self):
        hour = (self.current_step % 96) * 0.25
        hour_rad = 2 * np.pi * hour / 24.0
        return np.array([
            self.t_in,
            self.rh_in / 100.0,
            self.co2 / 2000.0,
            float(self.occupants) / 20.0,
            self.t_out,
            self.rh_out / 100.0,
            np.sin(hour_rad),
            np.cos(hour_rad)
        ], dtype=np.float32)

    def step(self, action):
        """
        Executes action [valve, fan, damper] in [0, 1]
        """
        action = np.clip(action, 0.0, 1.0)
        valve, fan, damper = float(action[0]), float(action[1]), float(action[2])

        # 1. Thermal Dynamics
        q_envelope = (self.t_out - self.t_in) / THERMAL_RESISTANCE
        q_occupants = self.occupants * HUMAN_SENSIBLE_HEAT
        q_solar = (1.0 - damper * 0.3) * (self.solar_kw * 150.0) # solar thermal load

        # HVAC thermal action: Cooling via chilled water heat exchanger
        effective_cooling = valve * (0.3 + 0.7 * fan) * MAX_COOLING_CAPACITY
        
        # Heating mode engagement if indoor temperature drops below 21°C
        effective_heating = 0.0
        if self.t_in < 21.0:
            effective_heating = (1.0 - valve) * 0.5 * MAX_HEATING_CAPACITY

        # Fresh air infiltration/ventilation thermal load
        v_fresh = damper * MAX_VENT_FLOW
        m_dot_fresh = v_fresh * AIR_DENSITY
        q_vent = m_dot_fresh * AIR_SPECIFIC_HEAT * (self.t_out - self.t_in)

        # Net thermal heat rate (Watts)
        q_net = q_envelope + q_occupants + q_solar + q_vent + effective_heating - effective_cooling
        delta_t = (q_net / HEAT_CAPACITANCE) * self.dt
        # Physical rate limit: room air + walls change by max ±1.2°C per 15 minutes
        delta_t = np.clip(delta_t, -1.2, 1.2)
        self.t_in += float(delta_t)

        # 2. Indoor Air Quality (CO2 Dynamics)
        # People generate CO2 (liters to ppm)
        co2_gen_ppm_per_sec = (self.occupants * HUMAN_CO2_GEN_RATE * 1000.0) / ZONE_AIR_VOLUME
        # Outdoor fresh air dilution
        air_changes_per_sec = v_fresh / ZONE_AIR_VOLUME
        delta_co2 = (co2_gen_ppm_per_sec - air_changes_per_sec * (self.co2 - OUTDOOR_CO2_PPM)) * self.dt
        self.co2 = float(np.clip(self.co2 + delta_co2, 380.0, 3000.0))

        # 3. Energy Consumption Model (kW)
        # Chiller compressor electrical power (COP ~ 3.5)
        p_chiller_kw = (effective_cooling / 3500.0)
        # Supply fan power follows cubic affinity law: P ~ fan^3
        p_fan_kw = 1.5 * (fan ** 3)
        # Damper motor power
        p_damper_kw = 0.05 * damper
        total_power_kw = p_chiller_kw + p_fan_kw + p_damper_kw
        step_energy_kwh = total_power_kw * (self.dt / 3600.0)
        self.total_energy_kwh += step_energy_kwh

        # 4. Map into GreenMind Discrete State
        discrete_state = classify_state(self.t_in, self.co2, 1 if self.occupants > 0 else 0)

        # 5. Reward Calculation from Formulation PDF
        terminated = False
        truncated = False
        reward = 0.0

        # Terminal Safety S7
        if discrete_state == 6: # S7 Safety boundary violated
            reward = -15.0
            terminated = True
        else:
            # Base step energy penalty
            r_energy = -WEIGHT_ENERGY * step_energy_kwh

            # Comfort penalty (outside 21-24 °C)
            r_comfort = 0.0
            if self.occupants > 0:
                if self.t_in < T_COMFORT_MIN:
                    r_comfort = -WEIGHT_COMFORT * (T_COMFORT_MIN - self.t_in)
                    self.comfort_violation_steps += 1
                elif self.t_in > T_COMFORT_MAX:
                    r_comfort = -WEIGHT_COMFORT * (self.t_in - T_COMFORT_MAX)
                    self.comfort_violation_steps += 1

            # IAQ penalty (CO2 > 1000 ppm)
            r_iaq = 0.0
            if self.occupants > 0 and self.co2 > CO2_SAFE_LIMIT:
                r_iaq = -WEIGHT_IAQ * ((self.co2 - CO2_SAFE_LIMIT) / 250.0)
                self.iaq_violation_steps += 1

            # Actuator Stability / Hunting Penalty (Section 11)
            action_diff = np.sum((action - self.prev_action) ** 2)
            r_smoothness = -WEIGHT_SMOOTHNESS * action_diff

            reward = r_energy + r_comfort + r_iaq + r_smoothness

            # Green Day progress (State S1)
            if discrete_state == 0:
                self.green_day_counter += 1
            else:
                self.green_day_counter = max(0, self.green_day_counter - 1)

        self.prev_action = action.copy()
        self.current_step += 1

        # Check episode completion (24 hours)
        if self.current_step % self.max_steps == 0:
            truncated = True
            # Green Day S8 Bonus if comfort was maintained for > 85% of operating day
            if self.green_day_counter >= int(0.80 * self.max_steps):
                reward += 15.0 # S8 Green Day reward!

        # Advance weather and occupancy from dataset
        if self.data_df is not None and self.current_step < len(self.data_df):
            row = self.data_df.iloc[self.current_step]
            self.t_out = float(row.get("outdoor_temp", self.t_out))
            self.rh_out = float(row.get("outdoor_humidity", self.rh_out))
            self.occupants = int(row.get("occupants_count", self.occupants))
            self.solar_kw = float(row.get("solar_renewable_kwh", 0.0))

        info = {
            "discrete_state": discrete_state,
            "indoor_temp": self.t_in,
            "co2_ppm": self.co2,
            "occupants": self.occupants,
            "outdoor_temp": self.t_out,
            "power_kw": total_power_kw,
            "step_energy_kwh": step_energy_kwh,
            "total_energy_kwh": self.total_energy_kwh,
            "valve": valve,
            "fan": fan,
            "damper": damper
        }

        return self._get_obs(), float(reward), terminated, truncated, info
