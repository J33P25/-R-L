"""
GreenMind: Physics-Informed Building Simulation Environment (Gymnasium-Compatible)
Implements:
1. Zone thermal dynamics (sensible heat gain, envelope conduction, solar gain, HVAC cooling/heating)
2. Indoor Air Quality (IAQ) CO2 mass balance with metabolic generation & fresh-air dilution
3. Fan affinity power laws and central chiller electrical consumption
4. Normalized Pareto Multi-Objective Reward Function (ASHRAE 55, ASHRAE 62.1, Energy, Actuator Hunting)
5. Grounded validation against BDG2 real-world commercial office meter telemetry
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from src.config import (
    TIME_STEP_SEC, STEPS_PER_DAY,
    T_COMFORT_MIN, T_COMFORT_MAX, CO2_SAFE_LIMIT, T_SAFETY_MIN, T_SAFETY_MAX,
    WEIGHT_ENERGY, WEIGHT_COMFORT, WEIGHT_IAQ, WEIGHT_SMOOTHNESS,
    REWARD_GREEN_DAY, PENALTY_SAFETY,
    ZONE_AIR_VOLUME, AIR_DENSITY, AIR_SPECIFIC_HEAT,
    HEAT_CAPACITANCE, THERMAL_RESISTANCE,
    HUMAN_SENSIBLE_HEAT, HUMAN_CO2_GEN_RATE, OUTDOOR_CO2_PPM,
    MAX_COOLING_CAPACITY, MAX_HEATING_CAPACITY, MAX_VENT_FLOW, MAX_FAN_FLOW,
    CHILLER_COP, MAX_FAN_POWER_KW, MAX_DAMPER_POWER_KW, MAX_TOTAL_POWER_KW,
    ZONE_MAX_CAPACITY
)
from src.mdp_formulation import classify_state

class GreenMindBuildingEnv:
    """
    Standardized Building HVAC Simulation Environment for Continuous Reinforcement Learning.
    
    Observation (8D continuous vector):
    [
        (T_in - 22.5) / 10.0,       # Normalized indoor temperature
        RH_in / 100.0,              # Normalized indoor relative humidity
        (CO2 - 400.0) / 1600.0,     # Normalized indoor CO2
        Occupancy / 15.0,           # Normalized occupant headcount
        (T_out - 22.5) / 15.0,      # Normalized outdoor temperature
        RH_out / 100.0,             # Normalized outdoor relative humidity
        sin(2*pi*hour / 24),        # Diurnal temporal feature (sin)
        cos(2*pi*hour / 24)         # Diurnal temporal feature (cos)
    ]

    Action (3D continuous vector in [0, 1]^3):
    [
        ChilledWaterValve (0..1),   # Modulates chilled water flow rate
        SupplyFanSpeed (0..1),      # Variable frequency drive (VFD) fan speed
        OutdoorAirDamper (0..1)     # Fresh air ventilation / economizer damper
    ]
    """
    def __init__(self, data_df: pd.DataFrame = None, time_step_sec: int = TIME_STEP_SEC):
        self.dt = time_step_sec
        self.data_df = data_df
        self.current_step = 0
        self.max_steps = STEPS_PER_DAY # 96 steps = 24 hours

        # Internal Environmental State Variables
        self.t_in = 22.5
        self.rh_in = 50.0
        self.co2 = 600.0
        self.occupants = 0
        self.t_out = 24.0
        self.rh_out = 60.0
        self.solar_irradiance = 0.0
        self.real_chiller_kwh = 0.0

        # Action tracking for mechanical hunting regularizer
        self.prev_action = np.array([0.2, 0.3, 0.2], dtype=np.float32)

        # Performance Metrics Tracking
        self.total_energy_kwh = 0.0
        self.comfort_violation_steps = 0
        self.iaq_violation_steps = 0
        self.occupied_steps = 0
        self.comfort_in_occupied_steps = 0
        self.action_jitter_sum = 0.0

    def reset(self, initial_step: int = None):
        """
        Resets environment to starting state.
        If initial_step is None and data_df is provided, picks a random 24h operating day.
        """
        if initial_step is not None:
            self.current_step = initial_step
        elif self.data_df is not None:
            max_start = max(0, len(self.data_df) - self.max_steps - 2)
            # Pick a day boundary (aligned with 96 steps)
            num_days = max_start // self.max_steps
            chosen_day = np.random.randint(0, max(1, num_days))
            self.current_step = chosen_day * self.max_steps
        else:
            self.current_step = 0

        # Load starting environmental condition
        if self.data_df is not None and self.current_step < len(self.data_df):
            row = self.data_df.iloc[self.current_step]
            self.t_out = float(row.get("outdoor_temp", 24.0))
            self.rh_out = float(row.get("outdoor_humidity", 60.0))
            self.solar_irradiance = float(row.get("solar_irradiance", 0.0))
            self.occupants = int(row.get("occupants_count", 0))
            self.real_chiller_kwh = float(row.get("real_chiller_kwh", 0.0))
            
            # Start indoor temperature close to comfort zone with mild offset
            self.t_in = float(np.clip(self.t_out * 0.3 + 22.5 * 0.7, 21.0, 25.0))
            self.rh_in = float(np.clip(self.rh_out * 0.4 + 50.0 * 0.6, 40.0, 65.0))
            self.co2 = 450.0 if self.occupants == 0 else 650.0
        else:
            self.t_in = 22.5
            self.rh_in = 50.0
            self.co2 = 550.0
            self.occupants = 0
            self.t_out = 25.0
            self.rh_out = 60.0
            self.solar_irradiance = 0.0
            self.real_chiller_kwh = 0.0

        self.prev_action = np.array([0.2, 0.3, 0.2], dtype=np.float32)
        self.total_energy_kwh = 0.0
        self.comfort_violation_steps = 0
        self.iaq_violation_steps = 0
        self.occupied_steps = 0
        self.comfort_in_occupied_steps = 0
        self.action_jitter_sum = 0.0

        return self._get_obs(), {}

    def _get_obs(self) -> np.ndarray:
        hour = (self.current_step % self.max_steps) * 0.25
        hour_rad = 2.0 * np.pi * hour / 24.0
        return np.array([
            (self.t_in - 22.5) / 10.0,
            self.rh_in / 100.0,
            (self.co2 - 400.0) / 1600.0,
            float(self.occupants) / float(ZONE_MAX_CAPACITY),
            (self.t_out - 22.5) / 15.0,
            self.rh_out / 100.0,
            np.sin(hour_rad),
            np.cos(hour_rad)
        ], dtype=np.float32)

    def step(self, action: np.ndarray):
        """
        Executes continuous control action [valve, fan, damper] in [0, 1]^3.
        Computes physical building thermal & IAQ updates and normalized reward.
        """
        action = np.clip(np.array(action, dtype=np.float32), 0.0, 1.0)
        valve, fan, damper = float(action[0]), float(action[1]), float(action[2])

        # ---------------------------------------------------------------------
        # 1. Thermal Physics Dynamics (ISO 13790 Sensible Heat Balance)
        # ---------------------------------------------------------------------
        # Envelope conductive heat transfer through walls & glazing (Watts)
        q_envelope = (self.t_out - self.t_in) / THERMAL_RESISTANCE

        # Internal occupant sensible metabolic heat gain (Watts)
        q_occupants = self.occupants * HUMAN_SENSIBLE_HEAT

        # Solar radiative heat gain through windows (Watts)
        # Windows ~15 m² with SHGC=0.45; damper shades slightly if closed
        effective_solar_area = 15.0 * 0.45
        q_solar = self.solar_irradiance * effective_solar_area

        # Fresh outdoor air ventilation thermal load (Coupled IAQ-Energy trade-off!)
        v_fresh = damper * MAX_VENT_FLOW # m³/s
        m_dot_fresh = v_fresh * AIR_DENSITY
        q_vent = m_dot_fresh * AIR_SPECIFIC_HEAT * (self.t_out - self.t_in)

        # Active HVAC heat extraction (Cooling via chilled water coil)
        # Cooling effectiveness depends non-linearly on water valve and supply air airflow
        effective_cooling = valve * (0.25 + 0.75 * fan) * MAX_COOLING_CAPACITY

        # Active heating / reheat coil engagement if zone drops below 20.5°C
        effective_heating = 0.0
        if self.t_in < 20.5:
            effective_heating = (1.0 - valve) * 0.6 * MAX_HEATING_CAPACITY

        # Net thermal heat exchange rate (Watts)
        q_net = q_envelope + q_occupants + q_solar + q_vent + effective_heating - effective_cooling

        # Zone temperature update via structural thermal capacitance
        delta_t = (q_net / HEAT_CAPACITANCE) * self.dt
        # Physical inertia rate limit: max +/- 1.2 °C per 15 minutes
        delta_t = np.clip(delta_t, -1.2, 1.2)
        self.t_in += float(delta_t)

        # ---------------------------------------------------------------------
        # 2. Indoor Air Quality (IAQ) CO2 Mass Balance Dynamics
        # ---------------------------------------------------------------------
        # Occupants generate CO2 (L/s -> m³/s)
        co2_gen_m3_per_sec = (self.occupants * HUMAN_CO2_GEN_RATE) / 1000.0
        # Fresh air dilution rate (air changes per second)
        air_exchange_rate = v_fresh / ZONE_AIR_VOLUME
        # Differential mass balance (ppm/s)
        co2_gen_ppm_rate = (co2_gen_m3_per_sec / ZONE_AIR_VOLUME) * 1e6
        delta_co2 = (co2_gen_ppm_rate - air_exchange_rate * (self.co2 - OUTDOOR_CO2_PPM)) * self.dt
        self.co2 = float(np.clip(self.co2 + delta_co2, 380.0, 3000.0))

        # Indoor relative humidity response
        rh_target = 0.7 * self.rh_in + 0.3 * (self.rh_out if damper > 0.4 else 50.0)
        self.rh_in = float(np.clip(self.rh_in + (rh_target - self.rh_in) * 0.05, 30.0, 85.0))

        # ---------------------------------------------------------------------
        # 3. Electrical Power & Energy Consumption Model (kW & kWh)
        # ---------------------------------------------------------------------
        # Chiller compressor electrical power (COP = 3.5)
        p_chiller_kw = (effective_cooling / 1000.0) / CHILLER_COP
        # Supply fan electrical power obeys cubic fan affinity law: P ~ fan^3
        p_fan_kw = MAX_FAN_POWER_KW * (fan ** 3)
        # Damper modulation motor power
        p_damper_kw = MAX_DAMPER_POWER_KW * damper

        total_power_kw = p_chiller_kw + p_fan_kw + p_damper_kw
        step_energy_kwh = total_power_kw * (self.dt / 3600.0)
        self.total_energy_kwh += step_energy_kwh

        # ---------------------------------------------------------------------
        # 4. Discrete State Mapping (GreenMind Formulation S1 - S8)
        # ---------------------------------------------------------------------
        discrete_state = classify_state(self.t_in, self.co2, 1 if self.occupants > 0 else 0)

        # ---------------------------------------------------------------------
        # 5. Mathematically Rigorous Multi-Objective Normalized Reward
        # ---------------------------------------------------------------------
        terminated = False
        truncated = False

        # Check Critical Safety Boundary Violation (S7)
        if discrete_state == 6: # T < 15°C or T > 32°C
            reward = PENALTY_SAFETY # -15.0
            terminated = True
        else:
            # Objective 1: Normalized Energy Loss L_E in [0, 1]
            loss_energy = np.clip(total_power_kw / MAX_TOTAL_POWER_KW, 0.0, 1.0)

            # Objective 2: Normalized Thermal Comfort Loss L_C in [0, 1] (ASHRAE Standard 55)
            loss_comfort = 0.0
            if self.occupants > 0:
                self.occupied_steps += 1
                if self.t_in < T_COMFORT_MIN:
                    dev = T_COMFORT_MIN - self.t_in
                    loss_comfort = np.clip((dev / 3.0) ** 2, 0.0, 1.0)
                    self.comfort_violation_steps += 1
                elif self.t_in > T_COMFORT_MAX:
                    dev = self.t_in - T_COMFORT_MAX
                    loss_comfort = np.clip((dev / 3.0) ** 2, 0.0, 1.0)
                    self.comfort_violation_steps += 1
                else:
                    self.comfort_in_occupied_steps += 1
            # Unoccupied room: loss_comfort = 0 (Deep setback permitted to save energy!)

            # Objective 3: Normalized IAQ Loss L_Q in [0, 1] (ASHRAE Standard 62.1)
            loss_iaq = 0.0
            if self.occupants > 0:
                if self.co2 > CO2_SAFE_LIMIT:
                    dev_co2 = self.co2 - CO2_SAFE_LIMIT
                    loss_iaq = np.clip(dev_co2 / 1000.0, 0.0, 1.0)
                    self.iaq_violation_steps += 1

            # Objective 4: Actuator Mechanical Wear / Hunting Jitter Loss L_S in [0, 1]
            action_diff = np.mean((action - self.prev_action) ** 2)
            loss_smoothness = np.clip(action_diff, 0.0, 1.0)
            self.action_jitter_sum += float(action_diff)

            # Combined Normalized Pareto Multi-Objective Step Reward
            step_penalty = (
                WEIGHT_ENERGY * loss_energy +
                WEIGHT_COMFORT * loss_comfort +
                WEIGHT_IAQ * loss_iaq +
                WEIGHT_SMOOTHNESS * loss_smoothness
            )
            reward = -float(step_penalty)

        self.prev_action = action.copy()
        self.current_step += 1

        # Check 24-hour Day Episode Horizon (96 steps)
        if self.current_step % self.max_steps == 0:
            truncated = True
            # Green Day Benchmark (S8): Maintained ideal comfort in >= 80% of occupied operating day
            if self.occupied_steps > 0 and (self.comfort_in_occupied_steps / self.occupied_steps) >= 0.80:
                reward += REWARD_GREEN_DAY # +15.0 Success bonus!

        # Advance Exogenous Weather and Telemetry from Dataset
        if self.data_df is not None and self.current_step < len(self.data_df):
            row = self.data_df.iloc[self.current_step]
            self.t_out = float(row.get("outdoor_temp", self.t_out))
            self.rh_out = float(row.get("outdoor_humidity", self.rh_out))
            self.solar_irradiance = float(row.get("solar_irradiance", 0.0))
            self.occupants = int(row.get("occupants_count", self.occupants))
            self.real_chiller_kwh = float(row.get("real_chiller_kwh", 0.0))

        info = {
            "discrete_state": discrete_state,
            "indoor_temp": self.t_in,
            "indoor_humidity": self.rh_in,
            "co2_ppm": self.co2,
            "occupants": self.occupants,
            "outdoor_temp": self.t_out,
            "outdoor_humidity": self.rh_out,
            "power_kw": total_power_kw,
            "step_energy_kwh": step_energy_kwh,
            "total_energy_kwh": self.total_energy_kwh,
            "real_chiller_kwh": self.real_chiller_kwh,
            "valve": valve,
            "fan": fan,
            "damper": damper
        }

        return self._get_obs(), float(reward), terminated, truncated, info
