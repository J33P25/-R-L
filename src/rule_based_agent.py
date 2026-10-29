"""
GreenMind: Industry Standard Rule-Based Controller (RBC / ASHRAE Baseline)
Implements traditional thermostat hysteresis and scheduled ventilation:
- Comfort deadband: 21°C - 24°C
- IAQ purge: Triggered when CO2 > 1000 ppm
- Occupancy setback: Standby when empty
"""
import numpy as np

class RuleBasedController:
    """
    Simulates commercial building management system (BMS) logic.
    Outputs action: [valve (0..1), fan (0..1), damper (0..1)]
    """
    def __init__(self, target_temp=22.5, t_high=24.0, t_low=21.0, co2_limit=1000.0):
        self.target_temp = target_temp
        self.t_high = t_high
        self.t_low = t_low
        self.co2_limit = co2_limit

    def act(self, obs, info=None):
        """
        Calculates reactive control command based on sensor readings.
        """
        # Extract from observation or info dict
        if info is not None and "indoor_temp" in info:
            t_in = info["indoor_temp"]
            co2 = info["co2_ppm"]
            occupants = info["occupants"]
        else:
            t_in = obs[0] * 10.0 + 22.5
            co2 = obs[2] * 1600.0 + 400.0
            occupants = int(round(obs[3] * 15.0))

        # 1. Unoccupied Rule: Deep Setback
        if occupants == 0:
            return np.array([0.0, 0.1, 0.05], dtype=np.float32)

        # 2. Occupied Rules
        # Thermal cooling requirement
        if t_in > self.t_high:
            # Overheated: proportional response
            excess = t_in - self.t_high
            valve = np.clip(0.5 + 0.25 * excess, 0.5, 1.0)
            fan = np.clip(0.6 + 0.2 * excess, 0.6, 1.0)
        elif t_in < self.t_low:
            # Overcooled: shut off cooling
            valve = 0.0
            fan = 0.2
        else:
            # Within comfort zone (21-24 °C): moderate floating mode
            valve = 0.25
            fan = 0.4

        # IAQ Ventilation requirement (Dilute CO2)
        if co2 > self.co2_limit:
            # Stuffy / Critical: Aggressive outdoor damper opening
            damper = 0.90
            fan = max(fan, 0.70)
        else:
            # Baseline minimum fresh air for indoor occupants
            damper = 0.25

        return np.array([valve, fan, damper], dtype=np.float32)
