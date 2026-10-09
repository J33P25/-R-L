"""
GreenMind: Deep Reinforcement Learning for Sustainable Buildings
Central Configuration and Hyperparameters
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
OCCUPANCY_DIR = BASE_DIR / "GreenMind-RL-Building-Control" / "occupancy+detection"
WEATHER_FILE = BASE_DIR / "weather.txt"
ENERGY_FILE = BASE_DIR / "Energy_consumption.csv"
SOLAR_FILE = BASE_DIR / "GreenMind-RL-Building-Control" / "solar_cleaned.txt"
DATA_EXPORT_DIR = BASE_DIR / "data_processed"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
DATA_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Comfort & Safety Thresholds (as specified in GreenMind formulation)
T_COMFORT_MIN = 21.0       # °C (Comfort lower bound)
T_COMFORT_MAX = 24.0       # °C (Comfort upper bound)
CO2_SAFE_LIMIT = 1000.0    # ppm (Indoor air quality limit)
T_SAFETY_MIN = 15.0        # °C (S7 Safety critical lower bound)
T_SAFETY_MAX = 32.0        # °C (S7 Safety critical upper bound)

# Reward Formulation Parameters (as defined in GreenMind PDF)
DISCOUNT_FACTOR = 0.95     # γ = 0.95 as specified in Section 5
REWARD_GREEN_DAY = 15.0    # S8 Positive terminal reward
PENALTY_SAFETY = -15.0     # S7 Negative terminal penalty

# Multi-objective weights for continuous Deep RL reward
WEIGHT_ENERGY = 1.0        # w_energy
WEIGHT_COMFORT = 2.0       # w_comfort
WEIGHT_IAQ = 1.5           # w_iaq
WEIGHT_SMOOTHNESS = 0.5    # w_smoothness (Actuator stability / hunting penalty)

# Physical Building Simulation Constants (ISO 13790 / ASHRAE Standard Thermal Mass)
ZONE_AREA = 100.0          # m² office zone
ZONE_AIR_VOLUME = 250.0    # m³ (100 m² x 2.5m ceiling)
AIR_DENSITY = 1.2          # kg/m³
AIR_SPECIFIC_HEAT = 1005.0 # J/(kg·K)
# Building zone thermal capacitance including concrete slab, walls, and structure:
# Real buildings have ~1.2e7 J/K (prevents instant temperature collapse)
HEAT_CAPACITANCE = 1.2e7   # J/K (Building structural thermal inertia)
THERMAL_RESISTANCE = 0.12  # K/W (Envelope insulation)

# Occupancy & Ventilation Constants
HUMAN_SENSIBLE_HEAT = 80.0 # Watts per person (metabolic heat gain)
HUMAN_CO2_GEN_RATE = 0.005 # L/s per person (~0.3 L/min)
OUTDOOR_CO2_PPM = 415.0    # Ambient outdoor fresh air CO2
MAX_VENT_FLOW = 0.25       # m³/s (Outdoor fresh air ventilation rate)
MAX_FAN_FLOW = 0.8         # m³/s (Supply fan capacity)
MAX_COOLING_CAPACITY = 4500.0 # Watts (4.5 kW cooling coil)
MAX_HEATING_CAPACITY = 3500.0 # Watts (3.5 kW heating / reheat coil)
