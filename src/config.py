"""
GreenMind: Deep Reinforcement Learning for Sustainable Buildings
Central Configuration and Physical Hyperparameters
"""
from pathlib import Path

# Project Directory Roots
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_EXPORT_DIR = BASE_DIR / "data_processed"
MODELS_DIR = BASE_DIR / "saved_models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure runtime directories exist
DATA_EXPORT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Raw Data Sources (Building Data Genome Project 2 - Panther Commercial Site)
WEATHER_FILE = BASE_DIR / "weather.txt"
CHILLED_WATER_FILE = BASE_DIR / "water_cleaned.txt"
UNIFIED_TELEMETRY_FILE = DATA_EXPORT_DIR / "unified_building_telemetry.csv"

# Target Building Baseline
PRIMARY_OFFICE_BUILDING = "Panther_office_Hannah"
BDG2_SITE_ID = "Panther"

# Simulation Time Horizon
TIME_STEP_SEC = 900  # 15-minute control time step (dt = 15 min = 900 s)
STEPS_PER_DAY = 96   # 24 hours * 4 steps/hour

# ASHRAE Standard Comfort & Safety Bounds
T_COMFORT_MIN = 21.0       # °C (ASHRAE 55 lower comfort bound)
T_COMFORT_MAX = 24.0       # °C (ASHRAE 55 upper comfort bound)
CO2_SAFE_LIMIT = 1000.0    # ppm (ASHRAE 62.1 indoor air quality limit)
T_SAFETY_MIN = 15.0        # °C (S7 Safety critical lower bound)
T_SAFETY_MAX = 32.0        # °C (S7 Safety critical upper bound)

# Multi-Objective Normalized Pareto Reward Weights (Sum = 1.0)
WEIGHT_ENERGY = 0.35       # w_energy: minimize HVAC electrical power
WEIGHT_COMFORT = 0.35      # w_comfort: ASHRAE 55 temperature compliance
WEIGHT_IAQ = 0.20          # w_iaq: ASHRAE 62.1 CO2 ventilation compliance
WEIGHT_SMOOTHNESS = 0.10   # w_smooth: Actuator hunting / mechanical wear penalty

# Terminal Reward / Penalty
DISCOUNT_FACTOR = 0.95     # γ = 0.95
REWARD_GREEN_DAY = 15.0    # S8 Positive terminal reward (sustained comfort >= 80% of operating day)
PENALTY_SAFETY = -15.0     # S7 Negative terminal penalty (safety bounds breached)

# Commercial Office Zone Physical Constants (ISO 13790 / ASHRAE Fundamentals)
ZONE_AREA = 120.0          # m² open office floor area
ZONE_CEILING_HEIGHT = 2.5  # m
ZONE_AIR_VOLUME = 300.0    # m³ (120 m² x 2.5 m)
ZONE_MAX_CAPACITY = 15     # Nominal maximum occupant headcount

AIR_DENSITY = 1.204        # kg/m³ at 20°C
AIR_SPECIFIC_HEAT = 1005.0 # J/(kg·K)

# Building Thermal Mass (Walls, concrete slab, furniture)
HEAT_CAPACITANCE = 1.5e7   # J/K (Effective structural thermal capacitance)
THERMAL_RESISTANCE = 0.15  # K/W (Building envelope thermal resistance)

# Human Occupant Metabolic Rates (ASHRAE Fundamentals)
HUMAN_SENSIBLE_HEAT = 90.0 # Watts per person
HUMAN_CO2_GEN_RATE = 0.0052# L/s per person (~0.31 L/min light office activity)
OUTDOOR_CO2_PPM = 415.0    # Ambient outdoor fresh air CO2 concentration

# HVAC Equipment Sizing & Capacities (ASHRAE Commercial Office Standards)
MAX_COOLING_CAPACITY = 10000.0 # Watts (10 kW chilled water cooling coil for 120 m² zone)
MAX_HEATING_CAPACITY = 6000.0  # Watts (6 kW reheat / heat recovery coil)
MAX_VENT_FLOW = 0.25          # m³/s outdoor fresh air damper capacity
MAX_FAN_FLOW = 1.0            # m³/s supply fan air volume flow
CHILLER_COP = 3.5             # Coefficient of Performance of central chiller
MAX_FAN_POWER_KW = 1.5        # kW peak electrical power of variable-speed supply fan
MAX_DAMPER_POWER_KW = 0.05    # kW actuator motor power
MAX_TOTAL_POWER_KW = 5.0      # kW normalization scale
