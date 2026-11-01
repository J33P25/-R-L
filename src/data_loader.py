"""
GreenMind: Multi-Source Dataset Loader & Synchronization Pipeline
Aligns and synchronizes:
1. BDG2 Weather Time-Series (Outdoor dry-bulb T, dew point RH, wind speed, solar radiation)
2. BDG2 Chilled Water Meter Data (Real measured commercial office cooling consumption)
3. ASHRAE 90.1 Commercial Office Occupancy Schedules & Metabolic CO2 Emission Profiles
All on a continuous, unified 15-minute time horizon (35,040 steps for 2016).
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
from src.config import (
    WEATHER_FILE, CHILLED_WATER_FILE, UNIFIED_TELEMETRY_FILE,
    PRIMARY_OFFICE_BUILDING, BDG2_SITE_ID,
    ZONE_MAX_CAPACITY, HUMAN_CO2_GEN_RATE
)

def calculate_relative_humidity(air_temp_c: np.ndarray, dew_temp_c: np.ndarray) -> np.ndarray:
    """
    Magnus-Tetens empirical formulation for ambient relative humidity from air and dew temperatures.
    """
    a = 17.27
    b = 237.7
    alpha_air = (a * air_temp_c) / (b + air_temp_c)
    alpha_dew = (a * dew_temp_c) / (b + dew_temp_c)
    rh = 100.0 * np.exp(alpha_dew - alpha_air)
    return np.clip(rh, 5.0, 100.0)

def calculate_solar_irradiance(datetimes: pd.DatetimeIndex, cloud_oktas: np.ndarray, latitude_deg: float = 28.5) -> np.ndarray:
    """
    Computes global horizontal solar irradiance (W/m²) accounting for solar zenith angle
    and Kasten-Czeplak cloud attenuation.
    """
    lat = np.radians(latitude_deg)
    day_of_year = datetimes.dayofyear.values
    hours = datetimes.hour.values + datetimes.minute.values / 60.0

    # Solar declination angle
    dec = np.radians(23.45 * np.sin(np.radians((360.0 / 365.0) * (284.0 + day_of_year))))
    # Solar hour angle (15 deg per hour from solar noon)
    hour_angle = np.radians(15.0 * (hours - 12.0))

    # Solar elevation angle
    sin_elev = np.sin(lat) * np.sin(dec) + np.cos(lat) * np.cos(dec) * np.cos(hour_angle)
    elev = np.maximum(0.0, sin_elev)

    # Clear-sky horizontal irradiance (~1050 W/m² peak)
    i_clear = 1050.0 * elev
    cloud_frac = np.clip(np.nan_to_num(cloud_oktas, nan=3.0) / 8.0, 0.0, 1.0)
    # Kasten-Czeplak cloud attenuation formula
    i_solar = i_clear * (1.0 - 0.75 * (cloud_frac ** 3.4))
    return np.where(elev > 0.0, i_solar, 0.0)

def generate_ashrae_occupancy(datetimes: pd.DatetimeIndex, zone_capacity: int = ZONE_MAX_CAPACITY, seed: int = 42) -> np.ndarray:
    """
    Generates realistic commercial office occupancy headcounts following ASHRAE 90.1
    standard commercial office schedules with stochastic Poisson noise.
    """
    np.random.seed(seed)
    n = len(datetimes)
    counts = np.zeros(n, dtype=int)

    # ASHRAE Standard Commercial Office Profile (Hourly Multiplier)
    weekday_profile = {
        0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0,
        6: 0.10, 7: 0.35, 8: 0.75, 9: 0.95, 10: 1.00, 11: 1.00,
        12: 0.50, 13: 0.90, 14: 1.00, 15: 1.00, 16: 0.90, 17: 0.50,
        18: 0.20, 19: 0.10, 20: 0.05, 21: 0.0, 22: 0.0, 23: 0.0
    }
    saturday_profile = {h: (0.15 if 8 <= h <= 13 else 0.0) for h in range(24)}
    sunday_profile = {h: 0.0 for h in range(24)}

    for i, dt in enumerate(datetimes):
        dow = dt.dayofweek
        h = dt.hour
        if dow < 5:
            frac = weekday_profile[h]
        elif dow == 5:
            frac = saturday_profile[h]
        else:
            frac = sunday_profile[h]

        lam = frac * zone_capacity
        if lam > 0.0:
            counts[i] = int(np.random.poisson(lam))
        else:
            counts[i] = 0

    return counts

def load_bdg2_weather(year: str = "2016") -> pd.DataFrame:
    """
    Loads and cleans BDG2 weather observations for the target site.
    """
    if not WEATHER_FILE.exists():
        raise FileNotFoundError(f"BDG2 Weather file not found at {WEATHER_FILE}")

    print(f"[DataLoader] Reading BDG2 weather telemetry from {WEATHER_FILE.name}...")
    df_raw = pd.read_csv(WEATHER_FILE, parse_dates=["timestamp"])
    site_df = df_raw[df_raw["site_id"] == BDG2_SITE_ID].copy()
    site_df.drop_duplicates(subset=["timestamp"], inplace=True)
    site_df.set_index("timestamp", inplace=True)
    site_df.sort_index(inplace=True)

    # Filter year
    start_ts = f"{year}-01-01 00:00:00"
    end_ts = f"{year}-12-31 23:00:00"
    df_year = site_df.loc[start_ts:end_ts].copy()

    # Clean and interpolate missing sensors
    df_year["airTemperature"] = df_year["airTemperature"].interpolate(method="time").bfill().ffill()
    df_year["dewTemperature"] = df_year["dewTemperature"].interpolate(method="time").bfill().ffill()
    df_year["cloudCoverage"] = df_year["cloudCoverage"].interpolate(method="time").fillna(3.0)
    df_year["windSpeed"] = df_year["windSpeed"].interpolate(method="time").fillna(2.0)

    # Resample to 15-minute intervals with smooth spline/linear interpolation
    resampled_15m = df_year[["airTemperature", "dewTemperature", "cloudCoverage", "windSpeed"]].resample("15min").interpolate(method="time")
    resampled_15m.bfill(inplace=True)
    resampled_15m.ffill(inplace=True)
    return resampled_15m

def load_bdg2_chilled_water(year: str = "2016") -> pd.DataFrame:
    """
    Loads and cleans BDG2 chilled water meter consumption for office buildings.
    """
    if not CHILLED_WATER_FILE.exists():
        raise FileNotFoundError(f"BDG2 Chilled Water file not found at {CHILLED_WATER_FILE}")

    print(f"[DataLoader] Reading BDG2 chilled water meter data from {CHILLED_WATER_FILE.name}...")
    df_water = pd.read_csv(CHILLED_WATER_FILE, parse_dates=["timestamp"])
    df_water.drop_duplicates(subset=["timestamp"], inplace=True)
    df_water.set_index("timestamp", inplace=True)
    df_water.sort_index(inplace=True)

    start_ts = f"{year}-01-01 00:00:00"
    end_ts = f"{year}-12-31 23:00:00"
    water_year = df_water.loc[start_ts:end_ts].copy()

    # Primary target office building
    if PRIMARY_OFFICE_BUILDING in water_year.columns:
        target_series = water_year[PRIMARY_OFFICE_BUILDING].interpolate(method="time").bfill().ffill()
    else:
        # Fallback to first available office column
        office_cols = [c for c in water_year.columns if "office" in c]
        target_series = water_year[office_cols[0]].interpolate(method="time").bfill().ffill()

    # Ensemble cluster mean across all office buildings for macro comparison
    office_cols = [c for c in water_year.columns if f"{BDG2_SITE_ID}_office" in c]
    if office_cols:
        cluster_mean = water_year[office_cols].mean(axis=1).interpolate(method="time").bfill().ffill()
    else:
        cluster_mean = target_series.copy()

    combined = pd.DataFrame({
        "real_chiller_kwh": target_series,
        "cluster_chiller_kwh": cluster_mean
    })

    # Resample to 15-minute intervals (hourly meter reading distributed evenly over four 15-min intervals)
    resampled_15m = combined.resample("15min").interpolate(method="time") / 4.0
    resampled_15m.bfill(inplace=True)
    resampled_15m.ffill(inplace=True)
    return resampled_15m

def build_unified_dataset(force_rebuild: bool = False) -> pd.DataFrame:
    """
    Builds a single, synchronized, physically consistent building telemetry dataset.
    """
    if UNIFIED_TELEMETRY_FILE.exists() and not force_rebuild:
        print(f"[DataLoader] Loading existing synchronized dataset from {UNIFIED_TELEMETRY_FILE}...")
        df = pd.read_csv(UNIFIED_TELEMETRY_FILE, parse_dates=["timestamp"])
        return df

    print("\n[DataLoader] >>> Constructing Synchronized BDG2 Telemetry Dataset (2016)...")
    weather_15m = load_bdg2_weather("2016")
    water_15m = load_bdg2_chilled_water("2016")

    # Inner join on timestamps
    merged = weather_15m.join(water_15m, how="inner")
    dts = merged.index

    # Physics computations
    outdoor_temp = merged["airTemperature"].values
    dew_temp = merged["dewTemperature"].values
    cloud_cov = merged["cloudCoverage"].values
    wind_spd = merged["windSpeed"].values

    outdoor_humidity = calculate_relative_humidity(outdoor_temp, dew_temp)
    solar_irradiance = calculate_solar_irradiance(dts, cloud_cov)
    occupants = generate_ashrae_occupancy(dts, zone_capacity=ZONE_MAX_CAPACITY)
    occupancy_binary = (occupants > 0).astype(int)
    co2_emission_rate = occupants * HUMAN_CO2_GEN_RATE

    df_clean = pd.DataFrame({
        "timestamp": dts,
        "outdoor_temp": np.round(outdoor_temp, 2),
        "outdoor_humidity": np.round(outdoor_humidity, 2),
        "cloud_coverage": np.round(cloud_cov, 1),
        "wind_speed": np.round(wind_spd, 2),
        "solar_irradiance": np.round(solar_irradiance, 2),
        "occupants_count": occupants,
        "occupancy_binary": occupancy_binary,
        "co2_emission_rate_lps": np.round(co2_emission_rate, 4),
        "real_chiller_kwh": np.round(merged["real_chiller_kwh"].values, 4),
        "cluster_chiller_kwh": np.round(merged["cluster_chiller_kwh"].values, 4)
    })

    df_clean.to_csv(UNIFIED_TELEMETRY_FILE, index=False)
    print(f"[DataLoader] >>> Successfully generated {len(df_clean)} synchronized timesteps.")
    print(f"[DataLoader] Saved to {UNIFIED_TELEMETRY_FILE}")
    return df_clean

if __name__ == "__main__":
    df = build_unified_dataset(force_rebuild=True)
    print("\n--- Summary Statistics ---")
    print(df[["outdoor_temp", "outdoor_humidity", "solar_irradiance", "occupants_count", "real_chiller_kwh"]].describe())
    print("\n--- First 5 Timesteps ---")
    print(df.head())
