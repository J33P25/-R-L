"""
GreenMind: Multi-Source Dataset Loader & Preprocessing Pipeline
Synchronizes and aligns:
1. Occupancy Detection data (Indoor T, RH, CO2, Binary Occ)
2. Weather time-series (Outdoor T, Dew point, Cloud coverage, Wind)
3. Energy Consumption data (Continuous Occupancy Headcount, Renewable Solar, Baseline Energy)
"""
import sys
from pathlib import Path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pandas as pd
import numpy as np
from src.config import OCCUPANCY_DIR, WEATHER_FILE, ENERGY_FILE, DATA_EXPORT_DIR

def calculate_relative_humidity(air_temp_c, dew_temp_c):
    """
    Magnus-Tetens formula to compute relative humidity from air and dew temperatures.
    """
    a = 17.27
    b = 237.7
    alpha_air = (a * air_temp_c) / (b + air_temp_c)
    alpha_dew = (a * dew_temp_c) / (b + dew_temp_c)
    rh = 100.0 * np.exp(alpha_dew - alpha_air)
    return np.clip(rh, 5.0, 100.0)

def load_occupancy_data():
    """
    Loads and concatenates the UCI Occupancy Detection environmental time-series.
    """
    files = ["datatraining.txt", "datatest.txt", "datatest2.txt"]
    dfs = []
    for f in files:
        file_path = OCCUPANCY_DIR / f
        if file_path.exists():
            df = pd.read_csv(file_path, parse_dates=["date"])
            dfs.append(df)
    
    if not dfs:
        raise FileNotFoundError(f"No occupancy data found in {OCCUPANCY_DIR}")
    
    combined = pd.concat(dfs, ignore_index=True)
    combined.sort_values(by="date", inplace=True)
    combined.rename(columns={"date": "timestamp"}, inplace=True)
    combined.drop_duplicates(subset=["timestamp"], inplace=True)
    return combined

def load_weather_data(limit_rows=50000):
    """
    Loads outdoor weather observations and derives outdoor relative humidity.
    """
    if not WEATHER_FILE.exists():
        raise FileNotFoundError(f"Weather file not found at {WEATHER_FILE}")

    # Read site observations (Panther site or first available)
    df = pd.read_csv(WEATHER_FILE, nrows=limit_rows, parse_dates=["timestamp"])
    df.sort_values(by="timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], inplace=True)
    
    # Fill or interpolate missing temperatures
    df["airTemperature"] = df["airTemperature"].interpolate(method="linear").bfill().ffill()
    df["dewTemperature"] = df["dewTemperature"].interpolate(method="linear").bfill().ffill()
    df["cloudCoverage"] = df["cloudCoverage"].fillna(4.0) # Medium cloud cover default
    df["windSpeed"] = df["windSpeed"].fillna(2.0)
    
    df["outdoor_humidity"] = calculate_relative_humidity(df["airTemperature"], df["dewTemperature"])
    df.rename(columns={"airTemperature": "outdoor_temp", "cloudCoverage": "cloud_coverage", "windSpeed": "wind_speed"}, inplace=True)
    return df[["timestamp", "outdoor_temp", "outdoor_humidity", "cloud_coverage", "wind_speed"]]

def load_energy_data():
    """
    Loads building energy consumption and occupancy headcount data.
    """
    if not ENERGY_FILE.exists():
        raise FileNotFoundError(f"Energy file not found at {ENERGY_FILE}")

    df = pd.read_csv(ENERGY_FILE, parse_dates=["Timestamp"])
    df.rename(columns={
        "Timestamp": "timestamp",
        "Occupancy": "occupancy_headcount",
        "EnergyConsumption": "baseline_energy_kwh",
        "RenewableEnergy": "solar_renewable_kwh"
    }, inplace=True)
    df.sort_values(by="timestamp", inplace=True)
    return df

def build_unified_dataset():
    """
    Aligns and blends the datasets into an unified environmental replay dataframe.
    """
    out_file = DATA_EXPORT_DIR / "unified_building_telemetry.csv"
    if out_file.exists():
        return pd.read_csv(out_file, parse_dates=["timestamp"])

    occ_df = load_occupancy_data()
    weather_df = load_weather_data()
    energy_df = load_energy_data()

    # Re-sample occupancy to 15-minute / hourly intervals for long-horizon simulation
    occ_resampled = occ_df.set_index("timestamp").resample("15min").agg({
        "Temperature": "mean",
        "Humidity": "mean",
        "CO2": "mean",
        "Occupancy": "max"
    }).dropna().reset_index()

    # Weather sequence
    weather_series = weather_df.set_index("timestamp").resample("15min").interpolate().dropna().reset_index()

    # Create composite synthetic aligned time-frame
    n_rows = min(len(occ_resampled), len(weather_series))
    merged = occ_resampled.iloc[:n_rows].copy()
    
    # Attach weather
    merged["outdoor_temp"] = weather_series["outdoor_temp"].iloc[:n_rows].values
    merged["outdoor_humidity"] = weather_series["outdoor_humidity"].iloc[:n_rows].values
    merged["cloud_coverage"] = weather_series["cloud_coverage"].iloc[:n_rows].values

    # Attach energy profiles
    if len(energy_df) > 0:
        energy_cycle = energy_df["baseline_energy_kwh"].values
        renew_cycle = energy_df["solar_renewable_kwh"].values
        headcount_cycle = energy_df["occupancy_headcount"].values

        merged["baseline_energy_kwh"] = np.resize(energy_cycle, n_rows)
        merged["solar_renewable_kwh"] = np.resize(renew_cycle, n_rows)
        # Combine binary occupancy with integer headcounts: if binary is 1, take headcount >= 1
        raw_headcount = np.resize(headcount_cycle, n_rows)
        merged["occupants_count"] = np.where(merged["Occupancy"] > 0, np.maximum(raw_headcount, 1), 0)
    else:
        merged["baseline_energy_kwh"] = 25.0
        merged["solar_renewable_kwh"] = 5.0
        merged["occupants_count"] = merged["Occupancy"] * 5

    # Renaming for clean access
    merged.rename(columns={
        "Temperature": "indoor_temp",
        "Humidity": "indoor_humidity",
        "CO2": "co2_ppm",
        "Occupancy": "occupancy_binary"
    }, inplace=True)

    # Save to processed data folder
    merged.to_csv(out_file, index=False)
    print(f"[DataLoader] Successfully unified {len(merged)} timesteps saved to {out_file}")
    return merged

if __name__ == "__main__":
    df = build_unified_dataset()
    print(df.head())
    print("\nSummary Statistics:")
    print(df[["indoor_temp", "indoor_humidity", "co2_ppm", "outdoor_temp", "occupants_count"]].describe())
