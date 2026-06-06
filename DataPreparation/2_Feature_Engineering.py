import pandas as pd
import math
from datetime import datetime
import glob
import numpy as np


file_path = r"clean_ft8.csv"

df = pd.read_csv(file_path)

print(df.head())
print("Satır sayısı:", len(df))
print(df.columns)

##DISTANCE##

def maidenhead_to_latlon(grid):
    if len(grid) < 4:
        return None, None

    A = ord('A')

    lon = (ord(grid[0]) - A) * 20 - 180
    lat = (ord(grid[1]) - A) * 10 - 90

    lon += int(grid[2]) * 2
    lat += int(grid[3]) * 1

    lon += 1
    lat += 0.5

    return lat, lon


def haversine(lat1, lon1, lat2, lon2):
    R = 6371

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))

    return R * c

istanbul_lat, istanbul_lon = 41.0, 29.0

def compute_distance(grid):
    lat, lon = maidenhead_to_latlon(grid)
    if lat is None:
        return None
    return haversine(istanbul_lat, istanbul_lon, lat, lon)

df["distance_km"] = df["grid"].apply(compute_distance)


def calculate_azimuth(lat1, lon1, lat2, lon2):
    import math

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_lon = math.radians(lon2 - lon1)

    x = math.sin(delta_lon) * math.cos(phi2)
    y = math.cos(phi1)*math.sin(phi2) - math.sin(phi1)*math.cos(phi2)*math.cos(delta_lon)

    azimuth = math.degrees(math.atan2(x, y))

    return (azimuth + 360) % 360


def compute_azimuth(grid):
    lat, lon = maidenhead_to_latlon(grid)
    if lat is None:
        return None
    return calculate_azimuth(istanbul_lat, istanbul_lon, lat, lon)

df["azimuth"] = df["grid"].apply(compute_azimuth)

df["azimuth_sin"] = np.sin(np.radians(df["azimuth"]))
df["azimuth_cos"] = np.cos(np.radians(df["azimuth"]))

print(df[["grid", "distance_km", "azimuth"]].head())


def parse_timestamp(ts):
    return datetime.strptime(ts, "%y%m%d_%H%M%S")

df["datetime"] = df["timestamp"].apply(parse_timestamp)
df["hour"] = df["datetime"].dt.floor("H")



print(df[["grid", "distance_km"]].head())
print(df["distance_km"].describe())

print(df[["timestamp", "datetime", "hour"]].head())

##NOAA MERGE##

noaa_path = r"Dataset\cleaned_data\space_weather.csv"

df_noaa = pd.read_csv(noaa_path)

print("NOAA veri:", len(df_noaa))
print(df_noaa.head())

# datetime parse
df_noaa["datetime"] = pd.to_datetime(df_noaa["datetime"])


df_noaa["datetime"] = df_noaa["datetime"].dt.tz_localize(None)


df_noaa["hour"] = df_noaa["datetime"]


df_noaa = df_noaa.drop_duplicates(subset=["hour"])

print("NOAA unique saat:", len(df_noaa))

print(df_noaa["hour"].value_counts().max())

df_merged = pd.merge(df, df_noaa, on="hour", how="inner")

print("FT8:", len(df))
print("Merged:", len(df_merged))

df_merged.to_csv("final_dataset.csv", index=False)

print("FINAL DATASET HAZIR")