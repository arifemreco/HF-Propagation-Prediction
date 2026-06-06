import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv(r"Dataset\cleaned_data\final_dataset.csv")
'''
summary = df[['snr','distance_km','kp','sfi']].describe().round(2)

print(summary)

fig, ax = plt.subplots()
ax.axis('off')
ax.table(cellText=summary.values,
         colLabels=summary.columns,
         rowLabels=summary.index,
         loc='center')

plt.savefig("summary_table.png", bbox_inches='tight')
plt.show()
'''
'''
plt.figure()
df['snr'].hist(bins=50)
plt.title("SNR Distribution")
plt.xlabel("SNR (dB)")
plt.ylabel("Frequency")
plt.savefig("snr_distribution.png")
plt.show()

plt.figure()
df['distance_km'].hist(bins=100)
plt.title("Distance Distribution")
plt.xlabel("Distance (km)")
plt.ylabel("Frequency")
plt.savefig("distance_distribution.png")
plt.show()

df["connection"] = (df["snr"] > -15).astype(int)
df['connection'].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Class")
plt.ylabel("Count")
plt.savefig("class_distribution.png")
plt.show()



plt.figure()
plt.scatter(df['distance_km'], df['snr'], alpha=0.3)
plt.xlabel("Distance (km)")
plt.ylabel("SNR")
plt.title("Distance vs SNR")
plt.savefig("distance_vs_snr.png")
plt.show()

plt.figure()
plt.scatter(df['audio_freq'], df['snr'], alpha=0.3)
plt.xlabel("Audio Frequency")
plt.ylabel("SNR")
plt.title("Audio Frequency vs SNR")
plt.savefig("audiofreq_vs_snr.png")
plt.show()

'''
'''
def maidenhead_to_latlon(grid):
    try:
        if not isinstance(grid, str) or len(grid) < 4:
            return None, None
        
        A = ord('A')

        lon = (ord(grid[0].upper()) - A) * 20 - 180
        lat = (ord(grid[1].upper()) - A) * 10 - 90

        lon += int(grid[2]) * 2
        lat += int(grid[3]) * 1

        # Eğer 6 karakter varsa devam et
        if len(grid) >= 6:
            lon += (ord(grid[4].upper()) - A) * 5 / 60
            lat += (ord(grid[5].upper()) - A) * 2.5 / 60

        return lat, lon

    except:
        return None, None
    
df[['lat','lon']] = df['grid'].apply(
    lambda x: pd.Series(maidenhead_to_latlon(x))
)

df = df.dropna(subset=['lat','lon'])

'''
'''
import plotly.express as px

sample_df = df.sample(5000)  

fig = px.scatter_geo(
    sample_df,
    lat="lat",
    lon="lon",
    color="snr",
    title="Global FT8 Signal Distribution",
)

fig.show()
'''
'''
import plotly.express as px

df_clean = df[(df['snr'] >= -25) & (df['snr'] <= 0)]
df_sample = df_clean.sample(100000)

fig = px.density_mapbox(
    df_sample,
    lat="lat",
    lon="lon",
    z="snr",
    radius=5,
    center=dict(lat=20, lon=0),
    zoom=1,
    mapbox_style="carto-positron",
    color_continuous_scale="viridis", 
    range_color=[-25, -5],  
    title="Global FT8 Signal Density"
)
fig.update_layout(
    margin=dict(l=0, r=0, t=50, b=0)
)
fig.update_layout(
    title_x=0.5
)
fig.show()
fig.write_image("heatmap_soft.png", width=1800, height=900, scale=2)


'''
'''
import geopandas as gpd
import matplotlib.pyplot as plt

world = gpd.read_file("https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip")

fig, ax = plt.subplots(figsize=(12,6))
world.plot(ax=ax, color='lightgray', edgecolor='white')

plt.title("Global FT8 Signal Distribution")
plt.show()

import pandas as pd
from shapely.geometry import Point

geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
gdf = gpd.GeoDataFrame(df, geometry=geometry)

fig, ax = plt.subplots(figsize=(12,6))
world.plot(ax=ax, color='lightgray', edgecolor='white')

gdf.sample(50000).plot(
    ax=ax,
    markersize=2,
    color='red',
    alpha=0.5
)

plt.title("Global FT8 Signal Locations")
plt.show()

fig, ax = plt.subplots(figsize=(12,6))
world.plot(ax=ax, color='lightgray', edgecolor='white')

gdf_sample = gdf.sample(50000)

gdf_sample.plot(
    ax=ax,
    column='snr',
    cmap='plasma',  # 
    markersize=3,
    alpha=0.6,
    legend=True
)

plt.title("Global FT8 Signal Distribution (SNR)")
plt.show()
plt.savefig("world_map_clean.png", dpi=300, bbox_inches='tight')

'''
'''
df_grouped = df.groupby(['lat','lon']).agg({
    'connection': 'mean'
}).reset_index()

fig = px.density_mapbox(
    df_grouped,
    lat="lat",
    lon="lon",
    z="connection",
    radius=6,
    center=dict(lat=20, lon=0),
    zoom=1,
    mapbox_style="carto-positron",
    color_continuous_scale="viridis",  
    range_color=[0, 1],
    title="Global Communication Success Rate"
)

fig.show()
'''
'''
fig.write_image("success_heatmap.png", width=1800, height=900, scale=2)
'''
df["connection"] = (df["snr"] > -15).astype(int)
'''
good = df[(df['kp'] <= 3) & (df['sfi'] >= 110)]
bad = df[(df['kp'] >= 4) & (df['sfi'] <= 100)]

good_success = good['connection'].mean()
bad_success = bad['connection'].mean()

good_max_dist = good['distance_km'].max()
bad_max_dist = bad['distance_km'].max()

print("Good success:", good_success)
print("Bad success:", bad_success)

print("Good max distance:", good_max_dist)
print("Bad max distance:", bad_max_dist)

import matplotlib.pyplot as plt

labels = ['Good', 'Bad']
success_values = [good_success, bad_success]

plt.figure()
plt.bar(labels, success_values, color=['green','red'])
plt.title("Success Rate under Different Conditions")
plt.ylabel("Success Rate")
plt.savefig("success_conditions.png")
plt.show()

distance_values = [good_max_dist, bad_max_dist]

plt.figure()
plt.bar(labels, distance_values, color=['green','red'])
plt.title("Maximum Communication Distance under Different Conditions")
plt.ylabel("Distance (km)")
plt.savefig("distance_conditions.png")
plt.show()
'''
df["datetime_x"] = pd.to_datetime(df["datetime_x"])

df["utc_hour"] = df["datetime_x"].dt.hour

hourly = df.groupby("utc_hour")["connection"].mean()

plt.figure(figsize=(10,5))

plt.plot(hourly.index, hourly.values, marker='o')

plt.xticks(range(0,24))
plt.xlabel("UTC Hour")
plt.ylabel("Average Connection Probability")

plt.title("40m FT8 Connection Probability vs UTC Hour")
plt.grid(True)
plt.savefig("grayline.png")
plt.show()


#FEATURE IMPORTANCE

import matplotlib.pyplot as plt
import pandas as pd

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": best_rf.feature_importances_
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

plt.figure(figsize=(8,5))

plt.barh(
    importance_df["Feature"],
    importance_df["Importance"]
)

plt.gca().invert_yaxis()

plt.xlabel("Importance Score")
plt.ylabel("Features")

plt.title("Random Forest Feature Importance")

plt.grid(axis="x")

plt.savefig(
    "rf_feature_importance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print(importance_df)