import folium
import pandas as pd
from pickle import load

df = pd.read_csv("./trace_data/DUJUAN_trace.csv")


stand = load(open("sample.pkl", "rb"))
df[["lat","lng"]] = stand.inverse_transform(df[["lat","lng"]])

sq = df[["lat","lng"]].to_numpy().tolist()

lat_m = df["lat"].mean()
lng_m = df["lng"].mean()
folium_map = folium.Map(location=[lat_m, lng_m], zoom_start=5)



folium.PolyLine(locations=sq).add_to(folium_map)



ddf = pd.read_csv("sample_result_actual.csv")
sq = ddf[["lat","lng"]].to_numpy().tolist()
folium.PolyLine(locations=sq,color="green").add_to(folium_map)

ddf = pd.read_csv("sample_result_predict.csv")
sq = ddf[["lat","lng"]].to_numpy().tolist()
folium.PolyLine(locations=sq,color="red").add_to(folium_map)

folium_map.save('shinjuku_station.html')