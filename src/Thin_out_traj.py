import pandas as pd
import tqdm

df = pd.read_csv("/data/original_pos_60min.csv")

tids = df["tid"].unique()

df_result = pd.DataFrame()

for tid in tqdm.tqdm(tids):
    df_tmp = df[df["tid"] == tid]
    if len(df_tmp) > 30:
        df_tmp = df_tmp[1::10]
        df_result = pd.concat([df_result, df_tmp])

df_result["lat"] = df_result["lat"] - 36
df_result["lon"] = df_result["lon"] - 139
df_result.to_csv("/data/original_pos_60min_thin_out.csv")