import pandas as pd
import numpy as np 

df_original = pd.read_csv("/src/dev_train_encoded_final.csv")
df_generated = pd.read_csv("/src/k-same-net_generated_traj.csv")



df_origin_day = df_original.groupby(["category"]).size().values
df_gen_day = df_generated.groupby(["category"]).size().values

print(np.corrcoef(df_origin_day, df_gen_day))
print("="*40)

df_origin_day = df_original.groupby(["hour"]).size().values
df_gen_day = df_generated.groupby(["hour"]).size()
gen_series = df_gen_day.append(pd.Series([0], index = [9])).sort_index()
gen_series = gen_series.values

print("original", df_origin_day)
print("gennerated", gen_series)

print(np.corrcoef(df_origin_day, gen_series))