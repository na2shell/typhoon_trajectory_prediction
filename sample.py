import pandas as pd
from sklearn.preprocessing import StandardScaler
from pickle import dump

df = pd.read_csv("./data/table2021.csv",
                 dtype={'年': "str", "月": "str", "日": "str", "時（UTC）": "str"})
df = df.rename(columns={'年': 'year',
                        '月': 'month',
                        "日": "day",
                        "時（UTC）": "time",
                        "緯度": "lat",
                        "経度": "lng",
                        "台風名":"name"})

names = df["name"].unique()

def concat_datetime(x):
    year = x["year"].zfill(4)
    month = x["month"].zfill(2)
    day = x["day"].zfill(2)
    time = x["time"].zfill(2)

    t = "{}-{}-{} {}".format(year, month, day, time)
    return pd.to_datetime(t, format='%Y-%m-%d %H')

stand = StandardScaler().fit(df[["lat","lng"]])
df[["lat","lng"]] = stand.transform(df[["lat","lng"]])
dump(stand, open("sample.pkl", "wb"))


for name in names:
    _df = df[df["name"] == name]
    _df["datetime"] = _df.apply(concat_datetime, axis=1)
    print(_df[["datetime","lat","lng"]])

    trace_data = _df[["datetime","lat","lng"]]
    trace_data.to_csv("./trace_data/{}_trace.csv".format(name), index=False)

