import pandas as pd

df = pd.read_csv("./dev_train_encoded_final.csv")

print(df["label"].unique().tolist(), df["label"].nunique())
