import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv("./dev_train_encoded_final.csv")

print(df.head())


days = ([i for i in range(7)])
time_slots = ([i for i in range(24)])
categorys = ([i for i in range(10)])


def build_encoder(original_vals):
    ### integer mapping using LabelEncoder
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(original_vals)
    print(integer_encoded)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    ### One hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    print(onehot_encoded)
    return onehot_encoder

day_one_hot_encoder = build_encoder(time_slots)

print(day_one_hot_encoder)