import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import torch
import torch.nn as nn


def build_encoder(original_vals):
    # integer mapping using LabelEncoder
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(original_vals)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

    # print(integer_encoded)

    # One hot encoding
    onehot_encoder = OneHotEncoder(sparse_output=False)
    onehot_encoded = onehot_encoder.fit(integer_encoded)

    return onehot_encoded


def convert_onehot(df, encoder, col_name="day"):
    onehot_encoded = encoder.transform(df[[col_name]])
    onehot_df = pd.DataFrame(
        onehot_encoded, columns=encoder.get_feature_names_out([col_name]))
    onehot_df = onehot_df.astype(int)

    return onehot_df


def main(device):
    df = pd.read_csv("./dev_train_encoded_final.csv")

    print(df.head())

    # df_onehot = pd.concat([df[["tid", "label", "lat", "lon"]],
    #                       pd.get_dummies(df["category"], prefix="category")], axis=1)

    # print(df_onehot.head())

    target_dict = {}
    target_dict["day"] = ([i for i in range(7)])
    target_dict["hour"] = ([i for i in range(24)])
    target_dict["category"] = ([i for i in range(10)])

    encoder_dict = {}

    for col in ["day", "category", "hour"]:
        target = target_dict[col]
        encoder_dict[col] = build_encoder(target)

    time_slots_encoder = encoder_dict["hour"]
    print(time_slots_encoder.transform([[12]]))

    df_target = df[["tid", "lat", "lon"]]
    for col in ["day", "category", "hour"]:
        _encoder = encoder_dict[col]
        onehot_df = convert_onehot(df=df, encoder=_encoder, col_name=col)
        df_target = pd.concat([df_target, onehot_df], axis=1)

    print(df_target.head())

    seq_len_list = []
    tensor_list = []
    for tid in df_target["tid"].unique():
        df_tmp = df_target[df_target["tid"] == tid]
        df_tmp = df_tmp.drop("tid", axis=1)
        traj = df_tmp.values
        seq_len_list.append(len(traj))
        tensor_list.append(torch.Tensor(traj))

    print(seq_len_list)

    traj_packed = nn.utils.rnn.pack_sequence(tensor_list, enforce_sorted=False)
    traj_padded, _ = nn.utils.rnn.pad_packed_sequence(
        traj_packed, batch_first=True, padding_value=-1)
    
    traj_padded = traj_padded.to(device)

    return traj_padded, seq_len_list


if __name__ == "__main__":
    traj_padded, _ = main()
    print(traj_padded.size())

    for i in range(3):
        print("-------------- hoge -----------------")
        print(traj_padded[i])
