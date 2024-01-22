import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import geohash


def time_collate_fn(batch):
    traj, seq_len, traj_class_indices, labels, tids = list(zip(*batch))
    x = torch.nn.utils.rnn.pad_sequence(traj, batch_first=True, padding_value=99)
    traj_class_indices = torch.nn.utils.rnn.pad_sequence(
        traj_class_indices, batch_first=True, padding_value=111
    )
    seq_len = torch.stack(seq_len)
    labels = torch.stack(labels)
    tids = torch.stack(tids)
    mask = x[:, :, 1] == 99

    return x, seq_len, traj_class_indices, labels, mask, tids

# it is hard to learn the hashed position for generator
# thus I convert actual latlon to geohash just before feeding it into discriminator
def convert_latlon_to_geohash(batched_position_array, precision=8):
    precision = 8

    base32 = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "j",
        "k",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    binary = [
        np.asarray(list("{0:05b}".format(x, "b")), dtype=int)
        for x in range(0, len(base32))
    ]
    
    base32toBin = dict(zip(base32, binary))

    batch_size = batched_position_array.shape[0]
    max_traj_len = batched_position_array.shape[1]
    onehot_latlon_dim = precision * 5
    traj_actual_latlon = np.full((batch_size, max_traj_len, onehot_latlon_dim), 99)

    for i, latlon_array in enumerate(batched_position_array):
        for j, latlon in enumerate(latlon_array):
            if 99 in latlon:
                break

            geo_code = geohash.encode(latlon[0], latlon[1], precision)
            geo_code_array = np.concatenate([base32toBin[x] for x in geo_code])
            traj_actual_latlon[i, j] = geo_code_array
    
    traj_actual_latlon = torch.tensor(traj_actual_latlon).to(torch.float)

    return traj_actual_latlon

def reverse_geohash_onehot_to_actual(batched_onehot_position_array, precision=8):
    precision = 8

    base32 = [
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "b",
        "c",
        "d",
        "e",
        "f",
        "g",
        "h",
        "j",
        "k",
        "m",
        "n",
        "p",
        "q",
        "r",
        "s",
        "t",
        "u",
        "v",
        "w",
        "x",
        "y",
        "z",
    ]
    binary = [
        np.asarray(list("{0:05b}".format(x, "b")), dtype=int)
        for x in range(0, len(base32))
    ]
    binary_str = [
        "".join(map(str, one_hot_encoded_val.tolist()))
        for one_hot_encoded_val in binary
    ]
    Bintobase32 = dict(zip(binary_str, base32))

    batch_size = batched_onehot_position_array.size(0)
    max_traj_len = batched_onehot_position_array.size(1)
    actual_latlon_dim = 2
    traj_actual_latlon = np.full((batch_size, max_traj_len, actual_latlon_dim), 99)
    for i, latlon_onehot_array in enumerate(batched_onehot_position_array):
        for j, latlon_onehot in enumerate(latlon_onehot_array):
            latlon_onehot = list(map(int, latlon_onehot))

            if 99 in latlon_onehot:
                break

            geo_code_str = ""
            for p in range(precision):
                one_character_list = latlon_onehot[p * 5 : (p + 1) * 5]
                one_character_bin = "".join(map(str, one_character_list))
                geo_code_str += Bintobase32[one_character_bin]

            actual_latlon = list(geohash.decode(geo_code_str))
            traj_actual_latlon[i, j] = actual_latlon

    return traj_actual_latlon


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


def build_int_encoder(labels):
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)

    return label_encoder


def convert_label_to_inger(label_encoder, df):
    df_tmp = label_encoder.transform(df["label"])
    return df_tmp


def convert_onehot(df, encoder, col_name="day"):
    onehot_encoded = encoder.transform(df[[col_name]])
    onehot_df = pd.DataFrame(
        onehot_encoded, columns=encoder.get_feature_names_out([col_name])
    )
    onehot_df = onehot_df.astype(int)

    return onehot_df
