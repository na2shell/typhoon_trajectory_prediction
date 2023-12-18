import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

def time_collate_fn(batch):
    traj, seq_len, traj_class_indices, labels = list(zip(*batch))
    x = torch.nn.utils.rnn.pad_sequence(
        traj, batch_first=True, padding_value=99)
    traj_class_indices = torch.nn.utils.rnn.pad_sequence(
        traj_class_indices, batch_first=True, padding_value=111)
    seq_len = torch.stack(seq_len)
    labels = torch.stack(labels)
    mask = x[:, :, 1] == 99
    
    return x, seq_len, traj_class_indices, labels, mask



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
        onehot_encoded, columns=encoder.get_feature_names_out([col_name]))
    onehot_df = onehot_df.astype(int)

    return onehot_df