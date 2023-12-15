from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import torch

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

class MyDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        self.data_path = data_path
        df = pd.read_csv(self.data_path)

        target_dict = {}
        target_dict["day"] = ([i for i in range(7)])
        target_dict["hour"] = ([i for i in range(24)])
        target_dict["category"] = ([i for i in range(10)])

        encoder_dict = {}

        for col in ["day", "category", "hour"]:
            target = target_dict[col]
            encoder_dict[col] = build_encoder(target)

        df_target = df[["tid", "lat", "lon"]]
        for col in ["day", "category", "hour"]:
            _encoder = encoder_dict[col]
            onehot_df = convert_onehot(df=df, encoder=_encoder, col_name=col)
            df_target = pd.concat([df_target, onehot_df], axis=1)
        
        self.df_target_class_indices = df[["tid", "lat", "lon", "day", "category", "hour"]]
        self.df_target = df_target
        self.len = df_target["tid"].nunique()
        self.tids = df_target["tid"].unique()
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        tid = self.tids[index]
        df_tmp = self.df_target[self.df_target["tid"] == tid]
        df_tmp = df_tmp.drop("tid", axis=1)
        traj = torch.tensor(df_tmp.values, dtype=torch.float)
        seq_len = torch.tensor(traj.size(0))

        df_tmp = self.df_target_class_indices[self.df_target_class_indices["tid"] == tid]
        df_tmp = df_tmp.drop("tid", axis=1)
        traj_class_indices = torch.tensor(df_tmp.values, dtype=torch.float)

        return traj, seq_len, traj_class_indices


if __name__=="__main__":
    data_path = "./dev_train_encoded_final.csv"
    data_set = MyDataset(data_path=data_path)
    traj = data_set[0]