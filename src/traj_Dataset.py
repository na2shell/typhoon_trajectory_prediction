from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from utils import convert_onehot, convert_label_to_inger, build_encoder, build_int_encoder, time_collate_fn
import geohash


class MyDataset(Dataset):
    def __init__(self, df, encoder_dict, int_label_encoder):
        super().__init__()
        self.if_geohash = True
        self.df = df

        df_one_hot_position = self.calculate_geohash_onehot()
        df_one_hot_position["tid"] = df["tid"]
        df_target = df_one_hot_position

        for col in ["day", "category", "hour"]:
            _encoder = encoder_dict[col]
            onehot_df = convert_onehot(df=df, encoder=_encoder, col_name=col)
            df_target = pd.concat([df_target, onehot_df], axis=1)

        self.df_target_class_indices = df[[
            "tid", "lat", "lon", "day", "category", "hour"]]
        self.df_target = df_target

        self.len = df_target["tid"].nunique()
        self.tids = df_target["tid"].unique()

        df["label_int"] = convert_label_to_inger(int_label_encoder, df)

        self.dict_tid_label = {}
        for tid in self.tids:
            label = df[df["tid"] == tid]["label_int"].head(1).values[0]
            self.dict_tid_label[tid] = label

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
        label = torch.tensor(self.dict_tid_label[tid])

        return traj, seq_len, traj_class_indices, label

    def calculate_geohash_onehot(self):
        precision = 8
        onehot_position_array = np.zeros((len(self.df), 5*precision))
        latlon_array = self.df[["lat", "lon"]].values

        base32 = ['0', '1', '2', '3', '4', '5', '6', '7',
                  '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',
                  'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r',
                  's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
        binary = [np.asarray(list('{0:05b}'.format(x, 'b')), dtype=int)
                  for x in range(0, len(base32))]
        base32toBin = dict(zip(base32, binary))

        for i, latlon in enumerate(latlon_array):
            geo_code = geohash.encode(latlon[0], latlon[1], precision)
            geo_code_array = np.concatenate([base32toBin[x] for x in geo_code])

            onehot_position_array[i] = geo_code_array

        print(onehot_position_array.shape)
        df_one_hot_position = pd.DataFrame(onehot_position_array, columns=[
                                           "position_onehot_{}".format(i) for i in range(5*precision)])

        return df_one_hot_position


if __name__ == "__main__":
    target_dict = {}
    target_dict["day"] = ([i for i in range(7)])
    target_dict["hour"] = ([i for i in range(24)])
    target_dict["category"] = ([i for i in range(10)])

    encoder_dict = {}

    for col in ["day", "category", "hour"]:
        target = target_dict[col]
        encoder_dict[col] = build_encoder(target)

    data_path = "./dev_train_encoded_final.csv"
    df = pd.read_csv(data_path)
    int_label_encoder = build_int_encoder(df["label"].unique())

    data_set = MyDataset(df=df, encoder_dict=encoder_dict,
                         int_label_encoder=int_label_encoder)
    traj, seq_len, traj_class_indices, label = data_set[0]
    print(label)

    BATCH_SIZE = 16

    train_dataloader = torch.utils.data.DataLoader(dataset=data_set,
                                                   batch_size=BATCH_SIZE,
                                                   shuffle=True,
                                                   collate_fn=time_collate_fn,
                                                   drop_last=True,
                                                   num_workers=2)
    
    for data, traj_len, traj_class_indices, label, mask in train_dataloader:
        print(mask.size())
        exit()

    data_path = "./dev_test_encoded_final.csv"
    df = pd.read_csv(data_path)
    data_set = MyDataset(df=df, encoder_dict=encoder_dict,
                         int_label_encoder=int_label_encoder)
    traj, seq_len, traj_class_indices, label = data_set[0]
    print(label)
