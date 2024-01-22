import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from traj_Dataset import MyDataset
from utils import (
    convert_onehot,
    convert_label_to_inger,
    build_encoder,
    build_int_encoder,
    time_collate_fn,
)
from utils_for_seq2seq import (
    create_mask,
    mse_loss_with_mask,
    generate_square_subsequent_mask,
    hinge_loss,
)
from GAN_seq2seq_model import Seq2SeqTransformer
import random
import tqdm

TEST_BATCH_SIZE = 1
is_check = False
k = 2
_k = k - 1
target_dict = {}
target_dict["day"] = [i for i in range(7)]
target_dict["hour"] = [i for i in range(24)]
target_dict["category"] = [i for i in range(10)]

encoder_dict = {}


def k_person_choose(src_uid, uid_list, k, method="random"):
    if method == "random":
        k_uid = random.sample(uid_list, k)
        return k_uid

    df = pd.read_csv("similar_user_data.csv")
    # print(df.head())
    similar_users = df[str(src_uid)].values
    similar_users_k = similar_users[1 : k + 1]
    return similar_users_k


for col in ["day", "category", "hour"]:
    target = target_dict[col]
    encoder_dict[col] = build_encoder(target)

data_path = "./dev_train_encoded_final.csv"
df = pd.read_csv(data_path)
int_label_encoder = build_int_encoder(df["label"].unique())
print(int_label_encoder.classes_)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

G_runtime = Seq2SeqTransformer(
    num_encoder_layers=2, num_decoder_layers=2, each_emb_size=64, nhead=2, DEVICE=DEVICE
)
G_runtime = G_runtime.to(DEVICE)
G_runtime.load_state_dict(torch.load("/src/generator_weight/generator_500_epoch.pt"))
G_runtime.eval()

train_data_set = MyDataset(
    df=df, encoder_dict=encoder_dict, int_label_encoder=int_label_encoder
)

test_dataloader = torch.utils.data.DataLoader(
    dataset=train_data_set,
    batch_size=TEST_BATCH_SIZE,
    shuffle=True,
    collate_fn=time_collate_fn,
    drop_last=True,
    num_workers=2,
)

each_person_latent_space_dict = {}
uid_set = set()
for data, traj_len, traj_class_indices, label, mask in test_dataloader:
    data = data.to(DEVICE)
    mask = mask.to(DEVICE)

    tgt_input = data[:, :-1, :]
    _src_for_making_mask = data[:, :, 0]
    _tgt_input_for_making_mask = _src_for_making_mask[:, :-1]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
        _src_for_making_mask, _tgt_input_for_making_mask, DEVICE=DEVICE
    )

    memory = G_runtime.encode(data, src_mask)

    uid = label.item()
    uid_set.add(uid)
    memory_array = memory.to("cpu").detach().numpy().copy()
    if uid in each_person_latent_space_dict:
        np.append(each_person_latent_space_dict[uid], memory_array)
    else:
        each_person_latent_space_dict[uid] = memory_array

    if len(uid_set) and is_check > 10:
        break

uid_list = list(uid_set)

df_result = pd.DataFrame()
for i, (data, traj_len, traj_class_indices, label, mask) in enumerate(tqdm.tqdm(test_dataloader)):
    tmp_df = pd.DataFrame()

    data = data.to(DEVICE)
    mask = mask.to(DEVICE)

    tgt_input = data[:, :-1, :]
    _src_for_making_mask = data[:, :, 0]
    _tgt_input_for_making_mask = _src_for_making_mask[:, :-1]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
        _src_for_making_mask, _tgt_input_for_making_mask, DEVICE=DEVICE
    )

    memory = G_runtime.encode(data, src_mask)
    uid = label.item()

    mixed_memory = memory
    if _k != 0:
        uids = k_person_choose(uid, uid_list, _k, method="latent")
        for uid in uids:
            another_memory = (
                torch.tensor(each_person_latent_space_dict[uid][0])
                .view(1, 1, 256)
                .to(DEVICE)
            )
            mixed_memory += another_memory

        mixed_memory /= _k

    gen_traj_emb = G_runtime.decode(tgt_input, mixed_memory, tgt_mask)
    lat_lon, day, hour, category = G_runtime.re_converter(gen_traj_emb)
    lat_lon = lat_lon.squeeze().to("cpu").detach().numpy().copy()
    day = torch.argmax(day, dim=2).squeeze().to("cpu").detach().numpy().copy()

    hour = torch.where(hour > 23, 23, hour)
    hour = hour.round().to(torch.int64).squeeze().to("cpu").detach().numpy().copy()
    category = torch.argmax(category, dim=2).squeeze().to("cpu").detach().numpy().copy()

    tmp_df[["lat", "lon"]] = lat_lon
    tmp_df["day"] = day
    tmp_df["hour"] = hour
    tmp_df["category"] = category
    tmp_df["uid"] = uid
    tmp_df["tid"] = i

    df_result = pd.concat([df_result, tmp_df], axis=0)
    if i > 10 and is_check:
        break


df_result.to_csv("k-same-net_generated_traj_k={}.csv".format(k), index=None)


traj = data.squeeze()

pred_y = lat_lon
data_y = traj.to("cpu").detach().numpy().copy()

df_pred = pd.DataFrame(pred_y)
df_data = pd.DataFrame(data_y)

print(df_data)
print(df_pred)

fig, ax = plt.subplots()
plt.plot(df_pred.iloc[:, 0], df_pred.iloc[:, 1], color="red", label="generate")
plt.plot(df_data.iloc[:, 0], df_data.iloc[:, 1], color="blue", label="real")
plt.legend()
plt.savefig("hoge.pdf")
