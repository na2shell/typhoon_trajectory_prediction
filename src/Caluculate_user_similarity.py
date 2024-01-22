import torch
import torch.nn as nn
from traj_Dataset import MyDataset
from utils import time_collate_fn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from utils import build_encoder, build_int_encoder
import pandas as pd
from TUL_model import TUL_attack_model
import numpy as np




BATCH_SIZE = 1
k = 3
is_caluculate_latent_feature = False


def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


target_dict = {}
target_dict["day"] = [i for i in range(7)]
target_dict["hour"] = [i for i in range(24)]
target_dict["category"] = [i for i in range(10)]

encoder_dict = {}

for col in ["day", "category", "hour"]:
    target = target_dict[col]
    encoder_dict[col] = build_encoder(target)

data_path = "./dev_train_encoded_final.csv"
df = pd.read_csv(data_path)
int_label_encoder = build_int_encoder(df["label"].unique())
print(int_label_encoder.classes_)

data_set = MyDataset(
    df=df,
    encoder_dict=encoder_dict,
    int_label_encoder=int_label_encoder,
    is_applied_geohash=True,
)


dataloader = torch.utils.data.DataLoader(
    dataset=data_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=time_collate_fn,
    drop_last=True,
    num_workers=2,
)


model = TUL_attack_model(input_dim=43, hidden_dim=128, output_dim=193)
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
model.to(DEVICE)

model.load_state_dict(
    torch.load("/src/TUL_model_weight/tul_model_weight_epoch_950.pth")
)
model.eval()

each_person_latent_space_dict = {}
each_person_latent_space_centeroid = {}
uid_set = set()

if is_caluculate_latent_feature:
    for data, traj_len, traj_class_indices, label, mask in dataloader:
        data = data.to(DEVICE)
        mask = mask.to(DEVICE)
        latent_feature = model.encode(data, mask)

        uid = label.item()
        uid_set.add(uid)
        memory_array = latent_feature.to("cpu").detach().numpy().copy()

        if uid in each_person_latent_space_dict:
            np.append(each_person_latent_space_dict[uid], memory_array)
        else:
            each_person_latent_space_dict[uid] = memory_array
        

    for uid, latent_features in each_person_latent_space_dict.items():
        average_feature = np.mean(latent_features, axis=0)
        print(average_feature.shape)
        each_person_latent_space_centeroid[uid] = average_feature.tolist()

    df = pd.DataFrame.from_dict(each_person_latent_space_centeroid)
    print(df.head())
    df.to_csv("user_center_point.csv", index=None)

similarity_user_dict = {}
similar_users = {}
df = pd.read_csv("user_center_point.csv")

for uid, vector_series in df.items():
    each_person_latent_space_centeroid[uid] = vector_series.values

for src_uid, src_val in each_person_latent_space_centeroid.items():
    similarity_user_dict[src_uid] = []
    for target_uid, target_val in each_person_latent_space_centeroid.items():
        similarity = cos_sim(src_val, target_val)
        similarity_user_dict[src_uid].append((target_uid, similarity))

    similarity_user_dict[src_uid].sort(key=lambda x: x[1], reverse=True)

for uid, user_list in similarity_user_dict.items():
    similar_users[uid] = [x[0] for x in user_list]

df = pd.DataFrame.from_dict(similar_users)
print(df.head())
df.to_csv("similar_user_data.csv", index=None)