import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from traj_Dataset import MyDataset
from Tras_GAN_model import discriminator
from utils import (
    build_encoder,
    build_int_encoder,
    time_collate_fn,
)
from utils_for_seq2seq import (
    create_mask,
    mse_loss_with_mask,
    generate_square_subsequent_mask,
)
from GAN_seq2seq_model import Seq2SeqTransformer
import os

# from soft_dtw import SoftDTW
from soft_dtw_cuda import SoftDTW

import argparse

parser = argparse.ArgumentParser(description="")
parser.add_argument("--train", action="store_true")
parser.add_argument("--over_write", action="store_true")
parser.add_argument("--dataset_name", default="CREST")

args = parser.parse_args()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# DEVICE = "cpu"
dataset_name = args.dataset_name

g_lr = 1e-4
d_lr = 1e-5
epoch_num = 500
BATCH_SIZE = 256
train = args.train
is_overwrite = args.over_write
latlon_corr = 10
hour_corr = 1
day_corr = 1
category_corr = 0
bce_loss_corr = 1

bce_loss = nn.BCEWithLogitsLoss()
ce_loss = nn.CrossEntropyLoss(ignore_index=111)
DTW_criterion = SoftDTW(
    gamma=1.0, normalize=True, use_cuda=True
)  # just like nn.MSELoss()
mean = 0
std = 0


G = Seq2SeqTransformer(
    num_encoder_layers=2,
    num_decoder_layers=2,
    each_emb_size=64,
    nhead=2,
    DEVICE=DEVICE,
)
G = G.to(DEVICE)
D = discriminator(input_dim=43, hidden_dim=128, output_dim=1).to(DEVICE)

G_optimizer = optim.Adam(G.parameters(), lr=g_lr)
D_optimizer = optim.Adam(D.parameters(), lr=d_lr)

target_dict = {}
target_dict["day"] = [i for i in range(7)]
target_dict["hour"] = [i for i in range(24)]
target_dict["category"] = [i for i in range(10)]

encoder_dict = {}

for col in ["day", "category", "hour"]:
    target = target_dict[col]
    encoder_dict[col] = build_encoder(target)


# data_path = "./dev_train_encoded_final.csv"
data_path = "/data/original_pos_60min_thin_out.csv"
# data_path = "/data/original_pos_60min.csv"
df = pd.read_csv(data_path)
int_label_encoder = build_int_encoder(df["label"].unique())
print(int_label_encoder.classes_)

train_data_set = MyDataset(
    df=df, encoder_dict=encoder_dict, int_label_encoder=int_label_encoder
)


dataloader = torch.utils.data.DataLoader(
    dataset=train_data_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=time_collate_fn,
    drop_last=True,
    num_workers=8,
)

# def anytime_zero(x, y):
#     return 0

# def match_or_not(x, y):
#     if x == y:
#         return 1
#     else:
#         return 0

# dist_functions = [anytime_zero for _ in range(2)] + [match_or_not for _ in range(41)]
# thresholds = [0.5 for _ in range(43)]
# features = [i for i in range(43)]
# weights = torch.tensor([1 for _ in range(43)])

# Utility_function_class = MUITAS(dist_functions, thresholds, features, weights)


norm_type = 2  # L2 norm
grad_clip = 0.1  # thereshold

weight_save_dir = "/src/generator_weight/{}/".format(dataset_name)
os.makedirs(weight_save_dir, exist_ok=True)

if train:
    G.train()
    D.train()
    for epoch in range(epoch_num + 1):
        g_losses = []
        g_losses_latlon = []
        g_losses_bce = []
        g_losses_hour = []
        g_losses_day = []
        g_losses_category = []
        g_losses_dtw = []
        d_losses = []
        for data, traj_len, traj_class_indices, label, mask, tid in dataloader:
            # print("data", torch.isnan(data).any())
            data = data.to(DEVICE)
            mask = mask.to(DEVICE)

            tgt_input = data[:, :-1, :]
            _src_for_making_mask = data[:, :, 0]
            _tgt_input_for_making_mask = _src_for_making_mask[:, :-1]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                _src_for_making_mask, _tgt_input_for_making_mask, DEVICE=DEVICE
            )

            real_outputs = D.forward(data.to(DEVICE), mask.to(DEVICE))
            real_label = 0.9 * torch.ones(BATCH_SIZE, 1).to(DEVICE)

            lat_lon, day, hour, category = G(
                data,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                None,
            )

            hour = hour.round().to(torch.int64)
            B, L, _ = hour.size()
            onehot = torch.LongTensor(B, L, 24).zero_().to(DEVICE)
            hour = torch.where(hour > 23, 23, hour)
            hour = torch.where(hour < 0, 0, hour)
            hour_one_hot = onehot.scatter_(2, hour, 1)

            _fake_inputs = torch.cat([lat_lon, day, hour_one_hot, category], dim=-1)
            # print("fake input", torch.isnan(_fake_inputs).any())

            fake_out = D(_fake_inputs, src_padding_mask[:, 1:])

            fake_label = 0.1 * torch.ones(BATCH_SIZE, 1).to(DEVICE)

            outputs = torch.cat((real_outputs, fake_out), 0)
            targets = torch.cat((real_label, fake_label), 0)
            D_loss = bce_loss(outputs, targets)

            # D_loss_real = hinge_loss(real_outputs, BATCH_SIZE, "for_real", DEVICE)
            # D_loss_fake = hinge_loss(fake_out, BATCH_SIZE, "for_fake", DEVICE)
            # D_loss = D_loss_real + D_loss_fake

            # print("D_loss", D_loss)
            D_optimizer.zero_grad()
            D_loss.backward()

            # nn.utils.clip_grad_norm_(
            #     parameters=D.parameters(), max_norm=grad_clip, norm_type=norm_type
            # )

            D_optimizer.step()

            d_losses.append(D_loss.item())

            # --- Generator ------

            lat_lon, day, hour, category = G(
                data,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                None,
            )

            hour_int = hour.round().to(torch.int64)
            B, L, _ = hour_int.size()
            onehot = torch.LongTensor(B, L, 24).zero_().to(DEVICE)
            hour_int = torch.where(hour_int > 23, 23, hour_int)
            hour_int = torch.where(hour_int < 0, 0, hour_int)
            hour_one_hot = onehot.scatter_(2, hour_int, 1)

            fake_inputs = torch.cat([lat_lon, day, hour_one_hot, category], dim=-1)
            # fake_inputs = lat_lon

            # lat, lng, day, hour, category = G(noise_data)

            # packed_fake_inputs = torch.nn.utils.rnn.pack_padded_sequence(
            #     fake_inputs, batch_first=True, lengths=traj_len, enforce_sorted=False)
            fake_outputs = D(fake_inputs, src_padding_mask[:, 1:])

            fake_targets = 0.9 * torch.ones([BATCH_SIZE, 1]).to(DEVICE)

            # print(fake_outputs.size(), fake_targets.size(),  fake_targets)

            G_loss_GAN = bce_loss(fake_outputs, fake_targets)
            # G_loss_GAN = hinge_loss(fake_outputs, BATCH_SIZE, "gen", DEVICE)

            g_losses_bce.append(G_loss_GAN.item())

            G_loss_equation = torch.sqrt(
                mse_loss_with_mask(
                    fake_inputs[:, :, 0], data[:, 1:, 0], ignored_index=99
                )
                + mse_loss_with_mask(
                    fake_inputs[:, :, 1], data[:, 1:, 1], ignored_index=99
                )
            )

            fake_inputs = fake_inputs.to(DEVICE)
            traj_class_indices = traj_class_indices.to(DEVICE)

            # print("size", torch.flatten(traj_class_indices[:, 1:, 2]).size())

            day_loss = ce_loss(
                day.view(-1, 7),
                torch.flatten(traj_class_indices[:, 1:, 2]).to(torch.long),
            )
            category_loss = ce_loss(
                category.view(-1, 10),
                torch.flatten(traj_class_indices[:, 1:, 3]).to(torch.long),
            )

            hour_loss = torch.sqrt(
                mse_loss_with_mask(
                    torch.flatten(hour),
                    torch.flatten(traj_class_indices[:, 1:, 4]).to(torch.long),
                    ignored_index=111,
                )
            )

            # B = data.size(0)
            # ignored_index = 99
            # _data = data.detach().clone()
            # _lat_lon = lat_lon.detach().clone()
            # # print(_data.requires_grad)

            # for b in range(B):
            #     pad_indeies = torch.flatten(_data[b, :, 0]) == ignored_index
            #     if pad_indeies.any():
            #         pad_index = _data[b : b + 1, pad_indeies, :2].size(1)
            #         _data[b : b + 1, pad_indeies, :2] = _lat_lon[
            #             b : b + 1, -pad_index:, :
            #         ].detach()

            # dtw_loss = DTW_criterion(_data[:, 1:, :2], lat_lon)

            B = data.size(0)
            ignored_index = 99

            dtw_loss = torch.zeros(B)
            for b in range(B):
                pad_indeies = torch.flatten(data[b, :, 0]) == ignored_index
                pad_index = lat_lon.size(1)
                if pad_indeies.any():
                    pad_index = data[b : b + 1, ~pad_indeies, :2].size(1)

                dtw_loss[b] = DTW_criterion(
                    data[b : b + 1, :pad_index, :2],
                    lat_lon[b : b + 1, :pad_index, :],
                )

            # print(dtw_loss.grad_fn)
            G_dtw_loss = torch.mean(dtw_loss)

            # similarity = Utility_function_class.similarity(fake_inputs[:, :, :], data[:, 1:, :])
            # print(similarity)
            # exit()

            # hour_loss = ce_loss(
            #     fake_inputs[:, :, 19:].view(-1, 24),
            #     torch.flatten(traj_class_indices[:, 1:, 4]).to(torch.long),
            # )

            G_loss = (
                bce_loss_corr * G_loss_GAN
                + latlon_corr * G_loss_equation
                + day_loss
                + category_loss * category_corr
                + hour_corr * hour_loss
                + G_dtw_loss
            )

            if torch.isnan(fake_outputs).any():
                exit()

            # G_loss = category_loss
            # G_loss = latlon_corr * G_loss_equation
            # G_loss = G_loss_GAN

            # print("G_loss", G_loss)
            G_optimizer.zero_grad()
            G_loss.backward()

            # nn.utils.clip_grad_norm_(
            #     parameters=G.parameters(), max_norm=grad_clip, norm_type=norm_type
            # )

            G_optimizer.step()

            g_losses.append(G_loss.item())
            g_losses_latlon.append(G_loss_equation.item())
            g_losses_hour.append(hour_loss.item())
            g_losses_day.append(day_loss.item())
            g_losses_dtw.append(G_dtw_loss.item())
            g_losses_category.append(category_loss.item())

        print(
            "epoch: {} \n G loss sum: {:.3f} G bce loss: {:.3f} "
            "G lat lon loss: {:.3f} G hour loss: {:.3f} G day loss: {:.3f} G category loss: {:.3f} G dtw loss: {:.3f} D loss: {:.3f}".format(
                epoch,
                sum(g_losses) / len(g_losses),
                sum(g_losses_bce) / len(g_losses_bce),
                sum(g_losses_latlon) / len(g_losses_latlon),
                sum(g_losses_hour) / len(g_losses_hour),
                sum(g_losses_day) / len(g_losses_day),
                sum(g_losses_category) / len(g_losses_category),
                sum(g_losses_dtw) / len(g_losses_dtw),
                sum(d_losses) / len(d_losses),
            )
        )

        if epoch % 10 == 0 and epoch != 0 and is_overwrite:
            torch.save(
                G.state_dict(),
                weight_save_dir + "generator_{}_epoch.pt".format(epoch),
            )

    print(data[0, :, :])
    print(fake_inputs[0, :, :])
    print("finish learning")
    torch.save(G.state_dict(), "/src/generator_weight/generator_n.pt")


G_runtime = Seq2SeqTransformer(
    num_encoder_layers=2, num_decoder_layers=2, each_emb_size=64, nhead=2, DEVICE=DEVICE
)
G_runtime = G_runtime.to(DEVICE)
G_runtime.load_state_dict(torch.load(weight_save_dir + "generator_20_epoch.pt"))
G_runtime.eval()

params_G = G.state_dict()
params_G_runtime = G_runtime.state_dict()

if train:
    for key in params_G.keys():
        assert torch.equal(params_G[key], params_G_runtime[key])

    print("all model weight is same")

TEST_BATCH_SIZE = 1

test_dataloader = torch.utils.data.DataLoader(
    dataset=train_data_set,
    batch_size=TEST_BATCH_SIZE,
    shuffle=True,
    collate_fn=time_collate_fn,
    drop_last=True,
    num_workers=2,
)

for data, traj_len, traj_class_indices, label, mask, tid in test_dataloader:
    data = data.to(DEVICE)
    mask = mask.to(DEVICE)

    tgt_input = data[:, :-1, :]
    _src_for_making_mask = data[:, :, 0]
    _tgt_input_for_making_mask = _src_for_making_mask[:, :-1]

    src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
        _src_for_making_mask, _tgt_input_for_making_mask, DEVICE=DEVICE
    )

    # lat_lon, day, hour, category = G_runtime(
    #     data,
    #     tgt_input,
    #     src_mask,
    #     tgt_mask,
    #     src_padding_mask,
    #     tgt_padding_mask,
    #     None,
    # )

    # print(data[:, :, :2], tgt_input)
    # lat_lon = lat_lon.squeeze()
    # print(lat_lon)
    # break

    memory = G_runtime.encode(data, src_mask)
    max_len = data.size(1)

    gen_traj_emb = G_runtime.decode(tgt_input, memory, tgt_mask)
    lat_lon, day, hour, category = G_runtime.re_converter(gen_traj_emb)
    lat_lon = lat_lon.squeeze()
    print(lat_lon)
    break

    ys = data[:, :2, :]
    for i in range(max_len - 1):
        memory = memory.to(DEVICE)
        tgt_mask = (
            generate_square_subsequent_mask(ys.size(1), DEVICE).type(torch.bool)
        ).to(DEVICE)
        tgt_mask = None

        next_point_emb = G_runtime.decode(ys, memory, tgt_mask)
        print("out emb", next_point_emb[:, :, :2])
        lat_lon, day, hour, category = G_runtime.re_converter(next_point_emb[:, -1:, :])

        next_point = torch.cat([lat_lon, day, hour, category], dim=-1)
        ys = torch.cat([ys, next_point], dim=1)

    print(data[:, :, :2], ys[:, :, :2])
    print("memory", memory[:, :, :2])

    lat_lon = ys[:, :, :2].squeeze()
    break

# for i in range(20, 23):
#     data, traj_len, traj_class_indices, label = train_data_set[i]

#     data = data.view(1, data.size(0), data.size(1)).to(DEVICE)

#     tgt_input = data[:, :-1, :]
#     _src_for_making_mask = data[:, :, 0]
#     _tgt_input_for_making_mask = _src_for_making_mask[:, :-1]

#     src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
#         _src_for_making_mask, _tgt_input_for_making_mask, DEVICE=DEVICE
#     )

#     lat_lon, day, hour, category = G_runtime(
#         data,
#         tgt_input,
#         src_mask,
#         tgt_mask,
#         src_padding_mask,
#         tgt_padding_mask,
#         src_padding_mask,
#     )
#     print(data[:, :, :2], tgt_input)
#     print(lat_lon)


traj = data.squeeze()

pred_y = lat_lon.to("cpu").detach().numpy().copy()
data_y = traj.to("cpu").detach().numpy().copy()

df_pred = pd.DataFrame(pred_y)
df_data = pd.DataFrame(data_y)

print(df_data)
print(df_pred)

fig, ax = plt.subplots()
plt.plot(df_pred.iloc[:, 0], df_pred.iloc[:, 1], color="red", label="generate")
plt.plot(df_data.iloc[:, 0], df_data.iloc[:, 1], color="blue", label="real")
# plt.plot(data_y[0], data_y[1], color="blue")
# plt.plot(left, data_y, color="blue")
plt.legend()
plt.savefig("hoge.pdf")
