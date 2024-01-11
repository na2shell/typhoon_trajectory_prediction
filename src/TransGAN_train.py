from Tras_GAN_model import (
    generator,
    discriminator,
    generator_decoder,
    generator_encoder,
)
import numpy as np
import torch
from torch import nn
import torch.optim as optim
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
from utils_for_seq2seq import create_mask, mse_loss_with_mask
from GAN_seq2seq_model import Seq2SeqTransformer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g_lr = 1e-3
d_lr = 1e-4
epoch_num = 50
BATCH_SIZE = 128
train = True
latlon_corr = 100
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss(ignore_index=111)
mean = 0
std = 0

# mse_loss = nn.MSELoss()


ge = generator_encoder(hidden_dim=128)
gd = generator_decoder()

G = generator(encoder=ge, decoder=gd).to(DEVICE)
D = discriminator(input_dim=43, hidden_dim=128, output_dim=1).to(DEVICE)

G_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

epoch_num = 20

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

train_data_set = MyDataset(
    df=df, encoder_dict=encoder_dict, int_label_encoder=int_label_encoder
)


dataloader = torch.utils.data.DataLoader(
    dataset=train_data_set,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=time_collate_fn,
    drop_last=True,
    num_workers=2,
)


if train:
    for epoch in range(epoch_num):
        g_losses = []
        g_losses_bce = []
        d_losses = []
        for data, traj_len, traj_class_indices, label, mask in dataloader:
            data = data.to(DEVICE)
            mask = mask.to(DEVICE)

            tgt_input = data[:, :-1, :]
            _src_for_making_mask = data[:, :, 0]
            _tgt_input_for_making_mask = _src_for_making_mask[:, :-1]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
                _src_for_making_mask, _tgt_input_for_making_mask, DEVICE=DEVICE
            )

            G = Seq2SeqTransformer(
                num_encoder_layers=4, num_decoder_layers=4, each_emb_size=128, nhead=4
            )
            G = G.to(DEVICE)

            real_outputs = D.forward(data.to(DEVICE), mask.to(DEVICE))
            real_label = 0.9 * torch.ones(BATCH_SIZE, 1).to(DEVICE)

            noise = torch.tensor(
                np.random.normal(mean, std, data.size()), dtype=torch.float
            ).to(DEVICE)

            noise_data = data + noise
            # noise_data = data
            # fake_input = G(noise_data)

            print(
                data.device, tgt_input.device, src_mask.device, src_padding_mask.device
            )

            lat_lon, day, hour, category = G(
                data,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )

            print(lat_lon.size(), day.size(), hour.size(), category.size())
            # print(lat_lon)

            day_onehot = torch.nn.functional.one_hot(
                torch.argmax(day, dim=2), num_classes=7
            )
            hour_onehot = torch.nn.functional.one_hot(
                torch.argmax(hour, dim=2), num_classes=24
            )
            category_onehot = torch.nn.functional.one_hot(
                torch.argmax(category, dim=2), num_classes=10
            )

            _fake_inputs = torch.cat(
                [lat_lon, day_onehot, hour_onehot, category_onehot], dim=-1
            )

            # # fake_input = lat_lon

            # fake_input = fake_input.detach()
            # packed_fake_input = torch.nn.utils.rnn.pack_padded_sequence(
            #     fake_input, batch_first=True, lengths=traj_len, enforce_sorted=False)

            # # print(fake_input.size())

            # last_pos = data.size(1)

            fake_out = D(_fake_inputs, src_padding_mask[:, 1:])
            fake_label = torch.zeros(BATCH_SIZE, 1).to(DEVICE)

            # print(real_outputs.size(), fake_out.size())
            outputs = torch.cat((real_outputs, fake_out), 0)
            targets = torch.cat((real_label, fake_label), 0)

            # print(outputs.size(), targets.size())
            D_loss = bce_loss(outputs, targets)
            # print("D_loss", D_loss)
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            d_losses.append(D_loss.item())

            # --- Generator ------
            noise = torch.tensor(
                np.random.normal(mean, std, data.size()), dtype=torch.float
            ).to(DEVICE)
            noise_data = data + noise
            noise_data = data

            lat_lon, day, hour, category = G(
                data,
                tgt_input,
                src_mask,
                tgt_mask,
                src_padding_mask,
                tgt_padding_mask,
                src_padding_mask,
            )

            day_onehot = torch.nn.functional.one_hot(
                torch.argmax(day, dim=2), num_classes=7
            )
            hour_onehot = torch.nn.functional.one_hot(
                torch.argmax(hour, dim=2), num_classes=24
            )
            category_onehot = torch.nn.functional.one_hot(
                torch.argmax(category, dim=2), num_classes=10
            )

            fake_inputs = torch.cat(
                [lat_lon, day_onehot, hour_onehot, category_onehot], dim=-1
            )
            # fake_inputs = lat_lon

            # lat, lng, day, hour, category = G(noise_data)

            # packed_fake_inputs = torch.nn.utils.rnn.pack_padded_sequence(
            #     fake_inputs, batch_first=True, lengths=traj_len, enforce_sorted=False)
            fake_outputs = D(fake_inputs, src_padding_mask[:, 1:])

            fake_targets = 0.9 * torch.ones([BATCH_SIZE, 1]).to(DEVICE)

            # print(fake_outputs.size(), fake_targets.size(),  fake_targets)

            G_loss_GAN = bce_loss(fake_outputs, fake_targets)
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
                fake_inputs[:, :, 2:9].view(-1, 7),
                torch.flatten(traj_class_indices[:, 1:, 2]).to(torch.long),
            )
            category_loss = ce_loss(
                fake_inputs[:, :, 9:19].view(-1, 10),
                torch.flatten(traj_class_indices[:, 1:, 3]).to(torch.long),
            )
            hour_loss = ce_loss(
                fake_inputs[:, :, 19:].view(-1, 24),
                torch.flatten(traj_class_indices[:, 1:, 4]).to(torch.long),
            )

            G_loss = (
                G_loss_GAN
                + latlon_corr * G_loss_equation
                + day_loss
                + category_loss
                + hour_loss
            )

            # print("G_loss", G_loss)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            g_losses.append(G_loss.item())

        print(
            "epoch: {} \n G loss sum: {:.3f} G bce loss: {:.3f} D loss: {:.3f}".format(
                epoch,
                sum(g_losses) / len(g_losses),
                sum(g_losses_bce) / len(g_losses_bce),
                sum(d_losses) / len(d_losses),
            )
        )

    print("finish learning")
    torch.save(G.state_dict(), "generator.pt")


G.load_state_dict(torch.load("generator.pt"))
G.eval()

for i in range(20, 23):
    data, traj_len, traj_class_indices, label = train_data_set[i]

    data = data.to(DEVICE)
    noise = torch.tensor(
        np.random.normal(mean, std, data.size()), dtype=torch.float
    ).to(DEVICE)
    noised_traj = data + noise
    noised_traj = noised_traj.view(1, noised_traj.size(0), noised_traj.size(1))
    mask = torch.zeros(noised_traj.size(0), noised_traj.size(1)).to(DEVICE)
    lat_lon, day, hour, category = G.forward(noised_traj, mask)


lat_lon = lat_lon.squeeze()
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
