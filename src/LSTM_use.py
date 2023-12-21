import preprocessing as pp
from LSTM_model import LSTMTagger, generator, discriminator
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
from traj_Dataset import MyDataset

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
traj_data, seq_len_list = pp.main(device)
print(device)
g_lr = 1e-3
d_lr = 1e-4
epoch_num = 50
BATCH_SIZE = 128
train = False
latlon_corr = 100
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss(ignore_index=111)
mean = 0
std = 0.01

# mse_loss = nn.MSELoss()


def mse_loss_with_mask(input, target, ignored_index, reduction="mean"):
    mask = target == ignored_index
    out = (input[~mask]-target[~mask])**2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out


G = generator(input_dim=43, hidden_dim=128).to(device)
D = discriminator(input_dim=43, hidden_dim=128, output_dim=1).to(device)

G_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

epoch_num = 1

def time_collate_fn(batch):
    traj, seq_len, traj_class_indices = list(zip(*batch))
    x = torch.nn.utils.rnn.pad_sequence(
        traj, batch_first=True, padding_value=99)
    traj_class_indices = torch.nn.utils.rnn.pad_sequence(
        traj_class_indices, batch_first=True, padding_value=111)
    seq_len = torch.stack(seq_len)

    return x, seq_len, traj_class_indices


data_path = "./dev_train_encoded_final.csv"
data_set = MyDataset(data_path=data_path)


dataloader = torch.utils.data.DataLoader(dataset=data_set,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         collate_fn=time_collate_fn,
                                         drop_last=True,
                                         num_workers=2)

# Define the mean and standard deviation of the Gaussian noise


# Create a tensor of the same size as the original tensor with random noise


if train:
    for epoch in range(epoch_num):
        g_losses = []
        g_losses_bce = []
        d_losses = []
        for data, traj_len, traj_class_indices in dataloader:

            # just for exp
            # data_for_D = data[:, :, :2]
            packed_data = torch.nn.utils.rnn.pack_padded_sequence(
                data, batch_first=True, lengths=traj_len, enforce_sorted=False)
            real_outputs = D.forward(packed_data)
            real_label = 0.9*torch.ones(BATCH_SIZE, 1).to(device)

            noise = torch.tensor(
                np.random.normal(mean, std, data.size()), dtype=torch.float
            ).to(device)

            noise_data = data + noise
            # noise_data = data
            # fake_input = G(noise_data)
            lat_lon, day, hour, category = G(noise_data)

            # print(lat_lon.size(), day.size(), hour.size(), category.size())

            day_onehot = torch.nn.functional.one_hot(
                torch.argmax(day, dim=2), num_classes=7)
            hour_onehot = torch.nn.functional.one_hot(
                torch.argmax(hour, dim=2), num_classes=24)
            category_onehot = torch.nn.functional.one_hot(
                torch.argmax(category, dim=2), num_classes=10)

            fake_input = torch.cat(
                [lat_lon, day_onehot, hour_onehot, category_onehot], dim=-1)
            # fake_input = lat_lon

            fake_input = fake_input.detach()
            packed_fake_input = torch.nn.utils.rnn.pack_padded_sequence(
                fake_input, batch_first=True, lengths=traj_len, enforce_sorted=False)

            # print(fake_input.size())

            last_pos = data.size(1)

            fake_out = D(packed_fake_input, last_position=last_pos)
            fake_label = torch.zeros(BATCH_SIZE, 1).to(device)

            # print(real_outputs.size(), fake_out.size())
            outputs = torch.cat((real_outputs, fake_out), 0)
            targets = torch.cat((real_label, fake_label), 0)

            # print(outputs.size(), targets.size())
            D_loss = bce_loss(outputs, targets)
            # print("D_loss", D_loss)
            D_optimizer.zero_grad()
            D_loss.backward()
            D_optimizer.step()

            d_losses.append(D_loss)

            # --- Generator ------
            noise = torch.tensor(
                np.random.normal(mean, std, data.size()), dtype=torch.float
            ).to(device)
            noise_data = data + noise
            noise_data = data

            lat_lon, day, hour, category = G(noise_data)

            day_onehot = torch.nn.functional.one_hot(
                torch.argmax(day, dim=2), num_classes=7)
            hour_onehot = torch.nn.functional.one_hot(
                torch.argmax(hour, dim=2), num_classes=24)
            category_onehot = torch.nn.functional.one_hot(
                torch.argmax(category, dim=2), num_classes=10)

            fake_inputs = torch.cat(
                [lat_lon, day_onehot, hour_onehot, category_onehot], dim=-1)
            # fake_inputs = lat_lon

            # lat, lng, day, hour, category = G(noise_data)

            packed_fake_inputs = torch.nn.utils.rnn.pack_padded_sequence(
                fake_inputs, batch_first=True, lengths=traj_len, enforce_sorted=False)
            fake_outputs = D(packed_fake_inputs, last_position=last_pos)

            fake_targets = 0.9*torch.ones([BATCH_SIZE, 1]).to(device)

            # print(fake_outputs.size(), fake_targets.size(),  fake_targets)

            G_loss_GAN = bce_loss(fake_outputs, fake_targets)
            g_losses_bce.append(G_loss_GAN)

            # print(traj_len, fake_inputs.size())
            fake_inputs_T = fake_inputs
            data_T = data
            # print(fake_inputs_T.size())

            # print("day size", fake_inputs[:,:,2:9].view(-1,7).size(), traj_class_indices[:,:,2].view(-1).size())

            G_loss_equation = torch.sqrt(
                mse_loss_with_mask(
                    fake_inputs[:, :, 0], data[:, :, 0], ignored_index=99)
                + mse_loss_with_mask(fake_inputs[:, :, 1], data[:, :, 1], ignored_index=99)
            )

            day_loss = ce_loss(
                fake_inputs[:, :, 2:9].view(-1, 7), traj_class_indices[:, :, 2].view(-1).to(torch.long))
            category_loss = ce_loss(
                fake_inputs[:, :, 9:19].view(-1, 10), traj_class_indices[:, :, 3].view(-1).to(torch.long))
            hour_loss = ce_loss(
                fake_inputs[:, :, 19:].view(-1, 24), traj_class_indices[:, :, 4].view(-1).to(torch.long))

            G_loss = G_loss_GAN + latlon_corr*G_loss_equation + \
                day_loss + category_loss + hour_loss

            # print("G_loss", G_loss)
            G_optimizer.zero_grad()
            G_loss.backward()
            G_optimizer.step()
            g_losses.append(G_loss)

        print("epoch :", epoch, sum(g_losses) / len(g_losses),
              sum(g_losses_bce) / len(g_losses_bce), sum(d_losses) / len(d_losses))

    print("finish learning")
    torch.save(G.state_dict(), 'generator.pt')


G.load_state_dict(torch.load('generator.pt'))
G.eval()

for i in range(20, 23):
    traj, seq_len, _ = data_set[i]
    print(traj.size())
    noise = torch.tensor(
        np.random.normal(mean, std, traj.size()), dtype=torch.float
    ).to(device)
    noised_traj = traj + noise
    noised_traj = noised_traj.view(1, noised_traj.size(0), noised_traj.size(1))
    lat_lon, day, hour, category = G.forward(noised_traj)
    print("pred", lat_lon, "data", traj)

lat_lon = lat_lon.squeeze()
traj = traj.squeeze()

pred_y = lat_lon.to("cpu").detach().numpy().copy()
data_y = traj.to("cpu").detach().numpy().copy()

df_pred = pd.DataFrame(pred_y)
df_data = pd.DataFrame(data_y)

fig, ax = plt.subplots()
plt.plot(df_pred.iloc[:, 0], df_pred.iloc[:, 1], color="red", label="generate")
plt.plot(df_data.iloc[:, 0], df_data.iloc[:, 1], color="blue", label="real")
# plt.plot(data_y[0], data_y[1], color="blue")
# plt.plot(left, data_y, color="blue")
plt.legend()
plt.savefig("hoge.pdf")
