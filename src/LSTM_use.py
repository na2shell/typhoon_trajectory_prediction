import preprocessing as pp
from LSTM_model import LSTMTagger, generator, discriminator
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cpu"
traj_data,  seq_len_list = pp.main(device)
print(device)
lr = 1e-4
loss = nn.BCELoss()
mse_loss = nn.MSELoss()


G = generator(input_dim=43, hidden_dim=32).to(device)
D = discriminator(input_dim=43, hidden_dim=32, output_dim=1).to(device)

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))


epoch_num = 400

for epoch in range(epoch_num):
    g_losses = []
    for i in range(30):
        data, seq_len = traj_data[i], seq_len_list[i]

        real_outputs = D.forward(data)
        real_label = torch.ones(1, 1).to(device)

        # Define the mean and standard deviation of the Gaussian noise
        mean = 0
        std = 1

        # Create a tensor of the same size as the original tensor with random noise
        noise = torch.tensor(np.random.normal(mean, std, data.size()), dtype=torch.float).to(device)
        noise_data = data + noise
        noise_data = data
        fake_in = G.forward(noise_data)
        fake_out = D.forward(fake_in.detach(), last_position=seq_len)
        fake_label = torch.zeros(1, 1).to(device)

        # print(real_outputs.size(), fake_out.size())
        outputs = torch.cat((real_outputs, fake_out), 0)
        targets = torch.cat((real_label, fake_label), 0).squeeze()

        # print(outputs.size(), targets.size())
        D_loss = loss(outputs, targets)
        # print("D_loss", D_loss)
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # --- Generator ------
        noise = torch.tensor(np.random.normal(mean, std, data.size()), dtype=torch.float).to(device)
        noise_data = data + noise
        noise_data = data

        fake_inputs = G(noise_data)
        # fake_outputs = D(fake_inputs, last_position=seq_len)
        # fake_targets = torch.ones([1, 1]).to(device)
        # fake_targets = fake_targets[0]

        # print(fake_outputs.size(), fake_targets.size(),  fake_targets)

        # G_loss = loss(fake_outputs, fake_targets)
        fake_inputs_T = fake_inputs[:seq_len].T
        data_T = data[:seq_len].T
        # print(fake_inputs_T.size())


        G_loss = torch.sqrt(mse_loss(fake_inputs_T[0], data_T[0]) + mse_loss(fake_inputs_T[1], data_T[1]))
        g_losses.append(G_loss)
        # print("G_loss", G_loss)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()
    
    print("epoch :", epoch, sum(g_losses)/len(g_losses))

print("finish learning")

G.eval()
torch.save(G, "./generator.pt")

for i in range(20,25):
    data, seq_len = traj_data[i], seq_len_list[i]
    
    pred = G.forward(data)

    pred, data = pred[:seq_len], data[:seq_len]
    print("pred", pred, "data", data)


pred_y = pred.to('cpu').detach().numpy().copy()
data_y = data.to('cpu').detach().numpy().copy()

df_pred = pd.DataFrame(pred_y)
df_data = pd.DataFrame(data_y)

# fig, ax = plt.subplots()
plt.plot(df_pred.iloc[:,0], df_pred.iloc[:,1], color="red", label="generate")
plt.plot(df_data.iloc[:,0], df_data.iloc[:,1], color="blue", label="real")
# plt.plot(data_y[0], data_y[1], color="blue")
# plt.plot(left, data_y, color="blue")
plt.legend()
plt.savefig("hoge.pdf")