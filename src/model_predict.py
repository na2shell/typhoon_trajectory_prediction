import preprocessing as pp
from LSTM_model import LSTMTagger, generator, discriminator
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd

device = "cpu"
G = torch.load("./generator.pt")
G.eval()

traj_data,  seq_len_list = pp.main(device)

for i in range(20,24):
    data, seq_len = traj_data[i], seq_len_list[i]

    mean = 0
    std = 0.01
    noise = torch.tensor(np.random.normal(mean, std, data.size()), dtype=torch.float).to(device)
    noise_data = data + noise
    pred = G.forward(noise_data)

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