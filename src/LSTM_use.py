import preprocessing as pp
from LSTM_model import LSTMTagger, generator, discriminator
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
traj_data,  seq_len_list = pp.main(device)
print(device)
lr = 1e-4
loss = nn.BCELoss()

G = generator(input_dim=43, hidden_dim=32).to(device)
D = discriminator(input_dim=43, hidden_dim=32, output_dim=1).to(device)

G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))



epoch_num = 10

for _ in range(epoch_num):
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
        fake_in = G.forward(noise_data)
        fake_out = D.forward(fake_in, last_position=seq_len)
        fake_label = torch.zeros(1, 1).to(device)

        # print(real_outputs.size(), fake_out.size())
        outputs = torch.cat((real_outputs, fake_out), 0)
        targets = torch.cat((real_label, fake_label), 0).squeeze()

        print(outputs.size(), targets.size())
        D_loss = loss(outputs, targets)
        print("D_loss", D_loss)
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # --- Generator ------
        noise = torch.tensor(np.random.normal(mean, std, data.size()), dtype=torch.float).to(device)
        noise_data = data + noise

        fake_inputs = G(noise_data)
        fake_outputs = D(fake_inputs, last_position=seq_len)
        fake_targets = torch.ones([1, 1]).to(device)
        fake_targets = fake_targets[0]

        # print(fake_outputs.size(), fake_targets.size(),  fake_targets)
        G_loss = loss(fake_outputs, fake_targets)
        print("G_loss", G_loss)
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

exit(0)
G.eval()

for i in range(31,50):
    data = x[i]
    data = torch.from_numpy(data.astype(np.float32)).clone()
    data = data.view(20, 1).to(device)
    # actual_y = torch.from_numpy(actual_y.astype(np.float32)).clone()

    pred = G.forward(data)
    print("pred", pred, "data", data)
    data = data.view(20).to(device)



pred_y = pred.to('cpu').detach().numpy().copy()
data_y = data.to('cpu').detach().numpy().copy()
left = np.array([i for i in range(20)])
plt.plot(left, pred_y, color="red")
plt.plot(left, data_y, color="blue")
# plt.legend()
plt.savefig("hoge.pdf")