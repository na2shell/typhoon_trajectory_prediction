import numpy as np
from LSTM_model import LSTMTagger
import torch
from torch import nn

class SimpleFormula():
    def __init__(self, sin_a=2.0, cos_a = 2.0, sin_t=25.0, cos_t = 25.0):
        self.sin_a = sin_a
        self.cos_a = cos_a
        self.sin_t = sin_t
        self.cos_t = cos_t

    def sin(self, input):
        return self.sin_a * np.sin(2.0 * np.pi / self.sin_t * (input))

    def cos(self, input):
        return self.cos_a * np.cos(2.0 * np.pi / self.cos_t * (input))


simple_fom =  SimpleFormula()
sin_result = simple_fom.sin(input=2)

print(sin_result)

import numpy as np
origin_x =np.zeros((50, 20))
x =np.zeros((50, 20))
y =np.zeros((50, 1))

for i in range(50):
    for j in range(20):
        origin_x[i][j] = i+j
        x[i][j] = simple_fom.sin(input=origin_x[i][j])

    y[i] = simple_fom.sin(input=i+j+1)

print(origin_x)
print(y)

model = LSTMTagger(embedding_dim=1, hidden_dim=32, output_size=1)
model.train()

loss_fn = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epoch_num = 100

for _ in range(epoch_num):
    for i in range(30):
        data = x[i]
        actual_y = y[i]

        data = torch.from_numpy(data.astype(np.float32)).clone()
        data = data.view(20, 1)
        actual_y = torch.from_numpy(actual_y.astype(np.float32)).clone()
        out = model.forward(data)

        loss = loss_fn(out, actual_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

for i in range(31,50):
    data = x[i]
    actual_y = y[i]
    data = torch.from_numpy(data.astype(np.float32)).clone()
    data = data.view(20, 1)
    actual_y = torch.from_numpy(actual_y.astype(np.float32)).clone()

    pred = model.forward(data)
    print(pred, actual_y)