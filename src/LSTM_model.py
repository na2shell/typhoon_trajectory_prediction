# Author: Robert Guthrie

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_packed_sequence

torch.manual_seed(1)


class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        # self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_size)

        # # The linear layer that maps from hidden state space to tag space
        # self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sequence):
        # print("seq", sequence)
        lstm_out, _ = self.lstm(sequence)
        # print("lstm", lstm_out[-1].size())
        output = self.output_layer(lstm_out[-1])
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return output


class discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(discriminator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, last_position=-1):
        # print("x", x.size())
        x, (hn, cn) = self.lstm(x)
        seq_unpacked, lens_unpacked = pad_packed_sequence(x, batch_first=True)

        # print("lstm output", hn[0], seq_unpacked[0, 11, :], lens_unpacked[0])
        # print(x[:, -1, 1], hn.size(), hn)
        x = self.output_layer(hn.squeeze())
        return nn.Sigmoid()(x)


class generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(generator, self).__init__()
        lstm_input_dim = 43
        self.dense = nn.Linear(input_dim, lstm_input_dim)  # [B, L, C]
        self.lstm = nn.LSTM(lstm_input_dim, hidden_dim)

        self.fc_latlon = nn.Linear(hidden_dim, 2)
        self.fc_day = nn.Linear(hidden_dim, 7)
        self.fc_hour = nn.Linear(hidden_dim, 24)
        self.fc_category = nn.Linear(hidden_dim, 10)

        self.lat_center = 35
        self.lng_center = 135

    def forward(self, x):
        # print("input", x.size())

        # shape = x.shape # (B, L, C)
        # x = x.reshape(-1, shape[-1])

        x = self.dense(x)
        # print("dense", x.size())

        # x = x.reshape(shape[0], shape[1], -1) # reshaping
        # print("after dense size", x.size())

        x, _ = self.lstm(x)

        lat_lon = self.fc_latlon(x)
        day = self.fc_day(x)
        hour = self.fc_hour(x)
        category = self.fc_category(x)

        return lat_lon, day, hour, category
