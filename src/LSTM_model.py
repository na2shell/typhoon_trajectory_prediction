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
        x, _ = self.lstm(x)
        x = self.output_layer(x[last_position])
        return nn.Sigmoid()(x)


class generator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(generator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, 43)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc1(x)
        return x