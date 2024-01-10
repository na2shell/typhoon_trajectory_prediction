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
import numpy as np
import geohash

torch.manual_seed(1)


class discriminator(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(discriminator, self).__init__()
        self.precision = 8
        each_hidden_dim = 128

        self.postion_last_index = 5 * self.precision
        self.day_last_index = self.postion_last_index + 7
        self.category_last_index = self.day_last_index + 10

        self.latlon_embbedding = nn.Linear(5 * self.precision, each_hidden_dim)
        self.day_embbedding = nn.Linear(7, each_hidden_dim)
        self.hour_embbedding = nn.Linear(24, each_hidden_dim)
        self.category_embbedding = nn.Linear(10, each_hidden_dim)

        self.dense = nn.Linear(each_hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=each_hidden_dim * 4, nhead=4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.output_layer = nn.Linear(each_hidden_dim * 4, output_dim)

    def forward(self, x, mask):
        latlon_e = self.latlon_embbedding(x[:, :, : self.postion_last_index])
        day_e = self.day_embbedding(
            x[:, :, self.postion_last_index : self.postion_last_index + 7]
        )
        category_e = self.category_embbedding(
            x[:, :, self.day_last_index : self.day_last_index + 10]
        )
        hour_e = self.hour_embbedding(x[:, :, self.category_last_index :])

        embedding_all = torch.cat([latlon_e, day_e, category_e, hour_e], dim=2)

        x = self.transformer_encoder(embedding_all, src_key_padding_mask=mask)
        x = torch.mean(x, dim=1)
        x = self.output_layer(x)
        x = nn.Sigmoid()(x)
        return x


class generator_encoder(nn.Module):
    def __init__(self, hidden_dim):
        super(generator_encoder, self).__init__()

        self.precision = 8
        each_hidden_dim = 128

        self.postion_last_index = 5 * self.precision
        self.day_last_index = self.postion_last_index + 7
        self.category_last_index = self.day_last_index + 10

        self.latlon_embbedding = nn.Linear(5 * self.precision, each_hidden_dim)
        self.day_embbedding = nn.Linear(7, each_hidden_dim)
        self.hour_embbedding = nn.Linear(24, each_hidden_dim)
        self.category_embbedding = nn.Linear(10, each_hidden_dim)

        self.dense = nn.Linear(each_hidden_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=each_hidden_dim * 4, nhead=4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

    def forward(self, x, mask):
        latlon_e = self.latlon_embbedding(x[:, :, : self.postion_last_index])
        day_e = self.day_embbedding(
            x[:, :, self.postion_last_index : self.postion_last_index + 7]
        )
        category_e = self.category_embbedding(
            x[:, :, self.day_last_index : self.day_last_index + 10]
        )
        hour_e = self.hour_embbedding(x[:, :, self.category_last_index :])

        embedding_all = torch.cat([latlon_e, day_e, category_e, hour_e], dim=2)

        x = self.transformer_encoder(embedding_all, src_key_padding_mask=mask)

        return x  # [B, L, H], H is hiden


class generator_decoder(nn.Module):
    def __init__(self):
        super(generator_decoder, self).__init__()
        each_hidden_dim = 128
        hidden_dim = each_hidden_dim * 4

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=each_hidden_dim * 4, nhead=4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        self.fc_latlon_geohash = nn.Linear(hidden_dim, 8 * 5)
        self.fc_latlon = nn.Linear(hidden_dim, 2)
        self.fc_day = nn.Linear(hidden_dim, 7)
        self.fc_hour = nn.Linear(hidden_dim, 24)
        self.fc_category = nn.Linear(hidden_dim, 10)

    def forward(self, x, mask):
        x = self.transformer_encoder(x, src_key_padding_mask=mask)

        lat_lon_geohash = self.fc_latlon_geohash(x)
        lat_lon_geohash = nn.Sigmoid()(lat_lon_geohash)

        lat_lon = self.fc_latlon(x)
        day = self.fc_day(x)
        hour = self.fc_hour(x)
        category = self.fc_category(x)

        return lat_lon_geohash, lat_lon, day, hour, category


class generator(nn.Module):
    def __init__(self, encoder, decoder):
        super(generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, mask):
        h = self.encoder(x, mask)
        lat_lon_geohash, lat_lon, day, hour, category = self.decoder(h, mask)

        return lat_lon_geohash, lat_lon, day, hour, category
