import torch
import torch.nn as nn


class TUL_attack_model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(TUL_attack_model, self).__init__()
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

        self.lstm = nn.LSTM(each_hidden_dim * 4, hidden_dim, batch_first=False)

        self.output_layer = nn.Linear(each_hidden_dim * 4, output_dim)
        self.soft_max = nn.Softmax(dim=1)

    def forward(self, x, traj_len, mask):
        latlon_e = self.latlon_embbedding(x[:, :, : self.postion_last_index])

        day_e = self.day_embbedding(
            x[:, :, self.postion_last_index : self.postion_last_index + 7]
        )
        category_e = self.category_embbedding(
            x[:, :, self.day_last_index : self.day_last_index + 10]
        )
        hour_e = self.hour_embbedding(x[:, :, self.category_last_index :])

        embedding_all = torch.cat([latlon_e, day_e, category_e, hour_e], dim=2)

        B, L, C = embedding_all.size(0), embedding_all.size(1), embedding_all.size(2)

        x = self.transformer_encoder(embedding_all, src_key_padding_mask=mask)

        x = torch.mean(x, dim=1)

        x = self.output_layer(x)

        x = self.soft_max(x)
        return x

    def encode(self, x, mask):
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

        return x
