from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer, LayerNorm
import math
from utils_for_seq2seq import PositionalEncoding
import torch.nn.functional as F

class Traj_Embedding(nn.Module):
    def __init__(self, emb_size):
        super(Traj_Embedding, self).__init__()

        self.latlon_embbedding = nn.Linear(2, emb_size)
        self.day_embbedding = nn.Linear(7, emb_size)
        self.hour_embbedding = nn.Linear(24, emb_size)
        self.category_embbedding = nn.Linear(10, emb_size)

        self.postion_last_index = 2
        self.day_last_index = self.postion_last_index + 7
        self.category_last_index = self.day_last_index + 10

    def forward(self, x):
        # print("input x size:", x.size())
        latlon_e = self.latlon_embbedding(x[:, :, : self.postion_last_index])
        day_e = self.day_embbedding(
            x[:, :, self.postion_last_index : self.postion_last_index + 7]
        )
        category_e = self.category_embbedding(
            x[:, :, self.day_last_index : self.day_last_index + 10]
        )
        hour_e = self.hour_embbedding(x[:, :, self.category_last_index :])

        embedding_all = torch.cat([latlon_e, day_e, category_e, hour_e], dim=2)

        return embedding_all


class re_converter(nn.Module):
    def __init__(self, emb_size):
        super(re_converter, self).__init__()
        self.fc_latlon = nn.Linear(emb_size, 2)
        self.fc_day = nn.Linear(emb_size, 7)
        self.fc_hour = nn.Linear(emb_size, 1)
        self.fc_category = nn.Linear(emb_size, 10)

    def forward(self, outs):
        lat_lon = self.fc_latlon(outs)
        day = self.fc_day(outs)
        hour = F.relu(self.fc_hour(outs))
        category = self.fc_category(outs)

        return lat_lon, day, hour, category


class My_encoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, DEVICE):
        super(My_encoder, self).__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, batch_first=True)
        encoder_norm = LayerNorm(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        self.DEVICE = DEVICE

    def forward(self, src, mask=None, src_key_padding_mask=None, is_causal=None):
        memory = self.encoder(src, mask, src_key_padding_mask, is_causal)
        noise = torch.randn(memory.size()).to(self.DEVICE)
        _memory = memory + noise

        x = torch.mean(memory, dim=1)
        x = x.view(x.size(0), 1, x.size(1))

        return _memory


class Seq2SeqTransformer(nn.Module):
    def __init__(
        self,
        num_encoder_layers: int,
        num_decoder_layers: int,
        each_emb_size: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        DEVICE: str = "cpu"
    ):
        super(Seq2SeqTransformer, self).__init__()
        emb_size = 4 * each_emb_size

        noised_encoder = My_encoder(
            d_model=emb_size, nhead=nhead, num_encoder_layers=num_encoder_layers, DEVICE=DEVICE
        )
        self.transformer = Transformer(
            d_model=emb_size,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            custom_encoder=noised_encoder,
        )

        self.src_emb = Traj_Embedding(emb_size=each_emb_size)
        self.tgt_emb = Traj_Embedding(emb_size=each_emb_size)

        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)
        self.re_converter = re_converter(emb_size=emb_size)

    def forward(
        self,
        src: Tensor,
        trg: Tensor,
        src_mask: Tensor,
        tgt_mask: Tensor,
        src_padding_mask: Tensor,
        tgt_padding_mask: Tensor,
        memory_key_padding_mask: Tensor,
    ):
        src_emb = self.positional_encoding(self.src_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_emb(trg))

        outs = self.transformer(
            src_emb,
            tgt_emb,
            src_mask,
            tgt_mask,
            None,
            src_padding_mask,
            tgt_padding_mask,
            None,
        )

        # reverse => day, latlon, category, hour

        lat_lon, day, hour, category = self.re_converter(outs)

        return lat_lon, day, hour, category

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(
            self.positional_encoding(self.src_emb(src)), src_mask
        )

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        # print("decode seq input", tgt[0, :, :2])
        return self.transformer.decoder(
            self.positional_encoding(self.tgt_emb(tgt)), memory, tgt_mask
        )
