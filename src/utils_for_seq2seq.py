import torch
from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import numpy as np


def generate_square_subsequent_mask(sz, DEVICE):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def create_mask(src, tgt, DEVICE):
    PAD_IDX = 99
    src_seq_len = src.size(1)
    tgt_seq_len = tgt.size(1)

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, DEVICE=DEVICE)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    # src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    # tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)

    src_padding_mask = (src == PAD_IDX).type(torch.bool)
    tgt_padding_mask = tgt == PAD_IDX

    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(1, max_len, d_model)  # [batch_size, seq_len, embedding_dim]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


def mse_loss_with_mask(input, target, ignored_index, reduction="mean"):
    mask = target == ignored_index
    out = (input[~mask] - target[~mask]) ** 2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out


def hinge_loss(logits, batch_size, condition, DEVICE):
    if condition == "gen":
        return -torch.mean(logits)
    elif condition == "for_real":
        minval = torch.min(logits - 1, torch.zeros(batch_size).to(DEVICE))
        return -torch.mean(minval)
    else:
        minval = torch.min(-logits - 1, torch.zeros(batch_size).to(DEVICE))
        return -torch.mean(minval)


class MUITAS:
    """Multiple-Aspect Trajectory Similarity Measure.

    Parameters
    ----------
    dist_functions : array-like, shape (n_features)
        Specifies the distance functions used for each trajectory
        attribute.
    thresholds : array-like, shape (n_features)
        Specifies the thresholds used for each trajectory attribute.
    features : list or array-like
        The groups of features (indices) to be considered for computing
        similarity (a list of arrays/lists). For each group, if at least one
        feature does not match, no score is assigned to the whole group of
        features.
    weights : array-like, shape (n_groups)
        Specifies the weight (importance) of each feature group.

    References
    ----------
    `Petry, L. M., Ferrero, C. A., Alvares, L. O., Renso, C., & Bogorny, V.
    (2019). Towards Semantic-Aware Multiple-Aspect Trajectory Similarity
    Measuring. Transactions in GIS (accepted), XX(X), XXX-XXX.
    <https://onlinelibrary.wiley.com/journal/14679671>`__
    """

    def __init__(self, dist_functions, thresholds, features, weights):
        self.dist_functions = dist_functions
        self.thresholds = thresholds
        self.features = np.array([[f] for f in features])

        if torch.is_tensor(weights):
            weights_sum = weights.sum()
        else:
            weights_sum = sum(weights)

        self.weights = weights.div(weights_sum)

    def similarity(self, t1, t2):
        # t1 shape: B, L, F
        batched_similarity = torch.zeros((t1.size(0),))
        for b in range(t1.size(0)):
            _t1 = t1[b]
            _t2 = t2[b]
            matrix = torch.zeros((t1.size(1), t2.size(1)))

            for i, p1 in enumerate(_t1):
                # print(p1.size(), _t2.size())
                matrix[i] = torch.tensor([self._score(p1, p2) for p2 in _t2])

            parity1 = torch.sum(torch.max(matrix, dim=1)[0])
            parity2 = torch.sum(torch.max(matrix, dim=0)[0])
            sim = (parity1 + parity2) / (len(_t1) + len(_t2))
            batched_similarity[b] = sim

        average_similarity = torch.mean(batched_similarity)
        # print(average_similarity)

        return average_similarity

    def _score(self, p1, p2):
        matches = torch.zeros((p1.size(0),))
        for i, _ in enumerate(p1):
            matches[i] = self.dist_functions[i](p1[i], p2[i]) <= self.thresholds[i]

        groups = torch.tensor([int(torch.all(matches[g])) for g in self.features])
        return torch.sum(groups * self.weights)
