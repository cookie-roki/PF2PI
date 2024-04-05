# --------------------------------------------------------
# part of codes borrowed from Quert2Label
# --------------------------------------------------------

import torch
import torch.nn as nn
import math

from multihead_attention import _get_activation_fn


class GroupWiseLinear(nn.Module):
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x


class FC_Decoder(nn.Module):
    def __init__(self, num_class, dim_feedforward, activation, dropout):
        super().__init__()
        self.num_class = num_class

        self.output_layer1 = nn.Linear(dim_feedforward * 2, dim_feedforward // 2)
        self.activation1 = torch.nn.Sigmoid()
        self.dropout1 = torch.nn.Dropout(dropout)

        self.output_layer3 = nn.Linear(dim_feedforward // 2, num_class)

    def forward(self, hs):
        hs = hs.permute(1, 0, 2)
        hs = hs.flatten(1)

        hs = self.output_layer1(hs)
        hs = self.activation1(hs)
        hs = self.dropout1(hs)

        out = self.output_layer3(hs)
        return out


class Predictor(nn.Module):
    def __init__(self, pre_model, num_class, dim_feedforward, activation, dropout):
        super().__init__()
        self.pre_model = pre_model
        self.fc_decoder = FC_Decoder(
            num_class=num_class,
            dim_feedforward=dim_feedforward,
            activation=activation,
            dropout=dropout
        )

    def forward(self, src):
        rec, hs = self.pre_model(src)
        out = self.fc_decoder(hs)
        return rec, out


def build_predictor(pre_model, args):
    predictor = Predictor(
        pre_model=pre_model,
        num_class=args.num_class,
        dim_feedforward=args.dim_feedforward,
        activation=_get_activation_fn(args.activation),
        dropout=args.dropout
    )

    return predictor
