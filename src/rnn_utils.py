import torch
import numpy as np
from torch import nn
from typing import *
from enum import IntEnum
from torch.nn import Parameter
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RnnEncoder(nn.Module):
    def __init__(self, seq_len, input_dim, hiddens, dropout_rate=0.25, rnn_cell_type='GRU', use_input_attention=False,
                 normalize=False):
        """
        RNN structure (MLP+seq2seq) (\theta_1: RNN parameters).
        :param seq_len: trajectory length.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action).
        :param hiddens: hidden layer dimensions.
        :param dropout_rate: dropout rate.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param use_input_attention: Whether to use the input cell attention.
        :param normalize: whether to normalize the inputs.
        """
        super(RnnEncoder, self).__init__()
        self.normalize = normalize
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hiddens[-1]
        self.rnn_cell_type = rnn_cell_type
        self.mlp_encoder = nn.Sequential()
        for i in range(len(hiddens) - 1):
            if i == 0:
                self.mlp_encoder.add_module('mlp_%d' % i, nn.Linear(input_dim, hiddens[i]))
            else:
                self.mlp_encoder.add_module('mlp_%d' % i, nn.Linear(hiddens[i - 1], hiddens[i]))
            self.mlp_encoder.add_module('relu_%d' % i, nn.ReLU())
            self.mlp_encoder.add_module('dropout_%d' % i, nn.Dropout(dropout_rate))

        if self.rnn_cell_type == 'GRU':
            print('Using GRU as the recurrent layer.')
            if use_input_attention:
                print('Using the input cell attention for saliency methods to prevent gradient vanish/explosion.')
                self.rnn = GRUWithInputCellAttention(input_sz=hiddens[-2], hidden_sz=hiddens[-1])
            else:
                self.rnn = nn.GRU(input_size=hiddens[-2], hidden_size=hiddens[-1], batch_first=True)
        elif self.rnn_cell_type == 'LSTM':
            print('Using LSTM as the recurrent layer.')
            if use_input_attention:
                print('Using the input cell attention for saliency methods to prevent gradient vanish/explosion.')
                self.rnn = LSTMWithInputCellAttention(input_sz=hiddens[-2], hidden_sz=hiddens[-1])
            else:
                self.rnn = nn.LSTM(input_size=hiddens[-2], hidden_size=hiddens[-1], batch_first=True)
        else:
            print('Using the default recurrent layer: GRU.')
            if use_input_attention:
                print('Using the input cell attention for saliency methods to prevent gradient vanish/explosion.')
                self.rnn = GRUWithInputCellAttention(input_sz=hiddens[-2], hidden_sz=hiddens[-1])
            else:
                self.rnn = nn.GRU(input_size=hiddens[-2], hidden_size=hiddens[-1], batch_first=True)
            self.rnn_cell_type = 'GRU'

    def forward(self, x, h0=None, c0=None):
        # forward function: given an input, return the model output (output at each time and the final time step).
        """
        :param x: input trajectories (Batch_size, seq_len, input_dim).
        :param h0: Initial hidden state at time t_0 (Batch_size, 1, hidden_dim).
        :param c0: Initial cell state at time t_0 (Batch_size, 1, hidden_dim).
        :return step_embed: the latent representation of each time step (batch_size, seq_len, hidden_dim).
        :return traj_embed: the latend representation of each trajectory (batch_size, hidden_dim).
        """
        if self.normalize:
            mean = torch.mean(x, dim=(0, 1))[None, None, :]
            std = torch.std(x, dim=(0, 1))[None, None, :]
            x = (x - mean) / std
        mlp_encoded = self.mlp_encoder(x)  # (N, T, Hiddens[-2]) get the hidden representation of every time step.
        if self.rnn_cell_type == 'GRU':
            step_embed, _ = self.rnn(mlp_encoded, h0)
        else:
            step_embed, _, _ = self.rnn(mlp_encoded, h0, c0)
        return step_embed


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class LSTMWithInputCellAttention(nn.Module):

    def __init__(self, input_sz: int, hidden_sz: int, r=10, d_a=30):
        """
        :param input_sz: input dim.
        :param hidden_sz: hidden layer dim.
        :param r: input-cell output dim.
        :param d_a: input-cell first layer dim.
        """
        super().__init__()
        self.r = r
        self.input_sz = input_sz
        self.hidden_size = hidden_sz
        self.weight_iBarh = Parameter(torch.Tensor(input_sz, hidden_sz * 4))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 4))
        self.bias = Parameter(torch.Tensor(hidden_sz * 4))
        self.r = r
        self.linear_first = torch.nn.Linear(input_sz, d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a, r)
        self.linear_second.bias.data.fill_(0)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def getMatrixM(self, pastTimeSteps):
        # input attention layer
        # x: [Batch, t, N]
        # M = softmax(W_2tanh(W_1X)): [Batch, r, t]
        # M = Mx: [Batch, r, N]
        # M = 1/r \sum_{i} M: [Batch, N].

        x = self.linear_first(pastTimeSteps)

        x = torch.tanh(x)
        x = self.linear_second(x)
        x = self.softmax(x, 1)
        attention = x.transpose(1, 2)
        matrixM = attention @ pastTimeSteps
        matrixM = torch.sum(matrixM, 1) / self.r

        return matrixM

    def softmax(self, input, axis=1):
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size) - 1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])

        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size) - 1)

    def forward(self, x: torch.Tensor,
                init_states: Optional[Tuple[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t, c_t = (torch.zeros(self.hidden_size).to(x.device),
                        torch.zeros(self.hidden_size).to(x.device))
        else:
            h_t, c_t = init_states

        HS = self.hidden_size
        batchSize = x[:, 0, :].size()[0]

        M = torch.zeros(batchSize, self.input_sz).double()

        for t in range(seq_sz):
            x_t = x[:, t, :]
            if (t == 0):
                H = x[:, 0, :].view(batchSize, 1, self.input_sz)

                M = self.getMatrixM(H)
            elif (t > 0):
                H = x[:, :t + 1, :]

                M = self.getMatrixM(H)

            gates = M @ self.weight_iBarh + h_t @ self.weight_hh + self.bias

            i_t, f_t, g_t, o_t = (
                torch.sigmoid(gates[:, :, :HS]),  # input
                torch.sigmoid(gates[:, :, HS:HS * 2]),  # forget
                torch.tanh(gates[:, :, HS * 2:HS * 3]),
                torch.sigmoid(gates[:, :, HS * 3:]),  # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        hidden_seq = hidden_seq.squeeze(1)

        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, (h_t, c_t)


class GRUWithInputCellAttention(LSTMWithInputCellAttention):

    def __init__(self, input_sz: int, hidden_sz: int, r=10, d_a=30):
        """
        :param input_sz: input dim.
        :param hidden_sz: hidden layer dim.
        :param r: input-cell output dim.
        :param d_a: input-cell first layer dim.
        """
        super().__init__(input_sz, hidden_sz, r, d_a)
        self.weight_iBarh = Parameter(torch.Tensor(input_sz, hidden_sz * 3))
        self.weight_hh = Parameter(torch.Tensor(hidden_sz, hidden_sz * 3))
        self.bias_i = Parameter(torch.Tensor(hidden_sz * 3))
        self.bias_h = Parameter(torch.Tensor(hidden_sz * 3))

    def forward(self, x: torch.Tensor,
                init_states: Optional[Tuple[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assumes x is of shape (batch, sequence, feature)"""
        bs, seq_sz, _ = x.size()
        hidden_seq = []

        if init_states is None:
            h_t = torch.zeros(self.hidden_size).to(x.device)
        else:
            h_t = init_states

        batchSize = x[:, 0, :].size()[0]
        M = torch.zeros(batchSize, self.input_sz).double()

        for t in range(seq_sz):
            x_t = x[:, t, :]
            if (t == 0):
                H = x[:, 0, :].view(batchSize, 1, self.input_sz)

                M = self.getMatrixM(H)
            elif (t > 0):
                H = x[:, :t + 1, :]

                M = self.getMatrixM(H)

            gi = M @ self.weight_iBarh + self.bias_i
            gh = h_t @ self.weight_hh + self.bias_h

            i_r, i_i, i_n = gi.chunk(3, 2)
            h_r, h_i, h_n = gh.chunk(3, 2)

            r_t = torch.sigmoid(i_r + h_r)
            z_t = torch.sigmoid(i_i + h_i)
            n_t = torch.tanh(i_n + r_t * h_n)
            h_t = n_t + z_t * (h_t - n_t)

            hidden_seq.append(h_t.unsqueeze(Dim.batch))

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        hidden_seq = hidden_seq.squeeze(1)

        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq, h_t


class TanhAttention(nn.Module):
    def __init__(self, input_dim, hidden_size):
        """
        :param input_dim: input dim.
        :param hidden_size: hidden size.
        """
        super().__init__()
        self.attn1 = nn.Linear(input_dim, hidden_size)
        self.attn2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, x):
        """
        :param x: X (B, L, D)
        :return: attn * X (B, D)
        """

        attn1 = torch.tanh(self.attn1(x)) # (B, L, H)
        attn2 = self.attn2(attn1) # (B, L, 1)
        attn = torch.softmax(attn2, 1) # (B, L, 1)
        attn_applied = (attn * x).sum(1) # (B, D)

        return attn, attn_applied


class DotAttention(nn.Module):
    def __init__(self, input_dim):
        """
        :param input_dim: input dim.
        """
        super().__init__()
        self.attn1 = nn.Linear(input_dim, 1, bias=False)
        self.input_dim = input_dim

    def forward(self, x):
        """
        :param x: X (B, L, D)
        :return: attn * X (B, D)
        """
        attn1 = self.attn1(x) / (self.input_dim)**0.5 # (B, L, 1)
        attn = torch.softmax(attn1, 1) # (B, L, 1)
        attn_applied = (attn * x).sum(1) # (B, D)

        return attn, attn_applied
