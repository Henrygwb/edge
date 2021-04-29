import torch
import numpy as np
from torch import nn
from typing import *
from enum import IntEnum
from torch.nn import Parameter
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CnnRnnEncoder(nn.Module):
    def __init__(self, seq_len, input_dim, input_channles, hidden_dim, n_action, embed_dim=16, rnn_cell_type='GRU',
                 use_input_attention=False, normalize=False):
        """
        RNN structure (CNN+seq2seq) (\theta_1: RNN parameters).
        :param seq_len: trajectory length.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action).
        :param input_channles: 1.
        :param hidden_dim: RNN output dim.
        :param n_action: total number of actions.
        :param embed_dim: action embedding dim.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param use_input_attention: Whether to use the input cell attention.
        :param normalize: whether to normalize the inputs.
        """

        super(CnnRnnEncoder, self).__init__()
        self.normalize = normalize
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell_type = rnn_cell_type

        self.act_embedding = nn.Embedding(n_action, embed_dim)

        self.cnn_encoder = nn.Sequential()

        self.cnn_encoder.add_module('cnn_%d' % 1, nn.Conv2d(input_channles, 32, kernel_size=(3, 3), stride=(2, 2)))
        self.cnn_encoder.add_module('relu_%d' % 1, nn.ReLU())

        self.cnn_encoder.add_module('cnn_%d' % 2, nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)))
        self.cnn_encoder.add_module('relu_%d' % 2, nn.ReLU())

        self.cnn_encoder.add_module('cnn_%d' % 3, nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)))
        self.cnn_encoder.add_module('relu_%d' % 3, nn.ReLU())

        self.cnn_encoder.add_module('cnn_%d' % 4, nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2)))
        self.cnn_encoder.add_module('relu_%d' % 4, nn.ReLU())

        self.cnn_encoder.add_module('flatten', nn.Flatten(start_dim=-3, end_dim=-1))

        if input_dim == 80 or 84:
            self.cnn_out_dim = 4 * 4 * 16 + embed_dim
        else:
            raise ValueError ('input dim does not support.')

        if self.rnn_cell_type == 'GRU':
            print('Using GRU as the recurrent layer.')
            if use_input_attention:
                print('Using the input cell attention for saliency methods to prevent gradient vanish/explosion.')
                self.rnn = GRUWithInputCellAttention(input_sz=self.cnn_out_dim, hidden_sz=hidden_dim)
            else:
                self.rnn = nn.GRU(input_size=self.cnn_out_dim, hidden_size=hidden_dim, batch_first=True)
        elif self.rnn_cell_type == 'LSTM':
            print('Using LSTM as the recurrent layer.')
            if use_input_attention:
                print('Using the input cell attention for saliency methods to prevent gradient vanish/explosion.')
                self.rnn = LSTMWithInputCellAttention(input_sz=self.cnn_out_dim, hidden_sz=hidden_dim)
            else:
                self.rnn = nn.LSTM(input_size=self.cnn_out_dim, hidden_size=hidden_dim, batch_first=True)
        else:
            print('Using the default recurrent layer: GRU.')
            if use_input_attention:
                print('Using the input cell attention for saliency methods to prevent gradient vanish/explosion.')
                self.rnn = GRUWithInputCellAttention(input_sz=self.cnn_out_dim, hidden_sz=hidden_dim)
            else:
                self.rnn = nn.GRU(input_size=self.cnn_out_dim, hidden_size=hidden_dim, batch_first=True)
            self.rnn_cell_type = 'GRU'

    def forward(self, x, y, h0=None, c0=None):
        # forward function: given an input, return the model output (output at each time and the final time step).
        """
        :param x: input observations (Batch_size, seq_len, 1, input_dim, input_dim).
        :param y: input actions (Batch_size, seq_len).
        :param h0: Initial hidden state at time t_0 (Batch_size, 1, hidden_dim).
        :param c0: Initial cell state at time t_0 (Batch_size, 1, hidden_dim).
        :return step_embed: the latent representation of each time step (batch_size, seq_len, hidden_dim).
        :return traj_embed: the latend representation of each trajectory (batch_size, hidden_dim).
        """
        num_traj = x.size(0)
        if self.normalize:
            mean = torch.mean(x, dim=(0, 1))[None, None, :]
            std = torch.std(x, dim=(0, 1))[None, None, :]
            x = (x - mean) / std
        x = x.view(-1, 1, self.input_dim, self.input_dim)
        obs_encoded = self.cnn_encoder(x)  # (N, T, D1) get the hidden representation of every time step.
        obs_encoded = obs_encoded.view(num_traj, self.seq_len, obs_encoded.size(-1))
        act_encoded = self.act_embedding(y)
        cnn_encoded = torch.cat((obs_encoded, act_encoded), -1)
        if self.rnn_cell_type == 'GRU':
            step_embed, _ = self.rnn(cnn_encoded, h0)
        else:
            if h0 is None or c0 is None:
                step_embed, _ = self.rnn(cnn_encoded, None)
            else:
                step_embed, _ = self.rnn(cnn_encoded, (h0, c0))
        return step_embed


# self.model = MlpRnnEncoder(seq_len, input_dim, hiddens, n_action, embed_dim, dropout_rate,
#                            rnn_cell_type, normalize=normalize)

class MlpRnnEncoder(nn.Module):
    def __init__(self, seq_len, input_dim, hiddens, n_action=0, embed_dim=4, dropout_rate=0.25,
                 rnn_cell_type='GRU', use_input_attention=False, normalize=False):
        """
        RNN structure (MLP+seq2seq) (\theta_1: RNN parameters).
        :param seq_len: trajectory length.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action).
        :param hiddens: hidden layer dimensions.
        :param n_action: num of possible input action, 0 if the action space is continuous.
        :param embed_dim: action embedding dim for discrete input action.
        :param dropout_rate: dropout rate.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param use_input_attention: Whether to use the input cell attention.
        :param normalize: whether to normalize the inputs.
        """
        super(MlpRnnEncoder, self).__init__()
        self.normalize = normalize
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hiddens[-1]
        self.rnn_cell_type = rnn_cell_type
        self.n_action = n_action

        if n_action != 0:
            self.act_embedding = nn.Embedding(n_action, embed_dim)

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

    def forward(self, x, y, h0=None, c0=None):
        # forward function: given an input, return the model output (output at each time and the final time step).
        """
        :param x: input observations (Batch_size, seq_len, input_dim).
        :param y: input actions (Batch_size, seq_len).
        :param h0: Initial hidden state at time t_0 (Batch_size, 1, hidden_dim).
        :param c0: Initial cell state at time t_0 (Batch_size, 1, hidden_dim).
        :return step_embed: the latent representation of each time step (batch_size, seq_len, hidden_dim).
        """
        if self.normalize:
            mean = torch.mean(x, dim=(0, 1))[None, None, :]
            std = torch.std(x, dim=(0, 1))[None, None, :]
            x = (x - mean) / std
        if self.n_action != 0:
            y = self.act_embedding(y)
        x = torch.cat((x, y), -1)
        mlp_encoded = self.mlp_encoder(x)  # (N, T, Hiddens[-2]) get the hidden representation of every time step.
        if self.rnn_cell_type == 'GRU':
            step_embed, _ = self.rnn(mlp_encoded, h0)
        else:
            if h0 is None or c0 is None:
                step_embed, _ = self.rnn(mlp_encoded, None)
            else:
                step_embed, _ = self.rnn(mlp_encoded, (h0, c0))
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

        soft_max_2d = F.softmax(input_2d, dim=1)
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
                torch.sigmoid(gates[..., :HS]),  # input
                torch.sigmoid(gates[..., HS:HS * 2]),  # forget
                torch.tanh(gates[..., HS * 2:HS * 3]),
                torch.sigmoid(gates[..., HS * 3:]),  # output
            )

            c_t = f_t * c_t + i_t * g_t
            h_t = o_t * torch.tanh(c_t)
            hidden_seq.append(h_t.unsqueeze(Dim.batch))

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # hidden_seq = hidden_seq.squeeze(1) # cause error when batch_size = 1.

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

            i_r, i_i, i_n = gi.chunk(3, -1)
            h_r, h_i, h_n = gh.chunk(3, -1)

            r_t = torch.sigmoid(i_r + h_r)
            z_t = torch.sigmoid(i_i + h_i)
            n_t = torch.tanh(i_n + r_t * h_n)
            h_t = n_t + z_t * (h_t - n_t)

            hidden_seq.append(h_t.unsqueeze(Dim.batch))

        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        # hidden_seq = hidden_seq.squeeze(1)

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


class RationaleNetGenerator(nn.Module):
    def __init__(self, seq_len, input_dim, hiddens, n_action, embed_dim=16, encoder_type='MLP',
                 dropout_rate=0.25, rnn_cell_type='GRU', normalize=False):

        super(RationaleNetGenerator, self).__init__()

        self.encoder_type = encoder_type

        if self.encoder_type == 'CNN':
            self.encoder = CnnRnnEncoder(seq_len, input_dim, input_channles=1, hidden_dim=hiddens[-1],
                                         n_action=n_action, embed_dim=embed_dim, rnn_cell_type=rnn_cell_type,
                                         normalize=normalize)
        else:
            self.encoder = MlpRnnEncoder(seq_len, input_dim, hiddens, n_action=n_action, embed_dim=embed_dim,
                                         dropout_rate=dropout_rate, rnn_cell_type=rnn_cell_type, normalize=normalize)
        self.z_dim = 2
        self.hidden = nn.Linear(hiddens[-1], self.z_dim)

        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

    @staticmethod
    def gumbel_softmax(input, temperature=1.0 / 10.0, cuda=False):
        """
        Concrete distribution q(z) = \prod_i z_i.
        :param input: z [B, L, 2]
        :param temperature: temp.
        :param cuda: cuda.
        :return: p: [B, L, 2], where [i, j, :] = [a1, a2], a1 + a2 = 1.
        """
        noise = torch.rand(input.size())
        noise.add_(1e-9).log_().neg_()
        noise.add_(1e-9).log_().neg_()
        noise = torch.autograd.Variable(noise)
        if input.is_cuda:
            noise = noise.cuda()
        x = (input + noise) / temperature # [B, L, 2]
        x = F.softmax(x.view(-1, x.size()[-1]), dim=-1) # [B*L, 2]
        return x.view_as(input)

    def __z_forward(self, activ):
        """
        :param activ: activation before the last layer.
        :return: prob of each token being selected.
        """
        logits = self.hidden(activ) # [B, L, 2]
        probs = self.gumbel_softmax(logits, cuda=self.use_cuda) # [B, L, 2]
        z = probs[:, : ,1] # [B, L]
        return z # [B, L]

    def forward(self, x, y):
        """
        :param x: input data [B, L, D].
        :return: z * x.
        """
        '''
            Given input x_indx of dim (batch, length), return z (batch, length) such that z
            can act as element-wise mask on x
        '''
        activ = self.encoder(x, y) # [B, L, H]
        z = self.__z_forward(F.relu(activ)) # [B, L]
        return z

    def loss(self, z):
        """
        :param z: [B, L].
        :return: Compute the generator specific costs, i.e selection cost, continuity cost, and global vocab cost.
        """
        selection_cost = torch.mean(torch.sum(z, dim=1))
        l_padded_mask = torch.cat([z[:, 0].unsqueeze(1), z], dim=1)
        r_padded_mask = torch.cat([z, z[:, -1].unsqueeze(1)], dim=1)
        continuity_cost = torch.mean(torch.sum(torch.abs(l_padded_mask - r_padded_mask), dim=1))
        return selection_cost, continuity_cost


class RationaleNetEncoder(nn.Module):
    def __init__(self, seq_len, input_dim, hiddens, n_action, embed_dim=16, encoder_type='MLP',
                 dropout_rate=0.25, rnn_cell_type='GRU', normalize=False):

        super(RationaleNetEncoder, self).__init__()

        self.encoder_type = encoder_type

        if self.encoder_type == 'CNN':
            self.encoder = CnnRnnEncoder(seq_len, input_dim, input_channles=1, hidden_dim=hiddens[-1],
                                         n_action=n_action, embed_dim=embed_dim, rnn_cell_type=rnn_cell_type,
                                         normalize=normalize)
        else:
            self.encoder = MlpRnnEncoder(seq_len, input_dim, hiddens, n_action=n_action, embed_dim=embed_dim,
                                         dropout_rate=dropout_rate, rnn_cell_type=rnn_cell_type, normalize=normalize)

        if torch.cuda.is_available():
            self.use_cuda = True
        else:
            self.use_cuda = False

    def forward(self, x, y, z=None):
        """
        :param x: [B, L, D].
        :param y: [B, L].
        :param z: [B, L].
        :return: logit: [B, L, H]
        """
        if z is not None:
            if self.encoder_type == 'CNN':
                x = x * z[..., None, None, None]
            else:
                x = x * z[..., None]
        logit = self.encoder(x, y)

        return logit
