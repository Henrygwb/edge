# Baseline explanation methods:
# 1. Vanilla RNN + Input mask [RUDDER];
# 2. Vanilla [input-cell] RNN + a saliency method (IG)
# [Input-Cell Attention Reduces Vanishing Saliency of Recurrent Neural Networks];
# 3. RNN with attention [Attention is not Explanation].
# 4. Self-explainable model: [Invariant Rationalization].

import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from .rnn_utils import RnnEncoder, TanhAttention, DotAttention


# Baseline 3. RNN with attention [Attention is not Explanation].
class RnnAttn(object):
    def __init__(self, seq_len, input_dim, hiddens, attention_type, likelihood_type, dropout_rate=0.25, num_class=0,
                 rnn_cell_type='GRU', use_input_attention=False, normalize=False):
        """
        :param seq_len: trajectory length.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action).
        :param attention_type: attention layer type.
        :param likelihood_type: likelihood type.
        :param hiddens: hidden layer dimensions.
        :param dropout_rate: dropout rate.
        :param num_class: number of output class.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param use_input_attention: Whether to use the input cell attention.
        :param normalize: whether to normalize the inputs.
        """
        self.attention_type = attention_type
        self.likelihood_type = likelihood_type
        self.model = RnnEncoder(seq_len, input_dim, hiddens, dropout_rate, rnn_cell_type, use_input_attention,
                                normalize)
        if self.attention_type == 'tanh':
            print('Using tanh attention.')
            self.attention = TanhAttention(hiddens[-1], hiddens[-1]//2)
        else:
            print('Using Dot attention.')
            self.attention = DotAttention(hiddens[-1])

        if self.likelihood_type == 'classification':
            self.likelihood = nn.Sequential()
            self.likelihood.add_module('linear_out', nn.Linear(hiddens[-1], num_class))
            self.likelihood.add_module('linear_out', nn.Softmax())
        else:
            self.likelihood = nn.Linear(hiddens[-1], 1)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.attention = self.attention.cuda()

    def load(self, load_path):
        dicts = torch.load(load_path)
        model_dict = dicts['model']
        likelihood_dict = dicts['likelihood']
        self.model.load_state_dict(model_dict)
        self.likelihood.load_state_dict(likelihood_dict)
        return self.model, self.likelihood

    def save(self, save_path):
        state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, save_path + '/checkpoint.data')
        return 0

    def train(self, train_loader, n_epoch, lr=0.01, gamma=0.1, optimizer_choice='adam', save_path=None):
        """
        :param train_loader: training data loader.
        :param n_epoch: number of training epoch.
        :param likelihood: Likelihood type, 'classification' or 'regression'.
        :param lr: learning rate.
        :param gamma: learning rate decay rate.
        :param optimizer_choice: training optimizer, 'adam' or 'sgd'.
        :param save_path: model save path.
        :return: trained model.
        """
        self.model.train()
        self.likelihood.train()
        self.attention.train()

        if optimizer_choice == 'adam':
            optimizer = optim.Adam([{'params': self.model.parameters(), 'weight_decay': 1e-4},
                                    {'params': self.likelihood.parameters()},
                                    {'params': self.attention.parameters()}], lr=lr)
        else:
            optimizer = optim.SGD([{'params': self.model.parameters(), 'weight_decay': 1e-4},
                                   {'params': self.likelihood.parameters()},
                                   {'params': self.attention.parameters()}], lr=lr, momentum=0.9, nesterov=True)

        # Learning rate decay schedule.
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epoch, 0.75 * n_epoch],
                                                   gamma=gamma)

        for epoch in range(1, n_epoch + 1):
            minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
            mse = 0
            mae = 0
            correct = 0
            for data, target in minibatch_iter:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                optimizer.zero_grad()
                output = self.model(data)[..., -1]
                _, output = self.attention(output)
                output = self.likelihood(output)

                if self.likelihood == 'classification':
                    loss = nn.CrossEntropyLoss(output, target)
                else:
                    loss = nn.MSELoss(output, target)
                loss.backward()
                optimizer.step()
                if self.likelihood == 'classification':
                    _, preds = torch.max(output, 1)
                    correct += preds.eq(target.view_as(preds)).cpu().sum()
                else:
                    mae += torch.sum(torch.abs(output - target))
                    mse += torch.sum(torch.square(output - target))
                minibatch_iter.set_postfix(loss=loss.item())

            if self.likelihood == 'classification':
                print('Test set: Accuracy: {}/{} ({}%)'.format(
                    correct, len(train_loader.dataset), 100. * correct / float(len(train_loader.dataset))
                ))
            else:
                print('Test MAE: {}'.format(mae / float(len(train_loader.dataset))))
                print('Test MSE: {}'.format(mse / float(len(train_loader.dataset))))
            scheduler.step()

        if save_path:
            self.save(save_path)
        return self.model

    def test(self, test_loader):
        """
        :param test_loader: testing data loader.
        :return: prediction error.
        """
        self.model.eval()
        self.likelihood.eval()
        self.attention.eval()

        mse = 0
        mae = 0
        correct = 0
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            preds = self.model(data)[..., -1]
            _, preds = self.attention(preds)
            preds = self.likelihood(preds)

            if self.likelihood == 'classification':
                _, preds = torch.max(preds, 1)
                correct += preds.eq(target.view_as(preds)).cpu().sum()
            else:
                mae += torch.sum(torch.abs(preds - target))
                mse += torch.sum(torch.square(preds - target))

        if self.likelihood == 'classification':
            print('Test set: Accuracy: {}/{} ({}%)'.format(
                correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
            ))
            return correct

        else:
            print('Test MAE: {}'.format(mae / float(len(test_loader.dataset))))
            print('Test MSE: {}'.format(mse / float(len(test_loader.dataset))))

            return mse, mae

    def get_explanations(self, data, normalize=True):
        """
        :param data: input trajectories.
        :param target: trajectory rewards.
        :param normalize: Normalization or not.
        :return: time step importance.
        """
        self.model.eval()
        self.attention.eval()

        output = self.model(data)
        saliency, _ = self.attention(output)

        if normalize:
            saliency = (saliency - np.min(saliency, axis=1)[:, None])
            saliency = saliency / (np.max(saliency, axis=1)[:, None] - np.min(saliency, axis=1)[:, None])

        return saliency
