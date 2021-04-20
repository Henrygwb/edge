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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from .rnn_utils import MlpRnnEncoder, CnnRnnEncoder, TanhAttention, DotAttention


# Baseline 3. RNN with attention [Attention is not Explanation].
class RnnAttn(object):
    def __init__(self, seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action=0, num_class=0, embed_dim=16,
                 encoder_type='MLP', attention_type='tanh', dropout_rate=0.25, rnn_cell_type='GRU',
                 use_input_attention=False, normalize=False):
        """
        :param seq_len: trajectory length.
        :param len_diff: trajectory len_diff.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action).
        :param likelihood_type: likelihood type.
        :param hiddens: hidden layer dimensions.
        :param num_class: number of output class.
        :param n_action: number of actions.
        :param embed_dim: actions embedding dim.
        :param encoder_type: encoder type ('MLP' or 'CNN').
        :param attention_type: attention layer type.
        :param dropout_rate: dropout rate.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param use_input_attention: Whether to use the input cell attention.
        :param normalize: whether to normalize the inputs.
        """
        self.attention_type = attention_type
        self.likelihood_type = likelihood_type
        self.encoder_type = encoder_type
        self.len_diff = len_diff
        self.n_action = n_action

        if self.encoder_type == 'CNN':
            self.model = CnnRnnEncoder(seq_len, input_dim, input_channles=1, hidden_dim=hiddens[-1],
                                       n_action=n_action, embed_dim=embed_dim, rnn_cell_type=rnn_cell_type,
                                       use_input_attention=use_input_attention, normalize=normalize)
        else:
            self.model = MlpRnnEncoder(seq_len, input_dim, hiddens, dropout_rate, rnn_cell_type,
                                       use_input_attention=use_input_attention, normalize=normalize)

        if self.attention_type == 'tanh':
            print('Using tanh attention.')
            self.attention = TanhAttention(hiddens[-1], hiddens[-1]//2)
        else:
            print('Using Dot attention.')
            self.attention = DotAttention(hiddens[-1])

        if self.likelihood_type == 'classification':
            self.likelihood = nn.Sequential()
            self.likelihood.add_module('linear_out', nn.Linear(hiddens[-1], num_class))
            self.likelihood.add_module('linear_out_soft', nn.Softmax(dim=1))
        else:
            self.likelihood = nn.Linear(hiddens[-1], 1)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()
            self.attention = self.attention.cuda()

    def save(self, save_path):
        state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        attention_dict = self.attention.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict, 'attention': attention_dict}, save_path)
        return 0

    def load(self, load_path):
        dicts = torch.load(load_path)
        model_dict = dicts['model']
        likelihood_dict = dicts['likelihood']
        attention_dict = dicts['attention']
        self.model.load_state_dict(model_dict)
        self.likelihood.load_state_dict(likelihood_dict)
        self.attention.load_state_dict(attention_dict)
        return self.model, self.likelihood, self.attention

    def train(self, train_idx, batch_size, n_epoch, traj_path, lr=0.01, gamma=0.1, optimizer_choice='adam',
              save_path=None):
        """
        :param train_idx: training traj index.
        :param batch_size: training batch size.
        :param n_epoch: number of training epoch.
        :param traj_path: training traj path.
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

        n_batch = int(train_idx.shape[0] / batch_size) + 1
        for _ in tqdm.tqdm(range(1, n_epoch + 1)):
            mse = 0
            mae = 0
            loss_sum = 0
            preds_all = []
            rewards_all = []
            for batch in range(n_batch):
                batch_obs = []
                batch_acts = []
                batch_rewards = []
                for idx in train_idx[batch * batch_size:min((batch + 1) * batch_size, train_idx.shape[0]), ]:
                    batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                    batch_acts.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['actions'])
                    batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])

                obs = torch.tensor(np.array(batch_obs)[:, self.len_diff:, ...], dtype=torch.float32)
                if self.n_action == 0:
                    acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=torch.float32)
                else:
                    acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=torch.long)

                if self.likelihood_type == 'classification':
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
                else:
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

                if torch.cuda.is_available():
                    obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
                optimizer.zero_grad()
                output = self.model(obs, acts)
                _, output = self.attention(output)
                output = self.likelihood(output)

                if self.likelihood_type == 'classification':
                    loss_fn = nn.CrossEntropyLoss()
                else:
                    loss_fn = nn.MSELoss()
                    output = output.flatten()
                    rewards = rewards.float()
                loss = loss_fn(output, rewards)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                if self.likelihood_type == 'classification':
                    _, preds = torch.max(output, 1)
                    preds_all.extend(preds.cpu().detach().numpy().tolist())
                    rewards_all.extend(rewards.cpu().detach().numpy().tolist())
                else:
                    mae += torch.sum(torch.abs(output - rewards))
                    mse += torch.sum(torch.square(output - rewards))

            if self.likelihood_type == 'classification':
                preds_all = np.array(preds_all)
                rewards_all = np.array(rewards_all)
                precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all)
                acc = accuracy_score(rewards_all, preds_all)
                for cls in range(len(precision)):
                    print('Train results of class {}: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                          format(cls, precision[cls], recall[cls], f1[cls], acc))
            else:
                print('Train MAE: {}'.format(mae / float(train_idx.shape[0])))
                print('Train MSE: {}'.format(mse / float(train_idx.shape[0])))
            scheduler.step()

        if save_path:
            self.save(save_path)
        return self.model

    def test(self, test_idx, batch_size, traj_path):
        """
        :param test_idx: training traj index.
        :param batch_size: training batch size.
        :param traj_path: training traj path.
        :return: prediction error.
        """
        self.model.eval()
        self.likelihood.eval()
        self.attention.eval()

        mse = 0
        mae = 0
        preds_all = []
        rewards_all = []

        n_batch = int(test_idx.shape[0] / batch_size) + 1
        for batch in range(n_batch):
            batch_obs = []
            batch_acts = []
            batch_rewards = []
            for idx in test_idx[batch * batch_size:min((batch + 1) * batch_size, test_idx.shape[0]), ]:
                batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                batch_acts.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['actions'])
                batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])

            obs = torch.tensor(np.array(batch_obs)[:, self.len_diff:, ...], dtype=torch.float32)
            if self.n_action == 0:
                acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=torch.float32)
            else:
                acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=torch.long)

            if self.likelihood_type == 'classification':
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
            else:
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

            if torch.cuda.is_available():
                obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
            preds = self.model(obs, acts)
            _, preds = self.attention(preds)
            preds = self.likelihood(preds)

            if self.likelihood_type == 'classification':
                _, preds = torch.max(preds, 1)
                preds_all.extend(preds.cpu().detach().numpy().tolist())
                rewards_all.extend(rewards.cpu().detach().numpy().tolist())
            else:
                mae += torch.sum(torch.abs(preds - rewards))
                mse += torch.sum(torch.square(preds - rewards))

        if self.likelihood_type == 'classification':
            preds_all = np.array(preds_all)
            rewards_all = np.array(rewards_all)
            precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all)
            acc = accuracy_score(rewards_all, preds_all)
            for cls in range(len(precision)):
                print('Test results of class {}: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                      format(cls, precision[cls], recall[cls], f1[cls], acc))
            return precision, recall, f1, acc
        else:
            print('Test MAE: {}'.format(mae / float(test_idx.shape[0])))
            print('Test MSE: {}'.format(mse / float(test_idx.shape[0])))
            return mse, mae

    def get_explanations(self, exp_idx, batch_size, traj_path, normalize=True):
        """
        :param exp_idx: training traj index.
        :param batch_size: training batch size.
        :param traj_path: training traj path.
        :param normalize: normalize.
        :return: time step importance.
        """
        self.model.eval()
        self.attention.eval()

        n_batch = int(exp_idx.shape[0] / batch_size) + 1

        for batch in range(n_batch):
            batch_obs = []
            batch_acts = []
            batch_rewards = []
            for idx in exp_idx[batch * batch_size:min((batch + 1) * batch_size, exp_idx.shape[0]), ]:
                batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                batch_acts.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['actions'])
                batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])

            obs = torch.tensor(np.array(batch_obs)[:, self.len_diff:, ...], dtype=torch.float32)

            if self.n_action == 0:
                acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=torch.float32)
            else:
                acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=torch.long)

            if self.likelihood_type == 'classification':
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
            else:
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

            if torch.cuda.is_available():
                obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()

            output = self.model(obs, acts)
            saliency, _ = self.attention(output)
            saliency = saliency.squeeze(-1)

            if batch == 0:
                saliency_all = saliency.cpu().detach().numpy()
            else:
                saliency_all = np.vstack((saliency_all, saliency.cpu().detach().numpy()))

        if normalize:
            saliency_all = (saliency_all - np.min(saliency_all, axis=1)[:, None]) \
                       / (np.max(saliency_all, axis=1)[:, None] - np.min(saliency_all, axis=1)[:, None])

        return saliency_all
