# Baseline explanation methods:
# 1. Vanilla RNN + Input mask [RUDDER];
# 2. Vanilla [input-cell] RNN + a saliency method (IG)
# [Input-Cell Attention Reduces Vanishing Saliency of Recurrent Neural Networks];
# 3. RNN with attention [Attention is not Explanation].
# 4. Self-explainable model (Input perturbation):
#    [Invariant Rationalization, require env related prior, we trace back to rationalizing neural predictions].

import tqdm
import torch
import timeit
import numpy as np
import torch.nn as nn
import torch.optim as optim
from .rnn_utils import RationaleNetEncoder, RationaleNetGenerator
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .quantitative_test import exp_fid2nn_topk, exp_fid2nn_zero_one, exp_stablity


# Baseline 4. Self-explainable model: [Rationalizing Neural Predictions].
class RationaleNet(object):
    def __init__(self, seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action=0, num_class=0, embed_dim=16,
                 encoder_type='MLP', dropout_rate=0.25, rnn_cell_type='GRU', normalize=False):

        """
        :param seq_len: trajectory length.
        :param len_diff: trajectory len_diff.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action).
        :param hiddens: hidden layer dimensions.
        :param likelihood_type: likelihood type.
        :param num_class: number of output class.
        :param n_action: number of actions.
        :param embed_dim: actions embedding dim.
        :param encoder_type: encoder type ('MLP' or 'CNN').
        :param dropout_rate: dropout rate.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param normalize: whether to normalize the inputs.
        """
        self.len_diff = len_diff
        self.n_action = n_action

        self.generator = RationaleNetGenerator(seq_len, input_dim, hiddens, n_action, embed_dim, encoder_type,
                                               dropout_rate, rnn_cell_type, normalize)
        self.encoder = RationaleNetEncoder(seq_len, input_dim, hiddens, n_action, embed_dim, encoder_type,
                                           dropout_rate, rnn_cell_type, normalize)

        self.likelihood_type = likelihood_type
        if self.likelihood_type == 'classification':
            self.likelihood = nn.Sequential()
            self.likelihood.add_module('linear_out', nn.Linear(hiddens[-1], num_class))
            self.likelihood.add_module('linear_out_softmax', nn.Softmax(dim=1))
        else:
            self.likelihood = nn.Linear(hiddens[-1], 1)

        if torch.cuda.is_available():
            self.generator = self.generator.cuda()
            self.encoder = self.encoder.cuda()
            self.likelihood = self.likelihood.cuda()

    def save(self, save_path):
        encoder_state_dict = self.encoder.state_dict()
        generator_state_dict = self.generator.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        torch.save({'encoder': encoder_state_dict, 'generator': generator_state_dict,
                    'likelihood': likelihood_state_dict}, save_path)
        return 0

    def load(self, load_path):
        dicts = torch.load(load_path)
        encoder_dict = dicts['encoder']
        generator_dict = dicts['generator']
        likelihood_dict = dicts['likelihood']
        self.encoder.load_state_dict(encoder_dict)
        self.generator.load_state_dict(generator_dict)
        self.likelihood.load_state_dict(likelihood_dict)
        return self.encoder, self.generator, self.likelihood

    def train(self, train_idx, batch_size, n_epoch, traj_path, lr=0.01, gamma=0.1, optimizer_choice='adam',
              lambda_selection=0.005, lambda_continuity=0.005, save_path=None):
        """
        :param train_idx: training traj index.
        :param batch_size: training batch size.
        :param n_epoch: number of training epoch.
        :param traj_path: training traj path.
        :param lr: learning rate.
        :param gamma: learning rate decay rate.
        :param optimizer_choice: training optimizer, 'adam' or 'sgd'.
        :param lambda_selection: .
        :param lambda_continuity: .
        :param save_path: model save path.
        :return: trained model.
        """
        self.encoder.train()
        self.generator.train()
        self.likelihood.train()

        if optimizer_choice == 'adam':
            optimizer = optim.Adam([{'params': self.encoder.parameters(), 'weight_decay': 1e-4},
                                    {'params': self.likelihood.parameters()},
                                    {'params': self.generator.parameters(), 'weight_decay': 1e-4}], lr=lr)
        else:
            optimizer = optim.SGD([{'params': self.encoder.parameters(), 'weight_decay': 1e-4},
                                   {'params': self.likelihood.parameters()},
                                   {'params': self.generator.parameters(), 'weight_decay': 1e-4}],
                                  lr=lr, momentum=0.9, nesterov=True)

        # Learning rate decay schedule.
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epoch, 0.75 * n_epoch],
                                                   gamma=gamma)

        if train_idx.shape[0] % batch_size == 0:
            n_batch = int(train_idx.shape[0] / batch_size)
        else:
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
                z = self.generator(obs, acts)
                output = self.encoder(obs, acts, z)[:, -1, :]
                output = self.likelihood(output)
                selection_cost, continuity_cost = self.generator.loss(z)
                if self.likelihood_type == 'classification':
                    loss_fn = nn.CrossEntropyLoss()
                else:
                    loss_fn = nn.MSELoss()
                    output = output.flatten()
                    rewards = rewards.float()
                loss = loss_fn(output, rewards)
                loss = loss + lambda_selection*selection_cost + lambda_continuity*continuity_cost
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
                precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all, average='micro')
                print('Overall training results: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                      format(precision, recall, f1, acc))
            else:
                print('Train MAE: {}'.format(mae / float(train_idx.shape[0])))
                print('Train MSE: {}'.format(mse / float(train_idx.shape[0])))
            scheduler.step()

        if save_path:
            self.save(save_path)
        return 0

    def predict(self, obs, acts, rewards):
        """
        :param obs: input observations.
        :param acts: input actions.
        :param rewards: trajectory rewards.
        :return: predicted outputs.
        """

        self.encoder.eval()
        self.likelihood.eval()
        self.generator.eval()

        if torch.cuda.is_available():
            obs, acts = obs.cuda(), acts.cuda()

        z = self.generator(obs, acts)
        output = self.encoder(obs, acts, z)[:, -1, :]
        preds = self.likelihood(output)

        rewards = rewards.cpu().detach().numpy()
        preds = preds.cpu().detach().numpy()

        if self.likelihood_type == 'classification':
            preds_labels = np.argmax(preds, 1)
            preds = preds[list(range(rewards.shape[0])), rewards]
            acc = accuracy_score(rewards, preds_labels)
            if len(preds.shape) == 2:
                preds = preds.flatten()
            return preds, acc
        else:
            if len(preds.shape) == 2:
                preds = preds.flatten()
            return preds

    def test(self, test_idx, batch_size, traj_path):
        """
        :param test_idx: training traj index.
        :param batch_size: training batch size.
        :param traj_path: training traj path.
        :return: prediction error.
        """
        self.encoder.eval()
        self.likelihood.eval()
        self.generator.eval()

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

            z = self.generator(obs, acts)
            output = self.encoder(obs, acts, z)[:, -1, :]
            preds = self.likelihood(output)

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
            precision, recall, f1, _ = precision_recall_fscore_support(rewards_all, preds_all, average='micro')
            print('Overall test results: Precision: {}, Recall: {}, F1: {}, Accuracy: {}.'.
                  format(precision, recall, f1, acc))
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
        self.encoder.eval()
        self.generator.eval()
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

            saliency = self.generator(obs, acts)

            if batch == 0:
                saliency_all = saliency.cpu().detach().numpy()
            else:
                saliency_all = np.vstack((saliency_all, saliency.cpu().detach().numpy()))

        if normalize:
            saliency_all = (saliency_all - np.min(saliency_all, axis=1)[:, None]) \
                           / (np.max(saliency_all, axis=1)[:, None] - np.min(saliency_all, axis=1)[:, None])

        return saliency_all

    def train_by_tensor(self, train_loader, n_epoch, lr=0.01, gamma=0.1, optimizer_choice='adam', lambda_selection=0.005,
                        lambda_continuity=0.005, save_path=None):
        """
        :param train_loader: training data loader.
        :param n_epoch: number of training epoch.
        :param lr: learning rate.
        :param gamma: learning rate decay rate.
        :param optimizer_choice: training optimizer, 'adam' or 'sgd'.
        :param lambda_selection: .
        :param lambda_continuity: .
        :param save_path: model save path.
        :return: trained model.
        """
        self.encoder.train()
        self.generator.train()
        self.likelihood.train()

        if optimizer_choice == 'adam':
            optimizer = optim.Adam([{'params': self.encoder.parameters(), 'weight_decay': 1e-4},
                                    {'params': self.likelihood.parameters()},
                                    {'params': self.generator.parameters(), 'weight_decay': 1e-4}], lr=lr)
        else:
            optimizer = optim.SGD([{'params': self.encoder.parameters(), 'weight_decay': 1e-4},
                                   {'params': self.likelihood.parameters()},
                                   {'params': self.generator.parameters(), 'weight_decay': 1e-4}],
                                  lr=lr, momentum=0.9, nesterov=True)

        # Learning rate decay schedule.
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epoch, 0.75 * n_epoch],
                                                   gamma=gamma)

        for epoch in range(1, n_epoch + 1):
            minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
            mse = 0
            mae = 0
            correct = 0
            for obs, acts, rewards in minibatch_iter:
                if torch.cuda.is_available():
                    obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
                optimizer.zero_grad()
                z = self.generator(obs, acts)
                output = self.encoder(obs, acts, z)[:, -1, :]
                output = self.likelihood(output)
                selection_cost, continuity_cost = self.generator.loss(z)
                if self.likelihood_type == 'classification':
                    loss_fn = nn.CrossEntropyLoss()
                else:
                    loss_fn = nn.MSELoss()
                    output = output.flatten()
                    rewards = rewards.float()
                loss = loss_fn(output, rewards)
                loss = loss + lambda_selection * selection_cost + lambda_continuity * continuity_cost
                loss.backward()
                optimizer.step()
                if self.likelihood_type == 'classification':
                    _, preds = torch.max(output, 1)
                    correct += preds.eq(rewards.view_as(preds)).cpu().sum()
                else:
                    mae += torch.sum(torch.abs(output - rewards))
                    mse += torch.sum(torch.square(output - rewards))
                minibatch_iter.set_postfix(loss=loss.item())

            if self.likelihood_type == 'classification':
                print('Test set: Accuracy: {}/{} ({}%)'.format(
                    correct, len(train_loader.dataset), 100. * correct / float(len(train_loader.dataset))
                ))
            else:
                print('Test MAE: {}'.format(mae / float(len(train_loader.dataset))))
                print('Test MSE: {}'.format(mse / float(len(train_loader.dataset))))
            scheduler.step()

        if save_path:
            self.save(save_path)
        return 0

    def test_by_tensor(self, test_loader):
        """
        :param test_loader: testing data loader.
        :return: prediction error.
        """
        self.encoder.eval()
        self.likelihood.eval()
        self.generator.eval()

        mse = 0
        mae = 0
        correct = 0
        for obs, acts, rewards in test_loader:
            if torch.cuda.is_available():
                obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
            z = self.generator(obs, acts)
            output = self.encoder(obs, acts, z)[:, -1, :]
            output = self.likelihood(output)

            if self.likelihood_type == 'classification':
                _, preds = torch.max(output, 1)
                correct += preds.eq(rewards.view_as(preds)).cpu().sum()
            else:
                mae += torch.sum(torch.abs(output - rewards))
                mse += torch.sum(torch.square(output - rewards))

        if self.likelihood_type == 'classification':
            print('Test set: Accuracy: {}/{} ({}%)'.format(
                correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
            ))
            return correct

        else:
            print('Test MAE: {}'.format(mae / float(len(test_loader.dataset))))
            print('Test MSE: {}'.format(mse / float(len(test_loader.dataset))))

            return mse, mae

    def get_explanations_by_tensor(self, obs, acts, rewards, normalize=True):
        """
        :param obs: input observations.
        :param acts: input actions.
        :param rewards: trajectory rewards.
        :param normalize: Normalization or not.
        :return: time step importance.
        """
        self.encoder.eval()
        self.generator.eval()
        if torch.cuda.is_available():
            obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()

        saliency = self.generator(obs, acts)
        saliency = saliency.cpu().detach().numpy()

        if normalize:
            saliency = (saliency - np.min(saliency, axis=1)[:, None]) \
                       / (np.max(saliency, axis=1)[:, None] - np.min(saliency, axis=1)[:, None])

        return saliency

    def exp_fid_stab(self, exp_idx, batch_size, traj_path, n_stab_samples=5):

        n_batch = int(exp_idx.shape[0] / batch_size)
        sum_time = 0
        acc_1 = 0
        acc_2 = 0
        acc_3 = 0
        acc_4 = 0

        for batch in range(n_batch):
            batch_obs = []
            batch_acts = []
            batch_rewards = []
            for idx in exp_idx[batch * batch_size:(batch + 1) * batch_size, ]:
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

            start = timeit.default_timer()
            sal = self.get_explanations_by_tensor(obs, acts, rewards)
            stop = timeit.default_timer()
            # print('Explanation time of {} samples: {}.'.format(obs.shape[0], (stop - start)))
            sum_time += (stop - start)
            if self.likelihood_type == 'classification':
                fid_1, acc_1_temp = exp_fid2nn_zero_one(obs, acts, rewards, self, sal)
                fid_2, acc_2_temp = exp_fid2nn_topk(obs, acts, rewards, self, sal, 10)
                fid_3, acc_3_temp = exp_fid2nn_topk(obs, acts, rewards, self, sal, 25)
                fid_4, acc_4_temp = exp_fid2nn_topk(obs, acts, rewards, self, sal, 50)
                acc_1 += acc_1_temp
                acc_2 += acc_2_temp
                acc_3 += acc_3_temp
                acc_4 += acc_4_temp
            else:
                fid_1 = exp_fid2nn_zero_one(obs, acts, rewards, self, sal)
                fid_2 = exp_fid2nn_topk(obs, acts, rewards, self, sal, 10)
                fid_3 = exp_fid2nn_topk(obs, acts, rewards, self, sal, 25)
                fid_4 = exp_fid2nn_topk(obs, acts, rewards, self, sal, 50)

            stab = exp_stablity(obs, acts, rewards, self, sal, n_stab_samples)
            fid = np.concatenate((fid_1[None,], fid_2[None,], fid_3[None,], fid_4[None,]))

            if batch == 0:
                sal_all = sal
                fid_all = fid
                stab_all = stab
            else:
                sal_all = np.vstack((sal_all, sal))
                fid_all = np.concatenate((fid_all, fid), axis=1)
                stab_all = np.concatenate((stab_all, stab))
        mean_time = sum_time / exp_idx.shape[0]
        acc_1 = acc_1 / n_batch
        acc_2 = acc_2 / n_batch
        acc_3 = acc_3 / n_batch
        acc_4 = acc_4 / n_batch

        return sal_all, fid_all, stab_all, [acc_1, acc_2, acc_3, acc_4], mean_time
