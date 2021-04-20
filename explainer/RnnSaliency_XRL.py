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
from .rnn_utils import MlpRnnEncoder, CnnRnnEncoder
from sklearn.metrics import precision_recall_fscore_support, accuracy_score


# Baseline 2: Vanilla [input-cell] RNN + a saliency method (IG).
# [Input-Cell Attention Reduces Vanishing Saliency of Recurrent Neural Networks].
class RnnSaliency(object):
    def __init__(self, seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action=0, num_class=0, embed_dim=16,
                 encoder_type='MLP', dropout_rate=0.25,  rnn_cell_type='GRU', use_input_attention=False,
                 normalize=False):
        """
        :param seq_len: trajectory length.
        :param len_diff: trajectory len_diff.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action).
        :param likelihood_type: likelihood type.
        :param hiddens: hidden layer dimensions.
        :param n_action: number of actions.
        :param embed_dim: actions embedding dim.
        :param encoder_type: encoder type ('MLP' or 'CNN').
        :param dropout_rate: dropout rate.
        :param num_class: number of output class.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param use_input_attention: Whether to use the input cell attention.
        :param normalize: whether to normalize the inputs.
        """
        self.len_diff = len_diff
        self.n_action = n_action
        self.encoder_type = encoder_type
        self.likelihood_type = likelihood_type
        if self.encoder_type == 'CNN':
            self.model = CnnRnnEncoder(seq_len, input_dim, input_channles=1, hidden_dim=hiddens[-1],
                                       n_action=n_action, embed_dim=embed_dim, rnn_cell_type=rnn_cell_type,
                                       use_input_attention=use_input_attention, normalize=normalize)
        else:
            self.model = MlpRnnEncoder(seq_len, input_dim, hiddens, dropout_rate, rnn_cell_type,
                                       use_input_attention=use_input_attention, normalize=normalize)

        if self.likelihood_type == 'classification':
            self.likelihood = nn.Sequential()
            self.likelihood.add_module('linear_out', nn.Linear(hiddens[-1], num_class))
            self.likelihood.add_module('linear_out_soft', nn.Softmax(dim=1))
        else:
            self.likelihood = nn.Linear(hiddens[-1], 1)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

    def save(self, save_path):
        state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, save_path)
        return 0

    def load(self, load_path):
        dicts = torch.load(load_path)
        model_dict = dicts['model']
        likelihood_dict = dicts['likelihood']
        self.model.load_state_dict(model_dict)
        self.likelihood.load_state_dict(likelihood_dict)
        return self.model, self.likelihood

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

        if optimizer_choice == 'adam':
            optimizer = optim.Adam([{'params': self.model.parameters(), 'weight_decay': 1e-4},
                                    {'params': self.likelihood.parameters()}], lr=lr)
        else:
            optimizer = optim.SGD([{'params': self.model.parameters(), 'weight_decay': 1e-4},
                                    {'params': self.likelihood.parameters()}], lr=lr, momentum=0.9, nesterov=True)

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
                output = self.model(obs, acts)[:, -1, :]
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
            preds = self.model(obs, acts)[:, -1, :]
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

    def compute_encoded(self, obs, acts):
        self.model.eval()
        self.likelihood.eval()

        obs.requires_grad_()

        if torch.cuda.is_available():
            obs, acts = obs.cuda(), acts.cuda()
        if self.encoder_type == 'CNN':
            obs = obs.view(-1, 1, obs.shape[-1], obs.shape[-1])
            obs_encoded = self.model.cnn_encoder(obs)  # (N, T, D1) get the hidden representation of every time step.
            obs_encoded = obs_encoded.view(int(obs.shape[0]/self.model.seq_len), self.model.seq_len,
                                           obs_encoded.size(-1))
            act_encoded = self.model.act_embedding(acts)
            encoded = torch.cat((obs_encoded, act_encoded), -1)
        else:
            x = torch.cat((obs, acts[..., None]), -1)
            encoded = self.model.mlp_encoder(x)  # (N, T, Hiddens[-2]) get the hidden representation of every time step.
        return encoded.cpu()

    def compute_gradient_rnn(self, cnn_encoded, rewards):

        self.model.eval()
        self.likelihood.eval()

        cnn_encoded.requires_grad_()

        if torch.cuda.is_available():
            cnn_encoded, rewards = cnn_encoded.cuda(), rewards.cuda()

        step_embed, _ = self.model.rnn(cnn_encoded, None)
        preds = step_embed[:, -1, :]

        preds = self.likelihood(preds)

        if preds.shape[1] > 1:
            preds = preds[list(range(preds.shape[0])), rewards]
        else:
            preds = preds.flatten()

        grad = torch.autograd.grad(torch.unbind(preds), cnn_encoded, retain_graph=True)[0]
        return grad.cpu()

    def compute_gradient_input(self, obs, acts, rewards):

        self.model.eval()
        self.likelihood.eval()

        obs.requires_grad_()

        if torch.cuda.is_available():
            obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()

        preds = self.model(obs, acts)[:, -1, :]
        preds = self.likelihood(preds)

        if preds.shape[1] > 1:
            preds = preds[list(range(preds.shape[0])), rewards]
        else:
            preds = preds.flatten()

        grad = torch.autograd.grad(torch.unbind(preds), obs, retain_graph=True)[0]

        return grad.cpu()

    def get_explanations(self, exp_idx, batch_size, traj_path, saliency_method='integrated_gradient', back2rnn=True,
                         n_samples=2, stdev_spread=0.15, normalize=True):
        """
        :param exp_idx: training traj index.
        :param batch_size: training batch size.
        :param traj_path: training traj path.
        :param saliency_method: choice of saliency method.
        :param back2rnn: Compute the gradient of the rnn layer.
        :param n_samples: number of reference samples.
        :param stdev_spread: std spread.
        :param normalize: Normalization or not.
        :return: time step importance.
        """
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

            if saliency_method == 'gradient':
                print('Using vanilla gradient.')
                if back2rnn:
                    cnn_encoded = self.compute_encoded(obs, acts)
                    saliency = self.compute_gradient_rnn(cnn_encoded, rewards)
                else:
                    saliency = self.compute_gradient_input(obs, acts, rewards)

            elif saliency_method == 'integrated_gradient':
                print('Using integrated gradient.')

                if back2rnn:
                    cnn_encoded = self.compute_encoded(obs, acts)
                    baseline = torch.zeros_like(cnn_encoded)
                    assert baseline.shape == cnn_encoded.shape
                    x_diff = cnn_encoded - baseline
                    saliency = torch.zeros_like(cnn_encoded)
                    for alpha in np.linspace(0, 1, n_samples):
                        x_step = baseline + alpha * x_diff
                        grads = self.compute_gradient_rnn(x_step, rewards)
                        saliency += grads
                    saliency = saliency * x_diff
                else:
                    baseline = torch.zeros_like(obs)
                    assert baseline.shape == obs.shape
                    x_diff = obs - baseline
                    saliency = torch.zeros_like(obs)
                    for alpha in np.linspace(0, 1, n_samples):
                        x_step = baseline + alpha * x_diff
                        grads = self.compute_gradient_input(x_step, acts, rewards)
                        saliency += grads
                    saliency = saliency * x_diff

            # elif saliency_method == 'maxdistintgrad':
            #     print('Using Maxdistintgrad.')
            #     dist_to_top = torch.fabs(torch.max(obs) - obs)
            #     dist_to_bottom = torch.fabs(torch.min(obs) - obs)
            #     baseline = torch.ones_like(obs) * torch.max(obs)
            #     baseline[dist_to_bottom > dist_to_top] = np.min(obs)
            #     assert baseline.shape == obs.shape
            #
            #     x_diff = obs - baseline
            #     saliency = np.zeros_like(obs)
            #     for alpha in np.linspace(0, 1, n_samples):
            #         x_step = baseline + alpha * x_diff
            #         grads = self.compute_gradient(x_step, acts, rewards)
            #         saliency += grads
            #     saliency = saliency * x_diff

            elif saliency_method == 'unifintgrad':
                print('Using Unifintgrad.')

                if back2rnn:
                    cnn_encoded = self.compute_encoded(obs, acts)
                    baseline = torch.rand(cnn_encoded.shape)
                    baseline = (torch.max(cnn_encoded) - torch.min(cnn_encoded)) * baseline + torch.min(cnn_encoded)
                    assert baseline.shape == cnn_encoded.shape
                    x_diff = cnn_encoded - baseline
                    saliency = torch.zeros_like(cnn_encoded)
                    for alpha in np.linspace(0, 1, n_samples):
                        x_step = baseline + alpha * x_diff
                        grads = self.compute_gradient_rnn(x_step, rewards)
                        saliency += grads
                    saliency = saliency * x_diff
                else:
                    baseline = torch.rand(obs.shape)
                    baseline = (torch.max(obs) - torch.min(obs)) * baseline + torch.min(obs)
                    assert baseline.shape == obs.shape
                    x_diff = obs - baseline
                    saliency = torch.zeros_like(obs)
                    for alpha in np.linspace(0, 1, n_samples):
                        x_step = baseline + alpha * x_diff
                        grads = self.compute_gradient_input(x_step, acts, rewards)
                        saliency += grads
                    saliency = saliency * x_diff

            elif saliency_method == 'smoothgrad':
                print('Using smooth gradient.')

                if back2rnn:
                    cnn_encoded = self.compute_encoded(obs, acts)
                    stdev = stdev_spread / (torch.max(cnn_encoded) - torch.min(cnn_encoded)).item()
                    saliency = torch.zeros_like(cnn_encoded)
                    for x in range(n_samples):
                        noise = torch.normal(0, stdev, cnn_encoded.shape)
                        noisy_data = cnn_encoded + noise
                        grads = self.compute_gradient_rnn(noisy_data, rewards)
                        saliency = saliency + grads
                    saliency = saliency / n_samples
                else:
                    stdev = stdev_spread / (torch.max(obs) - torch.min(obs)).item()
                    saliency = torch.zeros_like(obs)
                    for x in range(n_samples):
                        noise = torch.normal(0, stdev, obs.shape)
                        noisy_data = obs + noise
                        grads = self.compute_gradient_input(noisy_data, acts, rewards)
                        saliency = saliency + grads
                    saliency = saliency / n_samples

            elif saliency_method == 'expgrad':
                print('Using Expgrad.')
                if back2rnn:
                    cnn_encoded = self.compute_encoded(obs, acts)
                    stdev = stdev_spread / (torch.max(cnn_encoded) - torch.min(cnn_encoded)).item()
                    saliency = torch.zeros_like(cnn_encoded)
                    for x in range(n_samples):
                        noise = torch.normal(0, stdev, cnn_encoded.shape)
                        noisy_data = cnn_encoded + noise * torch.rand(1)[0]
                        grads = self.compute_gradient_rnn(noisy_data, rewards)
                        saliency = saliency + grads * noise
                else:
                    stdev = stdev_spread / (torch.max(obs) - torch.min(obs)).item()
                    saliency = torch.zeros_like(obs)
                    for x in range(n_samples):
                        noise = torch.normal(0, stdev, obs.shape)
                        noisy_data = obs + noise * torch.rand(1)[0]
                        grads = self.compute_gradient_input(noisy_data, acts, rewards)
                        saliency = saliency + grads * noise

            elif saliency_method == 'vargrad':
                print('Using vargrad.')
                saliency = []
                if back2rnn:
                    cnn_encoded = self.compute_encoded(obs, acts)
                    stdev = stdev_spread / (torch.max(cnn_encoded) - torch.min(cnn_encoded)).item()
                    for x in range(n_samples):
                        noise = torch.normal(0, stdev, cnn_encoded.shape)
                        noisy_data = cnn_encoded + noise
                        grads = self.compute_gradient_rnn(noisy_data, rewards)
                        saliency.append(grads[None, ...])
                else:
                    stdev = stdev_spread / (torch.max(obs) - torch.min(obs)).item()
                    for x in range(n_samples):
                        noise = torch.normal(0, stdev, obs.shape)
                        noisy_data = obs + noise
                        grads = self.compute_gradient_input(noisy_data, acts, rewards)
                        saliency.append(grads[None, ...])

                saliency = torch.cat(saliency, dim=0)
                saliency = torch.var(saliency, dim=0)

            else:
                print('Using vanilla gradient.')
                if back2rnn:
                    cnn_encoded = self.compute_encoded(obs, acts)
                    saliency = self.compute_gradient_rnn(cnn_encoded, rewards)
                else:
                    saliency = self.compute_gradient_input(obs, acts, rewards)

            if back2rnn or self.encoder_type == 'MLP':
                saliency = saliency.sum(-1)
            else:
                saliency = saliency.sum([-3, -2, -1])

            if batch == 0:
                saliency_all = saliency.cpu().detach().numpy()
            else:
                saliency_all = np.vstack((saliency_all, saliency.cpu().detach().numpy()))

        if normalize:
            saliency_all = (saliency_all - np.min(saliency_all, axis=1)[:, None]) \
                       / (np.max(saliency_all, axis=1)[:, None] - np.min(saliency_all, axis=1)[:, None])

        return saliency_all
