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
from .rnn_utils import MlPRnnEncoder, CnnRnnEncoder


# Baseline 2: Vanilla [input-cell] RNN + a saliency method (IG).
# [Input-Cell Attention Reduces Vanishing Saliency of Recurrent Neural Networks].
class RnnSaliency(object):
    def __init__(self, seq_len, input_dim, likelihood_type, hiddens, n_action, num_class=0, embed_dim=16,
                 encoder_type='MLP', dropout_rate=0.25,  rnn_cell_type='GRU', use_input_attention=False,
                 normalize=False):
        """
        :param seq_len: trajectory length.
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

        self.encoder_type = encoder_type
        self.likelihood_type = likelihood_type
        if self.encoder_type == 'CNN':
            self.model = CnnRnnEncoder(seq_len, input_dim, input_channles=1, hidden_dim=hiddens[-1],
                                       n_action=n_action, embed_dim=embed_dim, rnn_cell_type=rnn_cell_type,
                                       use_input_attention=use_input_attention, normalize=normalize)
        else:
            self.model = MlPRnnEncoder(seq_len, input_dim, hiddens, dropout_rate, rnn_cell_type,
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

    def train(self, train_loader, n_epoch, lr=0.01, gamma=0.1, optimizer_choice='adam', save_path=None):
        """
        :param train_loader: training data loader.
        :param n_epoch: number of training epoch.
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

        for epoch in range(1, n_epoch + 1):
            minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
            mse = 0
            mae = 0
            correct = 0
            for obs, acts, rewards in minibatch_iter:
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
                if self.likelihood_type == 'classification':
                    _, preds = torch.max(output, 1)
                    correct += preds.eq(rewards.view_as(preds)).cpu().sum()
                else:
                    mae += torch.sum(torch.abs(output - rewards))
                    mse += torch.sum(torch.square(output - rewards))
                minibatch_iter.set_postfix(loss=loss.item())

            if self.likelihood_type == 'classification':
                print('Train Accuracy: {}/{} ({}%)'.format(
                    correct, len(train_loader.dataset), 100. * correct / float(len(train_loader.dataset))
                ))
            else:
                print('Train MAE: {}'.format(mae / float(len(train_loader.dataset))))
                print('Train MSE: {}'.format(mse / float(len(train_loader.dataset))))
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

        mse = 0
        mae = 0
        correct = 0
        for obs, acts, rewards in test_loader:
            if torch.cuda.is_available():
                obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
            preds = self.model(obs, acts)[:, -1, :]
            preds = self.likelihood(preds)

            if self.likelihood_type == 'classification':
                _, preds = torch.max(preds, 1)
                correct += preds.eq(rewards.view_as(preds)).cpu().sum()
            else:
                mae += torch.sum(torch.abs(preds - rewards))
                mse += torch.sum(torch.square(preds - rewards))

        if self.likelihood_type == 'classification':
            print('Test set: Accuracy: {}/{} ({}%)'.format(
                correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
            ))
            return correct

        else:
            print('Test MAE: {}'.format(mae / float(len(test_loader.dataset))))
            print('Test MSE: {}'.format(mse / float(len(test_loader.dataset))))

            return mse, mae

    def compute_cnn_encoded(self, obs, acts):
        self.model.eval()
        self.likelihood.eval()

        obs.requires_grad_()

        if torch.cuda.is_available():
            obs, acts = obs.cuda(), acts.cuda()

        obs = obs.view(-1, 1, obs.shape[-1], obs.shape[-1])
        obs_encoded = self.model.cnn_encoder(obs)  # (N, T, D1) get the hidden representation of every time step.
        obs_encoded = obs_encoded.view(int(obs.shape[0]/self.model.seq_len), self.model.seq_len,
                                       obs_encoded.size(-1))
        act_encoded = self.model.act_embedding(acts)
        cnn_encoded = torch.cat((obs_encoded, act_encoded), -1)
        return cnn_encoded

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
        return grad

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

        return grad

    def get_explanations(self, obs, acts, rewards, saliency_method='integrated_gradient', back2rnn=True, n_samples=10,
                         stdev_spread=0.15, normalize=True):
        """
        :param obs: input observations.
        :param acts: input actions.
        :param rewards: trajectory rewards.
        :param normalize: Normalization or not.
        :param saliency_method: choice of saliency method.
        :param back2rnn: Compute the gradient of the rnn layer.
        :param n_samples: number of reference samples.
        :param stdev_spread: std spread.
        :return: time step importance.
        """

        if saliency_method == 'gradient':
            print('Using vanilla gradient.')
            if back2rnn:
                cnn_encoded = self.compute_cnn_encoded(obs, acts)
                saliency = self.compute_gradient_rnn(cnn_encoded, rewards)
            else:
                saliency = self.compute_gradient_input(obs, acts, rewards)

        elif saliency_method == 'integrated_gradient':
            print('Using integrated gradient.')

            if back2rnn:
                cnn_encoded = self.compute_cnn_encoded(obs, acts)
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
                cnn_encoded = self.compute_cnn_encoded(obs, acts)
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
                cnn_encoded = self.compute_cnn_encoded(obs, acts)
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
                cnn_encoded = self.compute_cnn_encoded(obs, acts)
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
                cnn_encoded = self.compute_cnn_encoded(obs, acts)
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
                cnn_encoded = self.compute_cnn_encoded(obs, acts)
                saliency = self.compute_gradient_rnn(cnn_encoded, rewards)
            else:
                saliency = self.compute_gradient_input(obs, acts, rewards)

        if back2rnn:
            saliency = saliency.sum(-1)
        else:
            saliency = saliency.sum([-3, -2, -1])

        saliency = saliency.detach().numpy()
        if normalize:
            saliency = (saliency - np.min(saliency, axis=1)[:, None]) \
                       / (np.max(saliency, axis=1)[:, None] - np.min(saliency, axis=1)[:, None])

        return saliency
