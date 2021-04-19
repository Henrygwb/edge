# Baseline explanation methods:
# 1. Vanilla RNN + Input mask [RUDDER];
# 2. Vanilla [input-cell] RNN + a saliency method (IG)
# [Input-Cell Attention Reduces Vanishing Saliency of Recurrent Neural Networks];
# 3. RNN with attention [Attention is not Explanation].
# 4. Self-explainable model: [Invariant Rationalization].

import tqdm
import torch
import numpy as np
import torch.optim as optim
from .rnn_utils import CnnRnnEncoder, MlpRnnEncoder


# Baseline 1 [RUDDER]: Vanilla RNN + Input mask.
# Collect a set of trajectories of a pretrained agent.
# Concatenate action and observation as the input and the final reward as the output.
# Train the Seq2one + Seq2seq RNN, and use the prediction difference of p_t - p_{t-1} as the importance of r_t.
class Rudder(object):
    def __init__(self, seq_len, input_dim, hiddens, n_action, embed_dim=16, encoder_type='MLP', dropout_rate=0.25,
                 rnn_cell_type='GRU', normalize=False):
        """
        :param seq_len: trajectory length.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action).
        :param hiddens: hidden layer dimensions.
        :param n_action: number of actions.
        :param embed_dim: actions embedding dim.
        :param encoder_type: encoder type ('MLP' or 'CNN').
        :param dropout_rate: dropout rate.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param normalize: whether to normalize the inputs.
        """
        self.encoder_type = encoder_type
        if self.encoder_type == 'CNN':
            self.model = CnnRnnEncoder(seq_len, input_dim, input_channles=1, hidden_dim=hiddens[-1],
                                       n_action=n_action, embed_dim=embed_dim, rnn_cell_type=rnn_cell_type,
                                       normalize=normalize)
        else:
            self.model = MlpRnnEncoder(seq_len, input_dim, hiddens, dropout_rate, rnn_cell_type, normalize=normalize)
        self.fc_out = torch.nn.Sequential(
            torch.nn.Linear(hiddens[-1], 1),
            torch.nn.Flatten())

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.fc_out = self.fc_out.cuda()

    def loss(self, predicts, targets):
        """
        :param predicts: predicted final returns (N, seq_len, 1).
        :param targets: true final returns (N, 1)
        :return: final loss of seq2one prediction and seq2seq prediction.
        """
        # Main task: predicting return at last timestep
        main_loss = torch.mean(torch.square(predicts[:, -1] - targets))
        # Auxiliary task: predicting final return at every timestep ([..., None] is for correct broadcasting)
        aux_loss = torch.mean(torch.square(predicts[:, :] - targets[..., None]))
        # Combine losses
        loss = main_loss + aux_loss * 0.5
        return loss

    def save(self, save_path):
        state_dict = self.model.state_dict()
        fc_dict = self.fc_out.state_dict()
        torch.save({'model': state_dict, 'fc': fc_dict}, save_path)
        return 0

    def load(self, load_path):
        dicts = torch.load(load_path)
        model_dict = dicts['model']
        fc_dict = dicts['fc']
        self.fc_out.load_state_dict(fc_dict)
        self.model.load_state_dict(model_dict)
        return self.model, self.fc_out

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
        self.fc_out.train()

        if optimizer_choice == 'adam':
            optimizer = optim.Adam([{'params': self.model.parameters(), 'weight_decay': 1e-4},
                                    {'params': self.fc_out.parameters()}], lr=lr)
        else:
            optimizer = optim.SGD([{'params': self.model.parameters(), 'weight_decay': 1e-4},
                                    {'params': self.fc_out.parameters()}], lr=lr, momentum=0.9, nesterov=True)

        # Learning rate decay schedule.
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epoch, 0.75 * n_epoch], gamma=gamma)

        for epoch in range(1, n_epoch + 1):
            mse = 0
            mae = 0
            minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
            for obs, acts, rewards in minibatch_iter:
                if torch.cuda.is_available():
                    obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
                optimizer.zero_grad()
                output = self.model(obs, acts)
                output = self.fc_out(output)
                loss = self.loss(output, rewards)
                loss.backward()
                optimizer.step()
                minibatch_iter.set_postfix(loss=loss.item())
                mae += torch.sum(torch.abs(output[:, -1] - rewards))
                mse += torch.sum(torch.square(output[:, -1] - rewards))

            print('Training MAE: {}'.format(mae / float(len(train_loader.dataset))))
            print('Traing MSE: {}'.format(mse / float(len(train_loader.dataset))))
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
        self.fc_out.train()

        mse = 0
        mae = 0
        for obs, acts, rewards in test_loader:
            if torch.cuda.is_available():
                obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
            preds = self.model(obs, acts)
            preds = self.fc_out(preds)[..., -1]
            mae += torch.sum(torch.abs(preds - rewards))
            mse += torch.sum(torch.square(preds - rewards))

        print('Test MAE: {}'.format(mae/float(len(test_loader.dataset))))
        print('Test MSE: {}'.format(mse/float(len(test_loader.dataset))))
        return mse, mae

    def get_explanations(self, obs, acts, rewards, normalize=True):
        """
        :param obs: input observations.
        :param acts: input actions.
        :param rewards: trajectory rewards.
        :param normalize: Normalization or not.
        :return: time step importance.
        """
        self.model.eval()
        self.fc_out.eval()

        if torch.cuda.is_available():
            obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()

        # Apply our reward redistribution model to the samples
        preds = self.model(obs, acts)
        preds = self.fc_out(preds)

        # Use the differences of predictions as redistributed reward
        redistributed_reward = preds[:, 1:] - preds[:, :-1]

        # For the first timestep we will take (0-predictions[:, :1]) as redistributed reward
        redistributed_reward = torch.cat([preds[:, :1], redistributed_reward], dim=1)

        predicted_returns = redistributed_reward.sum(dim=1)
        prediction_error = rewards - predicted_returns

        # Distribute correction for prediction error equally over all sequence positions
        redistributed_reward += prediction_error[:, None] / redistributed_reward.shape[1]
        saliency = redistributed_reward.cpu().detach().numpy()
        print(saliency)
        if normalize:
            saliency = (saliency - np.min(saliency, axis=1)[:, None]) / \
                       (np.max(saliency, axis=1)[:, None] - np.min(saliency, axis=1)[:, None])

        return saliency
