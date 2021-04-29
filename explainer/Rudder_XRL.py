# Baseline explanation methods:
# 1. Vanilla RNN + Input mask [RUDDER];
# 2. Vanilla [input-cell] RNN + a saliency method (IG)
# [Input-Cell Attention Reduces Vanishing Saliency of Recurrent Neural Networks];
# 3. RNN with attention [Attention is not Explanation].
# 4. Self-explainable model: [Invariant Rationalization].

import tqdm
import torch
import timeit
import numpy as np
import torch.optim as optim
from .quantitative_test import exp_stablity
from .rnn_utils import CnnRnnEncoder, MlpRnnEncoder


# Baseline 1 [RUDDER]: Vanilla RNN + Input mask.
# Collect a set of trajectories of a pretrained agent.
# Concatenate action and observation as the input and the final reward as the output.
# Train the Seq2one + Seq2seq RNN, and use the prediction difference of p_t - p_{t-1} as the importance of r_t.
class Rudder(object):
    def __init__(self, seq_len, len_diff, input_dim, hiddens, n_action=0, embed_dim=16, encoder_type='MLP',
                 dropout_rate=0.25, rnn_cell_type='GRU', normalize=False):
        """
        :param seq_len: trajectory length.
        :param len_diff: trajectory length diff.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action).
        :param hiddens: hidden layer dimensions.
        :param n_action: number of actions.
        :param embed_dim: actions embedding dim.
        :param encoder_type: encoder type ('MLP' or 'CNN').
        :param dropout_rate: dropout rate.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param normalize: whether to normalize the inputs.
        """
        self.n_action = n_action
        self.len_diff = len_diff
        self.encoder_type = encoder_type
        if self.encoder_type == 'CNN':
            self.model = CnnRnnEncoder(seq_len, input_dim, input_channles=1, hidden_dim=hiddens[-1],
                                       n_action=n_action, embed_dim=embed_dim, rnn_cell_type=rnn_cell_type,
                                       normalize=normalize)
        else:
            self.model = MlpRnnEncoder(seq_len, input_dim, hiddens, n_action, embed_dim, dropout_rate,
                                       rnn_cell_type, normalize=normalize)

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
        dicts = torch.load(load_path, map_location=torch.device('cpu'))
        model_dict = dicts['model']
        fc_dict = dicts['fc']
        self.fc_out.load_state_dict(fc_dict)
        self.model.load_state_dict(model_dict)
        return self.model, self.fc_out

    def train(self, train_idx, test_idx, batch_size, n_epoch, traj_path, lr=0.01, gamma=0.1, optimizer_choice='adam',
              save_path=None):
        """
        :param train_idx: training traj index.
        :param test_idx: testing traj index.
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
        self.fc_out.train()

        if optimizer_choice == 'adam':
            optimizer = optim.Adam([{'params': self.model.parameters(), 'weight_decay': 1e-4},
                                    {'params': self.fc_out.parameters()}], lr=lr)
        else:
            optimizer = optim.SGD([{'params': self.model.parameters(), 'weight_decay': 1e-4},
                                    {'params': self.fc_out.parameters()}], lr=lr, momentum=0.9, nesterov=True)

        # Learning rate decay schedule.
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * n_epoch, 0.75 * n_epoch], gamma=gamma)
        if train_idx.shape[0] % batch_size == 0:
            n_batch = int(train_idx.shape[0] / batch_size)
        else:
            n_batch = int(train_idx.shape[0] / batch_size) + 1

        for epoch in range(1, n_epoch + 1):
            print('{} out of {} epochs.'.format(epoch, n_epoch))
            mse = 0
            mae = 0
            loss_sum = 0
            for batch in tqdm.tqdm(range(n_batch)):
                batch_obs = []
                batch_acts = []
                batch_rewards = []
                for idx in train_idx[batch * batch_size:min((batch + 1) * batch_size, train_idx.shape[0]), ]:
                    batch_obs.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['states'])
                    batch_acts.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['actions'])
                    batch_rewards.append(np.load(traj_path + '_traj_' + str(idx) + '.npz')['final_rewards'])

                obs = torch.tensor(np.array(batch_obs)[:, self.len_diff:, ...], dtype=torch.float32)

                if self.n_action == 0:
                    act_dtype = torch.float32
                else:
                    act_dtype = torch.long

                acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=act_dtype)

                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

                if torch.cuda.is_available():
                    obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
                optimizer.zero_grad()
                output = self.model(obs, acts)
                output = self.fc_out(output)
                loss = self.loss(output, rewards)
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                mae += torch.sum(torch.abs(output[:, -1] - rewards))
                mse += torch.sum(torch.square(output[:, -1] - rewards))

            print('Training loss: {}'.format(mae / float(train_idx.shape[0])))
            print('Training MAE: {}'.format(mae / float(train_idx.shape[0])))
            print('Training MSE: {}'.format(mse / float(train_idx.shape[0])))
            scheduler.step()
            # self.test(test_idx, batch_size, traj_path)
            # self.model.train()
            # self.fc_out.train()

        if save_path:
            self.save(save_path)
        return self.model

    def predict(self, obs, acts, rewards):
        """
        :param obs: input observations.
        :param acts: input actions.
        :param rewards: trajectory rewards.
        :return: predicted outputs.
        """

        self.model.eval()
        self.fc_out.eval()

        if torch.cuda.is_available():
            obs, acts = obs.cuda(), acts.cuda()

        preds = self.model(obs, acts)
        preds = self.fc_out(preds)[..., -1]

        return preds.cpu().detach().numpy()

    def test(self, test_idx, batch_size, traj_path):
        """
        :param test_idx: training traj index.
        :param batch_size: training batch size.
        :param traj_path: training traj path.
        :return: prediction error.
        """
        self.model.eval()
        self.fc_out.eval()

        mse = 0
        mae = 0

        if test_idx.shape[0] % batch_size == 0:
            n_batch = int(test_idx.shape[0] / batch_size)
        else:
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
                act_dtype = torch.float32
            else:
                act_dtype = torch.long

            acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=act_dtype)

            rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

            if torch.cuda.is_available():
                obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()

            preds = self.model(obs, acts)
            preds = self.fc_out(preds)[..., -1].detach()
            mae += torch.sum(torch.abs(preds - rewards))
            mse += torch.sum(torch.square(preds - rewards))

        print('Test MAE: {}'.format(mae/float(test_idx.shape[0])))
        print('Test MSE: {}'.format(mse/float(test_idx.shape[0])))
        return mse, mae

    def get_explanations(self, exp_idx, batch_size, traj_path, normalize=True):
        """
        :param exp_idx: training traj index.
        :param batch_size: training batch size.
        :param traj_path: training traj path.
        :param normalize: Normalization or not.
        :return: time step importance.
        """
        self.model.eval()
        self.fc_out.eval()

        n_batch = int(exp_idx.shape[0] / batch_size)

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
                act_dtype = torch.float32
            else:
                act_dtype = torch.long

            acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=act_dtype)

            rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

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

            if batch == 0:
                saliency = redistributed_reward.cpu().detach().numpy()
            else:
                saliency = np.vstack((saliency, redistributed_reward.cpu().detach().numpy()))

        if normalize:
            saliency = (saliency - np.min(saliency, axis=1)[:, None]) / \
                       (np.max(saliency, axis=1)[:, None] - np.min(saliency, axis=1)[:, None] + 1e-16)

        return saliency

    def get_explanations_by_tensor(self, obs, acts, rewards, normalize=True):
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

        if normalize:
            saliency = (saliency - np.min(saliency, axis=1)[:, None]) / \
                       (np.max(saliency, axis=1)[:, None] - np.min(saliency, axis=1)[:, None] + 1e-16)

        return saliency

    def train_by_tensor(self, train_loader, n_epoch, lr=0.01, gamma=0.1, optimizer_choice='adam', save_path=None):
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

    def test_by_tensor(self, test_loader):
        """
        :param test_loader: testing data loader.
        :return: prediction error.
        """
        self.model.eval()
        self.fc_out.eval()

        mse = 0
        mae = 0
        for obs, acts, rewards in test_loader:
            if torch.cuda.is_available():
                obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
            preds = self.model(obs, acts)
            preds = self.fc_out(preds)[..., -1]
            mae += torch.sum(torch.abs(preds - rewards))
            mse += torch.sum(torch.square(preds - rewards))

        print('Test MAE: {}'.format(mae / float(len(test_loader.dataset))))
        print('Test MSE: {}'.format(mse / float(len(test_loader.dataset))))
        return mse, mae

    def exp_fid_stab(self, exp_idx, batch_size, traj_path, task_type, n_stab_sample, eps=0.05):
        """
        return explanations, fidelity, stability, runtime.
        """

        n_batch = int(exp_idx.shape[0] / batch_size)
        sum_time = 0
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
                act_dtype = torch.float32
            else:
                act_dtype = torch.long

            acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=act_dtype)

            rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)
            start = timeit.default_timer()
            sal_rudder = self.get_explanations_by_tensor(obs, acts, rewards)
            stop = timeit.default_timer()
            # print('Rudder explanation time of {} samples: {}.'.format(obs.shape[0], (stop - start)))
            sum_time += (stop - start)
            preds_orin = self.predict(obs, acts, rewards)
            if task_type == 'classification':
                abs_diff_1, fid_1 = self.exp_fid2nn_zero_one(obs, acts, rewards, self, sal_rudder, preds_orin)
                abs_diff_2, fid_2 = self.exp_fid2nn_topk(obs, acts, rewards, self, sal_rudder, preds_orin, int(obs.shape[1] * 0.05))
                abs_diff_3, fid_3 = self.exp_fid2nn_topk(obs, acts, rewards, self, sal_rudder, preds_orin, int(obs.shape[1] * 0.15))
                abs_diff_4, fid_4 = self.exp_fid2nn_topk(obs, acts, rewards, self, sal_rudder, preds_orin, int(obs.shape[1] * 0.25))
            else:
                fid_1, _ = self.exp_fid2nn_zero_one(obs, acts, rewards, self, sal_rudder, preds_orin)
                fid_2, _ = self.exp_fid2nn_topk(obs, acts, rewards, self, sal_rudder, preds_orin, int(obs.shape[1] * 0.05))
                fid_3, _ = self.exp_fid2nn_topk(obs, acts, rewards, self, sal_rudder, preds_orin, int(obs.shape[1] * 0.15))
                fid_4, _ = self.exp_fid2nn_topk(obs, acts, rewards, self, sal_rudder, preds_orin, int(obs.shape[1] * 0.25))

            stab = exp_stablity(obs, acts, rewards, self, sal_rudder, n_stab_sample, eps)
            fid = np.concatenate((fid_1[None, ], fid_2[None, ], fid_3[None, ], fid_4[None, ]))
            if task_type == 'classification':
                abs_diff = np.concatenate((abs_diff_1[None, ], abs_diff_2[None, ], abs_diff_3[None, ],
                                           abs_diff_4[None, ]))
            else:
                abs_diff = fid

            if batch == 0:
                sal_rudder_all = sal_rudder
                fid_all = fid
                stab_all = stab
                abs_diff_all = abs_diff
            else:
                sal_rudder_all = np.vstack((sal_rudder_all, sal_rudder))
                fid_all = np.concatenate((fid_all, fid), axis=1)
                stab_all = np.concatenate((stab_all, stab))
                abs_diff_all = np.concatenate((abs_diff_all, abs_diff), axis=1)
        mean_time = sum_time/exp_idx.shape[0]

        return sal_rudder_all, fid_all, stab_all, abs_diff_all, mean_time

    @staticmethod
    def exp_fid2nn_zero_one(obs, acts, rewards, explainer, saliency, preds_orin):
        obs.requires_grad = False
        acts.requires_grad = False

        if type(saliency) == np.ndarray:
            saliency = torch.tensor(saliency, dtype=torch.float32)

        if len(obs.shape) == 5:
            saliency = saliency[:, :, None, None, None]
        else:
            saliency = saliency[:, :, None]

        preds_sal = explainer.predict(saliency * obs, acts, rewards)
        abs_diff = np.abs(preds_sal-preds_orin)
        rewards = rewards.cpu().detach().numpy()
        preds_sal = np.abs(preds_sal - rewards)
        preds_sal = 1 - ((np.exp(preds_sal) - np.exp(-preds_sal)) / (np.exp(preds_sal) + np.exp(-preds_sal)))
        return abs_diff, -np.log(preds_sal)

    @staticmethod
    def exp_fid2nn_topk(obs, acts, rewards, explainer, saliency, preds_orin, num_fea):
        obs.requires_grad = False
        acts.requires_grad = False

        if type(saliency) == torch.Tensor:
            saliency = saliency.cpu().detach().numpy()

        importance_id_sorted = np.argsort(saliency, axis=1)[:, ::-1] # high to low.
        nonimportance_id = importance_id_sorted[:, num_fea:]
        nonimportance_id = nonimportance_id.copy()

        mask_obs = torch.ones_like(obs, dtype=torch.float32)
        mask_acts = torch.ones_like(acts, dtype=torch.long)

        for j in range(acts.shape[0]):
            mask_acts[j, nonimportance_id[j,]] = 0
            mask_obs[j, nonimportance_id[j,]] = 0

        preds_sal = explainer.predict(obs * mask_obs, acts * mask_acts, rewards)

        abs_diff = np.abs(preds_sal-preds_orin)
        rewards = rewards.cpu().detach().numpy()
        preds_sal = np.abs(preds_sal - rewards)
        preds_sal = 1 - ((np.exp(preds_sal) - np.exp(-preds_sal)) / (np.exp(preds_sal) + np.exp(-preds_sal)))
        return abs_diff, -np.log(preds_sal)
