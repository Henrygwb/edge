# Implementation of our RNN-DGP classification/regression model .
# 1. (Capturing the sequential dependency) Use a RNN (seq2seq) inside the RBF kernel function.
# 2. (Non-Gaussian likelihood) Use the Variational inference with the variational distribution as q(u)~N(\mu, LL^T).
# 3. Additive Gaussian process with two kernels - data (trajectory/individual) level and feature level.
# 4. Standard variational strategy with whitening and SoR.
#    (Faster way of computing p(f|u)) Use the inducing point technique with the SoR approximation of
#    p(f|u)=K_{x,z}K_{z,z}u and p(f_*|u)=K_{x_*,z}K_{z,z}u.
# 5. Standard variational strategy with SoR.
# 6. Natural Gradient Descent: Second-order optimization with Hessian replaced by Fisher Information matrix.
#    Optimization in the distribution space with the KL divergence as the distance measure. Better choice for
#    Variational distribution parameter.
# 7. Standard variational strategy with Contour Integral Quadrature to approximate K_{zz}^{-1/2},
#    Use it together with NGD.
# 8. Grid variational strategy with KSI.
# (Faster way of computing K_{x,z} in p(f|u)) KISS-GP: Use the local kernel interpolation technique
# (Structured kernel interpolation) to approximate K_{x,z}K_{z,z} with the interpolation matrix M.
# 9. Orthogonally decoupled VGPs. Using a different set of inducing points for the mean and covariance functions.
#    Use more inducing points for the mean and fewer inducing points for the covariance.

import tqdm
import torch
import timeit
import gpytorch
import numpy as np
import torch.optim as optim
from .quantitative_test import exp_fid2nn_topk, exp_fid2nn_zero_one
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from .gp_utils import DGPXRLModel, CustomizedGaussianLikelihood, CustomizedSoftmaxLikelihood, NNSoftmaxLikelihood

# # Hyper-parameters
# HIDDENS = [10, 5, 3]
# NUM_INDUCING_POINTS = 20
# USING_SOR = False # Whether to use SoR approximation, not applicable for KSI and CIQ.
# USING_KSI = False # Whether to use KSI approximation, using this with other options as False.
# USING_NGD = False # Whether to use natural gradient descent.
# USING_CIQ = False # Whether to use Contour Integral Quadrature to approximate K_{zz}^{-1/2}, Use it together with NGD.
# USING_OrthogonallyDecouple = False # Using together NGD may cause numerical issue.
# GRID_BOUNDS = [(-3, 3)] * HIDDENS[-1] * 2
# LIKELIHOOD_TYPE = 'classification'
# # LIKELIHOOD_TYPE = 'regression'
# RNN_CELL_TYPE = 'GRU' # 'LSTM'
# DROPOUT_RATE = 0.0
# NUM_CLASSES = 2
# N_EPOCHS = 5
# N_BATCHS = 4
# LR = 0.01
# OPTIMIZER = 'adam' # 'sgd'
# GAMMA = 0.1 # LR decay multiplicative factor.
# SAVE_PATH = ' '
# LOAD_PATH = 'checkpoint.data'


class DGPXRL(object):
    def __init__(self, train_len, seq_len, len_diff, input_dim, hiddens, likelihood_type, lr, optimizer_type, n_epoch,
                 gamma, num_inducing_points, n_action=0, encoder_type='MLP', embed_dim=16, inducing_points=None,
                 mean_inducing_points=None, dropout_rate=0.25, num_class=None, rnn_cell_type='GRU', normalize=False,
                 grid_bounds=None, using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
                 using_OrthogonallyDecouple=False, weight_x=False, lambda_1=0.01):
        """
        Define the full model.
        :param train_len: training data length.
        :param seq_len: trajectory length.
        :param len_diff: trajectory len_diff.
        :param input_dim: input state/action dimension.
        :param hiddens: hidden layer dimentions.
        :param likelihood_type: likelihood type.
        :param lr: learning rate.
        :param optimizer_type.
        :param n_epoch.
        :param gamma.
        :param num_inducing_points: number of inducing points.
        :param n_action: number of actions.
        :param embed_dim: actions embedding dim.
        :param encoder_type: encoder type ('MLP' or 'CNN').
        :param inducing_points: inducing points at the latent space Z (num_inducing_points, 2*hiddens[-1]).
        :param mean_inducing_points: mean inducing points, used for orthogonally decoupled VGP.
        :param dropout_rate: MLP dropout rate.
        :param rnn_cell_type: the RNN cell type.
        :param normalize: whether to normalize the input.
        :param grid_bounds: grid bounds.
        :param using_ngd: Whether to use natural gradient descent.
        :param using_ksi: Whether to use KSI approximation, using this with other options as False.
        :param using_ciq: Whether to use Contour Integral Quadrature to approximate K_{zz}^{-1/2}, Use it together with NGD.
        :param using_sor: Whether to use SoR approximation, not applicable for KSI and CIQ.
        :param using_OrthogonallyDecouple
        :param weight_x: whether the mixing weights depend on inputs.
        :param lambda_1: coefficient before the lasso/local linear regularization (here we time it to lr).
        """
        self.len_diff = len_diff
        self.train_len = train_len
        self.n_action = n_action
        self.likelihood_type = likelihood_type
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.n_epoch = n_epoch
        self.gamma = gamma
        self.using_ngd = using_ngd
        self.weight_x = weight_x
        # Build the likelihood layer (Regression and classification).
        if self.likelihood_type == 'regression':
            print('Conduct regression and use GaussianLikelihood')
            self.likelihood = CustomizedGaussianLikelihood(num_features=seq_len, weight_x=weight_x, hidden_dims=2*hiddens[-1])
        elif self.likelihood_type == 'classification':
            print('Conduct classification and use softmaxLikelihood')
            if self.weight_x:
                print('Varying the mixing weights with an ')
                self.likelihood = NNSoftmaxLikelihood(num_features=seq_len, num_classes=num_class,
                                                      input_encoding_dim=hiddens[-1]*2)
            else:
                self.likelihood = CustomizedSoftmaxLikelihood(num_features=seq_len, num_classes=num_class)
        else:
            print('Default choice is regression and use GaussianLikelihood')
            self.likelihood = CustomizedGaussianLikelihood(num_features=seq_len)

        # Compute the loss (ELBO) likelihood + KL divergence.
        self.model = DGPXRLModel(seq_len=seq_len, input_dim=input_dim, hiddens=hiddens, likelihood_type=likelihood_type,
                                 n_action=n_action, encoder_type=encoder_type, embed_dim=embed_dim, dropout_rate=dropout_rate,
                                 rnn_cell_type=rnn_cell_type, normalize=normalize, num_inducing_points=num_inducing_points,
                                 inducing_points=inducing_points, mean_inducing_points=mean_inducing_points,
                                 grid_bounds=grid_bounds, using_ngd=using_ngd, using_ksi=using_ksi, using_ciq=using_ciq,
                                 using_sor=using_sor, using_OrthogonallyDecouple=using_OrthogonallyDecouple)
        print(self.model)

        # First, sampling from q(f) with shape [n_sample, n_data].
        # Then, the likelihood function times it with the mixing weight and get the marginal likelihood p(y|f).
        # VariationalELBO will call _ApproximateMarginalLogLikelihood, which then computes the marginal likelihood by
        # calling the likelihood function (the expected_log_prob in the likelihood class)
        # and the KL divergence (VariationalStrategy.kl_divergence()).
        # ELBO = E_{q(f)}[p(y|f)] - KL[q(u)||p(u)].
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model.gp_layer,
                                                 num_data=self.train_len)

        # Define the optimizer over the parameters.
        # (RNN parameters, RBF kernel parameters, Z, variational parameters, mixing weight).
        if self.using_ngd:
            self.variational_ngd_optimizer = gpytorch.optim.NGD(self.model.gp_layer.variational_parameters(),
                                                                num_data=self.train_len, lr=self.lr*10)
            if self.optimizer_type == 'adam':
                self.hyperparameter_optimizer = optim.Adam([
                    {'params': self.model.encoder.parameters(), 'weight_decay': 1e-4},
                    {'params': self.model.gp_layer.hyperparameters(), 'lr': self.lr * 0.01},
                    {'params': self.likelihood.parameters()}, ], lr=self.lr, weight_decay=0)
            else:
                self.hyperparameter_optimizer = optim.SGD([
                    {'params': self.model.encoder.parameters(), 'weight_decay': 1e-4},
                    {'params': self.model.gp_layer.hyperparameters(), 'lr': self.lr * 0.01},
                    {'params': self.likelihood.parameters()}, ], lr=self.lr, weight_decay=0)
            # Learning rate decay schedule.
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.variational_ngd_optimizer,
                                                            milestones=[0.5 * self.n_epoch, 0.75 * self.n_epoch],
                                                            gamma=self.gamma)
            self.scheduler_hyperparameter = optim.lr_scheduler.MultiStepLR(self.hyperparameter_optimizer,
                                                                           milestones=[0.5 * self.n_epoch, 0.75 * self.n_epoch],
                                                                           gamma=self.gamma)
        else:
            if self.optimizer_type == 'adam':
                self.optimizer = optim.Adam([
                    {'params': self.model.encoder.parameters(), 'weight_decay': 1e-4},
                    {'params': self.model.gp_layer.hyperparameters(), 'lr': self.lr * 0.01},
                    {'params': self.model.gp_layer.variational_parameters()},
                    {'params': self.likelihood.parameters()}, ], lr=self.lr, weight_decay=0)
            else:
                self.optimizer = optim.SGD([
                    {'params': self.model.encoder.parameters(), 'weight_decay': 1e-4},
                    {'params': self.model.gp_layer.hyperparameters(), 'lr': self.lr * 0.01},
                    {'params': self.model.gp_layer.variational_parameters()},
                    {'params': self.likelihood.parameters()}, ], lr=self.lr,
                    momentum=0.9, nesterov=True, weight_decay=0)

            # Learning rate decay schedule.
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                            milestones=[0.5 * self.n_epoch, 0.75 * self.n_epoch],
                                                            gamma=self.gamma)

        self.likelihood_regular_optimizer = optim.Adam([{'params': self.likelihood.parameters()}],
                                                       lr=self.lr * lambda_1, weight_decay=0) # penalize the lr with lambda_1.

        self.scheduler_regular = optim.lr_scheduler.MultiStepLR(self.likelihood_regular_optimizer,
                                                                milestones=[0.5 * self.n_epoch, 0.75 * self.n_epoch],
                                                                gamma=self.gamma)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

    def save(self, save_path):
        state_dict = self.model.state_dict()
        likelihood_state_dict = self.likelihood.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, save_path)
        return 0

    # Load a pretrained model.
    def load(self, load_path):
        """
        :param load_path: load model path.
        :return: model, likelihood.
        """
        dicts = torch.load(load_path, map_location=torch.device('cpu'))
        model_dict = dicts['model']
        likelihood_dict = dicts['likelihood']
        self.model.load_state_dict(model_dict)
        self.likelihood.load_state_dict(likelihood_dict)
        return self.model, self.likelihood

    def train(self, train_idx, test_idx, batch_size, traj_path, eps=0.01, local_samples=10,
              save_path=None, likelihood_sample_size=8):
        """
        :param train_idx: training traj index.
        :param test_idx: testing traj index.
        :param batch_size: training batch size.
        :param eps: local linear region.
        :param local_samples: .
        :param traj_path: training traj path.
        :param save_path: model save path.
        :param likelihood_sample_size: .
        :return: trained model.
        """

        # specify the mode of the modules in the model; some modules may have different behaviors under training and eval
        # module, such as dropout and batch normalization.
        self.model.train()
        self.likelihood.train()

        if train_idx.shape[0] % batch_size == 0:
            n_batch = int(train_idx.shape[0] / batch_size)
        else:
            n_batch = int(train_idx.shape[0] / batch_size) + 1

        for epoch in range(1, self.n_epoch + 1):
            print('{} out of {} epochs.'.format(epoch, self.n_epoch))
            mse = 0
            mae = 0
            loss_sum = 0
            loss_reg_sum = 0
            preds_all = []
            rewards_all = []
            with gpytorch.settings.use_toeplitz(False):
                with gpytorch.settings.num_likelihood_samples(likelihood_sample_size):
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

                        if self.likelihood_type == 'classification':
                            rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
                        else:
                            rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

                        if torch.cuda.is_available():
                            obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()

                        if self.using_ngd:
                            self.variational_ngd_optimizer.zero_grad()
                            self.hyperparameter_optimizer.zero_grad()
                        else:
                            self.optimizer.zero_grad()

                        output, features = self.model(obs, acts)  # marginal variational posterior, q(f|x).
                        if self.weight_x:
                            loss = -self.mll(output, rewards, input_encoding=features)  # approximated ELBO.
                        else:
                            loss = -self.mll(output, rewards)  # approximated ELBO.

                        loss.backward()

                        if self.using_ngd:
                            self.variational_ngd_optimizer.step()
                            self.hyperparameter_optimizer.step()
                        else:
                            self.optimizer.step()

                        loss_sum += loss.item()

                        self.likelihood_regular_optimizer.zero_grad()
                        if self.weight_x and self.likelihood_type == 'classification':
                            # lasso.
                            output, features = self.model(obs, acts)  # marginal variational posterior, q(f|x).
                            features_sum = features.detach().sum(-1)
                            weight_output = self.likelihood.weight_encoder(features_sum)
                            lasso_term = torch.norm(weight_output, p=1) # lasso
                            lasso_term.backward()
                            loss_reg_sum += lasso_term

                            # local linear regularization.
                            # output, features = self.model(obs, acts)  # marginal variational posterior, q(f|x).
                            # features_sum = features.detach().sum(-1)
                            # weight_output = self.likelihood.weight_encoder(features_sum)
                            # weight_output = weight_output.detach()
                            # eps = eps * weight_output.abs().max()
                            # local_perturbations = torch.rand(local_samples, features.shape[0], features.shape[1])
                            # if torch.cuda.is_available():
                            #     local_perturbations = local_perturbations.cuda()
                            # local_perturbations = local_perturbations * 2 * eps - eps
                            # perturbed_output = self.likelihood.weight_encoder(features_sum+local_perturbations)
                            # local_linear_loss = torch.abs(weight_output-perturbed_output)
                            # local_linear_loss = local_linear_loss.sum((1, 2)).mean()
                            # # local_linear_loss.backward()
                            # loss_reg_sum += local_linear_loss
                        else:
                            lasso_term = torch.norm(self.likelihood.mixing_weights, p=1) # lasso
                            lasso_term.backward()
                            loss_reg_sum += lasso_term
                            self.likelihood_regular_optimizer.step()

                        if self.weight_x:
                            output = self.likelihood(output, input_encoding=features)
                        else:
                            output = self.likelihood(output)  # This gives us 8 samples from the predictive distribution (q(y|f_*)).
                        if self.likelihood_type == 'classification':
                            preds = output.probs.mean(0).argmax(-1)  # Take the mean over all of the sample we've drawn (y = E_{f_* ~ q(f_*)}[y|f_*]).
                            preds_all.extend(preds.cpu().detach().numpy().tolist())
                            rewards_all.extend(rewards.cpu().detach().numpy().tolist())
                        else:
                            preds = output.mean
                            mae += torch.sum(torch.abs(preds - rewards))
                            mse += torch.sum(torch.square(preds - rewards))
                    print('ELBO loss {}'.format(loss_sum/n_batch))
                    print('Regularization loss {}'.format(loss_reg_sum/n_batch))

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
                        print('Train MAE: {}'.format(mae / float(self.train_len)))
                        print('Train MSE: {}'.format(mse / float(self.train_len)))

            if self.using_ngd:
                self.scheduler_hyperparameter.step()

            self.scheduler.step()
            self.scheduler_regular.step()
            # self.test(test_idx, batch_size, traj_path)
            # self.model.train()
            # self.likelihood.train()
            # if epoch % 20 == 0:
            self.save(save_path + '_' + str(epoch) + '_model.data')
        if save_path:
            self.save(save_path)
        return self.model

    def predict(self, obs, acts, rewards, likelihood_sample_size=16):
        """
        :param obs: input observations.
        :param acts: input actions.
        :param rewards: trajectory rewards.
        :return: predicted outputs.
        """
        self.model.eval()
        self.likelihood.eval()

        if torch.cuda.is_available():
            obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
            self.model, self.likelihood = self.model.cuda(), self.likelihood.cuda()

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(likelihood_sample_size):
            f_predicted, features = self.model(obs, acts)
            if self.weight_x:
                output = self.likelihood(f_predicted, input_encoding=features)
            else:
                output = self.likelihood(f_predicted)  # This gives us 8 samples from the predictive distribution (q(y|f_*)).

        if self.likelihood_type == 'classification':
            preds = output.probs.mean(0)
        else:
            preds = output.mean

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

    def test(self, test_idx, batch_size, traj_path, likelihood_sample_size=16):
        """
        :param test_idx: training traj index.
        :param batch_size: training batch size.
        :param traj_path: training traj path.
        :param likelihood_sample_size: .
        :return: prediction error.
        """

        # Specify that the model is in eval mode.
        self.model.eval()
        self.likelihood.eval()

        mse = 0
        mae = 0
        preds_all = []
        rewards_all = []

        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(likelihood_sample_size):

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

                if self.likelihood_type == 'classification':
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
                else:
                    rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

                if torch.cuda.is_available():
                    obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()

                f_predicted, features = self.model(obs, acts)
                if self.weight_x:
                    output = self.likelihood(f_predicted, input_encoding=features)
                else:
                    output = self.likelihood(
                        f_predicted)  # This gives us 8 samples from the predictive distribution (q(y|f_*)).

                if self.likelihood_type == 'classification':
                    preds = output.probs.mean(0).argmax(-1)  # Take the mean over all of the sample we've drawn (y = E_{f_* ~ q(f_*)}[y|f_*]).
                    preds_all.extend(preds.cpu().detach().numpy().tolist())
                    rewards_all.extend(rewards.cpu().detach().numpy().tolist())
                else:
                    preds = output.mean.detach()
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

    def get_explanations(self, exp_idx, batch_size, traj_path, logit=True, normalize=True):
        """
        :param exp_idx: training traj index.
        :param batch_size: training batch size.
        :param traj_path: training traj path.
        :param normalize: normalize.
        :param logit: .
        :return: time step importance.
        """
        if torch.cuda.is_available():
            self.model, self.likelihood = self.model.cuda(), self.likelihood.cuda()

        self.model.eval()
        self.likelihood.eval()
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

            if self.likelihood_type == 'classification':
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
            else:
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

            if torch.cuda.is_available():
                obs, acts = obs.cuda(), acts.cuda()
            step_embedding, traj_embedding = self.model.encoder(obs, acts)  # (N, T, P) -> (N, T, D), (N, D).
            traj_embedding = traj_embedding[:, None, :].repeat(1, obs.shape[1], 1)  # (N, D) -> (N, T, D)
            features = torch.cat([step_embedding, traj_embedding], dim=-1)  # (N, T, 2D)
            features = features.view(obs.size(0) * obs.size(1), features.size(-1))
            covar_all = self.model.gp_layer.covar_module(features)
            covar_step = self.model.gp_layer.step_kernel(features)
            covar_traj = self.model.gp_layer.traj_kernel(features)

            if self.weight_x:
                input_encoding = features.sum(-1)
                importance_all = self.likelihood.weight_encoder(input_encoding)
                importance_all = importance_all.reshape(importance_all.shape[0], self.likelihood.num_features,
                                                        self.likelihood.num_classes)
            else:
                importance_all = self.likelihood.mixing_weights
                importance_all = importance_all.transpose(1, 0)

            importance_all = importance_all.cpu().detach().numpy()

            if len(importance_all.shape) == 2:
                importance_all = np.repeat(importance_all[None, ...], rewards.shape[0], axis=0)

            if importance_all.shape[-1] > 1:
                # back from the logit i.e., WF -> W^T
                importance = importance_all[list(range(rewards.shape[0])), :, rewards]
                if not logit:
                    # todo: support multi-classes (larger than two).
                    # back from the logit i.e., WF -> W^T
                    importance_oppo = importance_all[list(range(rewards.shape[0])), :, (1 - rewards)]
                    importance = importance - importance_oppo
            else:
                importance = np.squeeze(importance_all, -1)

            # TODO: combine importance weight with covariance structure.

            if batch == 0:
                saliency_all = importance
                covar_all_all = covar_all.numpy()[None, ...]
                covar_step_all = covar_step.numpy()[None, ...]
                covar_traj_all = covar_traj.numpy()[None, ...]
            else:
                saliency_all = np.vstack((saliency_all, importance))
                covar_all_all = np.concatenate((covar_all_all, covar_all.numpy()[None, ...]))
                covar_step_all = np.concatenate((covar_step_all, covar_all.numpy()[None, ...]))
                covar_traj_all = np.concatenate((covar_traj_all, covar_all.numpy()[None, ...]))

        if normalize:
            saliency_all = (saliency_all - np.min(saliency_all, axis=1)[:, None]) \
                         / (np.max(saliency_all, axis=1)[:, None] - np.min(saliency_all, axis=1)[:, None] + 1e-16)

        return saliency_all, (covar_all_all, covar_traj_all, covar_step_all)

    def train_by_tensor(self, save_path=None, likelihood_sample_size=8):

        # specify the mode of the modules in the model; some modules may have different behaviors under training and eval
        # module, such as dropout and batch normalization.
        self.model.train()
        self.likelihood.train()

        for epoch in range(1, self.n_epoch + 1):
            mse = 0
            mae = 0
            correct = 0
            with gpytorch.settings.use_toeplitz(False):
                minibatch_iter = tqdm.tqdm(self.train_loader, desc=f"(Epoch {epoch}) Minibatch")
                with gpytorch.settings.num_likelihood_samples(likelihood_sample_size):
                    for obs, acts, rewards in minibatch_iter:
                        if torch.cuda.is_available():
                            obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
                        if self.using_ngd:
                            self.variational_ngd_optimizer.zero_grad()
                            self.hyperparameter_optimizer.zero_grad()
                        else:
                            self.optimizer.zero_grad()
                        output = self.model(obs, acts)  # marginal variational posterior, q(f|x).
                        loss = -self.mll(output, rewards)  # approximated ELBO.
                        loss.backward()
                        if self.using_ngd:
                            self.variational_ngd_optimizer.step()
                            self.hyperparameter_optimizer.step()
                        else:
                            self.optimizer.step()

                        output = self.likelihood(
                            output)  # This gives us 8 samples from the predictive distribution (q(y|f_*)).
                        if self.likelihood_type == 'classification':
                            pred = output.probs.mean(0).argmax(
                                -1)  # Take the mean over all of the sample we've drawn (y = E_{f_* ~ q(f_*)}[y|f_*]).
                            correct += pred.eq(rewards.view_as(pred)).cpu().sum()
                        else:
                            preds = output.mean
                            mae += torch.sum(torch.abs(preds - rewards))
                            mse += torch.sum(torch.square(preds - rewards))

                        minibatch_iter.set_postfix(loss=loss.item())

                    if self.likelihood_type == 'classification':
                        print('Train set: Accuracy: {}/{} ({}%)'.format(
                            correct, len(self.train_loader.dataset),
                            100. * correct / float(len(self.train_loader.dataset))
                        ))
                    else:
                        print('Train MAE: {}'.format(mae / float(len(self.train_loader.dataset))))
                        print('Train MSE: {}'.format(mse / float(len(self.train_loader.dataset))))

            self.scheduler.step()

        if save_path:
            self.save(save_path)
        return self.model

    # test function.
    def test_by_tensor(self, test_loader, likelihood_sample_size=16):
        # Specify that the model is in eval mode.
        self.model.eval()
        self.likelihood.eval()

        correct = 0
        mse = 0
        mae = 0
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(likelihood_sample_size):
            for obs, acts, rewards in test_loader:
                if torch.cuda.is_available():
                    obs, acts, rewards = obs.cuda(), acts.cuda(), rewards.cuda()
                f_predicted = self.model(obs, acts)
                output = self.likelihood(
                    f_predicted)  # This gives us 16 samples from the predictive distribution (q(y|f_*)).
                if self.likelihood_type == 'classification':
                    pred = output.probs.mean(0).argmax(
                        -1)  # Take the mean over all of the sample we've drawn (y = E_{f_* ~ q(f_*)}[y|f_*]).
                    correct += pred.eq(rewards.view_as(pred)).cpu().sum()
                else:
                    preds = output.mean
                    mae += torch.sum(torch.abs(preds - rewards))
                    mse += torch.sum(torch.square(preds - rewards))
            if self.likelihood_type == 'classification':
                print('Test set: Accuracy: {}/{} ({}%)'.format(
                    correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
                ))
            else:
                print('Test MAE: {}'.format(mae / float(len(test_loader.dataset))))
                print('Test MSE: {}'.format(mse / float(len(test_loader.dataset))))

    def get_explanations_by_tensor(self, obs, acts, rewards, logit=True, normalize=True):
        """
        :param obs: input observations.
        :param acts: input actions.
        :param rewards: trajectory rewards.
        :param logit: whether to back to logit.
        :return: time step importance.
        """

        self.model.eval()
        self.likelihood.eval()

        if torch.cuda.is_available():
            obs, acts = obs.cuda(), acts.cuda()
            self.model, self.likelihood = self.model.cuda(), self.likelihood.cuda()

        step_embedding, traj_embedding = self.model.encoder(obs, acts)  # (N, T, P) -> (N, T, D), (N, D).
        traj_embedding = traj_embedding[:, None, :].repeat(1, obs.shape[1], 1)  # (N, D) -> (N, T, D)
        features = torch.cat([step_embedding, traj_embedding], dim=-1)  # (N, T, 2D)
        features_reshaped = features.view(obs.size(0) * obs.size(1), features.size(-1))
        covar_all = self.model.gp_layer.covar_module(features_reshaped)
        covar_step = self.model.gp_layer.step_kernel(features_reshaped)
        covar_traj = self.model.gp_layer.traj_kernel(features_reshaped)

        if self.weight_x and self.likelihood_type == 'classification':
            input_encoding = features.sum(-1)
            importance_all = self.likelihood.weight_encoder(input_encoding)
            importance_all = importance_all.view(importance_all.shape[0], self.likelihood.num_features,
                                                 self.likelihood.num_classes)
        else:
            importance_all = self.likelihood.mixing_weights.clone()
            importance_all = importance_all.transpose(1, 0)

        importance_all = importance_all.cpu().detach().numpy()

        if len(importance_all.shape) == 2:
            importance_all = np.repeat(importance_all[None, ...], rewards.shape[0], axis=0)

        if importance_all.shape[-1] > 1:
            importance = importance_all[list(range(rewards.shape[0])), :, rewards]  # back from the logit i.e., WF -> W^T
            if not logit:
                # todo: support multi-classes (larger than two).
                # back from the logit i.e., WF -> W^T
                importance_oppo = importance_all[list(range(rewards.shape[0])), :, (1-rewards)]
                importance = importance - importance_oppo
        else:
            importance = np.squeeze(importance_all, -1)

        # TODO: combine importance weight with covariance structure.

        if normalize:
            importance = (importance - np.min(importance, axis=1)[:, None]) \
                         / (np.max(importance, axis=1)[:, None] - np.min(importance, axis=1)[:, None] + 1e-16)

        return importance, (covar_all.cpu().detach().numpy(), covar_traj.cpu().detach().numpy(),
                            covar_step.cpu().detach().numpy())

    def exp_fid_stab(self, exp_idx, batch_size, traj_path, logit=True, n_stab_samples=5, eps=0.05):

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
                act_dtype = torch.float32
            else:
                act_dtype = torch.long

            acts = torch.tensor(np.array(batch_acts)[:, self.len_diff:], dtype=act_dtype)

            if self.likelihood_type == 'classification':
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.long)
            else:
                rewards = torch.tensor(np.array(batch_rewards), dtype=torch.float32)

            start = timeit.default_timer()
            sal, cov = self.get_explanations_by_tensor(obs, acts, rewards, logit)
            stop = timeit.default_timer()
            # print('Explanation time of {} samples: {}.'.format(obs.shape[0], (stop - start)))
            sum_time += (stop - start)
            preds_orin = self.predict(obs, acts, rewards)
            if self.likelihood_type == 'classification':
                fid_1, acc_1_temp, abs_diff_1 = exp_fid2nn_zero_one(obs, acts, rewards, self, sal, preds_orin)
                fid_2, acc_2_temp, abs_diff_2 = self.exp_fid2nn_topk(obs, acts, rewards, sal, preds_orin,
                                                                     int(obs.shape[1] * 0.05))
                fid_3, acc_3_temp, abs_diff_3 = self.exp_fid2nn_topk(obs, acts, rewards, sal, preds_orin,
                                                                     int(obs.shape[1] * 0.10))
                fid_4, acc_4_temp, abs_diff_4 = self.exp_fid2nn_topk(obs, acts, rewards, sal, preds_orin,
                                                                     int(obs.shape[1] * 0.15))
                acc_1 += acc_1_temp
                acc_2 += acc_2_temp
                acc_3 += acc_3_temp
                acc_4 += acc_4_temp
            else:
                fid_1 = exp_fid2nn_zero_one(obs, acts, rewards, self, sal, preds_orin)
                fid_2 = self.exp_fid2nn_topk(obs, acts, rewards, sal, preds_orin, int(obs.shape[1] * 0.05))
                fid_3 = self.exp_fid2nn_topk(obs, acts, rewards, sal, preds_orin, int(obs.shape[1] * 0.10))
                fid_4 = self.exp_fid2nn_topk(obs, acts, rewards, sal, preds_orin, int(obs.shape[1] * 0.15))

            stab = self.exp_stablity(obs, acts, rewards, sal, n_stab_samples, eps)
            fid = np.concatenate((fid_1[None,], fid_2[None,], fid_3[None,], fid_4[None,]))

            if self.likelihood_type == 'classification':
                abs_diff = np.concatenate((abs_diff_1[None, ], abs_diff_2[None, ], abs_diff_3[None, ],
                                           abs_diff_4[None, ]))
            else:
                abs_diff = fid

            if batch == 0:
                sal_all = sal
                fid_all = fid
                stab_all = stab
                covar_all_all = cov[0][None, ...]
                covar_traj_all = cov[1][None, ...]
                covar_step_all = cov[2][None, ...]
                abs_diff_all = abs_diff
            elif batch < 5:
                sal_all = np.vstack((sal_all, sal))
                fid_all = np.concatenate((fid_all, fid), axis=1)
                stab_all = np.concatenate((stab_all, stab))
                covar_all_all = np.concatenate((covar_all_all, cov[0][None, ...]))
                covar_traj_all = np.concatenate((covar_traj_all, cov[1][None, ...]))
                covar_step_all = np.concatenate((covar_step_all, cov[2][None, ...]))
                abs_diff_all = np.concatenate((abs_diff_all, abs_diff), axis=1)
            else:
                abs_diff_all = np.concatenate((abs_diff_all, abs_diff), axis=1)
                sal_all = np.vstack((sal_all, sal))
                fid_all = np.concatenate((fid_all, fid), axis=1)
                stab_all = np.concatenate((stab_all, stab))

        mean_time = sum_time / exp_idx.shape[0]
        acc_1 = acc_1 / n_batch
        acc_2 = acc_2 / n_batch
        acc_3 = acc_3 / n_batch
        acc_4 = acc_4 / n_batch
        print(sal_all.shape)
        print(covar_all_all.shape)
        print(covar_traj_all.shape)
        print(covar_step_all.shape)
        print(fid_all.shape)
        print(stab_all.shape)
        print(abs_diff_all.shape)

        return sal_all, (covar_all_all, covar_traj_all, covar_step_all), fid_all, stab_all, \
               [acc_1, acc_2, acc_3, acc_4], abs_diff_all, mean_time

    def exp_stablity(self, obs, acts, rewards, saliency, num_sample=5, eps=0.05):
        def get_l2_diff(x, y):
            diff_square = np.square((x - y))
            if len(x.shape) > 2:
                diff_square_sum = np.apply_over_axes(np.sum, diff_square, list(range(2, len(x.shape))))
                diff_square_sum = diff_square_sum.reshape(x.shape[0], x.shape[1])
            else:
                diff_square_sum = diff_square
            diff_square_sum = diff_square_sum.sum(-1)
            diff = np.sqrt(diff_square_sum)
            return diff
        stab = []
        for _ in range(num_sample):
            noise = torch.rand(obs.shape) * eps
            noisy_obs = obs + noise
            diff_x = get_l2_diff(obs.cpu().detach().numpy(), noisy_obs.cpu().detach().numpy())
            diff_x[diff_x==0] = 1e-8
            noisy_saliency, _ = self.get_explanations_by_tensor(noisy_obs, acts, rewards)
            diff_x_sal = get_l2_diff(saliency, noisy_saliency)
            stab.append((diff_x_sal / diff_x))
        stab = np.array(stab)
        stab = np.max(stab, axis=0)

        return stab

    def exp_fid2nn_topk(self, obs, acts, rewards, saliency, preds_orin, num_fea):
        obs.requires_grad = False
        acts.requires_grad = False

        if type(saliency) == torch.Tensor:
            saliency = saliency.cpu().detach().numpy()

        importance_id_sorted = np.argsort(saliency, axis=1)[:, ::-1] # high to low.
        nonimportance_id = importance_id_sorted[:, num_fea:]
        nonimportance_id = nonimportance_id.copy()

        self.model.eval()
        self.likelihood.eval()

        if torch.cuda.is_available():
            obs, acts = obs.cuda(), acts.cuda()
            self.model, self.likelihood = self.model.cuda(), self.likelihood.cuda()

        f_predicted, features = self.model(obs, acts)
        if self.likelihood_type == 'classification':
            function_samples = f_predicted.rsample(torch.Size([8])).cpu().detach()
            function_samples = function_samples.view(function_samples.size(0), rewards.shape[0], obs.shape[1])
            if self.weight_x:
                input_encoding = features.sum(-1)
                mixing_weights = self.likelihood.weight_encoder(input_encoding)
                mixing_weights = mixing_weights.view(mixing_weights.shape[0], obs.shape[1], self.likelihood.num_classes)
                mixing_weights = mixing_weights.cpu().detach()
            else:
                mixing_weights = self.likelihood.mixing_weights.cpu().detach()
                mixing_weights = mixing_weights.t()
                mixing_weights = mixing_weights.repeat(rewards.shape[0], 1, 1)
            for i in range(rewards.shape[0]):
                mixing_weights[i, nonimportance_id[i], rewards[i]] = 0 # maybe no need to select the last dim.
            mixed_fs = torch.einsum('bxy, xyk->bxk', (function_samples, mixing_weights)) # num_classes x num_data

            output = torch.nn.functional.softmax(mixed_fs, dim=-1)
            preds = output.mean(0)
        else:
            mean = f_predicted.mean.cpu().detach()
            mean = mean.view(rewards.shape[0], obs.shape[1])
            mean_bias = self.likelihood.weight_encoder(features)#.squeeze(-1)
            mean_bias = mean_bias.cpu().detach()
            mixing_weights = self.likelihood.mixing_weights.cpu().detach()
            mixing_weights = mixing_weights.t()
            mixing_weights = mixing_weights.repeat(rewards.shape[0], 1, 1)
            for i in range(rewards.shape[0]):
                mixing_weights[i, nonimportance_id[i], 0] = 0 # maybe no need to select the last dim.

            """
            Old model with input-dependent bias
            mean = mean + mean_bias
            output = torch.einsum('xy, xyk->xk', (mean, mixing_weights)) # num_classes x num_data
            """
            output = torch.einsum('xy, xyk->xk', (mean, mixing_weights)) + mean_bias# num_classes x num_data
            preds = output.flatten()

        preds = preds.cpu().detach().numpy()

        if self.likelihood_type == 'classification':
            preds_labels = np.argmax(preds, 1)
            preds = preds[list(range(rewards.shape[0])), rewards]
            acc = accuracy_score(rewards, preds_labels)
            if len(preds.shape) == 2:
                preds = preds.flatten()
            return -np.log(preds), acc, np.abs(preds - preds_orin[0])
        else:
            if len(preds.shape) == 2:
                preds = preds.flatten()
            return np.abs(preds-preds_orin)


# All the keys in the state_dict.
# encoder.mlp_encoder.mlp_0.weight (\Theta_1)
# encoder.mlp_encoder.mlp_0.bias (\Theta_1)
# encoder.mlp_encoder.mlp_1.weight (\Theta_1)
# encoder.mlp_encoder.mlp_1.bias (\Theta_1)
# encoder.rnn.weight_ih_l0 (\Theta_1)
# encoder.rnn.weight_hh_l0 (\Theta_1)
# encoder.rnn.bias_ih_l0 (\Theta_1)
# encoder.rnn.bias_hh_l0 (\Theta_1)
# encoder.traj_embed_layer.weight (\Theta_1)
# encoder.traj_embed_layer.bias (\Theta_1)
# gp_layer.variational_strategy.inducing_points (\Theta_2)
# gp_layer.variational_strategy.variational_params_initialized
# gp_layer.variational_strategy.updated_strategy
# gp_layer.variational_strategy._variational_distribution.variational_mean (\Theta_2)
# gp_layer.variational_strategy._variational_distribution.chol_variational_covar (\Theta_2)
# gp_layer.mean_module.constant
# gp_layer.step_kernel.raw_outputscale (\Theta_2)
# gp_layer.step_kernel.active_dims
# gp_layer.step_kernel.base_kernel.raw_lengthscale (\Theta_2)
# gp_layer.step_kernel.base_kernel.active_dims
# gp_layer.step_kernel.base_kernel.lengthscale_prior.a
# gp_layer.step_kernel.base_kernel.lengthscale_prior.b
# gp_layer.step_kernel.base_kernel.lengthscale_prior.sigma
# gp_layer.step_kernel.base_kernel.lengthscale_prior.tails.loc
# gp_layer.step_kernel.base_kernel.lengthscale_prior.tails.scale
# gp_layer.step_kernel.base_kernel.raw_lengthscale_constraint.lower_bound
# gp_layer.step_kernel.base_kernel.raw_lengthscale_constraint.upper_bound
# gp_layer.step_kernel.raw_outputscale_constraint.lower_bound
# gp_layer.step_kernel.raw_outputscale_constraint.upper_bound
# gp_layer.traj_kernel.raw_outputscale (\Theta_2)
# gp_layer.traj_kernel.active_dims
# gp_layer.traj_kernel.base_kernel.raw_lengthscale (\Theta_2)
# gp_layer.traj_kernel.base_kernel.active_dims
# gp_layer.traj_kernel.base_kernel.lengthscale_prior.a
# gp_layer.traj_kernel.base_kernel.lengthscale_prior.b
# gp_layer.traj_kernel.base_kernel.lengthscale_prior.sigma
# gp_layer.traj_kernel.base_kernel.lengthscale_prior.tails.loc
# gp_layer.traj_kernel.base_kernel.lengthscale_prior.tails.scale
# gp_layer.traj_kernel.base_kernel.raw_lengthscale_constraint.lower_bound
# gp_layer.traj_kernel.base_kernel.raw_lengthscale_constraint.upper_bound
# gp_layer.traj_kernel.raw_outputscale_constraint.lower_bound
# gp_layer.traj_kernel.raw_outputscale_constraint.upper_bound
# gp_layer.covar_module.kernels.0.raw_outputscale
# gp_layer.covar_module.kernels.0.active_dims
# gp_layer.covar_module.kernels.0.base_kernel.raw_lengthscale
# gp_layer.covar_module.kernels.0.base_kernel.active_dims
# gp_layer.covar_module.kernels.0.base_kernel.lengthscale_prior.a
# gp_layer.covar_module.kernels.0.base_kernel.lengthscale_prior.b
# gp_layer.covar_module.kernels.0.base_kernel.lengthscale_prior.sigma
# gp_layer.covar_module.kernels.0.base_kernel.lengthscale_prior.tails.loc
# gp_layer.covar_module.kernels.0.base_kernel.lengthscale_prior.tails.scale
# gp_layer.covar_module.kernels.0.base_kernel.raw_lengthscale_constraint.lower_bound
# gp_layer.covar_module.kernels.0.base_kernel.raw_lengthscale_constraint.upper_bound
# gp_layer.covar_module.kernels.0.raw_outputscale_constraint.lower_bound
# gp_layer.covar_module.kernels.0.raw_outputscale_constraint.upper_bound
# gp_layer.covar_module.kernels.1.raw_outputscale
# gp_layer.covar_module.kernels.1.active_dims
# gp_layer.covar_module.kernels.1.base_kernel.raw_lengthscale
# gp_layer.covar_module.kernels.1.base_kernel.active_dims
# gp_layer.covar_module.kernels.1.base_kernel.lengthscale_prior.a
# gp_layer.covar_module.kernels.1.base_kernel.lengthscale_prior.b
# gp_layer.covar_module.kernels.1.base_kernel.lengthscale_prior.sigma
# gp_layer.covar_module.kernels.1.base_kernel.lengthscale_prior.tails.loc
# gp_layer.covar_module.kernels.1.base_kernel.lengthscale_prior.tails.scale
# gp_layer.covar_module.kernels.1.base_kernel.raw_lengthscale_constraint.lower_bound
# gp_layer.covar_module.kernels.1.base_kernel.raw_lengthscale_constraint.upper_bound
# gp_layer.covar_module.kernels.1.raw_outputscale_constraint.lower_bound
# gp_layer.covar_module.kernels.1.raw_outputscale_constraint.upper_bound

# Key in the likelihood_state_dict.
# mixing_weights (\Theta_3).
