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

import torch
import tqdm
import gpytorch
import numpy as np
import torch.optim as optim
from .gp_utils import DGPXRLModel, CustomizedGaussianLikelihood, CustomizedSoftmaxLikelihood

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
    def __init__(self, train_loader, seq_len, input_dim, hiddens, likelihood_type, lr, optimizer_type, n_epoch, gamma,
                 num_inducing_points, inducing_points=None, mean_inducing_points=None, dropout_rate=0.25, num_class=None,
                 rnn_cell_type='GRU', normalize=False, grid_bounds=None, using_ngd=False, using_ksi=False,
                 using_ciq=False, using_sor=False, using_OrthogonallyDecouple=False):
        """
        Define the full model.
        :param train_loader: training data loader.
        :param seq_len: trajectory length.
        :param input_dim: input state/action dimension.
        :param hiddens: hidden layer dimentions.
        :param likelihood_type: likelihood type.
        :param lr: learning rate.
        :param optimizer_type.
        :param n_epoch.
        :param gamma.
        :param num_inducing_points: number of inducing points.
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
        """
        self.train_loader = train_loader
        self.likelihood_type = likelihood_type
        self.lr = lr
        self.optimizer_type = optimizer_type
        self.n_epoch = n_epoch
        self.gamma = gamma
        self.using_ngd = using_ngd

        # Build the likelihood layer (Regression and classification).
        if self.likelihood_type == 'regression':
            print('Conduct regression and use GaussianLikelihood')
            self.likelihood = CustomizedGaussianLikelihood(num_features=seq_len)
        elif self.likelihood_type == 'classification':
            print('Conduct classification and use softmaxLikelihood')
            self.likelihood = CustomizedSoftmaxLikelihood(num_features=seq_len, num_classes=num_class)
        else:
            print('Default choice is regression and use GaussianLikelihood')
            self.likelihood = CustomizedGaussianLikelihood(num_features=seq_len)

        # Compute the loss (ELBO) likelihood + KL divergence.
        self.model = DGPXRLModel(seq_len=seq_len, input_dim=input_dim, hiddens=hiddens, likelihood_type=likelihood_type,
                                 dropout_rate=dropout_rate, rnn_cell_type=rnn_cell_type, normalize=normalize,
                                 num_inducing_points=num_inducing_points, inducing_points=inducing_points,
                                 mean_inducing_points=mean_inducing_points, grid_bounds=grid_bounds, using_ngd=using_ngd,
                                 using_ksi=using_ksi, using_ciq=using_ciq, using_sor=using_sor,
                                 using_OrthogonallyDecouple=using_OrthogonallyDecouple)
        print(self.model)

        # First, sampling from q(f) with shape [n_sample, n_data].
        # Then, the likelihood function times it with the mixing weight and get the marginal likelihood p(y|f).
        # VariationalELBO will call _ApproximateMarginalLogLikelihood, which then computes the marginal likelihood by
        # calling the likelihood function (the expected_log_prob in the likelihood class)
        # and the KL divergence (VariationalStrategy.kl_divergence()).
        # ELBO = E_{q(f)}[p(y|f)] - KL[q(u)||p(u)].
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model.gp_layer,
                                                 num_data=len(train_loader.dataset))

        # Define the optimizer over the parameters.
        # (RNN parameters, RBF kernel parameters, Z, variational parameters, mixing weight).
        if self.using_ngd:
            self.variational_ngd_optimizer = gpytorch.optim.NGD(self.model.gp_layer.variational_parameters(),
                                                                num_data=len(train_loader.dataset), lr=self.lr*10)
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
            self.scheduler = optim.lr_scheduler.MultiStepLR(self.hyperparameter_optimizer,
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

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.likelihood = self.likelihood.cuda()

    # Load a pretrained model.
    def load(self, load_path):
        """
        :param load_path: load model path.
        :return: model, likelihood.
        """
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

    # train function.
    def train(self, save_path=None, likelihood_sample_size=8):

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
                    for data, target in minibatch_iter:
                        if torch.cuda.is_available():
                            data, target = data.cuda(), target.cuda()
                        if self.using_ngd:
                            self.variational_ngd_optimizer.zero_grad()
                            self.hyperparameter_optimizer.zero_grad()
                        else:
                            self.optimizer.zero_grad()
                        output = self.model(data)  # marginal variational posterior, q(f|x).
                        loss = -self.mll(output, target)  # approximated ELBO.
                        loss.backward()
                        if self.using_ngd:
                            self.variational_ngd_optimizer.step()
                            self.hyperparameter_optimizer.step()
                        else:
                            self.optimizer.step()

                        if self.likelihood_type == 'classification':
                            _, preds = torch.max(preds, 1)
                            correct += preds.eq(target.view_as(preds)).cpu().sum()
                        else:
                            mae += torch.sum(torch.abs(preds - target))
                            mse += torch.sum(torch.square(preds - target))
                        minibatch_iter.set_postfix(loss=loss.item())

                    if self.likelihood_type == 'classification':
                        print('Test set: Accuracy: {}/{} ({}%)'.format(
                            correct, len(self.train_loader.dataset), 100. * correct / float(len(self.train_loader.dataset))
                        ))
                    else:
                        print('Test MAE: {}'.format(mae / float(len(self.train_loader.dataset))))
                        print('Test MSE: {}'.format(mse / float(len(self.train_loader.dataset))))

            self.scheduler.step()

        if save_path:
            self.save(save_path)
        return self.model

    # test function.
    def test(self, test_loader, likelihood_sample_size=16):
        # Specify that the model is in eval mode.
        self.model.eval()
        self.likelihood.eval()

        correct = 0
        mse = 0
        mae = 0
        with torch.no_grad(), gpytorch.settings.num_likelihood_samples(likelihood_sample_size):
            for data, target in test_loader:
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                f_predicted = self.model(data)
                output = self.likelihood(f_predicted)  # This gives us 16 samples from the predictive distribution (q(y|f_*)).
                if self.likelihood_type == 'classification':
                    pred = output.probs.mean(0).argmax(-1)  # Take the mean over all of the sample we've drawn (y = E_{f_* ~ q(f_*)}[y|f_*]).
                    correct += pred.eq(target.view_as(pred)).cpu().sum()
                else:
                    preds = output.mean
                    mae += torch.sum(torch.abs(preds - target))
                    mse += torch.sum(torch.square(preds - target))
            if self.likelihood_type == 'classification':
                print('Test set: Accuracy: {}/{} ({}%)'.format(
                    correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
                ))
            else:
                print('Test MAE: {}'.format(mae/float(len(test_loader.dataset))))
                print('Test MSE: {}'.format(mse/float(len(test_loader.dataset))))

    def get_explanations(self, data, normalize=True):
        self.model.eval()
        self.likelihood.eval()

        if len(data.shape) == 2:
            data = data[None,:,:]
        importance = self.likelihood.mixing_weights
        step_embedding, traj_embedding = self.model.encoder(data)  # (N, T, P) -> (N, T, D), (N, D).
        traj_embedding = traj_embedding[:, None, :].repeat(1, data.shape[1], 1)  # (N, D) -> (N, T, D)
        features = torch.cat([step_embedding, traj_embedding], dim=-1)  # (N, T, 2D)
        features = features.view(data.size(0) * data.size(1), features.size(-1))
        covar_all = self.model.gp_layer.covar_module(features)
        covar_step = self.model.gp_layer.ste_kernel(features)
        covar_traj = self.model.gp_layer.traj_kernel(features)
        # TODO: combine importance weight with covariance structure.
        importance += importance[:, None] / importance.shape[1]
        if normalize:
            importance = (importance - np.min(importance, axis=1)[:, None])
            importance = importance / (np.max(importance, axis=1)[:, None] - np.min(importance, axis=1)[:, None])

        return importance, (covar_all, covar_traj, covar_step)


# if __name__ == '__main__':
#     explainer = DGPXRL(train_loader=train_loader, seq_len=train_x.size(1), input_dim=train_x.size(2), hiddens=HIDDENS,
#                        likelihood_type=LIKELIHOOD_TYPE, lr=LR, optimizer_type=OPTIMIZER, n_epoch=N_EPOCHS, gamma=GAMMA,
#                        dropout_rate=DROPOUT_RATE, num_inducing_points=NUM_INDUCING_POINTS)
#
#     if LOAD_PATH:
#         explainer.load(LOAD_PATH)
#
#     explainer.train(save_path=SAVE_PATH)
#     explainer.test(test_loader)
#
#     # Get the explanations and covariance.
#     importance, covariance = explainer.get_explanations(data=train_x)
#
#     VisualizeCovar(covariance[0], SAVE_PATH+'/full_covar.pdf')
#     VisualizeCovar(covariance[1], SAVE_PATH+'/traj_covar.pdf')
#     VisualizeCovar(covariance[2], SAVE_PATH+'/step_covar.pdf')


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
