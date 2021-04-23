#!/usr/bin/env python3

import math
import torch
import warnings
import gpytorch
import numpy as np
import seaborn as sns
import torch.nn as nn
from matplotlib import pyplot as plt
from gpytorch.likelihoods.likelihood import Likelihood
from gpytorch.distributions import base_distributions, MultivariateNormal
from gpytorch.likelihoods.noise_models import HomoskedasticNoise
from gpytorch import settings
from gpytorch.utils.errors import CachingError
from gpytorch.utils.cholesky import psd_safe_cholesky
from gpytorch.variational._variational_strategy import _VariationalStrategy
from gpytorch.variational.variational_strategy import _ensure_updated_strategy_flag_set
from gpytorch.utils.memoize import cached, clear_cache_hook, pop_from_cache_ignore_args
from gpytorch.lazy import DiagLazyTensor, MatmulLazyTensor, RootLazyTensor, SumLazyTensor, TriangularLazyTensor, delazify


# Build the RNN model.
class CnnRnnEncoder(nn.Module):
    def __init__(self, seq_len, input_dim, input_channles, hidden_dim, n_action, embed_dim=16,
                 rnn_cell_type='GRU', normalize=False):
        """
        RNN structure (CNN+seq2seq) (\theta_1: RNN parameters).
        :param seq_len: trajectory length.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action)
        :param input_channles: 1.
        :param hidden_dim: RNN output dim.
        :param n_action: total number of actions.
        :param embed_dim: action embedding dim.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param normalize: whether to normalize the inputs.
        """
        super(CnnRnnEncoder, self).__init__()
        self.normalize = normalize
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell_type = rnn_cell_type

        self.act_embedding = nn.Embedding(n_action, embed_dim)

        self.encoder = nn.Sequential()

        self.encoder.add_module('cnn_%d' % 1, nn.Conv2d(input_channles, 32, kernel_size=(3, 3), stride=(2, 2)))
        self.encoder.add_module('relu_%d' % 1, nn.ReLU())

        self.encoder.add_module('cnn_%d' % 2, nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)))
        self.encoder.add_module('relu_%d' % 2, nn.ReLU())

        self.encoder.add_module('cnn_%d' % 3, nn.Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2)))
        self.encoder.add_module('relu_%d' % 3, nn.ReLU())

        self.encoder.add_module('cnn_%d' % 4, nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(2, 2)))
        self.encoder.add_module('relu_%d' % 4, nn.ReLU())

        self.encoder.add_module('flatten', nn.Flatten(start_dim=-3, end_dim=-1))

        if input_dim == 80 or 84:
            self.cnn_out_dim = 4 * 4 * 16 + embed_dim
        else:
            raise ValueError ('input dim does not support.')

        if self.rnn_cell_type == 'GRU':
            print('Using GRU as the recurrent layer.')
            self.rnn = nn.GRU(input_size=self.cnn_out_dim, hidden_size=hidden_dim, batch_first=True)
        elif self.rnn_cell_type == 'LSTM':
            print('Using LSTM as the recurrent layer.')
            self.rnn = nn.LSTM(input_size=self.cnn_out_dim, hidden_size=hidden_dim, batch_first=True)
        else:
            print('Using the default recurrent layer: GRU.')
            self.rnn = nn.GRU(input_size=self.cnn_out_dim, hidden_size=hidden_dim, batch_first=True)
            self.rnn_cell_type = 'GRU'

        self.traj_embed_layer = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, y, h0=None, c0=None):
        # forward function: given an input, return the model output (output at each time and the final time step).
        """
        :param x: input observations (Batch_size, seq_len, 1, input_dim, input_dim).
        :param y: input actions (Batch_size, seq_len).
        :param c0: Initial cell state at time t_0 (Batch_size, 1, hidden_dim).
        :return step_embed: the latent representation of each time step (batch_size, seq_len, hidden_dim).
        :return traj_embed: the latend representation of each trajectory (batch_size, hidden_dim).
        """
        num_traj = x.size(0)

        if self.normalize:
            mean = torch.mean(x, dim=(0, 1))[None, None, :]
            std = torch.std(x, dim=(0, 1))[None, None, :]
            x = (x - mean)/std
        x = x.view(-1, 1, self.input_dim, self.input_dim)
        obs_encoded = self.encoder(x)  # (N, T, D1) get the hidden representation of every time step.
        obs_encoded = obs_encoded.view(num_traj, self.seq_len, obs_encoded.size(-1))
        act_encoded = self.act_embedding(y)
        cnn_encoded = torch.cat((obs_encoded, act_encoded), -1)
        if self.rnn_cell_type == 'GRU':
            step_embed, traj_embed = self.rnn(cnn_encoded, h0)
        else:
            if h0 is None or c0 is None:
                step_embed, traj_embed = self.rnn(cnn_encoded, None)
            else:
                step_embed, traj_embed = self.rnn(cnn_encoded, (h0, c0))
        traj_embed = torch.squeeze(traj_embed, 0) # (1, batch_size, hidden_dim) -> (batch_size, hidden_dim)
        traj_embed = self.traj_embed_layer(traj_embed) # (batch_size, hidden_dim)
        return step_embed, traj_embed


class MlpRnnEncoder(nn.Module):
    def __init__(self, seq_len, input_dim, hiddens, dropout_rate=0.25, rnn_cell_type='GRU', normalize=False):
        """
        RNN structure (MLP+seq2seq) (\theta_1: RNN parameters).
        :param seq_len: trajectory length.
        :param input_dim: the dimensionality of the input (Concatenate of observation and action)
        :param hiddens: hidden layer dimensions.
        :param dropout_rate: dropout rate.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param normalize: whether to normalize the inputs.
        """
        super(MlpRnnEncoder, self).__init__()
        self.normalize = normalize
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hiddens[-1]
        self.rnn_cell_type = rnn_cell_type
        self.encoder = nn.Sequential()
        for i in range(len(hiddens)-1):
            if i == 0:
                self.encoder.add_module('mlp_%d' % i, nn.Linear(input_dim, hiddens[i]))
            else:
                self.encoder.add_module('mlp_%d' % i, nn.Linear(hiddens[i-1], hiddens[i]))
            self.encoder.add_module('relu_%d' % i, nn.ReLU())
            self.encoder.add_module('dropout_%d' % i, nn.Dropout(dropout_rate))

        if self.rnn_cell_type == 'GRU':
            print('Using GRU as the recurrent layer.')
            self.rnn = nn.GRU(input_size=hiddens[-2], hidden_size=hiddens[-1], batch_first=True)
        elif self.rnn_cell_type == 'LSTM':
            print('Using LSTM as the recurrent layer.')
            self.rnn = nn.LSTM(input_size=hiddens[-2], hidden_size=hiddens[-1], batch_first=True)
        else:
            print('Using the default recurrent layer: GRU.')
            self.rnn = nn.GRU(input_size=hiddens[-2], hidden_size=hiddens[-1], batch_first=True)
            self.rnn_cell_type = 'GRU'

        self.traj_embed_layer = nn.Linear(hiddens[-1], hiddens[-1])

    def forward(self, x, y, h0=None, c0=None):
        # forward function: given an input, return the model output (output at each time and the final time step).
        """
        :param x: input observations (Batch_size, seq_len, input_dim).
        :param y: input actions (Batch_size, seq_len).
        :param h0: Initial hidden state at time t_0 (Batch_size, 1, hidden_dim).
        :param c0: Initial cell state at time t_0 (Batch_size, 1, hidden_dim).
        :return step_embed: the latent representation of each time step (batch_size, seq_len, hidden_dim).
        :return traj_embed: the latend representation of each trajectory (batch_size, hidden_dim).
        """
        if self.normalize:
            mean = torch.mean(x, dim=(0, 1))[None, None, :]
            std = torch.std(x, dim=(0, 1))[None, None, :]
            x = (x - mean)/std
        x = torch.cat((x, y[..., None]), -1)
        mlp_encoded = self.encoder(x) # (N, T, Hiddens[-2]) get the hidden representation of every time step.
        if self.rnn_cell_type == 'GRU':
            step_embed, traj_embed = self.rnn(mlp_encoded, h0)
        else:
            step_embed, traj_embed, _ = self.rnn(mlp_encoded, h0, c0)
        traj_embed = torch.squeeze(traj_embed, 0) # (1, batch_size, hidden_dim) -> (batch_size, hidden_dim)
        traj_embed = self.traj_embed_layer(traj_embed) # (batch_size, hidden_dim)
        return step_embed, traj_embed


# Build the GP layer.
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim_step, input_dim_traj, num_inducing_points, inducing_points, mean_inducing_points,
                 grid_bounds, likelihood_type, using_ngd, using_ksi, using_ciq, using_sor, using_OrthogonallyDecouple):
        """
        Define the mean and kernel function: Constant mean, and additive RBF kernel (step kernel + traj kernel).
        variational distribution: Cholesky Multivariate Gaussian q(u) ~ N(\mu, LL^T).
        variational strategy: VariationalStrategy with whitening.
        Standard variational strategy, add KSI.
        p(u) ~ N(0, K_{zz})
        Define u = Lv, where v ~ N(0, 1), LL^T = K_{zz}
        q(v) ~ N(\mu, S) -> q(u) ~ N(L\mu, LSL^T)
        q(f|v) ~ N(K_{xz}K_{zz}^(-1)Lv, K_{xx} - K_{xz}K_{zz}^{-1}K_{xz}^{T})
        q(f) ~ N(K_{xz}K_{zz}^(-1/2)\mu, K_{xx} + K_{xz}K_{zz}^(-1/2)(S-I)K_{zz}^(-1/2)K_{xz}^{T})
        inducing points Z (n, input_dim_step+input_dim_traj),
        \theta_2: kernel parameters, variational parameters, and Z.

        :param input_dim_step: step embedding dim.
        :param input_dim_traj: traj embedding dim.
        :param inducing_points: inducing points at the latent space (n, input_dim_step+input_dim_traj).
        :param mean_inducing_points: mean inducing points, used for orthogonally decoupled VGP.
        :param grid_bounds: grid bounds.
        :param likelihood_type: likelihood type.
        :param using_ngd: Whether to use natural gradient descent.
        :param using_ksi: Whether to use KSI approximation, using this with other options as False.
        :param using_ciq: Whether to use Contour Integral Quadrature to approximate K_{zz}^{-1/2}, Use it together with NGD.
        :param using_sor: Whether to use SoR approximation, not applicable for KSI and CIQ.
        :param using_OrthogonallyDecouple
        """
        if using_ngd:
            print('Using Natural Gradient Descent.')
            if likelihood_type == 'regression':
                print('Conjugate likelihood: using NaturalVariationalDistribution.')
                variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
                    num_inducing_points=num_inducing_points)
            else:
                print('Non-conjugate likelihood: using TrilNaturalVariationalDistribution.')
                variational_distribution = gpytorch.variational.TrilNaturalVariationalDistribution(
                    num_inducing_points=num_inducing_points)
        else:
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=num_inducing_points)

        if using_ksi:
            print('Using KSI.')
            variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
                num_inducing_points=int(pow(num_inducing_points, len(grid_bounds))))
            variational_strategy = gpytorch.variational.GridInterpolationVariationalStrategy(
                self, num_inducing_points, grid_bounds, variational_distribution)
        elif using_ciq:
            print('Using CIQ.')
            variational_strategy = gpytorch.variational.CiqVariationalStrategy(
                self, inducing_points, variational_distribution, learn_inducing_locations=True)
        else:
            variational_strategy = CustomizedVariationalStrategy(self, inducing_points, variational_distribution,
                                                                 learning_inducing_locations=True,
                                                                 using_sor=using_sor)

        if using_OrthogonallyDecouple:
            print('Using Orthogonally Decouple.')
            variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
                variational_strategy, mean_inducing_points,
                gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2)))

        super(GaussianProcessLayer, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.step_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim_step, active_dims=tuple(range(input_dim_step)),
                                       lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                                           math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp)))
        self.traj_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim_traj,
                                       active_dims=tuple(range(input_dim_step, input_dim_traj+input_dim_step)),
                                       lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                                           math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp)))
        self.covar_module = self.step_kernel + self.traj_kernel

    def forward(self, x):
        """
        Compute the prior distribution.
        :param x: input data (n, d).
        :return: Prior distribution of x (MultivariateNormal).
        """
        mean = self.mean_module(x)
        covar = self.covar_module(x) # Checked this part, covar = self.step_kernel(x) + self.traj_kernel(x).
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    # __call__: compute the conditional/marginal posterior.


# Build the full model.
class DGPXRLModel(gpytorch.Module):
    def __init__(self, seq_len, input_dim, hiddens, likelihood_type, n_action,  num_inducing_points, embed_dim=16,
                 encoder_type='MLP', inducing_points=None, mean_inducing_points=None, dropout_rate=0.25,
                 rnn_cell_type='GRU', normalize=False, grid_bounds=None, using_ngd=False, using_ksi=False,
                 using_ciq=False, using_sor=False, using_OrthogonallyDecouple=False):
        """
        Define the full model.
        :param seq_len: trajectory length.
        :param input_dim: input state/action dimension.
        :param hiddens: hidden layer dimentions.
        :param likelihood_type: likelihood type.
        :param n_action: number of actions.
        :param num_inducing_points: number of inducing points.
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
        """
        super().__init__()
        self.seq_len = seq_len
        self.encoder_type = encoder_type

        if self.encoder_type == 'CNN':
            self.encoder = CnnRnnEncoder(seq_len, input_dim, input_channles=1, hidden_dim=hiddens[-1],
                                       n_action=n_action, embed_dim=embed_dim, rnn_cell_type=rnn_cell_type,
                                       normalize=normalize)
        else:
            self.encoder = MlpRnnEncoder(seq_len, input_dim, hiddens, dropout_rate, rnn_cell_type, normalize=normalize)

        if inducing_points is None:
            inducing_points = torch.randn(num_inducing_points, 2*hiddens[-1])
        if mean_inducing_points is None:
            mean_inducing_points = torch.randn(num_inducing_points*5, 2*hiddens[-1])

        self.gp_layer = GaussianProcessLayer(input_dim_step=hiddens[-1], input_dim_traj=hiddens[-1],
                                             num_inducing_points=num_inducing_points, inducing_points=inducing_points,
                                             mean_inducing_points=mean_inducing_points, grid_bounds=grid_bounds,
                                             likelihood_type=likelihood_type, using_ngd=using_ngd, using_ksi=using_ksi,
                                             using_ciq=using_ciq, using_sor=using_sor,
                                             using_OrthogonallyDecouple=using_OrthogonallyDecouple)

    def forward(self, x, y):
        """
        Compute the marginal posterior q(f) ~ N(\mu_f, \sigma_f), \mu_f (N*T, 1), \sigma_f(N*T, N*T).
        Later, when computing the marginal loglikelihood, we sample multiple set of data from the marginal loglikelihood.
        :param x: input data x (N, T, P).
        :param x: input data y (N, T).
        :return: q(gy_layer(Encoder(x))).
        """
        step_embedding, traj_embedding = self.encoder(x, y)  # (N, T, P) -> (N, T, D), (N, D).
        traj_embedding = traj_embedding[:, None, :].repeat(1, self.seq_len, 1) # (N, D) -> (N, T, D)
        features = torch.cat([step_embedding, traj_embedding], dim=-1) # (N, T, 2D)
        features_reshaped = features.view(x.size(0)*x.size(1), features.size(-1))
        res = self.gp_layer(features_reshaped)
        return res, features


class ConstantMean(nn.Module):
    """
    Constant mean function of a Multivariate Gaussian.
    """
    def __init__(self, prior=None, learnable=False):
        """
        :param prior: non-zero mean vector (1, ).
        """
        super(ConstantMean, self).__init__()
        if prior is None:
            prior = torch.zeros(1,)
        self.register_parameter(name="constant_mean", param=nn.Parameter(prior, requires_grad=learnable))

    def forward(self, input):
        """
        :param input: input samples, (batch_size, input_dim).
        :return: zero mean vectors (batch_size,).
        """
        return self.constant.expand(input.shape[:-1])

    # def __call__(self, input):
    #     return self.forward(input)


class AdditiveRBFKernel(nn.Module):
    """
    Addition of two RBF kernel functions at the trajectory and the time step level.
    RBF kernel:
    k_{\text{RBF}}(\mathbf{x_1}, \mathbf{x_2}) = \exp \left( -\frac{1}{2}(\mathbf{x_1} - \mathbf{x_2})^\top
                                                  \Theta^{-2} (\mathbf{x_1} - \mathbf{x_2}) \right)
    k_{\text{RBF}}(\mathbf{x_1}, \mathbf{x_2}) = \text{outputscale} k_{\text{RBF}}(\mathbf{x_1}, \mathbf{x_2})
    """
    def __init__(self, input_dim_step, input_dim_traj, ard_num_dims_step=True, ard_num_dims_traj=True):
        """
        Define the parameters in the kernel functions.
        :param input_dim_step: time step kernel input dim.
        :param input_dim_traj: trajectory step kernel input dim.
        :param ard_num_dims_step: whether to use an unique ls_step for each input dim.
        :param ard_num_dims_traj: whether to use an unique ls_traj for each input dim.
        """

        super().__init__()

        ls_num_dims_step = 1 if ard_num_dims_step is None else input_dim_step
        self.register_parameter(name="ls_step", param=torch.nn.Parameter(torch.zeros(1, ls_num_dims_step)))
        self.register_parameter(name="outputscale_step", param=torch.nn.Parameter(torch.tensor(0.0)))

        ls_num_dims_traj = 1 if ard_num_dims_traj is None else input_dim_traj
        self.register_parameter(name="ls_traj", param=torch.nn.Parameter(torch.zeros(1, ls_num_dims_traj)))
        self.register_parameter(name="outputscale_traj", param=torch.nn.Parameter(torch.tensor(0.0)))

    @staticmethod
    def _sq_dist(x1, x2=None):
        """
        :param x1: data matrix (n1, d).
        :param x2: data matrix (n2, d).
        :return: square distance matrix: res (n1, n2), res_{ij} = (x1_i - x2_j)^2.
        """
        x1_norm = x1.pow(2).sum(dim=-1, keepdim=True) # x1_norm: (n1, 1), each element is \sum_{d} x_{ij}^2.
        x1_pad = torch.ones_like(x1_norm) # x1_pad(n1, 1)
        if x2 is None:
            x2 = torch.clone(x1)
            x2_norm, x2_pad = x1_norm, x1_pad
        else:
            x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
            x2_pad = torch.ones_like(x2_norm)
        x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1) # x1_: [x1, x1^2, 1] (n1, d+2).
        x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1) # x2_: [x2, 1, x2^2] (n2, d+2).
        res = x1_.matmul(x2_.transpose(-2, -1)) # res: x1^2 + x2^2 - 2x1x2 (n1, n2).

        if x2 is None:
            res.diagonal(dim1=-2, dim2=-1).fill_(0)

        # Zero out negative values.
        return res.clamp(min=0)

    @staticmethod
    def _rbf_covar_mat(x1, x2, lengthscale, outputscale, sq_dist_func):
        """
        Compute the covariance matrix using a RBF kernel.
        :param x1: data matrix (n1, d).
        :param x2: data matrix (n2, d).
        :param sq_dist_func: square distance function.
        :param lengthscale: ls.
        :param outputscale: output scale.
        :return:
        """
        x1_ = x1.div(lengthscale[None, :])
        x2_ = x2.div(lengthscale[None, :])
        unitless_sq_dist = sq_dist_func(x1_, x2_)
        covar_mat = unitless_sq_dist.div_(-2.0).exp_()
        return covar_mat*outputscale

    def forward(self, x1_step, x1_traj, x2_step, x2_traj):
        """
        Compute the additive covariance matrix for a pair of inputs.
        :param x1_step: the time step embedding of x1 (n1, input_dim_step).
        :param x1_traj: the trajectory embedding of x1 (n1, input_dim_traj).
        :param x2_step: the time step embedding of x2 (n2, input_dim_step).
        :param x2_traj: the time step embedding of x2 (n2, input_dim_traj).
        :return: the covariance matrix (n1, n2).
        """
        step_covar_mat = self._rbf_covar_mat(x1_step, x2_step, self.ls_step, self.outputscale_step, self._sq_dist)
        traj_covar_mat = self._rbf_covar_mat(x1_traj, x2_traj, self.ls_traj, self.outputscale_traj, self._sq_dist)

        return step_covar_mat + traj_covar_mat


class CustomizedSoftmaxLikelihood(Likelihood):
    """
    Customized softmax (multiclass) likelihood used for GP classification.
    q(f) ~ N(\mu_f, \Sigma_f), \mu (N*T, 1), \migma_f (N*T, N*T)
    Sample f from q(f) and reshape it into (N, T), \mathbf W (T, C)
    p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf W \mathbf f \right)
    """
    def __init__(self, num_features=None, num_classes=None, mixing_weights_prior=None):
        """
        :param num_features: Dimensionality of latent function
        :param num_classes: Number of classes
        :param mixing_weights_prior: Prior to use over the mixing weights
        """
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes is required")
        self.num_classes = num_classes
        self.num_features = num_features
        if num_features is None:
            raise ValueError("num_features is required with mixing weights")
        self.register_parameter(
            name="mixing_weights",
            parameter=torch.nn.Parameter(torch.randn(num_classes, num_features).div_(num_features)),
        )
        if mixing_weights_prior is not None:
            self.register_prior("mixing_weights_prior", mixing_weights_prior, "mixing_weights")

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        likelihood_samples = self._draw_likelihood_samples(function_dist, *args, **kwargs)
        res = likelihood_samples.log_prob(observations).mean(dim=0)
        return res * self.num_features # times seq_len, because the log_likelihood in _approximate_mll will divide (n_traj*seq_len).

    def forward(self, function_samples, *params, **kwargs):
        """
        :param function_samples: samples from q(f).
        :return: Conditional likelihood distribution.
        """
        num_data = int(function_samples.shape[-1]/self.num_features)

        # Catch legacy mode
        if num_data == self.num_features:
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                DeprecationWarning,
            )
            function_samples = function_samples.transpose(-1, -2)
            num_data, num_features = function_samples.shape[-2:]

        function_samples = function_samples.view(function_samples.size(0),
                                                   num_data, self.num_features)
        # print('Check the shape of f, should be [n_likelihood_sample, n_traj, traj_length]:')
        # print(function_samples.shape)
        if self.mixing_weights is not None:
            mixed_fs = function_samples @ self.mixing_weights.t()  # num_classes x num_data
            # print('Check the shape of fW, should be [n_likelihood_sample, n_traj, n_classes]:')
            # print(mixed_fs.shape)
        else:
            mixed_fs = function_samples
        res = base_distributions.Categorical(logits=mixed_fs)
        return res

    def __call__(self, function, *params, **kwargs):
        return super().__call__(function, *params, **kwargs)


class NNSoftmaxLikelihood(Likelihood):
    """
    Customized softmax (multiclass) likelihood used for GP classification.
    q(f) ~ N(\mu_f, \Sigma_f), \mu (N*T, 1), \migma_f (N*T, N*T)
    Sample f from q(f) and reshape it into (N, T), \mathbf W (T, C)
    p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf W \mathbf f \right)
    """
    def __init__(self, num_features=None, num_classes=None, input_encoding_dim=None):
        """
        :param num_features: Dimensionality of latent function
        :param num_classes: Number of classes
        :param mixing_weights_prior: Prior to use over the mixing weights
        """
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes is required")
        self.num_classes = num_classes
        self.num_features = num_features
        if num_features is None:
            raise ValueError("num_features is required with mixing weights")
        self.weight_encoder = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.LeakyReLU(),
            nn.Linear(num_features, num_features * num_classes)
        )

    def expected_log_prob(self, observations, function_dist, *args, **kwargs):
        likelihood_samples = self._draw_likelihood_samples(function_dist, *args, **kwargs)
        res = likelihood_samples.log_prob(observations).mean(dim=0)
        return res * self.num_features # times seq_len, because the log_likelihood in _approximate_mll will divide (n_traj*seq_len).

    def forward(self, function_samples, *args, **kwargs):
        """
        :param function_samples: samples from q(f).
        :return: Conditional likelihood distribution.
        """
        num_data = int(function_samples.shape[-1]/self.num_features)

        # Catch legacy mode
        if num_data == self.num_features:
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                DeprecationWarning,
            )
            function_samples = function_samples.transpose(-1, -2)
            num_data, num_features = function_samples.shape[-2:]

        function_samples = function_samples.view(function_samples.size(0),
                                                   num_data, self.num_features)

        # print('Check the shape of f, should be [n_likelihood_sample, n_traj, traj_length]:')
        # print(function_samples.shape)
        input_encoding = kwargs['input_encoding'].sum(-1)
        mixing_weights = self.weight_encoder(input_encoding)
        mixing_weights = mixing_weights.view(mixing_weights.shape[0], self.num_features, self.num_classes)
        mixed_fs = torch.einsum('bxy, xyk->bxk', (function_samples, mixing_weights)) # num_classes x num_data
            # print('Check the shape of fW, should be [n_likelihood_sample, n_traj, n_classes]:')
            # print(mixed_fs.shape)

        res = base_distributions.Categorical(logits=mixed_fs)
        return res


class CustomizedGaussianLikelihood(Likelihood):
    """
    Customized Gaussian likelihood for GP regression.
    Assumes a standard homoskedastic (variance of the residual is constant) and IID noise model.
    p(y \mid f) = fW + \epsilon, \quad \epsilon \sim \mathcal N (0, \sigma^2)
    """
    def __init__(self, num_features, noise_prior=None, noise_constraint=None, batch_shape=torch.Size(), **kwargs):
        """
        :param num_features: trajectory length.
        :param noise_prior: Prior for noise parameter :math:`\sigma^2`.
        :param noise_constraint: Constraint for noise parameter :math:`\sigma^2`.
        :param batch_shape: The batch shape of the learned noise parameter (default: []).
        """
        super().__init__()
        self.noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        self.register_parameter(
            name="mixing_weights",
            parameter=torch.nn.Parameter(torch.randn(1, num_features).div_(num_features)),
        )
        self.num_features = num_features

    def _shaped_noise_covar(self, base_shape: torch.Size, *params, **kwargs):
        """
        Sample noise variance with a specific shape.
        :param base_shape: batch size.
        :param params: any.
        :param kwargs: any.
        :return: shaped noise covariance.
        """
        return self.noise_covar(*params, shape=base_shape, **kwargs)

    def expected_log_prob(self, target, input, *params, **kwargs):
        """
        Compute the expected conditional log likelihood (likelihood term in ELBO) E_{f~q(f)}[log p(y|f)], f ~ N(\mu, \Sigma).
        For standard SVGP regression, the likelihood layer is GaussianLikelihood: y = f + \epsilon, \epsilon ~ N(0, \sigma^{2}).
        E_{f~q(f)}[log p(y|f)] = -N/2(log 2\pi + log \sigma^{2} + \sum_{i=1}^{N} [(y_i - \mu_i)^2 + \Sigma_{ii}] /sigma^{2}).
        In the matrix form:
        E_{f~q(f)}[log p(y|f)] = -N/2(log 2\pi + log \sigma^{2} + [(Y - \mu)^T(Y - \mu) + \sum_{i} \Sigma_{ii}] /sigma^{2}.
        Code implementation:
        def expected_log_prob(self, target: Tensor, input: MultivariateNormal)
            mean, variance = input.mean, input.variance # \mu (N, ), diag(\Sigma) (N, ).
            num_event_dim = len(input.event_shape)

            noise = self._shaped_noise_covar(mean.shape, *params, **kwargs).diag()
            # Potentially reshape the noise to deal with the multitask case
            noise = noise.view(*noise.shape[:-1], *input.event_shape) # (N, )

            res = ((target - mean) ** 2 + variance) / noise + noise.log() + math.log(2 * math.pi) # (N, )
            res = res.mul(-0.5) # (N, )
            if num_event_dim > 1:  # Do appropriate summation for multitask Gaussian likelihoods
                res = res.sum(list(range(-1, -num_event_dim, -1)))
            return res # Vector with each element as the exact Expected log likelihood of each target.

        For our model, the liklihood term is: y = f_reshaped W^{T} + \epsilon.
        q(f)=N(\mu_f (n*T, 1), \Sigma_f(n*T, n*T)),
        p(y|f)=N(fW^{T}, \sigma^{2}), f: (n, T) W: (T, 1).

        E_{f~q(f)}[log p(y|f)] = -N/2(log 2\pi + log \sigma^{2} + E_{f~q(f)}[(y - fW^{T})^{t}(y - fW^{T})]/\sigma^{2})
        E_{f~q(f)}[(y - fW^{T})^{t}(y - fW^{T})]) = y^{T}y - W\mu^{T}y - y^{T}\muW^{T} - WE[f^{T}f]W^{T}
        E_{f~q(f)}[f^{T}f] = \sum_{n}[Sigma_{i}+\mu_{i}^{T}\mu_{i}]
        In the vector form:
        E_{f~q(f)}[log p(y|f)] = -N/2(log 2\pi + log \sigma^{2} + \sum_{i=1}^{N} [(y_i - \mu_{i}W^T)^2 + \Sigma_{i}] /sigma^{2}).
        Called by variational_elbo for computing the likelihood term in elbo.
        :param target: y (n, 1).
        :param input: q(f). 
        :param params: any
        :param kwargs: any
        :return: E_{f~q(f)}[log p(y|f)].
        """
        mean, covar = input.mean, input.covariance_matrix # mean and covar of q(f).
        # num_event_dim = len(input.event_shape)
        num_data = int(mean.shape[0]/self.num_features)
        noise = self._shaped_noise_covar(torch.Size([num_data]), *params, **kwargs).diag() # noise variance.

        # Potentially reshape the noise to deal with the multitask case
        # noise = noise.view(*noise.shape[:-1], *input.event_shape)
        w_transpose = self.mixing_weights.transpose(-2, -1)

        # Computing N WE[f_{i}^{T}f_{i}]W^T with for loop.
        # exp_ff_1 = torch.zeros(num_data)
        # for i in range(num_data):
        #     exp_ff_tmp = covar[self.num_features * i:self.num_features * (i + 1),
        #                  self.num_features * i:self.num_features * (i + 1)] \
        #                  + mean[self.num_features * i:self.num_features * (i + 1)][:, None] @
        #                  mean[self.num_features * i:self.num_features * (i + 1)][None, :]
        #     exp_ff_1[i] = self.mixing_weights @ exp_ff_tmp @ w_transpose

        # Computing N WE[f_{i}^{T}f_{i}]W^T with matrix computation.
        # Suppose num_data = t
        # left = [W, 0, 0, 0,
        #         0, W, 0, 0,
        #         0, 0, W, 0,
        #         0, 0, 0, W] (N, N*T)
        # middle = covar + mean[:, None] @ mean[None, :] (N*T, N*T)
        # right = [W^T, W^T, W^T, W^T]^T (N*T, 1)

        exp_ff = covar + mean[:, None] @ mean[None, :]
        left_weight = torch.zeros(num_data, num_data*self.num_features)
        mask = torch.zeros(num_data, num_data*self.num_features)
        for i in range(num_data):
            left_weight[i, i*self.num_features:(i+1)*self.num_features] = self.mixing_weights.flatten()
            mask[i, i*self.num_features:(i+1)*self.num_features] = torch.ones_like(self.mixing_weights.flatten())
        if torch.cuda.is_available():
            left_weight = left_weight.cuda()
            mask = mask.cuda()
        right_weight = w_transpose.repeat(num_data, 1)
        exp_ff = left_weight @ exp_ff
        exp_ff = exp_ff * mask
        exp_ff = exp_ff @ right_weight

        mean_matrix = mean.view(num_data, self.num_features)
        mean_w = mean_matrix @ w_transpose
        exp_ff = exp_ff.flatten()
        mean_w = mean_w.flatten()
        exp_y_f = (target - mean_w)**2 + exp_ff
        res = (exp_y_f) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5)

        # matrix form: summation of res.
        # target_transpose = target.transpose(-2, -1)
        # exp_y_f = target_transpose @ target - self.mixing_weights @ mean_matrix_transpose @ target - \
        #           target_transpose @ mean_matrix @ w_transpose - exp_ff.sum()

        return res * self.num_features # times seq_len, because the log_likelihood in _approximate_mll will divide (n_traj*seq_len).

    def forward(self, function_samples, *params, **kwargs):
        """
        Compute the conditional distribution p(y|f) = N(fW^{T}, \epsilon)
        :param function_samples: f sampled from q(f).
        :return: p(y|f).
        """
        num_data, num_features = function_samples.shape[-2:]
        function_samples = function_samples.view(function_samples.size(0),
                                                   num_data/self.num_features, self.num_features) # (N_sample, n, T)
        function_samples = function_samples @ self.mixing_weights # (N_sample, n, 1)
        noise = self._shaped_noise_covar(function_samples.shape, *params, **kwargs).diag()
        return base_distributions.Normal(function_samples, noise.sqrt())

    def log_marginal(self, observations, function_dist, *params, **kwargs):
        """
        Compute the log marginal likelihood of the approximate predictive distribution log E_{f~q(f)}[p(y|f)]
        :param observations: target y.
        :param function_dist: marginal likelihood.
        :return: log E_{f~q(f)}[p(y|f)] = log p(y).
        """
        marginal = self.marginal(function_dist, *params, **kwargs)
        # We're making everything conditionally independent
        indep_dist = base_distributions.Normal(marginal.mean, marginal.variance.clamp_min(1e-8).sqrt())
        res = indep_dist.log_prob(observations)

        # Do appropriate summation for multitask Gaussian likelihoods
        num_event_dim = len(function_dist.event_shape)
        if num_event_dim > 1:
            res = res.sum(list(range(-1, -num_event_dim, -1)))
        return res

    def marginal(self, function_dist, *params, **kwargs):
        """
        Compute p(y^{*}|x^{*}).
        :param function_dist: q(f*).
        :return: p(y^{*}|x^{*}) = N(fw^(T), \epsilon), fw^(T) ~ N(E[fW^{t}], Var[fW^{t}]),
                 cov[fW^{t}]_{ij}= W\Sigma_{i*T:(i+1)*T, j*T:(j+1)*T}W^{T}.
        """
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        num_data = int(mean.shape[0]/self.num_features)
        mean = mean.view(num_data, self.num_features)
        mean = mean @ self.mixing_weights.transpose(-2, -1)
        mean = mean.flatten()
        # Matrix form of the covariance matrix.
        # left = [W, 0, 0, 0,
        #         0, W, 0, 0,
        #         0, 0, W, 0,
        #         0, 0, 0, W] (N, N*T)
        # middle = covar (N*T, N*T)
        # right = [W^T, 0, 0, 0,
        #          0, W^T, 0, 0,
        #          0, 0, W^T, 0,
        #          0, 0, 0, W^T] (N*T, N)

        left_weight = torch.zeros(num_data, num_data*self.num_features)
        for i in range(num_data):
            left_weight[i, i*self.num_features:(i+1)*self.num_features] = self.mixing_weights.flatten()
        if torch.cuda.is_available():
            left_weight = left_weight.cuda()

        right_weight = left_weight.transpose(-2, -1)

        if settings.trace_mode.on():
            covar_fw = left_weight @ covar.evaluate() @ right_weight
        else:
            covar_fw = MatmulLazyTensor(left_weight, covar @ right_weight)

        noise_covar = self._shaped_noise_covar(mean.shape, *params, **kwargs)
        full_covar = covar_fw + noise_covar
        return function_dist.__class__(mean, full_covar)

    @property
    def noise(self):
        return self.noise_covar.noise

    @noise.setter
    def noise(self, value):
        self.noise_covar.initialize(noise=value)

    @property
    def raw_noise(self):
        return self.noise_covar.raw_noise

    @raw_noise.setter
    def raw_noise(self, value):
        self.noise_covar.initialize(raw_noise=value)


class CustomizedVariationalStrategy(_VariationalStrategy):
    """
    Implementation of our customized variational inference strategy.
    This strategy is based on the standard variational strategy with whitening,
    with the option for SoR approximation and KSI (Structured kernel interpolation).
    Standard variational strategy with whitening:
        Prior of u=f(z): p(u) ~ N(0, K_{zz})
        Let u = Lv, where v ~ N(0, 1), LL^T = K_{zz}, change of variable from u to v.
        Define the variational distribution of v as q(v) = N(\mu, S).
        Then, we can compute the variational distribution of u as q(u) ~ N(L\mu, LSL^T).
    Standard strategy:
        q(f|v) = N(K_{xz}K_{zz}^(-1)Lv, K_{xx} - K_{xz}K_{zz}^{-1}K_{xz}^{T})
        q(f) = N(K_{xz}K_{zz}^{-1/2}\mu, K_{xx} + K_{xz}K_{zz}^{-1/2}(S-I)K_{zz}^{-1/2}K_{xz}^{T})
    With the SoR approximation:
        q(f|v) \approx K_{xz}K_{zz}^(-1)Lv (Omit the variance of the conditional distribution q(f|u)).
        Here q(f) = N(K_{xz}K_{zz}^{-1/2}\mu, K_{xz}K_{zz}^{-1/2}SK_{zz}^{-1/2}K_{xz}^{T})
        K_{xx} can be approximated as  K_{xz}K_{zz}^{-1}K_{xz}^{T}.
    """

    def __init__(self, model, inducing_points, variational_distribution, learning_inducing_locations=True,
                 using_sor=False):
        """
        :param model: Model this strategy is applied to (ApproximateGP).
        :param inducing_points: z.
        :param variational_distribution: q(u) or q(v) if whitening.
        :param learning_inducing_locations: Whether to learn/udpate z.
        :param using_sor: Whether to use SoR.
        """
        super(CustomizedVariationalStrategy, self).__init__(model, inducing_points, variational_distribution,
                                                            learning_inducing_locations)
        self.register_buffer("updated_strategy", torch.tensor(True))
        self.using_sor = using_sor
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(delazify(induc_induc_covar).double())
        return TriangularLazyTensor(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        """
        Prior of v rather than u due to changing of variables.
        :return: v ~ N(0, I)
        """
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

    @property
    @cached(name="variational_distribution_memo")
    def variational_distribution(self):
        """
        Variational distribution of v.
        :return: v ~ N(\mu, S), S=LL^{T}.
        """
        return self._variational_distribution()

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        """
        Computing the mariginal posterior of f: q(f) = \int p(f|x, u)q(u)du
        :param x: Locations to get the variational posterior of the function values at.
        :param inducing_points: Locations of the inducing points (z).
        :param inducing_values: Samples of the inducing function values (u), or the mean of the distribution q(u)
                                if q is a Gaussian, here it should be q(v).
        :param variational_inducing_covar (gpytorch.lazy.LazyTensor): If the distribuiton q(u)/q(v) is Gaussian,
        then this variable is the covariance matrix of that Gaussian. Otherwise, it will be :attr:`None`.
        :return: q(f(x)).
        """
        # Compute full prior distribution
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs)
        full_covar = full_output.lazy_covariance_matrix

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:] # \mu_x
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter() # K_{zz}.
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate() # K_{xz}.
        data_data_covar = full_covar[..., num_induc:, num_induc:] # K_{xx}.

        # Compute interpolation terms
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar) # LL^T = K_{zz}
        if L.shape != induc_induc_covar.shape:
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.inv_matmul(induc_data_covar.double()).to(full_inputs.dtype) # K_{zz}^{-1/2} K_{xz}^T

        # Compute the mean of q(f)
        # K_{xz} K_{zz}^{-1/2} \mu_z + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        middle_term = self.prior_distribution.lazy_covariance_matrix # I
        if self.using_sor and self.training:
            # print('Using SoR.')
            # K_XZ K_ZZ^{-1/2} S K_ZZ^{-1/2} K_ZX
            if variational_inducing_covar is not None:
                middle_term = MatmulLazyTensor(variational_inducing_covar, middle_term)
        else:
            # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX
            middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1)  # -I
            if variational_inducing_covar is not None:
                middle_term = SumLazyTensor(variational_inducing_covar, middle_term) # S-I

        if settings.trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(1e-4).evaluate()
                + interp_term.transpose(-1, -2) @ middle_term.evaluate() @ interp_term
            )
        else:
            predictive_covar = SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term),
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def kl_divergence(self):
        """
        Compute the KL divergence between the variational inducing distribution q(v).
        and the prior inducing distribution p(v), the KL term in ELBO.
        KL(q(v)||p(v)) = \int q(v) log(p(v)/q(v)) dv.
        :rtype: torch.Tensor
        """
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence

    def __call__(self, x, prior=False, **kwargs):
        """
        call function VariationalStrategy(x, prior)
        :param x: Location X.
        :param prior: whether to output the prior distribution.
        :return: marginal prior/posterior distribution.
        """
        if not self.updated_strategy.item() and not prior:
            with torch.no_grad():
                # Get unwhitened p(u)
                prior_function_dist = self(self.inducing_points, prior=True)
                prior_mean = prior_function_dist.loc
                L = self._cholesky_factor(prior_function_dist.lazy_covariance_matrix.add_jitter())

                # Temporarily turn off noise that's added to the mean
                orig_mean_init_std = self._variational_distribution.mean_init_std
                self._variational_distribution.mean_init_std = 0.0

                # Change the variational parameters to be whitened
                variational_dist = self.variational_distribution
                mean_diff = (variational_dist.loc - prior_mean).unsqueeze(-1).double()
                whitened_mean = L.inv_matmul(mean_diff).squeeze(-1).to(variational_dist.loc.dtype)
                covar_root = variational_dist.lazy_covariance_matrix.root_decomposition().root.evaluate().double()
                whitened_covar = RootLazyTensor(L.inv_matmul(covar_root).to(variational_dist.loc.dtype))
                whitened_variational_distribution = variational_dist.__class__(whitened_mean, whitened_covar)
                self._variational_distribution.initialize_variational_distribution(whitened_variational_distribution)

                # Reset the random noise parameter of the model
                self._variational_distribution.mean_init_std = orig_mean_init_std

                # Reset the cache
                clear_cache_hook(self)

                # Mark that we have updated the variational strategy
                self.updated_strategy.fill_(True)

        return super().__call__(x, prior=prior, **kwargs)


def VisualizeCovar(covariance, save_path):
    plt.figure()
    heat = sns.heatmap(
        covariance,
        cmap="YlGnBu",
        square=True,
        robust=True,
        xticklabels=False,
        yticklabels=False,
    )
    if save_path[-3:] != 'pdf':
        raise TypeError('Output format should be pdf.')
    plt.savefig(save_path, bbox_inches='tight')
    return 0
