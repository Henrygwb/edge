#!/usr/bin/env python3

import math
import torch
import warnings
import torch.nn as nn
from gpytorch.likelihoods.likelihood import Likelihood
# from gpytorch.likelihoods.gaussian_likelihood import _GaussianLikelihoodBase
from gpytorch.distributions import Distribution, MultitaskMultivariateNormal, base_distributions
from gpytorch.likelihoods.noise_models import HomoskedasticNoise


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

    def forward(self, function_samples, *params, **kwargs):
        """
        :param function_samples: samples from q(f).
        :return: Marginal likelihood distribution.
        """
        num_data, num_features = function_samples.shape[-2:]

        # Catch legacy mode
        if num_data == self.num_features:
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                DeprecationWarning,
            )
            function_samples = function_samples.transpose(-1, -2)
            num_data, num_features = function_samples.shape[-2:]

        if num_features != self.num_features:
            raise RuntimeError("There should be %d features" % self.num_features)

        function_samples = function_samples.resize(function_samples.size(0),
                                                   num_data/self.num_features, self.num_features)
        print('Check the shape of f, should be [n_likelihood_sample, n_traj, traj_length]:')
        print(function_samples.shape)
        if self.mixing_weights is not None:
            mixed_fs = function_samples @ self.mixing_weights.t()  # num_classes x num_data
            print('Check the shape of fW, should be [n_likelihood_sample, n_traj, n_classes]:')
            print(mixed_fs.shape)
        else:
            mixed_fs = function_samples
        res = base_distributions.Categorical(logits=mixed_fs)
        return res

    def __call__(self, function, *params, **kwargs):
        if isinstance(function, Distribution) and not isinstance(function, MultitaskMultivariateNormal):
            warnings.warn(
                "The input to SoftmaxLikelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                DeprecationWarning,
            )
            function = MultitaskMultivariateNormal.from_batch_mvn(function)
        return super().__call__(function, *params, **kwargs)


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
        self.noise_covar = HomoskedasticNoise(
            noise_prior=noise_prior, noise_constraint=noise_constraint, batch_shape=batch_shape
        )
        super().__init__()
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
        Compute the expected conditional log likelihood (likelihood term in ELBO) E_{f~q(f)}[log p(y|f)].
        q(f)=N(\mu_f (n*T, 1), \Sigma_f(n*T, n*T)), p(y|f)=N(fW^{T}, \epsilon), f: (1, T) W: (T, 1) \epsilon ~ N(0, \sigma_{noise}).
        E_{f~q(f)}[log p(y|f)] = -N/2(log 2\pi + log \Sigma_{noise} + \Sigma_{noise}E_{f~q(f)}[(y - fW^{T})^{t}(y - fW^{T})])
        E_{f~q(f)}[(y - fW^{T})^{t}(y - fW^{T})]) = y^{T}y - W\mu^{T}y - y^{T}\muW^{T} - WE[f^{T}f]W^{T}
        E_{f~q(f)}[f^{T}f] = \sum_{n}[Sigma_{i}+\mu_{i}\mu_{i}^{T}]
        :param target: y (n, 1).
        :param input: q(f). 
        :param params: any
        :param kwargs: any
        :return: E_{f~q(f)}[log p(y|f)].
        """
        mean, variance = input.mean, input.variance # mean and var of q(f).
        num_event_dim = len(input.event_shape)

        noise = self._shaped_noise_covar(mean.shape, *params, **kwargs).diag() # noise variance.
        # Potentially reshape the noise to deal with the multitask case
        noise = noise.view(*noise.shape[:-1], *input.event_shape)

        exp_ff = variance[0:self.num_features, 0:self.num_features] \
                 + mean[0:self.num_features] @ mean[0:self.num_features].transpose(-2, -1)
        for i in range(num_event_dim-1):
            i = i+1
            exp_ff_tmp = variance[self.num_features*i:self.num_features*(i+1), self.num_features*i:self.num_features*(i+1)] \
                         + mean[self.num_features*i:self.num_features*(i+1)] @ mean[self.num_features*i:self.num_features*(i+1)].transpose(-2, -1)
            exp_ff += exp_ff_tmp

        target_transpose = target.transpose(-2, -1)
        w_transpose = self.mixing_weights.transpose(-2, -1)
        mean_matrix = mean.resize(num_event_dim, self.num_features)
        mean_matrix_transpose = mean_matrix.transpose(-2, -1)
        exp_y_f = target_transpose @ target - self.mixing_weights @ mean_matrix_transpose @ target - \
                  target_transpose @ mean_matrix @ w_transpose - self.mixing_weights @ exp_ff @ w_transpose
        res = (exp_y_f) / noise + noise.log() + math.log(2 * math.pi)
        res = res.mul(-0.5 * num_event_dim)
        return res

    def forward(self, function_samples, *params, **kwargs):
        """
        Compute the conditional distribution p(y|f) = N(fW^{T}, \epsilon)
        :param function_samples: f sampled from q(f).
        :return: p(y|f).
        """
        num_data, num_features = function_samples.shape[-2:]
        function_samples = function_samples.resize(function_samples.size(0),
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
        :param function_dist: q(f).
        :return: p(y^{*}|x^{*}) = N(fw^(T), \epsilon), fw^(T) ~ N(E[fW^{t}], Var[fW^{t}]), Var[fW^{t}]=\sum_{n}[Sigma_{i}.
        """
        num_event_dim = function_dist.event_shape
        mean, covar = function_dist.mean, function_dist.lazy_covariance_matrix
        mean = mean.resize(num_event_dim, self.num_features)
        mean = mean @ self.mixing_weights

        covar_fw = covar[0:self.num_features, 0:self.num_features]
        for i in range(num_event_dim-1):
            i = i+1
            covar_fw_tmp = covar[self.num_features*i:self.num_features*(i+1), self.num_features*i:self.num_features*(i+1)]
            covar_fw += covar_fw_tmp

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
