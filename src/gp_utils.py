#!/usr/bin/env python3

import torch
import torch.nn as nn


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

        self.eps = eps

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
