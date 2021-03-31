# Implementation of [Stochastic Variational Deep Kernel Learning].
# With the following techniques:
# 1. (Better feature representation) Use a DNN inside the RBF kernel function.
# 2. (Faster way of computing p(f|u)) Use the inducing point technique with the SoR approximation of
#    p(f|u)=K_{x,z}K_{z,z}u and p(f_*|u)=K_{x_*,z}K_{z,z}u.
# 3. (Non-Gaussion likelihood) Use the Variational inference with the variational distribution as q(u)~N(\mu, LL^T).
# 4. (Faster way of computing K_{x,z} in p(f|u)) KISS-GP: Use the local kernel interpolation technique
# (Structured kernel interpolation) to approximate K_{x,z}K_{z,z} with the interpolation matrix M.
# 5. (Construct K_{z,z} with a specific structure) Place inducting point (Z) on a grid. Here, the inducing points are
# fixed after defining the grid boundary and the grid size.

import math
import tqdm
import torch
import gpytorch
from torch import nn
from torch.optim import SGD
from torchsummary import summary
from torch.optim.lr_scheduler import MultiStepLR

# load the dataset.
train_set = torch.utils.data.TensorDataset(torch.randn(8, 256), torch.rand(8).round().long()) # torch.round(): round to the closest int.
test_set = torch.utils.data.TensorDataset(torch.randn(4, 256), torch.rand(4).round().long()) # torch.long(): change the data type to int64.
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False)
num_classes = 2


# Build the model with DNN and the GP layer.
# DNN. # define the model structure and the forward function.
class Encoder(nn.Module):
    def __init__(self, input_dim, z_dim, hidden_dim):
        super(Encoder, self).__init__()
        self.num_dim = z_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.CELU(),
            nn.Dropout(.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.CELU(),
            nn.Dropout(.2),
            nn.Linear(hidden_dim, z_dim)
        )

    def forward(self, x):
        return self.net(x)


# GP layer, inherited from the Approximated GP class.
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds, grid_size=64):

        # Define the variational posterior distribution. q(u) = N(\mu, S), S = LL^T.
        # L: lower triangular and positive diagonal.
        # Parameters: \mu (num_inducing_samples, 1) and L (num_inducing_samples, num_inducing_samples).
        # Define one GP/variational distribution for each input dim. Here the number of GP is the batch_size.
        # member function: forward() -> output a multivariate Gaussian distribution.
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP

        # All the strategies are inherited from the variational_strategy class. This class takes as input the inducing
        # points and the variational distribution.
        # Member functions: prior, forward (integrate out u and compute f),
        # kl_divergence (prior and variational distribution).

        variational_strategy = gpytorch.variational.IndependentMultitaskVariationalStrategy(
            gpytorch.variational.GridInterpolationVariationalStrategy(
                self, grid_size=grid_size, grid_bounds=[grid_bounds],
                variational_distribution=variational_distribution,
            ), num_tasks=num_dim,
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        # Compute the prior latent distribution on a given input x.
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    # The call function, defined in approximate_gp.
    # If prior = True, output the prior distribution of the input x.
    # If prior = False, output the predictive distribution of the input x.
    # Usage: define gp = GaussianProcessLayer(), then run gp(x).
    # It gives the marginal posterior p(f|x) = \int p(f, u|x)q(u) du.

    # def __call__(self, inputs, prior=False, **kwargs):
    #     if inputs.dim() == 1:
    #         inputs = inputs.unsqueeze(-1)
    #     return self.variational_strategy(inputs, prior=prior, **kwargs)

"""
# The implementation of Structure Kernel Interpolation (SKI), with multiple approximation strategies.
# Introducing points with SoR approximation of p(f|u) = N(K_{xz}K_{zz}^-1u, 0).
# Interpolation approximation of K_{xz} = MK_{zz} -> f = Mu. K_{xx} = M K_{zz} M^T.
# Grid structure of inducing point Z. Note that Z are fixed after defining the grid bound and size. No need to learn. 
# Variational inference with an variational distribution q(u) -> variation posterior: q(f, u|y) = p(f|u)q(u).

class GridInterpolationVariationalStrategy(_VariationalStrategy):
    # This strategy constrains the inducing points to a grid and applies a deterministic
    # relationship between :math:`\mathbf f` and :math:`\mathbf u`.
    # It was introduced by `Wilson et al. (2016)`_.
    # 
    # Here, the inducing points are not learned. Instead, the strategy
    # automatically creates inducing points based on a set of grid sizes and grid
    # bounds.
    # 
    # .. _Wilson et al. (2016): https://arxiv.org/abs/1611.00336
    # 
    # :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to. p(y|f)
    #     Typically passed in when the VariationalStrategy is created in the
    #     __init__ method of the user defined model.
    # :param int grid_size: Size of the grid. # Number of partitions of the grid of each dim.
    # :param list grid_bounds: Bounds of each dimension of the grid (should be a list of (float, float) tuples)
    # :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
    #     VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    

    def __init__(self, model, grid_size, grid_bounds, variational_distribution):
        # Computing the function value of inducing points z. Here, inducing point is computed by the given 
        # grid_size and grid_bounds, no need to train it. 
        
        grid = torch.zeros(grid_size, len(grid_bounds)) # grid (grid_size, num_dim): possible value of each dimension.
        for i in range(len(grid_bounds)):
            grid_diff = float(grid_bounds[i][1] - grid_bounds[i][0]) / (grid_size - 2)
            grid[:, i] = torch.linspace(grid_bounds[i][0] - grid_diff, grid_bounds[i][1] + grid_diff, grid_size)

        # Inducing points with shape (grid_size^(num_dim), num_dim), Combinations of each possible value in each dim.
        inducing_points = torch.zeros(int(pow(grid_size, len(grid_bounds))), len(grid_bounds))
        prev_points = None
        for i in range(len(grid_bounds)):
            for j in range(grid_size):
                inducing_points[j * grid_size ** i : (j + 1) * grid_size ** i, i].fill_(grid[j, i])
                if prev_points is not None:
                    inducing_points[j * grid_size ** i : (j + 1) * grid_size ** i, :i].copy_(prev_points)
            prev_points = inducing_points[: grid_size ** (i + 1), : (i + 1)]
        
        super(GridInterpolationVariationalStrategy, self).__init__(
            model, inducing_points, variational_distribution, learn_inducing_locations=False
        )
        object.__setattr__(self, "model", model)

        self.register_buffer("grid", grid) 

    def _compute_grid(self, inputs):
        # Given the input x and the inducing points z, computing the approximation of K_{xz} = M via interpolation.
        # x (batch_size, sample_size_in_batch, num_dim).
        # z ((grid_size)^(num_dim), num_dim).
        # M (sample_size_in_batch, (grid_size)^(num_dim)) repeat batch_size times -> (batch_size, sample_size_in_batch, (grid_size)^(num_dim))        

        n_data, n_dimensions = inputs.size(-2), inputs.size(-1)
        batch_shape = inputs.shape[:-2]

        inputs = inputs.reshape(-1, n_dimensions)
        interp_indices, interp_values = Interpolation().interpolate(self.grid, inputs)
        interp_indices = interp_indices.view(*batch_shape, n_data, -1)
        interp_values = interp_values.view(*batch_shape, n_data, -1)

        if (interp_indices.dim() - 2) != len(self._variational_distribution.batch_shape):
            batch_shape = _mul_broadcast_shape(interp_indices.shape[:-2], self._variational_distribution.batch_shape)
            interp_indices = interp_indices.expand(*batch_shape, *interp_indices.shape[-2:])
            interp_values = interp_values.expand(*batch_shape, *interp_values.shape[-2:])
        return interp_indices, interp_values

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        # p(u) ~ N(model(z).mean, model(z).covariance)
        out = self.model.forward(self.inducing_points)
        res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter())
        return res

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None):
        # :param torch.Tensor inducing_points: Locations :math:`\mathbf Z` of the inducing points
        # :param torch.Tensor inducing_values: Samples of the inducing function values :math:`\mathbf u`
        #     (or the mean of the distribution :math:`q(\mathbf u)` if q is a Gaussian.
        # :param ~gpytorch.lazy.LazyTensor variational_inducing_covar: If the distribuiton :math:`q(\mathbf u)`
        #     is Gaussian, then this variable is the covariance matrix of that Gaussian. Otherwise, it will be
        #     :attr:`None`.
        # Compute the variational posterior f, 
        # given x,  z (inducing_points), and \mu_z (inducing values).
        # q(f|x) = \int p(f|u, x)q(u) du.
        # prior of p(f|x) ~ N(0, \Sigma_{xx}).
        # prior of p(u|z) ~ N(0, \Sigma_{zz}).
        # variational posterior of q(u) ~ N(\mu, LL^T).
        # conditional posterior SoR approximation with KSI: p(f|x, u) = M\mu_z
        # marginal posterior of f: q(f|x) ~ N(M\mu, MLL^TM^T).
        
        if variational_inducing_covar is None:
            raise RuntimeError(
                "GridInterpolationVariationalStrategy is only compatible with Gaussian variational "
                f"distributions. Got ({self.variational_distribution.__class__.__name__}."
            )

        variational_distribution = self.variational_distribution # q(u)

        # Get interpolations
        interp_indices, interp_values = self._compute_grid(x) # Obtain M* from x_* and z. 

        # Compute test mean: MU.
        # Left multiply samples by interpolation matrix
        predictive_mean = left_interp(interp_indices, interp_values, inducing_values.unsqueeze(-1)) 
        predictive_mean = predictive_mean.squeeze(-1) # [n_gp(hidden_dim), batch_size]

        # Compute test covar: MK_{zz}M^T.
        predictive_covar = InterpolatedLazyTensor(
            variational_distribution.lazy_covariance_matrix,
            interp_indices,
            interp_values,
            interp_indices,
            interp_values,
        ) # [n_gp(hidden_dim), batch_size, batch_size]
        output = MultivariateNormal(predictive_mean, predictive_covar)
        return output
    
    # Defined in _variational_strategy, output the distribution of input x.
    # If in prior model, output the prior mean: model(x), else output the posterior(predictive) distribution by calling forward.
    # Usage: define s = GridInterpolationVariationalStrategy, and then call s(x).

    def __call__(self, x, prior=False, **kwargs):
        # If we're in prior mode, then we're done!
        if prior:
            return self.model.forward(x, **kwargs)

        # Delete previously cached items from the training distribution
        if self.training:
            self._clear_cache()
        # (Maybe) initialize variational distribution
        if not self.variational_params_initialized.item():
            prior_dist = self.prior_distribution
            self._variational_distribution.initialize_variational_distribution(prior_dist)
            self.variational_params_initialized.fill_(1)

        # Ensure inducing_points and x are the same size
        inducing_points = self.inducing_points
        if inducing_points.shape[:-2] != x.shape[:-2]:
            x, inducing_points = self._expand_inputs(x, inducing_points)

        # Get q(u)
        variational_dist_u = self.variational_distribution

        # Get q(f) = \int q(f|u)q(u) du
        if isinstance(variational_dist_u, MultivariateNormal):
            return super().__call__(
                x,
                inducing_points,
                inducing_values=variational_dist_u.mean,
                variational_inducing_covar=variational_dist_u.lazy_covariance_matrix,
                **kwargs,
            )
        elif isinstance(variational_dist_u, Delta):
            return super().__call__(
                x, inducing_points, inducing_values=variational_dist_u.mean, variational_inducing_covar=None, **kwargs
            )
        else:
            raise RuntimeError(
                f"Invalid variational distribuition ({type(variational_dist_u)}). "
                "Expected a multivariate normal or a delta distribution."
            )

    def kl_divergence(self):
        # Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        # and the prior inducing distribution :math:`p(\mathbf u)`.
    
        :rtype: torch.Tensor
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence
"""

# Final layer.
# Return a categorical distribution with parameter (softmax(X*W^T))
"""
class SoftmaxLikelihood(Likelihood):
    # Implements the Softmax (multiclass) likelihood used for GP classification.
    # 
    # .. math::
    #     p(\mathbf y \mid \mathbf f) = \text{Softmax} \left( \mathbf W \mathbf f \right)
    # 
    # :math:`\mathbf W` is a set of linear mixing weights applied to the latent functions :math:`\mathbf f`.
    # 
    # :param int num_features: Dimensionality of latent function :math:`\mathbf f`.
    # :param int num_classes: Number of classes.
    # :param bool mixing_weights: (Default: `True`) Whether to learn a linear mixing weight :math:`\mathbf W` applied to
    #     the latent function :math:`\mathbf f`. If `False`, then :math:`\mathbf W = \mathbf I`.
    # :param mixing_weights_prior: Prior to use over the mixing weights :math:`\mathbf W`.
    # :type mixing_weights_prior: ~gpytorch.priors.Prior, optional

    def __init__(self, num_features=None, num_classes=None, mixing_weights=True, mixing_weights_prior=None):
        super().__init__()
        if num_classes is None:
            raise ValueError("num_classes is required")
        self.num_classes = num_classes
        if mixing_weights:
            self.num_features = num_features
            if num_features is None:
                raise ValueError("num_features is required with mixing weights")
            # Define the trainable parameter 
            self.register_parameter(
                name="mixing_weights",
                parameter=torch.nn.Parameter(torch.randn(num_classes, num_features).div_(num_features)), # Initialize the softmax weight (num_classes, num_features), random initialization - Gaussian noise/2. 
            )
            if mixing_weights_prior is not None:
                self.register_prior("mixing_weights_prior", mixing_weights_prior, "mixing_weights") # Add a module with a prior of the correspond parameter.
        else:
            self.num_features = num_classes
            self.mixing_weights = None
    
    # Forward pass function, function_samples: input samples (N*D). 
    def forward(self, function_samples, *params, **kwargs):
        num_data, num_features = function_samples.shape[-2:]

        # Catch legacy mode
        if num_data == self.num_features:
            warnings.warn(
                "The input to Softmax Likelihood should be a MultitaskMultivariateNormal (num_data x num_tasks). "
                "Batch MultivariateNormal inputs (num_tasks x num_data) will be deprectated.",
                DeprecationWarning,
            )
            function_samples = function_samples.transpose(-1, -2) 
            num_data, num_features = function_samples.shape[-2:]

        if num_features != self.num_features:
            raise RuntimeError("There should be %d features" % self.num_features)

        if self.mixing_weights is not None:
            mixed_fs = function_samples @ self.mixing_weights.t()  # X (N*D) * W^T (D*Y_n) # num_classes x num_data
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
"""


# Create the final DGP model.
class DKLModel(gpytorch.Module):
    def __init__(self, feature_extractor, num_dim, grid_bounds=(-10., 10.)):
        super(DKLModel, self).__init__()
        self.feature_extractor = feature_extractor
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

    def forward(self, x):
        features = self.feature_extractor(x) # N*D -> N*D_1
        features = gpytorch.utils.grid.scale_to_bounds(features, self.grid_bounds[0], self.grid_bounds[1])
        # This next line makes it so that we learn a GP for each feature
        # N*D_1 -> D_1*N*1 (Fit D_1 GP with each one have N data and one dim)
        features = features.transpose(-1, -2).unsqueeze(-1)
        # The next line gives the N*D_1 output of the Gaussian process model. Here, the output is f|u = Mu.
        res = self.gp_layer(features) # res.mean.shape[batch_size, hidden_dim]; res.cov.shape[batch_size*hidden_dim, batch_size*hidden_dim]
        return res


hidden_dim = 5
feature_extractor = Encoder(input_dim=train_set.tensors[0].shape[-1], z_dim=hidden_dim, hidden_dim=16)
model = DKLModel(feature_extractor, num_dim=hidden_dim)
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)
print(model)
print(summary(model.feature_extractor, input_size=(train_set.tensors[0].shape[-1],)))

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

# Training process.
n_epochs = 5
lr = 0.1
optimizer = SGD([
    {'params': model.feature_extractor.parameters(), 'weight_decay': 1e-4}, # DNN parameters.
    {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01}, # RBF kernel parameters.
    {'params': model.gp_layer.variational_parameters()}, # Mean \mu and lower triangle matrix L in the q(u).
    {'params': likelihood.parameters()}, # Mixing matrix W in the likelihood.
], lr=lr, momentum=0.9, nesterov=True, weight_decay=0)
scheduler = MultiStepLR(optimizer, milestones=[0.5 * n_epochs, 0.75 * n_epochs], gamma=0.1)
# Define the final loss -> ELBO = E_{q(f, u)}[p(y|f)] - KL[q(u)||p(u)].
# As shown above, compute ELBO requires conditional likelihood p(y|f), prior p(u), and the variational posterior q(u).
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

"""
# The implementation of variational ELBO. Using MC approximation for the likelihood term.
# Inherited from the _ApproximateMarginalLogLikelihood class. 

class VariationalELBO(_ApproximateMarginalLogLikelihood):
    # The variational evidence lower bound (ELBO). This is used to optimize
    # variational Gaussian processes (with or without stochastic optimization).
    # 
    # .. math::
    # 
    #    \begin{align*}
    #       \mathcal{L}_\text{ELBO} &=
    #       \mathbb{E}_{p_\text{data}( y, \mathbf x )} \left[
    #         \mathbb{E}_{p(f \mid \mathbf u, \mathbf x) q(\mathbf u)} \left[  \log p( y \! \mid \! f) \right]
    #       \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
    #       \\
    #       &\approx \sum_{i=1}^N \mathbb{E}_{q( f_i)} \left[
    #         \log p( y_i \! \mid \! f_i) \right] - \beta \: \text{KL} \left[ q( \mathbf u) \Vert p( \mathbf u) \right]
    #    \end{align*}
    # 
    # where :math:`N` is the number of datapoints, :math:`q(\mathbf u)` is the variational distribution for
    # the inducing function values, :math:`q(f_i)` is the marginal of
    # :math:`p(f_i \mid \mathbf u, \mathbf x_i) q(\mathbf u)`,
    # and :math:`p(\mathbf u)` is the prior distribution for the inducing function values.
    # 
    # :math:`\beta` is a scaling constant that reduces the regularization effect of the KL
    # divergence. Setting :math:`\beta=1` (default) results in the true variational ELBO.
    # 
    # For more information on this derivation, see `Scalable Variational Gaussian Process Classification`_
    # (Hensman et al., 2015).
    # 
    # :param ~gpytorch.likelihoods.Likelihood likelihood: The likelihood for the model
    # :param ~gpytorch.models.ApproximateGP model: The approximate GP model
    # :param int num_data: The total number of training data points (necessary for SGD)
    # :param float beta: (optional, default=1.) A multiplicative factor for the KL divergence term.
    #     Setting it to 1 (default) recovers true variational inference
    #     (as derived in `Scalable Variational Gaussian Process Classification`_).
    #     Setting it to anything less than 1 reduces the regularization effect of the model
    #     (similarly to what was proposed in `the beta-VAE paper`_).
    # :param bool combine_terms: (default=True): Whether or not to sum the
    #     expected NLL with the KL terms (default True)
    # 
    # Example:
    #     >>> # model is a gpytorch.models.ApproximateGP
    #     >>> # likelihood is a gpytorch.likelihoods.Likelihood
    #     >>> mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=100, beta=0.5)
    #     >>>
    #     >>> output = model(train_x)
    #     >>> loss = -mll(output, train_y)
    #     >>> loss.backward()
    # 
    # .. _Scalable Variational Gaussian Process Classification:
    #     http://proceedings.mlr.press/v38/hensman15.pdf
    # .. _the beta-VAE paper:
    #     https://openreview.net/pdf?id=Sy2fzU9gl
    
    def _log_likelihood_term(self, variational_dist_f, target, **kwargs):
        
        return self.likelihood.expected_log_prob(target, variational_dist_f, **kwargs).sum(-1)

    def forward(self, variational_dist_f, target, **kwargs):

        # Computes the Variational ELBO given :math:`q(\mathbf f)` and :math:`\mathbf y`.
        # Calling this function will call the likelihood's :meth:`~gpytorch.likelihoods.Likelihood.expected_log_prob`
        # function.
        # 
        # :param ~gpytorch.distributions.MultivariateNormal variational_dist_f: :math:`q(\mathbf f)`
        #     the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
        # :param torch.Tensor target: :math:`\mathbf y` The target values
        # :param kwargs: Additional arguments passed to the
        #     likelihood's :meth:`~gpytorch.likelihoods.Likelihood.expected_log_prob` function.
        # :rtype: torch.Tensor
        # :return: Variational ELBO. Output shape corresponds to batch shape of the model/input data.

        return super().forward(variational_dist_f, target, **kwargs)
        
    # The implementation of forward in _ApproximateMarginalLogLikelihood.

    def forward(self, approximate_dist_f, target, **kwargs):

        # Computes the Variational ELBO given :math:`q(\mathbf f)` and `\mathbf y`.
        # Calling this function will call the likelihood's `expected_log_prob` function.
        # 
        # Args:
        #     :attr:`approximate_dist_f` (:obj:`gpytorch.distributions.MultivariateNormal`):
        #         :math:`q(\mathbf f)` the outputs of the latent function (the :obj:`gpytorch.models.ApproximateGP`)
        #     :attr:`target` (`torch.Tensor`):
        #         :math:`\mathbf y` The target values
        #     :attr:`**kwargs`:
        #         Additional arguments passed to the likelihood's `expected_log_prob` function.
        # Get likelihood term and KL term

        num_batch = approximate_dist_f.event_shape[0]
        log_likelihood = self._log_likelihood_term(approximate_dist_f, target, **kwargs).div(num_batch) 
        kl_divergence = self.model.variational_strategy.kl_divergence().div(self.num_data / self.beta)

        # Add any additional registered loss terms
        added_loss = torch.zeros_like(log_likelihood)
        had_added_losses = False
        for added_loss_term in self.model.added_loss_terms():
            added_loss.add_(added_loss_term.loss())
            had_added_losses = True

        # Log prior term
        log_prior = torch.zeros_like(log_likelihood)
        for name, module, prior, closure, _ in self.named_priors():
            log_prior.add_(prior.log_prob(closure(module)).sum().div(self.num_data))

        if self.combine_terms:
            return log_likelihood - kl_divergence + log_prior - added_loss
        else:
            if had_added_losses:
                return log_likelihood, kl_divergence, log_prior, added_loss
            else:
                return log_likelihood, kl_divergence, log_prior

"""


# Define the train and test function.
def train(epoch):
    # specify the mode of the modules in the model; some modules may have different behaviors under training and eval
    # module, such as dropout and batch normalization.
    model.train()
    likelihood.train()

    minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
    with gpytorch.settings.num_likelihood_samples(8):
        for data, target in minibatch_iter:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data) # marginal variational posterior, q(f|x).
            loss = -mll(output, target) # approximated ELBO.
            loss.backward()
            optimizer.step()
            minibatch_iter.set_postfix(loss=loss.item())


def test():
    # Specify that the model is in eval mode.
    model.eval()
    likelihood.eval()

    correct = 0
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(16):
        for data, target in test_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = likelihood(model(data))  # This gives us 16 samples from the predictive distribution (q(f_*)).
            pred = output.probs.mean(0).argmax(-1)  # Taking the mean over all of the sample we've drawn (y = E_{f_* ~ q(f_*)}[y|f_*]).
            correct += pred.eq(target.view_as(pred)).cpu().sum()
    print('Test set: Accuracy: {}/{} ({}%)'.format(
        correct, len(test_loader.dataset), 100. * correct / float(len(test_loader.dataset))
    ))


for epoch in range(1, n_epochs + 1):
    with gpytorch.settings.use_toeplitz(False):
        train(epoch)
        test()
    scheduler.step()
    state_dict = model.state_dict()
    likelihood_state_dict = likelihood.state_dict()
    torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, 'dkl_cifar_checkpoint.dat')