# Implementation of [Orthogonally Decoupled Variational Gaussian Processes].
# Decouple the computation of posterior mean and covariance with two sets of inducing points.
# Using the delta distribution (single point) as the variational distribution for mean.
# Using the standard variational strategy for covariance SVGP.
# No approximation of the K_{xx}
# No structure of the inducing points.
# With Whitening - Variational posterior q(f):
# \mathbb E[ f(\mathbf x) ] = \mathbf k_{\mathbf Z_\mu \mathbf x}^\top \mathbf K_{\mathbf Z_\mu \mathbf Z_\mu}^{-1} \mathbf m
# \text{Var}[ f(\mathbf x) ] = k_{\mathbf x \mathbf x} - \mathbf k_{\mathbf Z_\sigma \mathbf x}^\top
# \mathbf K_{\mathbf Z_\sigma \mathbf Z_\sigma}^{-1}\left( \mathbf K_{\mathbf Z_\sigma} -
# \mathbf S \right)\mathbf K_{\mathbf Z_\sigma \mathbf Z_\sigma}^{-1}\mathbf k_{\mathbf Z_\sigma \mathbf x}


import math
import tqdm
import torch
import gpytorch
from torch import nn
from torch.optim import SGD
from sklearn.cluster import KMeans
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
    def __init__(self, mean_inducing_points, covar_inducing_points):

        # covar variational distribution as the CholeskyVariationalDistribution q(u) ~ N(\mu, LL^T).
        # covar variational strategy: VariationalStrategy with whitening
        # Standard variational strategy, without KSI.
        # p(u) ~ N(0, K_{zz})
        # Define u = Lv, where v ~ N(0, 1), LL^T = K_{zz}
        # q(v) ~ N(\mu, S) -> q(u) ~ N(L\mu, LSL^T)
        # q(f|v) ~ N(K_{xz}K_{zz}^(-1)Lv, K_{xx} - K_{xz}K_{zz}^{-1}K_{xz}^{T})
        # q(f) ~ N(K_{xz}K_{zz}^(-1/2)\mu, K_{xx} + K_{xz}K_{zz}^(-1/2)(S-I)K_{zz}^(-1/2)K_{xz}^{T})

        covar_variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(0))
        covar_variational_strategy = gpytorch.variational.VariationalStrategy(
            self, covar_inducing_points, covar_variational_distribution,
            learn_inducing_locations=True
        )

        # mean variational posterior distribution as the delta distribution q(u) = delta(\mu).
        # mean variational strategy: OrthogonallyDecoupledVariationalStrategy
        mean_variational_distribution = gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(0))

        variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
            covar_variational_strategy, mean_inducing_points,
            mean_variational_distribution
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

    def forward(self, x):
        # Compute the prior distribution on of f/u a given input x/z.
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
class VariationalStrategy(_VariationalStrategy):

    # The standard variational strategy, as defined by `Hensman et al. (2015)`_.
    # This strategy takes a set of :math:`m \ll n` inducing points :math:`\mathbf Z`
    # and applies an approximate distribution :math:`q( \mathbf u)` over their function values.
    # (Here, we use the common notation :math:`\mathbf u = f(\mathbf Z)`.
    # The approximate function distribution for any arbitrary input :math:`\mathbf X` is given by:
    # 
    # .. math::
    # 
    #     q( f(\mathbf X) ) = \int p( f(\mathbf X) \mid \mathbf u) q(\mathbf u) \: d\mathbf u
    # 
    # This variational strategy uses "whitening" to accelerate the optimization of the variational
    # parameters. See `Matthews (2017)`_ for more info.
    # 
    # :param ~gpytorch.models.ApproximateGP model: Model this strategy is applied to.
    #     Typically passed in when the VariationalStrategy is created in the
    #     __init__ method of the user defined model. (The model should specify the mean and covariance function.)
    # :param torch.Tensor inducing_points: Tensor containing a set of inducing
    #     points to use for variational inference.
    # :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
    #     VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    # :param learn_inducing_locations: (Default True): Whether or not
    #     the inducing point locations :math:`\mathbf Z` should be learned (i.e. are they
    #     parameters of the model).
    # :type learn_inducing_locations: `bool`, optional
    # 
    # .. _Hensman et al. (2015):
    #     http://proceedings.mlr.press/v38/hensman15.pdf
    # .. _Matthews (2017):
    #     https://www.repository.cam.ac.uk/handle/1810/278022

    def __init__(self, model, inducing_points, variational_distribution, learn_inducing_locations=True):
        super().__init__(model, inducing_points, variational_distribution, learn_inducing_locations)
        self.register_buffer("updated_strategy", torch.tensor(True))
        self._register_load_state_dict_pre_hook(_ensure_updated_strategy_flag_set)

    @cached(name="cholesky_factor", ignore_args=True)
    def _cholesky_factor(self, induc_induc_covar):
        L = psd_safe_cholesky(delazify(induc_induc_covar).double())
        return TriangularLazyTensor(L)

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        # Whitening: prior distribution is N(0, 1). 
        zeros = torch.zeros(
            self._variational_distribution.shape(),
            dtype=self._variational_distribution.dtype,
            device=self._variational_distribution.device,
        )
        ones = torch.ones_like(zeros)
        res = MultivariateNormal(zeros, DiagLazyTensor(ones))
        return res

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):

        # Standard variational strategy with whitening, without KSI.
        # p(u) ~ N(0, K_{zz})
        # Define u = Lv, where v ~ N(0, 1), LL^T = K_{zz}
        # q(v) ~ N(\mu, S) -> q(u) ~ N(L\mu, LSL^T)
        # q(f|v) ~ N(K_{xz}K_{zz}^(-1)Lv, K_{xx} - K_{xz}K_{zz}^{-1}K_{xz}^{T})
        # q(f) ~ N(K_{xz}K_{zz}^(-1/2)\mu, K_{xx} + K_{xz}K_{zz}^(-1/2)(S-I)K_{zz}^(-1/2)K_{xz}^{T})

        # Compute full prior distribution 
        full_inputs = torch.cat([inducing_points, x], dim=-2)
        full_output = self.model.forward(full_inputs, **kwargs) # Passing the input through the covariance function and get the prior.
        full_covar = full_output.lazy_covariance_matrix # Full covariance.

        # Covariance terms
        num_induc = inducing_points.size(-2)
        test_mean = full_output.mean[..., num_induc:] # \mu_x. 
        induc_induc_covar = full_covar[..., :num_induc, :num_induc].add_jitter() # K_{zz}.
        induc_data_covar = full_covar[..., :num_induc, num_induc:].evaluate() # K_{xz}.
        data_data_covar = full_covar[..., num_induc:, num_induc:] # K_{xx}.

        # Compute interpolation terms
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        L = self._cholesky_factor(induc_induc_covar) K_{zz}^{1/2}
        if L.shape != induc_induc_covar.shape:
            # Aggressive caching can cause nasty shape incompatibilies when evaluating with different batch shapes
            # TODO: Use a hook fo this
            try:
                pop_from_cache_ignore_args(self, "cholesky_factor")
            except CachingError:
                pass
            L = self._cholesky_factor(induc_induc_covar)
        interp_term = L.inv_matmul(induc_data_covar.double()).to(full_inputs.dtype) # K_{zz}^{-1/2} K_{zx}

        # Compute the mean of q(f)
        # The only possible way is that \mu_z contains the K_ZZ^{-1/2}.
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        predictive_mean = (interp_term.transpose(-1, -2) @ inducing_values.unsqueeze(-1)).squeeze(-1) + test_mean

        # Compute the covariance of q(f)
        # K_XX + k_XZ K_ZZ^{-1/2} (S - I) K_ZZ^{-1/2} k_ZX  
        # The only possible way is that S contains the K_ZZ^{-1/2}.
        middle_term = self.prior_distribution.lazy_covariance_matrix.mul(-1) # - I
        if variational_inducing_covar is not None:
            middle_term = SumLazyTensor(variational_inducing_covar, middle_term) # S - I

        if trace_mode.on():
            predictive_covar = (
                data_data_covar.add_jitter(1e-4).evaluate()
                + interp_term.transpose(-1, -2) @ middle_term.evaluate() @ interp_term
            )
        else:
            predictive_covar = SumLazyTensor(
                data_data_covar.add_jitter(1e-4),
                MatmulLazyTensor(interp_term.transpose(-1, -2), middle_term @ interp_term), # K_{xx} +  K_{zx}K_{zz}^{-1/2}(S - I)K_{zz}^{-1/2}K_{zx}
            )

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def __call__(self, x, prior=False, **kwargs):
        # Including the whitening operation. 
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

    def kl_divergence(self):
        # Compute the KL divergence between the variational inducing distribution :math:`q(\mathbf u)`
        # and the prior inducing distribution :math:`p(\mathbf u)`.
    
        :rtype: torch.Tensor
        with settings.max_preconditioner_size(0):
            kl_divergence = torch.distributions.kl.kl_divergence(self.variational_distribution, self.prior_distribution)
        return kl_divergence
"""

"""
class OrthogonallyDecoupledVariationalStrategy(_VariationalStrategy):

    # Implements orthogonally decoupled VGPs as defined in `Salimbeni et al. (2018)`_.
    # This variational strategy uses a different set of inducing points for the mean and covariance functions.
    # The idea is to use more inducing points for the (computationally efficient) mean and fewer inducing points for the
    # (computationally expensive) covaraince.
    # 
    # This variational strategy defines the inducing points/:obj:`~gpytorch.variational._VariationalDistribution`
    # for the mean function.
    # It then wraps a different :obj:`~gpytorch.variational._VariationalStrategy` which
    # defines the covariance inducing points.
    # 
    # :param ~gpytorch.variational._VariationalStrategy covar_variational_strategy:
    #     The variational strategy for the covariance term.
    # :param torch.Tensor inducing_points: Tensor containing a set of inducing
    #     points to use for variational inference.
    # :param ~gpytorch.variational.VariationalDistribution variational_distribution: A
    #     VariationalDistribution object that represents the form of the variational distribution :math:`q(\mathbf u)`
    # 
    # Example:
    #     >>> mean_inducing_points = torch.randn(1000, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
    #     >>> covar_inducing_points = torch.randn(100, train_x.size(-1), dtype=train_x.dtype, device=train_x.device)
    #     >>>
    #     >>> covar_variational_strategy = gpytorch.variational.VariationalStrategy(
    #     >>>     model, covar_inducing_points,
    #     >>>     gpytorch.variational.CholeskyVariationalDistribution(covar_inducing_points.size(-2)),
    #     >>>     learn_inducing_locations=True
    #     >>> )
    #     >>>
    #     >>> variational_strategy = gpytorch.variational.OrthogonallyDecoupledVariationalStrategy(
    #     >>>     covar_variational_strategy, mean_inducing_points,
    #     >>>     gpytorch.variational.DeltaVariationalDistribution(mean_inducing_points.size(-2)),
    #     >>> )
    # 
    # .. _Salimbeni et al. (2018):
    #     https://arxiv.org/abs/1809.08820

    def __init__(self, covar_variational_strategy, inducing_points, variational_distribution):
        if not isinstance(variational_distribution, DeltaVariationalDistribution):
            raise NotImplementedError(
                "OrthogonallyDecoupledVariationalStrategy currently works with DeltaVariationalDistribution"
            )

        super().__init__(
            covar_variational_strategy, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        self.base_variational_strategy = covar_variational_strategy

    @property
    @cached(name="prior_distribution_memo")
    def prior_distribution(self):
        out = self.model(self.inducing_points)
        res = MultivariateNormal(out.mean, out.lazy_covariance_matrix.add_jitter())
        return res

    def forward(self, x, inducing_points, inducing_values, variational_inducing_covar=None, **kwargs):
        # \mathbb E[ f(\mathbf x) ] = \mathbf k_{\mathbf Z_\mu \mathbf x}^\top \mathbf K_{\mathbf Z_\mu \mathbf Z_\mu}^{-1} \mathbf m
        # \text{Var}[ f(\mathbf x) ] = k_{\mathbf x \mathbf x} - \mathbf k_{\mathbf Z_\sigma \mathbf x}^\top
        # \mathbf K_{\mathbf Z_\sigma \mathbf Z_\sigma}^{-1}\left( \mathbf K_{\mathbf Z_\sigma} - 
        # \mathbf S \right)\mathbf K_{\mathbf Z_\sigma \mathbf Z_\sigma}^{-1}\mathbf k_{\mathbf Z_\sigma \mathbf x}

        if variational_inducing_covar is not None:
            raise NotImplementedError(
                "OrthogonallyDecoupledVariationalStrategy currently works with DeltaVariationalDistribution"
            )

        num_data = x.size(-2)
        full_output = self.model(torch.cat([x, inducing_points], dim=-2), **kwargs)
        full_mean = full_output.mean # posterior mean of covariance_variational_strategy
        full_covar = full_output.lazy_covariance_matrix # posterior covariance of covariance_variational_strategy

        if self.training:
            induc_mean = full_mean[..., num_data:] # posterior_mean_u
            induc_induc_covar = full_covar[..., num_data:, num_data:] # posterior_var_u
            prior_dist = MultivariateNormal(induc_mean, induc_induc_covar)
            add_to_cache(self, "prior_distribution_memo", prior_dist)

        test_mean = full_mean[..., :num_data] # posterior_mean_x
        data_induc_covar = full_covar[..., :num_data, num_data:] # posterior_k_{xz}
        predictive_mean = (data_induc_covar @ inducing_values.unsqueeze(-1)).squeeze(-1).add(test_mean) # posterior_k_{xz} * \mu_{mean} + posterior_mean_x
        predictive_covar = full_covar[..., :num_data, :num_data] # posterior_k_{xx}

        # Return the distribution
        return MultivariateNormal(predictive_mean, predictive_covar)

    def kl_divergence(self):
        # kl of covariance variational distribution + kl of the mean variational distribution.
        mean = self.variational_distribution.mean
        induc_induc_covar = self.prior_distribution.lazy_covariance_matrix
        kl = self.model.kl_divergence() + ((induc_induc_covar @ mean.unsqueeze(-1)).squeeze(-1) * mean).sum(-1).mul(0.5)
        return kl
"""


class ODVGPModel(gpytorch.Module):
    def __init__(self, mean_inducing_points, covar_inducing_points, encoder, use_encoder=False):
        super(ODVGPModel, self).__init__()
        self.gp_layer = ODVGPModel(mean_inducing_points, covar_inducing_points, encoder, use_encoder)
        self.encoder = encoder
        self.use_encoder = use_encoder

    def forward(self, x):
        if self.use_encoder:
            x = self.encoder(x)
        res = self.gp_layer(x)
        return res


hidden_dim = 5
kmModel = KMeans(n_clusters=20)
kmModel.fit(train_set.tensors.numpy())
mean_inducing_points = torch.from_numpy(kmModel.cluster_centers_).type(torch.float)

kmModel = KMeans(n_clusters=10)
kmModel.fit(train_set.tensors.numpy())
covar_inducing_points = torch.from_numpy(kmModel.cluster_centers_).type(torch.float)

feature_extractor = Encoder(input_dim=train_set.tensors[0].shape[-1], z_dim=hidden_dim, hidden_dim=16)
model = ODVGPModel(mean_inducing_points, covar_inducing_points, feature_extractor, False)
likelihood = gpytorch.likelihoods.SoftmaxLikelihood(num_features=model.num_dim, num_classes=num_classes)
print(model)

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