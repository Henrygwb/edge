# Implementation of our RNN-DGP classification/regression model .
# 1. (Capturing the sequential dependency) Use a RNN (seq2seq) inside the RBF kernel function.
# 2. (Non-Gaussian likelihood) Use the Variational inference with the variational distribution as q(u)~N(\mu, LL^T).
# 3. Additive Gaussian process with two kernels - data (trajectory/individual) level and feature level.
# 4. Standard variational strategy with whitening and SoR.
#    (Faster way of computing p(f|u)) Use the inducing point technique with the SoR approximation of
#    p(f|u)=K_{x,z}K_{z,z}u and p(f_*|u)=K_{x_*,z}K_{z,z}u.
# 5. Or Grid variational strategy with KSI and SoR.
# (Faster way of computing K_{x,z} in p(f|u)) KISS-GP: Use the local kernel interpolation technique
# (Structured kernel interpolation) to approximate K_{x,z}K_{z,z} with the interpolation matrix M.

import math
import torch
import tqdm
import gpytorch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from gp_utils import CustomizedGaussianLikelihood, CustomizedSoftmaxLikelihood, CustomizedVariationalStrategy

# Hyper-parameters
HIDDENS = []
NUM_INDUCING_POINTS = 10
USING_SOR = False # Whether to use SoR approximation.
USING_KSI = False # Whether to use KSI approximation.
GRID_BOUNDS = [(-10, 10)]
LIKELIHOOD = 'Classification'
RNN_CELL_TYPE = 'GRU'
NUM_CLASSES = 2
N_EPOCHS = 5
N_BATCHS = 4
LR = 0.1
GAMMA = 0.1 # LR decay multiplicative factor.
SAVE_PATH = None


# load the dataset.
train_set = torch.utils.data.TensorDataset(torch.randn(20, 10, 20), torch.rand(8).round().long()) # torch.round(): round to the closest int.
test_set = torch.utils.data.TensorDataset(torch.randn(8, 10, 20), torch.rand(4).round().long()) # torch.long(): change the data type to int64.
train_loader = torch.utils.data.DataLoader(train_set, batch_size=N_BATCHS, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=N_BATCHS, shuffle=False)
train_x = train_set.tensors[0]
train_y = train_set.tensors[1]


# Build the RNN model.
class RNNEncoder(nn.Module):
    def __init__(self, seq_len, input_dim, hiddens, dropout_rate=0.25, rnn_cell_type='GRU', normalize=False):
        """
        RNN structure (MLP+seq2seq) (\theta_1: RNN parameters).
        :param seq_len: trajectory length.
        :param input_dim: the dimensionality of the
        :param hiddens: hidden layer dimensions.
        :param dropout_rate: dropout rate.
        :param rnn_cell_type: rnn layer type ('GRU' or 'LSTM').
        :param normalize: whether to normalize the inputs.
        """
        super(RNNEncoder, self).__init__()
        self.normalize = normalize
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = hiddens[-1]
        self.rnn_cell_type = rnn_cell_type
        self.mlp_encoder = nn.Sequential()
        for i in range(len(hiddens)-1):
            if i == 0:
                self.mlp_encoder.add_module('mlp_%d' % i, nn.Linear(input_dim, hiddens[i]))
            else:
                self.mlp_encoder.add_module('mlp_%d' % i, nn.Linear(hiddens[i-1], hiddens[i]))
            self.mlp_encoder.add_module('relu_%d' % i, nn.ReLU())
            self.mlp_encoder.add_module('dropout_%d' % i, nn.Dropout(dropout_rate))

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

    def forward(self, x, h0=None, c0=None):
        # forward function: given an input, return the model output (output at each time and the final time step).
        """
        :param x: input trajectories (Batch_size, seq_len, input_dim).
        :param h0: Initial hidden state at time t_0 (Batch_size, 1, hidden_dim).
        :param c0: Initial cell state at time t_0 (Batch_size, 1, hidden_dim).
        :return step_embed: the latent representation of each time step (batch_size, seq_len, hidden_dim).
        :return traj_embed: the latend representation of each trajectory (batch_size, hidden_dim).
        """
        if self.normalize:
            mean = np.mean(x, axis=(0, 1))[None, None, :]
            std = np.std(x, axis=(0, 1))[None, None, :]
            x = (x - mean)/std
        mlp_encoded = self.mlp_encoder(x)
        if self.rnn_cell_type == 'GRU':
            step_embed, traj_embed = self.rnn(mlp_encoded, h0)
        else:
            step_embed, traj_embed, _ = self.rnn(mlp_encoded, h0, c0)
        traj_embed = torch.squeeze(traj_embed, 1) # (batch_size, 1, hidden_dim) -> (batch_size, hidden_dim)
        traj_embed = self.traj_embed_layer(traj_embed) # (batch_size, hidden_dim)
        return step_embed, traj_embed


# Build the GP layer.
class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, input_dim_step, input_dim_traj, inducing_points):
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
        """
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0))
        if USING_KSI:
            variational_strategy = gpytorch.variational.GridInterpolationVariationalStrategy(
                self, NUM_INDUCING_POINTS, GRID_BOUNDS, variational_distribution)
        else:
            variational_strategy = CustomizedVariationalStrategy(self, inducing_points, variational_distribution,
                                                                 learning_inducing_locations=True,
                                                                 using_sor=USING_SOR)
        super(GaussianProcessLayer, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean()
        self.step_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim_step, active_dims=tuple(range(input_dim_step)),
                                       lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                                           math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp)))
        self.traj_kernel = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=input_dim_traj,
                                       active_dims=tuple(range(input_dim_step, input_dim_traj)),
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
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)

    # __call__: compute the conditional/marginal posterior.


# Build the full model.
class DGPXAIModel(gpytorch.Module):
    def __init__(self, seq_len, input_dim, hiddens, num_inducing_points, inducing_points=None, dropout_rate=0.25, rnn_cell_type='GRU', normalize=False):
        """
        Define the full model.
        :param seq_len: trajectory length.
        :param input_dim: input state/action dimension.
        :param hiddens: hidden layer dimentions.
        :param num_inducing_points: number of inducing points.
        :param inducing_points: inducing points at the latent space Z (num_inducing_points, 2*hiddens[-1]).
        :param dropout_rate: MLP dropout rate.
        :param rnn_cell_type: the RNN cell type.
        :param normalize: whether to normalize the input.
        """
        super().__init__()
        self.seq_len = seq_len
        self.encoder = RNNEncoder(seq_len, input_dim, hiddens, dropout_rate, rnn_cell_type, normalize)
        if inducing_points is None:
            inducing_points = torch.randn(num_inducing_points, 2*hiddens[-1])
        self.gp_layer = GaussianProcessLayer(hiddens[-1], hiddens[-1], inducing_points)

    def forward(self, x):
        """
        Compute the marginal posterior q(f) ~ N(\mu_f, \sigma_f), \mu_f (N*T, 1), \sigma_f(N*T, N*T).
        Later, when computing the marginal loglikelihood, we sample multiple set of data from the marginal loglikelihood.
        :param x: input data x (N, T, P).
        :return: q(gy_layer(Encoder(x))).
        """
        step_embedding, traj_embedding = self.feature_extractor(x)  # (N, T, P) -> (N, T, D), (N, D).
        traj_embedding = traj_embedding[:, None, :].repeat(1, self.seq_len, 1) # (N, D) -> (N, T, D)
        features = torch.cat([step_embedding, traj_embedding], dim=-1) # (N, T, 2D)
        features = features.reshape(x.size(0)*x.size(1), features.size(-1))
        res = self.gp_layer(features)
        return res


# Build the likelihood layer (Regression and classification).
if LIKELIHOOD == 'regression':
    print('Conduct regression and use GaussianLikelihood')
    likelihood = CustomizedGaussianLikelihood()
elif LIKELIHOOD == 'classification':
    print('Conduct classification and use softmaxLikelihood')
    likelihood = CustomizedSoftmaxLikelihood(num_features=train_x.size(1), num_classes=NUM_CLASSES)
else:
    print('Default choice is regression and use GaussianLikelihood')
    likelihood = CustomizedGaussianLikelihood()

# Compute the loss (ELBO) likelihood + KL divergence.
model = DGPXAIModel(seq_len=train_x.size(1), input_dim=train_x.size(2), hiddens=HIDDENS,
                    num_inducing_points=NUM_INDUCING_POINTS)
print(model)
# First, sampling from q(f) with shape [n_sample, n_data].
# Then, the likelihood function times it with the mixing weight and get the marginal likelihood distribution p(y|f).
# VariationalELBO will call _ApproximateMarginalLogLikelihood, which will then compute the marginal likelihood by
# calling the likelihood function (the expected_log_prob in the likelihood class)
# and the KL divergence (VariationalStrategy.kl_divergence()).
# ELBO = E_{q(f)}[p(y|f)] - KL[q(u)||p(u)].
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

if torch.cuda.is_available():
    model = model.cuda()
    likelihood = likelihood.cuda()

# Define the optimizer over the parameters
# (RNN parameters, RBF kernel parameters, Z, variational parameters, mixing weight).
optimizer = optim.Adam([
    {'params': model.encoder.parameters(), 'weight_decay': 1e-4},
    {'params': model.gp_layer.hyperparameters(), 'lr': LR * 0.01},
    {'params': model.gp_layer.variational_parameters()},
    {'params': likelihood.parameters()}, ], lr=LR, weight_decay=0)

# Learning rate decay schedule.
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[0.5 * N_EPOCHS, 0.75 * N_EPOCHS], gamma=GAMMA)


# train function.
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


# test function.
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


if __name__ == '__main__':
    for epoch in range(1, N_EPOCHS + 1):
        with gpytorch.settings.use_toeplitz(False):
            train(epoch)
            test()
        scheduler.step()
        state_dict = model.state_dict()
        likelihood_state_dict = likelihood.state_dict()
        torch.save({'model': state_dict, 'likelihood': likelihood_state_dict}, SAVE_PATH)
