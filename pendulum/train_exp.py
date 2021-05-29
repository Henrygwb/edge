import os, sys
from pathlib import Path

sys.path.append('..')
import numpy as np
import argparse
from utils import get_model
from explainer.DGP_XRL import DGPXRL
from explainer.Rudder_XRL import Rudder
from explainer.RnnAttn_XRL import RnnAttn
from explainer.RnnSaliency_XRL import RnnSaliency
from explainer.RationaleNet_XRL import RationaleNet

parser = argparse.ArgumentParser()
parser.add_argument("--explainer", type=str, required=True, default='dgp')
parser.add_argument("--game", type=str, required=True, default='Pendulum-v0')
parser.add_argument("--train_dir", type=str, required=True, help='Training traj dir')
parser.add_argument("--model_save_dir", type=str, required=True, help='dir to save explanation models')
parser.add_argument("--n_action", type=int, required=True, help='Cardinality of action space, 0 for continuous')
parser.add_argument("--epochs", type=int, required=True, help='num epochs')
parser.add_argument("--eps", type=float, required=False, default=0.01, help='fid eps')

args = parser.parse_args()

# Setup env, load the target agent, and collect the trajectories.
env_name = args.game
# model, env = get_model(args)
test_trajs = [x for x in (Path('logs/Pendulum-v0-2021-05-05-14:55:08-4000-episodes').iterdir()) if 'metadata' not in x.name]
all_trajs = [x for x in (Path(args.train_dir).iterdir()) if 'metadata' not in x.name]
print(f'{len(all_trajs)} total trajs')
num_traj = len(all_trajs)
traj_path = str(Path(args.train_dir).absolute()/f'{env_name}')
exp_traj_path = str(Path('logs/Pendulum-v0-2021-05-05-14:55:08-4000-episodes').absolute()/f'{env_name}')
max_ep_len = len(np.load(all_trajs[0])['actions'])

# Get the shared parameters, prepare training/testing data.
num_class = 1
seq_len = len(np.load(all_trajs[0])['actions'])
n_action = args.n_action
len_diff = max_ep_len - seq_len
total_data_idx = np.arange(len(all_trajs))  # np.arange(30)
train_idx = total_data_idx[0:int(total_data_idx.shape[0] * 0.7), ]
test_idx = total_data_idx[int(total_data_idx.shape[0] * 0.7):, ]
exp_idx = np.arange(len(test_trajs))
print(len(test_trajs))

hiddens = [32, 16, 4]
embed_dim = 3
input_dim = 1+embed_dim
encoder_type = 'MLP'
rnn_cell_type = 'GRU'
n_epoch = args.epochs
batch_size = 40
save_path = args.model_save_dir
likelihood_type = 'regression'
n_stab_samples = 10

if args.explainer == 'rudder':
    # Explainer 2 - Rudder.
    rudder_explainer = Rudder(seq_len=seq_len, len_diff=len_diff, input_dim=input_dim, hiddens=hiddens,
                              n_action=n_action, embed_dim=embed_dim, encoder_type=encoder_type)
    name = 'rudder_' + encoder_type + '_' + rnn_cell_type + '_' + str(embed_dim)
    rudder_explainer.train(train_idx, test_idx, batch_size, n_epoch, traj_path,
                           save_path=save_path + name + '_model.data')
    rudder_explainer.test(test_idx, batch_size, traj_path)

elif args.explainer == 'saliency':
    # Explainer 3 - RNN + Saliency.
    use_input_attention = True
    rnn_cell_type = 'LSTM'
    saliency_explainer = RnnSaliency(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                                     embed_dim=embed_dim, encoder_type=encoder_type, num_class=num_class,
                                     rnn_cell_type=rnn_cell_type, use_input_attention=use_input_attention,
                                     normalize=False)
    name = 'saliency_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type \
           + '_' + str(use_input_attention) + '_' + str(embed_dim)

    saliency_explainer.train(train_idx, test_idx, batch_size, n_epoch, traj_path,
                             save_path=save_path + name + '_model.data')
    saliency_explainer.test(test_idx, batch_size, traj_path)

elif args.explainer == 'attention':
    # Explainer 4 - AttnRNN.
    attention_type = 'tanh'
    name = 'attention_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + attention_type \
           + '_' + str(embed_dim)

    attention_explainer = RnnAttn(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action, embed_dim=embed_dim,
                                  encoder_type=encoder_type, num_class=num_class, attention_type=attention_type,
                                  normalize=False)

    attention_explainer.train(train_idx, test_idx, batch_size, n_epoch, traj_path, save_path=save_path+name+'_model.data')
    attention_explainer.test(test_idx, batch_size, traj_path)

elif args.explainer == 'rationale':
    # Explainer 5 - RationaleNet.
    name = 'rationale_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + str(embed_dim) + '_' + str(n_epoch)

    rationale_explainer = RationaleNet(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                                       embed_dim=embed_dim, encoder_type=encoder_type, num_class=num_class
                                       , normalize=False)

    rationale_explainer.train(train_idx, test_idx, batch_size, n_epoch, traj_path,
                              save_path=save_path + name + '_model.data')
    rationale_explainer.test(test_idx, batch_size, traj_path)

elif args.explainer == 'dgp':
    # Explainer 6 - DGP.
    optimizer = 'adam'
    num_inducing_points = 600
    using_ngd = False  # Whether to use natural gradient descent.
    using_ksi = False  # Whether to use KSI approximation, using this with other options as False.
    using_ciq = False  # Whether to use Contour Integral Quadrature to approximate K_{zz}^{-1/2}, Use it together with NGD.
    using_sor = False  # Whether to use SoR approximation, not applicable for KSI and CIQ.
    using_OrthogonallyDecouple = False  # Using together NGD may cause numerical issue.
    grid_bound = [(-3, 3)] * hiddens[-1] * 2
    weight_x = True
    logit = True
    lambda_1 = 0.01
    local_samples = 10
    likelihood_sample_size = 16

    dgp_explainer = DGPXRL(train_len=train_idx.shape[0], seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
                           hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type=optimizer,
                           n_action=n_action, embed_dim=embed_dim, n_epoch=n_epoch, gamma=0.1,
                           num_inducing_points=num_inducing_points, grid_bounds=grid_bound, encoder_type=encoder_type,
                           inducing_points=None, mean_inducing_points=None, num_class=num_class,
                           rnn_cell_type=rnn_cell_type, using_ngd=using_ngd, using_ksi=using_ksi, using_ciq=using_ciq,
                           using_sor=using_sor, using_OrthogonallyDecouple=using_OrthogonallyDecouple,
                           weight_x=weight_x, lambda_1=lambda_1)

    name = 'dgp_' + likelihood_type + '_' + rnn_cell_type + '_' + \
           str(num_inducing_points) + '_' + str(using_ngd) + '_' + str(using_ngd) + '_' \
           + str(using_ksi) + '_' + str(using_ciq) + '_' + str(using_sor) + '_' \
           + str(using_OrthogonallyDecouple) + '_' + str(weight_x) + '_' + str(lambda_1) + '_' \
           + str(local_samples) + '_' + str(likelihood_sample_size) + '_' + str(logit) + '_' + str(embed_dim)
    dgp_explainer.train(train_idx, test_idx, batch_size, traj_path, local_samples=local_samples,
                        likelihood_sample_size=likelihood_sample_size,
                        save_path=save_path + name + '_model.data')
    dgp_explainer.test(test_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
