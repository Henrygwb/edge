import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import numpy as np
import argparse
from explainer.DGP_XRL import DGPXRL
from explainer.Rudder_XRL import Rudder
from explainer.RnnAttn_XRL import RnnAttn
from explainer.RnnSaliency_XRL import RnnSaliency
from explainer.RationaleNet_XRL import RationaleNet

parser = argparse.ArgumentParser()
parser.add_argument("--explainer", type=str, default='dgp')
args = parser.parse_args()

# Setup env, load the target agent, and collect the trajectories.
env_name = 'Pong-v0'
agent_path = 'agents/{}/'.format(env_name.lower())
traj_path = 'trajs/' + env_name
max_ep_len = 200

# Get the shared parameters, prepare training/testing data.
num_class = 2
seq_len = int(np.load(traj_path + '_max_length.npy'))
input_dim = 80
n_action = 7
len_diff = max_ep_len - seq_len
total_data_idx = np.arange(int(np.load(traj_path + '_num_traj.npy')))
train_idx = total_data_idx[0:int(total_data_idx.shape[0]*0.5), ]
test_idx = total_data_idx[int(total_data_idx.shape[0]*0.7):, ]

hiddens = [4]
encoder_type = 'CNN'
rnn_cell_type = 'GRU'
n_epoch = 100
batch_size = 40
save_path = 'models/'
likelihood_type = 'classification'
n_stab_samples = 10

if args.explainer == 'rudder':
    # Explainer 2 - Rudder.
    rudder_explainer = Rudder(seq_len=seq_len, len_diff=len_diff, input_dim=input_dim, hiddens=hiddens,
                              n_action=n_action, encoder_type=encoder_type)
    name = 'rudder_' + encoder_type + '_' + rnn_cell_type
    rudder_explainer.train(train_idx, test_idx, batch_size, n_epoch, traj_path, save_path=save_path+name+'_model.data')
    rudder_explainer.test(test_idx, batch_size, traj_path)
    rudder_explainer.load(save_path+name+'_model.data')
    rudder_explainer.test(test_idx, batch_size, traj_path)

elif args.explainer == 'saliency':
    # Explainer 3 - RNN + Saliency.
    use_input_attention = True
    saliency_explainer = RnnSaliency(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                                     encoder_type=encoder_type, num_class=2, rnn_cell_type='LSTM',
                                     use_input_attention=use_input_attention, normalize=False)
    name = 'saliency_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + str(use_input_attention)

    saliency_explainer.train(train_idx, test_idx, batch_size, n_epoch, traj_path, save_path=save_path+name+'_model.data')

elif args.explainer == 'attention':
    # Explainer 4 - AttnRNN.
    attention_type = 'tanh'
    name = 'attention_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + attention_type

    attention_explainer = RnnAttn(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                                  encoder_type=encoder_type, num_class=2, attention_type=attention_type,
                                  normalize=False)

    attention_explainer.train(train_idx, test_idx, batch_size, n_epoch, traj_path, save_path=save_path+name+'_model.data')
    attention_explainer.test(test_idx, batch_size, traj_path)

elif args.explainer == 'rationale':
    # Explainer 5 - RationaleNet.
    name = 'rationale_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type

    rationale_explainer = RationaleNet(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
                                       encoder_type=encoder_type, num_class=2, normalize=False)

    rationale_explainer.train(train_idx, test_idx, batch_size, n_epoch, traj_path, save_path=save_path+name+'_model.data')
    rationale_explainer.test(test_idx, batch_size, traj_path)

elif args.explainer == 'dgp':
    # Explainer 6 - DGP.
    save_path = 'models/dgp/'
    optimizer = 'adam'
    num_inducing_points = 100
    using_ngd = False # Whether to use natural gradient descent.
    using_ksi = False # Whether to use KSI approximation, using this with other options as False.
    using_ciq = False # Whether to use Contour Integral Quadrature to approximate K_{zz}^{-1/2}, Use it together with NGD.
    using_sor = False # Whether to use SoR approximation, not applicable for KSI and CIQ.
    using_OrthogonallyDecouple = False # Using together NGD may cause numerical issue.
    grid_bound = [(-3, 3)] * hiddens[-1] * 2
    weight_x = False # True
    logit = True
    lambda_1 = 0.1 # 0.005
    local_samples = 10
    likelihood_sample_size = 8

    dgp_explainer = DGPXRL(train_len=train_idx.shape[0], seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
                           hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type=optimizer,
                           n_epoch=n_epoch, gamma=0.1, num_inducing_points=num_inducing_points, n_action=n_action,
                           grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
                           mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
                           using_ngd=using_ngd, using_ksi=using_ksi, using_ciq=using_ciq, using_sor=using_sor,
                           using_OrthogonallyDecouple=using_OrthogonallyDecouple, weight_x=weight_x, lambda_1=lambda_1)

    name = 'dgp_' + likelihood_type + '_' + rnn_cell_type + '_' + \
           str(num_inducing_points)+'_'+ str(using_ngd) + '_' + str(using_ngd) + '_' \
           + str(using_ksi) + '_' + str(using_ciq) + '_' + str(using_sor) + '_' \
           + str(using_OrthogonallyDecouple) + '_' + str(weight_x) + '_' + str(lambda_1) + '_' \
           + str(local_samples) + '_' + str(likelihood_sample_size) + '_' + str(logit)

    dgp_explainer.train(train_idx, test_idx, batch_size, traj_path, local_samples=local_samples,
                        likelihood_sample_size=likelihood_sample_size,
                        save_path=save_path+name+'_model.data')

    dgp_explainer.test(test_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
