import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import numpy as np
from explainer.DGP_XRL import DGPXRL
from explainer.Rudder_XRL import Rudder
from explainer.RnnAttn_XRL import RnnAttn
from explainer.RnnSaliency_XRL import RnnSaliency
from explainer.RationaleNet_XRL import RationaleNet


# Setup env, load the target agent, and collect the trajectories.
env_name = 'Pong-v0'
agent_path = 'agents/{}/'.format(env_name.lower())
traj_path = 'trajs_exp/' + env_name
max_ep_len = 200

# Get the shared parameters, prepare training/testing data.
num_class = 2
seq_len = int(np.load(traj_path + '_max_length.npy'))
input_dim = 80
n_action = 7
len_diff = max_ep_len - seq_len
exp_idx = np.arange(2120)

hiddens = [4]
encoder_type = 'CNN'
rnn_cell_type = 'GRU'
batch_size = 4
save_path = 'exp_model_results/'
likelihood_type = 'classification'
n_stab_samples = 10

# # Explainer 2 - Rudder.
# rudder_explainer = Rudder(seq_len=seq_len, len_diff=len_diff, input_dim=input_dim, hiddens=hiddens,
#                           n_action=n_action, encoder_type=encoder_type)
# name = 'rudder_' + encoder_type + '_' + rnn_cell_type
# rudder_explainer.load(save_path+name+'_model.data')
# rudder_explainer.test(exp_idx, batch_size, traj_path)
# rudder_fid_results = np.load(save_path + name + '_exp.npz')
# rudder_sal = rudder_fid_results['sal']
# rudder_fid = rudder_fid_results['fid']
# rudder_stab = rudder_fid_results['stab']
# rudder_diff = rudder_fid_results['abs_diff']
# rudder_time = rudder_fid_results['time']
#
#
# # Explainer 3 - RNN + Saliency.
# use_input_attention = True
# saliency_explainer = RnnSaliency(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
#                                  encoder_type=encoder_type, num_class=2, rnn_cell_type='LSTM',
#                                  use_input_attention=use_input_attention, normalize=False)
# name = 'saliency_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + str(use_input_attention)
#
# saliency_explainer.load(save_path+name+'_model.data')
# # saliency_explainer.test(exp_idx, batch_size, traj_path)
#
# sal_saliency_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = saliency_explainer.exp_fid_stab(
#     exp_idx[0:100], batch_size, traj_path, True, 'smoothgrad', n_samples=15, n_stab_samples=n_stab_samples)
#
# # Almost all the methods returns zero gradient, except smoothgrad, vargrad, and expgrad on the hidden layer.
# # Those three methods returns nan.
# # Reason: stdev = stdev_spread / (torch.max(cnn_encoded) - torch.min(cnn_encoded)).item() is too large, causing the
# # generated noise are inf and thus resulting nan gradient.
#
# saliency_fid_results = np.load(save_path + name + '_exp_best.npz')
# saliency_sal = saliency_fid_results['sal']
# saliency_fid = saliency_fid_results['fid']
# saliency_stab = saliency_fid_results['stab']
# saliency_diff = saliency_fid_results['abs_diff']
# saliency_time = saliency_fid_results['time']
# saliency_acc = saliency_fid_results['acc']
#
# # Explainer 4 - AttnRNN.
# attention_type = 'tanh'
# name = 'attention_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + attention_type
#
# attention_explainer = RnnAttn(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
#                               encoder_type=encoder_type, num_class=2, attention_type=attention_type,
#                               normalize=False)
#
# attention_explainer.load(save_path+name+'_model.data')
# attention_explainer.test(exp_idx, batch_size, traj_path)
#
# attn_fid_results = np.load(save_path + name + '_exp_best.npz')
# attn_sal = attn_fid_results['sal']
# attn_fid = attn_fid_results['fid']
# attn_stab = attn_fid_results['stab']
# attn_diff = attn_fid_results['abs_diff']
# attn_time = attn_fid_results['time']
# attn_acc = attn_fid_results['acc']
#
#
# # Explainer 5 - RationaleNet.
# name = 'rationale_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type
#
# rationale_explainer = RationaleNet(seq_len, len_diff, input_dim, likelihood_type, hiddens, n_action,
#                                    encoder_type=encoder_type, num_class=2, normalize=False)
#
# rationale_explainer.load(save_path+name+'_model.data')
# rationale_explainer.test(exp_idx, batch_size, traj_path)
#
# rat_fid_results = np.load(save_path + name + '_exp_best.npz')
# rat_sal = rat_fid_results['sal']
# rat_fid = rat_fid_results['fid']
# rat_stab = rat_fid_results['stab']
# rat_diff = rat_fid_results['abs_diff']
# rat_time = rat_fid_results['time']
# rat_acc = rat_fid_results['acc']


# Explainer 6 - DGP.
grid_bound = [(-3, 3)] * hiddens[-1] * 2
likelihood_sample_size = 8

model_1 = 'dgp_classification_GRU_100_False_False_False_False_False_False_model.data'
dgp_explainer = DGPXRL(train_len=30123, seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
                       hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type='adam',
                       n_epoch=2, gamma=0.1, num_inducing_points=100, n_action=n_action,
                       grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
                       mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
                       using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
                       using_OrthogonallyDecouple=False, weight_x=False, lambda_1=0.001)

dgp_explainer.load(save_path+'dgp/'+model_1)
# dgp_explainer.test(exp_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
name = model_1[:-11]
sal_rationale_all, covar_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = dgp_explainer.exp_fid_stab(
    exp_idx, batch_size, traj_path, logit=True, n_stab_samples=n_stab_samples)

np.savez_compressed(save_path + 'dgp/' + name + '_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all,
                    time=mean_time, acc=acc_all, full_covar=covar_all[0], traj_cova=covar_all[1],
                    step_covar=covar_all[2], abs_diff=abs_diff_all)


model_2 = 'dgp_classification_GRU_100_False_False_False_False_False_False_False_0.1_10_8_True_model.data'
dgp_explainer = DGPXRL(train_len=30123, seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
                       hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type='adam',
                       n_epoch=2, gamma=0.1, num_inducing_points=100, n_action=n_action,
                       grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
                       mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
                       using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
                       using_OrthogonallyDecouple=False, weight_x=False, lambda_1=0.1)


dgp_explainer.load(save_path+'dgp/'+model_2)
# dgp_explainer.test(exp_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
name = model_2[:-11]
sal_rationale_all, covar_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = dgp_explainer.exp_fid_stab(
    exp_idx, batch_size, traj_path, logit=True, n_stab_samples=n_stab_samples)

np.savez_compressed(save_path + 'dgp/' + name + '_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all,
                    time=mean_time, acc=acc_all, full_covar=covar_all[0], traj_cova=covar_all[1],
                    step_covar=covar_all[2], abs_diff=abs_diff_all)

model_3 = 'dgp_classification_GRU_100_False_False_False_False_False_False_True_0.001_10_8_True_model.data'
dgp_explainer = DGPXRL(train_len=30123, seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
                       hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type='adam',
                       n_epoch=2, gamma=0.1, num_inducing_points=100, n_action=n_action,
                       grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
                       mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
                       using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
                       using_OrthogonallyDecouple=False, weight_x=True, lambda_1=0.001)


dgp_explainer.load(save_path+'dgp/'+model_3)
# dgp_explainer.test(exp_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
name = model_3[:-11]
sal_rationale_all, covar_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = dgp_explainer.exp_fid_stab(
    exp_idx, batch_size, traj_path, logit=True, n_stab_samples=n_stab_samples)

np.savez_compressed(save_path + 'dgp/' + name + '_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all,
                    time=mean_time, acc=acc_all, full_covar=covar_all[0], traj_cova=covar_all[1],
                    step_covar=covar_all[2], abs_diff=abs_diff_all)
