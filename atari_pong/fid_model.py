import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import gym
import numpy as np
from atari_pong.utils import run_reproduce


# Setup env, load the target agent, and collect the trajectories.
env_name = 'Pong-v0'
agent_path = 'agents/{}/'.format(env_name.lower())
traj_path = 'trajs_exp/' + env_name
max_ep_len = 200

# Get the shared parameters, prepare training/testing data.
encoder_type = 'CNN'
rnn_cell_type = 'GRU'
save_path = 'exp_model_results/'
likelihood_type = 'classification'

# Explainer 1 - Value function.
sal_value = np.load(save_path + 'value_exp.npz')['sal'][0:2120]


# Explainer 2 - Rudder.
name = 'rudder_' + encoder_type + '_' + rnn_cell_type
rudder_fid_results = np.load(save_path + name + '_exp.npz')
rudder_sal = rudder_fid_results['sal']
rudder_fid = rudder_fid_results['fid']
rudder_stab = rudder_fid_results['stab']

# Explainer 3 - RNN + Saliency.
name = 'saliency_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + str(True)
saliency_fid_results = np.load(save_path + name + '_exp_best.npz')
saliency_sal = saliency_fid_results['sal']
saliency_fid = saliency_fid_results['fid']
saliency_stab = saliency_fid_results['stab']

# Explainer 4 - AttnRNN.
attention_type = 'tanh'
name = 'attention_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + attention_type
attn_fid_results = np.load(save_path + name + '_exp.npz')
attn_sal = attn_fid_results['sal']
attn_fid = attn_fid_results['fid']
attn_stab = attn_fid_results['stab']

# Explainer 5 - RationaleNet.
name = 'rationale_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type
rat_fid_results = np.load(save_path + name + '_exp.npz')
rat_sal = rat_fid_results['sal']
rat_fid = rat_fid_results['fid']
rat_stab = rat_fid_results['stab']

# Explainer 6 - DGP.
dgp_1_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_exp.npz')
dgp_1_sal = dgp_1_fid_results['sal']
dgp_1_fid = dgp_1_fid_results['fid']
dgp_1_stab = dgp_1_fid_results['stab']

dgp_2_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_False_0.1_10_8_True_exp.npz')
dgp_2_sal = dgp_2_fid_results['sal']
dgp_2_fid = dgp_2_fid_results['fid']
dgp_2_stab = dgp_2_fid_results['stab']

dgp_3_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_True_0.001_10_8_True_exp.npz')
dgp_3_sal = dgp_3_fid_results['sal']
dgp_3_fid = dgp_3_fid_results['fid']
dgp_3_stab = dgp_3_fid_results['stab']


# Model Fid/Stab figures.


# RL fid figures.
env = gym.make(env_name)
env.seed(1)
env.env.frameskip = 4
diff_all = []
for i in range(1100):
    print('============')
    print(i)
    original_traj = np.load('trajs_exp/Pong-v0_traj_{}.npz'.format(i))
    reply_reward = run_reproduce(env, original_traj=original_traj, max_ep_len=max_ep_len, render=False)
    orin_reward = original_traj['final_rewards']
    if orin_reward == 0:
        orin_reward = -1
    diff_all.append((reply_reward-orin_reward))
print(np.count_nonzero(diff_all))
print('...........')
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

# Explainer 6 - DGP.
# dgp_1_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_exp.npz')
# dgp_1_sal = dgp_1_fid_results['sal']
# dgp_1_fid = dgp_1_fid_results['fid']
# dgp_1_stab = dgp_1_fid_results['stab']
# dgp_1_diff = dgp_1_fid_results['abs_diff']
# dgp_1_time = dgp_1_fid_results['time']
# dgp_1_acc = dgp_1_fid_results['acc']
#
# print('=============================================')
# print('Mean fid of the zero-one normalization: {}'.format(np.mean(dgp_1_fid[0])))
# print('Std fid of the zero-one normalization: {}'.format(np.std(dgp_1_fid[0])))
# print('Acc fid of the zero-one normalization: {}'.format(dgp_1_acc[0]))
# print('Mean abs pred diff of the zero-one normalization: {}'.format(np.mean(dgp_1_diff[0])))
# print('Std abs pred diff of the zero-one normalization: {}'.format(np.std(dgp_1_diff[0])))
#
# print('=============================================')
# print('Mean fid of the top 5 normalization: {}'.format(np.mean(dgp_1_fid[1])))
# print('Std fid of the top 5 normalization: {}'.format(np.std(dgp_1_fid[1])))
# print('Acc fid of the top 5 normalization: {}'.format(dgp_1_acc[1]))
# print('Mean abs pred diff of the top 5 normalization: {}'.format(np.mean(dgp_1_diff[1])))
# print('Std abs pred diff of the top 5 normalization: {}'.format(np.std(dgp_1_diff[1])))
#
# print('=============================================')
# print('Mean fid of the top 15 normalization: {}'.format(np.mean(dgp_1_fid[2])))
# print('Std fid of the top 15 normalization: {}'.format(np.std(dgp_1_fid[2])))
# print('Acc fid of the top 15 normalization: {}'.format(dgp_1_acc[2]))
# print('Mean abs pred diff of the top 15 normalization: {}'.format(np.mean(dgp_1_diff[2])))
# print('Std abs pred diff of the top 15 normalization: {}'.format(np.std(dgp_1_diff[2])))
#
# print('=============================================')
# print('Mean fid of the top 25 normalization: {}'.format(np.mean(dgp_1_fid[3])))
# print('Std fid of the top 25 normalization: {}'.format(np.std(dgp_1_fid[3])))
# print('Acc fid of the top 25 normalization: {}'.format(dgp_1_acc[3]))
# print('Mean abs pred diff of the top 25 normalization: {}'.format(np.mean(dgp_1_diff[3])))
# print('Std abs pred diff of the top 25 normalization: {}'.format(np.std(dgp_1_diff[3])))
#
# print('=============================================')
# print('Mean stab: {}'.format(np.mean(dgp_1_stab)))
# print('Std stab: {}'.format(np.std(dgp_1_stab)))
#
# print('=============================================')
# print('Mean exp time: {}'.format(dgp_1_time))

# dgp_2_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_False_0.1_10_8_True_exp.npz')
# dgp_2_sal = dgp_2_fid_results['sal']
# dgp_2_fid = dgp_2_fid_results['fid']
# dgp_2_stab = dgp_2_fid_results['stab']
# dgp_2_diff = dgp_2_fid_results['abs_diff']
# dgp_2_time = dgp_2_fid_results['time']
# dgp_2_acc = dgp_2_fid_results['acc']

# print('=============================================')
# print('Mean fid of the zero-one normalization: {}'.format(np.mean(dgp_2_fid[0])))
# print('Std fid of the zero-one normalization: {}'.format(np.std(dgp_2_fid[0])))
# print('Acc fid of the zero-one normalization: {}'.format(dgp_2_acc[0]))
# print('Mean abs pred diff of the zero-one normalization: {}'.format(np.mean(dgp_2_diff[0])))
# print('Std abs pred diff of the zero-one normalization: {}'.format(np.std(dgp_2_diff[0])))
#
# print('=============================================')
# print('Mean fid of the top 5 normalization: {}'.format(np.mean(dgp_2_fid[1])))
# print('Std fid of the top 5 normalization: {}'.format(np.std(dgp_2_fid[1])))
# print('Acc fid of the top 5 normalization: {}'.format(dgp_2_acc[1]))
# print('Mean abs pred diff of the top 5 normalization: {}'.format(np.mean(dgp_2_diff[1])))
# print('Std abs pred diff of the top 5 normalization: {}'.format(np.std(dgp_2_diff[1])))
#
# print('=============================================')
# print('Mean fid of the top 15 normalization: {}'.format(np.mean(dgp_2_fid[2])))
# print('Std fid of the top 15 normalization: {}'.format(np.std(dgp_2_fid[2])))
# print('Acc fid of the top 15 normalization: {}'.format(dgp_2_acc[2]))
# print('Mean abs pred diff of the top 15 normalization: {}'.format(np.mean(dgp_2_diff[2])))
# print('Std abs pred diff of the top 15 normalization: {}'.format(np.std(dgp_2_diff[2])))
#
# print('=============================================')
# print('Mean fid of the top 25 normalization: {}'.format(np.mean(dgp_2_fid[3])))
# print('Std fid of the top 25 normalization: {}'.format(np.std(dgp_2_fid[3])))
# print('Acc fid of the top 25 normalization: {}'.format(dgp_2_acc[3]))
# print('Mean abs pred diff of the top 25 normalization: {}'.format(np.mean(dgp_2_diff[3])))
# print('Std abs pred diff of the top 25 normalization: {}'.format(np.std(dgp_2_diff[3])))
#
# print('=============================================')
# print('Mean stab: {}'.format(np.mean(dgp_2_stab)))
# print('Std stab: {}'.format(np.std(dgp_2_stab)))
#
# print('=============================================')
# print('Mean exp time: {}'.format(dgp_2_time))


# dgp_3_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_True_0.001_10_8_True_exp.npz')
# dgp_3_sal = dgp_3_fid_results['sal']
# dgp_3_fid = dgp_3_fid_results['fid']
# dgp_3_stab = dgp_3_fid_results['stab']
# dgp_3_diff = dgp_3_fid_results['abs_diff']
# dgp_3_time = dgp_3_fid_results['time']
# dgp_3_acc = dgp_3_fid_results['acc']
#
# print('=============================================')
# print('Mean fid of the zero-one normalization: {}'.format(np.mean(dgp_3_fid[0])))
# print('Std fid of the zero-one normalization: {}'.format(np.std(dgp_3_fid[0])))
# print('Acc fid of the zero-one normalization: {}'.format(dgp_3_acc[0]))
# print('Mean abs pred diff of the zero-one normalization: {}'.format(np.mean(dgp_3_diff[0])))
# print('Std abs pred diff of the zero-one normalization: {}'.format(np.std(dgp_3_diff[0])))
#
# print('=============================================')
# print('Mean fid of the top 5 normalization: {}'.format(np.mean(dgp_3_fid[1])))
# print('Std fid of the top 5 normalization: {}'.format(np.std(dgp_3_fid[1])))
# print('Acc fid of the top 5 normalization: {}'.format(dgp_3_acc[1]))
# print('Mean abs pred diff of the top 5 normalization: {}'.format(np.mean(dgp_3_diff[1])))
# print('Std abs pred diff of the top 5 normalization: {}'.format(np.std(dgp_3_diff[1])))
#
# print('=============================================')
# print('Mean fid of the top 15 normalization: {}'.format(np.mean(dgp_3_fid[2])))
# print('Std fid of the top 15 normalization: {}'.format(np.std(dgp_3_fid[2])))
# print('Acc fid of the top 15 normalization: {}'.format(dgp_3_acc[2]))
# print('Mean abs pred diff of the top 15 normalization: {}'.format(np.mean(dgp_3_diff[2])))
# print('Std abs pred diff of the top 15 normalization: {}'.format(np.std(dgp_3_diff[2])))
#
# print('=============================================')
# print('Mean fid of the top 25 normalization: {}'.format(np.mean(dgp_3_fid[3])))
# print('Std fid of the top 25 normalization: {}'.format(np.std(dgp_3_fid[3])))
# print('Acc fid of the top 25 normalization: {}'.format(dgp_3_acc[3]))
# print('Mean abs pred diff of the top 25 normalization: {}'.format(np.mean(dgp_3_diff[3])))
# print('Std abs pred diff of the top 25 normalization: {}'.format(np.std(dgp_3_diff[3])))
#
# print('=============================================')
# print('Mean stab: {}'.format(np.mean(dgp_3_stab)))
# print('Std stab: {}'.format(np.std(dgp_3_stab)))
#
# print('=============================================')
# print('Mean exp time: {}'.format(dgp_3_time))
#

# grid_bound = [(-3, 3)] * hiddens[-1] * 2
# likelihood_sample_size = 8
#
# model_1 = 'dgp_classification_GRU_100_False_False_False_False_False_False_model.data'
# dgp_explainer = DGPXRL(train_len=30123, seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
#                        hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type='adam',
#                        n_epoch=2, gamma=0.1, num_inducing_points=100, n_action=n_action,
#                        grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
#                        mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
#                        using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
#                        using_OrthogonallyDecouple=False, weight_x=False, lambda_1=0.001)
#
# dgp_explainer.load(save_path+'dgp/'+model_1)
# # dgp_explainer.test(exp_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
# name = model_1[:-11]
# sal_rationale_all, covar_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = dgp_explainer.exp_fid_stab(
#     exp_idx, batch_size, traj_path, logit=True, n_stab_samples=n_stab_samples)
#
# np.savez_compressed(save_path + 'dgp/' + name + '_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all,
#                     time=mean_time, acc=acc_all, full_covar=covar_all[0], traj_cova=covar_all[1],
#                     step_covar=covar_all[2], abs_diff=abs_diff_all)
#
#
# model_2 = 'dgp_classification_GRU_100_False_False_False_False_False_False_False_0.1_10_8_True_model.data'
# dgp_explainer = DGPXRL(train_len=30123, seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
#                        hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type='adam',
#                        n_epoch=2, gamma=0.1, num_inducing_points=100, n_action=n_action,
#                        grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
#                        mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
#                        using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
#                        using_OrthogonallyDecouple=False, weight_x=False, lambda_1=0.1)
#
#
# dgp_explainer.load(save_path+'dgp/'+model_2)
# # dgp_explainer.test(exp_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
# name = model_2[:-11]
# sal_rationale_all, covar_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = dgp_explainer.exp_fid_stab(
#     exp_idx, batch_size, traj_path, logit=True, n_stab_samples=n_stab_samples)
#
# np.savez_compressed(save_path + 'dgp/' + name + '_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all,
#                     time=mean_time, acc=acc_all, full_covar=covar_all[0], traj_cova=covar_all[1],
#                     step_covar=covar_all[2], abs_diff=abs_diff_all)
#
# model_3 = 'dgp_classification_GRU_100_False_False_False_False_False_False_True_0.001_10_8_True_model.data'
# dgp_explainer = DGPXRL(train_len=30123, seq_len=seq_len, len_diff=len_diff, input_dim=input_dim,
#                        hiddens=hiddens, likelihood_type=likelihood_type, lr=0.01, optimizer_type='adam',
#                        n_epoch=2, gamma=0.1, num_inducing_points=100, n_action=n_action,
#                        grid_bounds=grid_bound, encoder_type=encoder_type, inducing_points=None,
#                        mean_inducing_points=None, num_class=num_class, rnn_cell_type=rnn_cell_type,
#                        using_ngd=False, using_ksi=False, using_ciq=False, using_sor=False,
#                        using_OrthogonallyDecouple=False, weight_x=True, lambda_1=0.001)
#
#
# dgp_explainer.load(save_path+'dgp/'+model_3)
# # dgp_explainer.test(exp_idx, batch_size, traj_path, likelihood_sample_size=likelihood_sample_size)
# name = model_3[:-11]
# sal_rationale_all, covar_all, fid_all, stab_all, acc_all, abs_diff_all, mean_time = dgp_explainer.exp_fid_stab(
#     exp_idx, batch_size, traj_path, logit=True, n_stab_samples=n_stab_samples)
#
# np.savez_compressed(save_path + 'dgp/' + name + '_exp.npz', sal=sal_rationale_all, fid=fid_all, stab=stab_all,
#                     time=mean_time, acc=acc_all, full_covar=covar_all[0], traj_cova=covar_all[1],
#                     step_covar=covar_all[2], abs_diff=abs_diff_all)
