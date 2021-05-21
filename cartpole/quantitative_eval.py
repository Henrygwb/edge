import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import gym
import numpy as np
from utils import rl_fed
from explainer.quantitative_test import truncate_importance, draw_fid_fig, draw_stab_fig, draw_fid_fig_t, compute_rl_fid
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env


encoder_type = 'MLP'
rnn_cell_type = 'GRU'
save_path = 'exp_results/'
embed_dim = 4
likelihood_type = 'regression'

# Explainer 1 - Value function.
sal_value = np.load(save_path + 'value_exp.npz')['sal']

# Explainer 2 - Rudder.
name = 'rudder_' + encoder_type + '_' + rnn_cell_type + '_' + str(embed_dim)
rudder_fid_results = np.load(save_path + name + '_exp.npz')
rudder_sal = rudder_fid_results['sal']
rudder_fid = rudder_fid_results['fid']
rudder_stab = rudder_fid_results['stab']

# Explainer 3 - RNN + Saliency.
name = 'saliency_' + likelihood_type + '_' + encoder_type + '_' + 'LSTM' + '_' + str(True) + '_' + str(embed_dim)
saliency_fid_results = np.load(save_path + name + '_exp_best.npz')
saliency_sal = saliency_fid_results['sal']
saliency_fid = saliency_fid_results['fid']
saliency_stab = saliency_fid_results['stab']

# Explainer 4 - AttnRNN.
attention_type = 'tanh'
name = 'attention_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + attention_type + '_' + str(embed_dim)
attn_fid_results = np.load(save_path + name + '_exp.npz')
attn_sal = attn_fid_results['sal']
attn_fid = attn_fid_results['fid']
attn_stab = attn_fid_results['stab']

# Explainer 5 - RationaleNet.
name = 'rationale_' + likelihood_type + '_' + encoder_type + '_' + rnn_cell_type + '_' + str(embed_dim)
rat_fid_results = np.load(save_path + name + '_exp.npz')
rat_sal = rat_fid_results['sal']
rat_fid = rat_fid_results['fid']
rat_stab = rat_fid_results['stab']

# # Explainer 6 - DGP.
dgp_1_fid_results = np.load(save_path + 'dgp/dgp_regression_GRU_100_False_False_False_False_False_False_True_1e-05_10_16_True_4_exp.npz')
dgp_1_sal = dgp_1_fid_results['sal']
dgp_1_fid = dgp_1_fid_results['fid']
dgp_1_stab = dgp_1_fid_results['stab']

dgp_2_fid_results = np.load(save_path + 'dgp/dgp_regression_GRU_100_False_False_False_False_False_False_True_0.01_10_16_True_4_exp.npz')
dgp_2_sal = dgp_2_fid_results['sal']
dgp_2_fid = dgp_2_fid_results['fid']
dgp_2_stab = dgp_2_fid_results['stab']

# Model Fid/Stab figures.
#
# fid_all = np.vstack((rudder_fid[None, ], saliency_fid[None, ], attn_fid[None, ], rat_fid[None, ], dgp_1_fid[None, ],
#                      dgp_2_fid[None, ]))
#
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our', 'Our_x']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25']
# save_fig_path = save_path+'model_fid_bar.pdf'
# draw_fid_fig(fid_all, explainer_all, metrics_all, save_fig_path, box_plot=False, log_scale=False)
#
# stab_all = np.vstack((rudder_stab[None, ...], saliency_stab[None, ...], attn_stab[None,  ...], rat_stab[None,  ...],
#                       dgp_1_stab[None, ...], dgp_2_stab[None, ...]))
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our', 'Our_x']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25']
# save_stab_path = save_path+'model_stab_bar.pdf'
# draw_stab_fig(stab_all, explainer_all, save_stab_path, box_plot=False)
#
# rudder_fid = np.vstack((rudder_fid, rudder_stab[None, ...]))
# saliency_fid = np.vstack((saliency_fid, saliency_stab[None, ...]))
# attn_fid = np.vstack((attn_fid, attn_stab[None, ...]))
# rat_fid = np.vstack((rat_fid, rat_stab[None, ...]))
# dgp_1_fid = np.vstack((dgp_1_fid, dgp_1_stab[None, ...]))
# dgp_2_fid = np.vstack((dgp_2_fid, dgp_2_stab[None, ...]))

# fid_all = np.vstack((rudder_fid[None, ...], saliency_fid[None, ...], attn_fid[None,  ...], rat_fid[None,  ...],
#                      dgp_2_fid[None, ...]))
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25']
# save_fig_path = save_path+'figures_x_l1/model_fid_bar.pdf'
# draw_fid_fig(fid_all, explainer_all, metrics_all, save_fig_path, box_plot=False, log_scale=False)
#
# stab_all = np.vstack((rudder_stab[None, ...], saliency_stab[None, ...], attn_stab[None,  ...], rat_stab[None,  ...],
#                       dgp_2_stab[None, ...]))
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25']
# save_stab_path = save_path+'figures_x_l1/model_stab_bar.pdf'
# draw_stab_fig(stab_all, explainer_all, save_stab_path, box_plot=False)
#
# rudder_fid = np.vstack((rudder_fid, rudder_stab[None, ...]))
# saliency_fid = np.vstack((saliency_fid, saliency_stab[None, ...]))
# attn_fid = np.vstack((attn_fid, attn_stab[None, ...]))
# rat_fid = np.vstack((rat_fid, rat_stab[None, ...]))
# dgp_2_fid = np.vstack((dgp_2_fid, dgp_2_stab[None, ...]))
#
# fid_all = np.vstack((rudder_fid[None, ...], saliency_fid[None, ...], attn_fid[None,  ...], rat_fid[None,  ...],
#                      dgp_2_fid[None, ...]))
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25', 'Stability']
# save_stab_path = save_path+'figures_x_l1/model_fid_stab_bar.pdf'
# draw_fid_fig(fid_all, explainer_all, metrics_all, save_stab_path, box_plot=False, log_scale=False)

# Fid RL.fig

env_name = 'CartPole-v1'
max_ep_len = 200
agent_path = './agents/ppo2_cartpole.zip'

# load ppo2 model
model = PPO2.load(agent_path)
model = model.act_model

num_trajs = 4200
env = make_vec_env(env_name, n_envs=1)


# Baseline fidelity
diff_all_10 = np.zeros((5, num_trajs))
diff_all_20 = np.zeros((5, num_trajs))
diff_all_30 = np.zeros((5, num_trajs))

importance_len_10 = np.zeros((5, num_trajs))
importance_len_20 = np.zeros((5, num_trajs))
importance_len_30 = np.zeros((5, num_trajs))
finals_all = np.zeros(num_trajs)
exps_all = [sal_value, rudder_sal, saliency_sal, attn_sal, rat_sal]
for k in range(5):
    print(k)
    importance = exps_all[k]
    for i in range(num_trajs):
        print(i)
        value = importance[i,0]
        if np.sum(importance[i,:] == value) == importance.shape[1]:
            importance_traj = np.arange(max_ep_len)
            np.random.shuffle(importance_traj)
        else:
            importance_traj = np.argsort(importance[i,])[::-1]
        importance_traj_10 = truncate_importance(importance_traj, 10)
        importance_traj_20 = truncate_importance(importance_traj, 20)
        importance_traj_30 = truncate_importance(importance_traj, 30)
        original_traj = np.load('trajs_exp/CartPole-v1_traj_{}.npz'.format(i))
        orin_reward = original_traj['final_rewards']

        if k == 0:
            finals_all[i] = orin_reward
        orin_reward = int(orin_reward * (200 - 106) + 106)
        seed = int(original_traj['seeds'])
        # rl_fed(env=env, seed=seed, model=model, original_traj=original_traj, max_ep_len=max_ep_len, importance=None,
        #        render=False, mask_act=False)
        replay_reward_10 = rl_fed(env=env, seed=seed, model=model, 
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_10, 
                                  render=False, mask_act=True)
        replay_reward_20 = rl_fed(env=env, seed=seed, model=model,
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_20,
                                  render=False, mask_act=True)
        replay_reward_30 = rl_fed(env=env, seed=seed, model=model,
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_30,
                                  render=False, mask_act=True)

        diff_all_10[k, i] = np.abs(orin_reward-replay_reward_10)
        diff_all_20[k, i] = np.abs(orin_reward-replay_reward_20)
        diff_all_30[k, i] = np.abs(orin_reward-replay_reward_30)
        importance_len_10[k, i] = len(importance_traj_10)
        importance_len_20[k, i] = len(importance_traj_20)
        importance_len_30[k, i] = len(importance_traj_30)

np.savez('fid_baselines.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_20=diff_all_20,
         len_10=importance_len_10, len_30=importance_len_30, len_20=importance_len_20, rewards=finals_all)

print(np.sum(diff_all_10, 1))
print(np.sum(diff_all_20, 1))
print(np.sum(diff_all_30, 1))

# DGP fidelity
diff_all_10 = np.zeros((2, num_trajs))
diff_all_20 = np.zeros((2, num_trajs))
diff_all_30 = np.zeros((2, num_trajs))

importance_len_10 = np.zeros((2, num_trajs))
importance_len_20 = np.zeros((2, num_trajs))
importance_len_30 = np.zeros((2, num_trajs))
finals_all = np.zeros(num_trajs)
exps_all = [dgp_1_sal, dgp_2_sal]
for k in range(2):
    print(k)
    importance = exps_all[k]
    for i in range(num_trajs):
        print(i)
        value = importance[i,0]
        if np.sum(importance[i,:] == value) == importance.shape[1]:
            importance_traj = np.arange(max_ep_len)
            np.random.shuffle(importance_traj)
        else:
            importance_traj = np.argsort(importance[i,])[::-1]
        importance_traj_10 = truncate_importance(importance_traj, 10)
        importance_traj_20 = truncate_importance(importance_traj, 20)
        importance_traj_30 = truncate_importance(importance_traj, 30)
        original_traj = np.load('trajs_exp/CartPole-v1_traj_{}.npz'.format(i))
        orin_reward = original_traj['final_rewards']
        print(orin_reward)
        if k == 0:
            finals_all[i] = orin_reward
        seed = int(original_traj['seeds'])
        orin_reward = int(orin_reward * (200 - 106) + 106)
        # rl_fed(env=env, seed=seed, model=model, original_traj=original_traj, max_ep_len=max_ep_len, importance=None,
        #                                   render=False, mask_act=False)
        replay_reward_10 = rl_fed(env=env, seed=seed, model=model, 
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_10,
                                  render=False, mask_act=True)
        replay_reward_20 = rl_fed(env=env, seed=seed, model=model,
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_20,
                                  render=False, mask_act=True)
        replay_reward_30 = rl_fed(env=env, seed=seed, model=model,
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_30,
                                  render=False, mask_act=True)

        diff_all_10[k, i] = np.abs(orin_reward-replay_reward_10)
        diff_all_20[k, i] = np.abs(orin_reward-replay_reward_20)
        diff_all_30[k, i] = np.abs(orin_reward-replay_reward_30)
        importance_len_10[k, i] = len(importance_traj_10)
        importance_len_20[k, i] = len(importance_traj_20)
        importance_len_30[k, i] = len(importance_traj_30)

np.savez('fid_dgp.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_20=diff_all_20,
         len_10=importance_len_10, len_30=importance_len_30, len_20=importance_len_20, rewards=finals_all)

print(np.sum(diff_all_10, 1))
print(np.sum(diff_all_20, 1))
print(np.sum(diff_all_30, 1))
"""

a1 = np.load('exp_results/fid_baselines.npz')['diff_10']
b1 = np.load('exp_results/fid_dgp.npz')['diff_10']
diff_10 = np.vstack((a1, b1))

a2 = np.load('exp_results/fid_baselines.npz')['diff_30']
b2 = np.load('exp_results/fid_dgp.npz')['diff_30']
diff_30 = np.vstack((a2, b2))

a3 = np.load('exp_results/fid_baselines.npz')['diff_50']
b3 = np.load('exp_results/fid_dgp.npz')['diff_50']
diff_50 = np.vstack((a3, b3))

a1 = np.load('exp_results/fid_baselines.npz')['len_10']
b1 = np.load('exp_results/fid_dgp.npz')['len_10']
len_10 = np.vstack((a1, b1))

a2 = np.load('exp_results/fid_baselines.npz')['len_30']
b2 = np.load('exp_results/fid_dgp.npz')['len_30']
len_30 = np.vstack((a2, b2))

a3 = np.load('exp_results/fid_baselines.npz')['len_50']
b3 = np.load('exp_results/fid_dgp.npz')['len_50']
len_50 = np.vstack((a3, b3))

# Reward diff and explanation len figures
# explainer_all = ['Value', 'Rudder', 'Saliency', 'Attention', 'RatNet', 'Our_1', 'Our_2']
# metrics_all = ['Top5', 'Top15', 'Top25']
#
# diff_all = np.vstack((diff_10[None, ...], diff_30[None, ...], diff_50[None,  ...]))
# draw_fid_fig_t(diff_all, explainer_all, metrics_all, save_path+'rl_fid_diff_bar.pdf', box_plot=False, log_scale=False)
# draw_fid_fig_t(diff_all, explainer_all, metrics_all, save_path+'rl_fid_diff_box.pdf', box_plot=True, log_scale=False)
#
# len_all = np.vstack((len_10[None, ...], len_30[None, ...], len_50[None,  ...]))
# draw_fid_fig_t(len_all, explainer_all, metrics_all, save_path+'rl_fid_len_bar.pdf', box_plot=False, log_scale=False)
# draw_fid_fig_t(len_all, explainer_all, metrics_all, save_path+'rl_fid_len_box.pdf', box_plot=True, log_scale=False)

eps = 0.001
rl_fid_10 = compute_rl_fid(diff_10, len_10, diff_max=192.0, eps=eps)
rl_fid_30 = compute_rl_fid(diff_30, len_30, diff_max=192.0, eps=eps)
rl_fid_50 = compute_rl_fid(diff_50, len_50, diff_max=192.0, eps=eps)

print(np.mean(rl_fid_10, 1))
print(np.std(rl_fid_10, 1))
print(np.mean(rl_fid_30, 1))
print(np.std(rl_fid_30, 1))
print(np.mean(rl_fid_50, 1))
print(np.std(rl_fid_50, 1))


explainer_all = ['Value', 'Rudder', 'Saliency', 'Attention', 'RatNet', 'Our_1', 'Our_2']
metrics_all = ['Top5', 'Top15', 'Top25']
rl_fid_all = np.vstack((rl_fid_10[None, ...], rl_fid_30[None, ...], rl_fid_50[None,  ...]))
draw_fid_fig_t(rl_fid_all, explainer_all, metrics_all, save_path+'rl_fid_bar.pdf', box_plot=False, log_scale=False)

rl_fid_10 = np.vstack((rl_fid_10[0:5], rl_fid_10[6:]))
rl_fid_30 = np.vstack((rl_fid_30[0:5], rl_fid_30[6:]))
rl_fid_50 = np.vstack((rl_fid_50[0:5], rl_fid_50[6:]))
rl_fid_all = np.vstack((rl_fid_10[None, ...], rl_fid_30[None, ...], rl_fid_50[None,  ...]))

explainer_all = ['Value', 'Rudder', 'Saliency', 'Attention', 'RatNet', 'Our']
metrics_all = ['Top5', 'Top15', 'Top25']
draw_fid_fig_t(rl_fid_all, explainer_all, metrics_all, save_path+'figures_x_l1/rl_fid_bar.pdf',
               box_plot=False, log_scale=False)
"""
