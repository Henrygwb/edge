import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import gym
import numpy as np
import gym_compete
from mujoco.utils import rl_fed, load_agent, load_from_file
from explainer.quantitative_test import truncate_importance #, draw_fid_fig, draw_stab_fig, draw_fid_fig_t, compute_rl_fid


encoder_type = 'MLP'
rnn_cell_type = 'GRU'
save_path = 'exp_results/'
likelihood_type = 'classification'

# Explainer 1 - Value function.
sal_value = np.load(save_path + 'value_exp.npz')['sal'][0:2000]

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

# # Explainer 6 - DGP.
dgp_1_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_600_False_False_False_False_False_False_False_1e-05_10_16_True_exp.npz')
dgp_1_sal = dgp_1_fid_results['sal']
dgp_1_fid = dgp_1_fid_results['fid']
dgp_1_stab = dgp_1_fid_results['stab']

dgp_2_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_600_False_False_False_False_False_False_False_0.01_10_16_True_exp.npz')
dgp_2_sal = dgp_2_fid_results['sal']
dgp_2_fid = dgp_2_fid_results['fid']
dgp_2_stab = dgp_2_fid_results['stab']

dgp_3_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_600_False_False_False_False_False_False_True_1e-05_10_16_True_exp.npz')
dgp_3_sal = dgp_3_fid_results['sal']
dgp_3_fid = dgp_3_fid_results['fid']
dgp_3_stab = dgp_3_fid_results['stab']

# Model Fid/Stab figures.

# Model fid box/bar plot.

# fid_all = np.vstack((rudder_fid[None, ], saliency_fid[None, ], attn_fid[None, ], rat_fid[None, ],
#                      dgp_2_fid[None, ], dgp_3_fid[None, ]))
#
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our', 'Our_x']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25']
# save_fig_path = save_path+'figures_weight_x_true_false/model_fid_bar.pdf'
# draw_fid_fig(fid_all, explainer_all, metrics_all, save_fig_path, box_plot=False)
#
# stab_all = np.vstack((rudder_stab[None, ...], saliency_stab[None, ...], attn_stab[None,  ...], rat_stab[None,  ...],
#                       dgp_2_stab[None, ...], dgp_3_stab[None, ...]))
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our', 'Our_x']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25']
# save_stab_path = save_path+'figures_weight_x_true_false/model_stab_bar.pdf'
# draw_stab_fig(stab_all, explainer_all, save_stab_path, box_plot=False)
#
# rudder_fid = np.vstack((rudder_fid, rudder_stab[None, ...]))
# saliency_fid = np.vstack((saliency_fid, saliency_stab[None, ...]))
# attn_fid = np.vstack((attn_fid, attn_stab[None, ...]))
# rat_fid = np.vstack((rat_fid, rat_stab[None, ...]))
# dgp_2_fid = np.vstack((dgp_2_fid, dgp_2_stab[None, ...]))
# dgp_3_fid = np.vstack((dgp_3_fid, dgp_3_stab[None, ...]))
#
# fid_all = np.vstack((rudder_fid[None, ...], saliency_fid[None, ...], attn_fid[None,  ...], rat_fid[None,  ...],
#                      dgp_2_fid[None, ...], dgp_3_fid[None, ...]))
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our', 'Our_x']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25', 'Stability']
# save_stab_path = save_path+'figures_weight_x_true_false/model_fid_stab_bar.pdf'
# draw_fid_fig(fid_all, explainer_all, metrics_all, save_stab_path, box_plot=False)


# fid_all = np.vstack((rudder_fid[None, ...], saliency_fid[None, ...], attn_fid[None,  ...], rat_fid[None,  ...],
#                      dgp_3_fid[None, ...]))
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25']
# save_fig_path = save_path+'figures_best_weight_x_true/model_fid_bar.pdf'
# draw_fid_fig(fid_all, explainer_all, metrics_all, save_fig_path, box_plot=False)
#
# stab_all = np.vstack((rudder_stab[None, ...], saliency_stab[None, ...], attn_stab[None,  ...], rat_stab[None,  ...],
#                       dgp_3_stab[None, ...]))
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25']
# save_stab_path = save_path+'figures_best_weight_x_true/model_stab_bar.pdf'
# draw_stab_fig(stab_all, explainer_all, save_stab_path, box_plot=False)
#
# rudder_fid = np.vstack((rudder_fid, rudder_stab[None, ...]))
# saliency_fid = np.vstack((saliency_fid, saliency_stab[None, ...]))
# attn_fid = np.vstack((attn_fid, attn_stab[None, ...]))
# rat_fid = np.vstack((rat_fid, rat_stab[None, ...]))
# dgp_3_fid = np.vstack((dgp_3_fid, dgp_3_stab[None, ...]))
#
# fid_all = np.vstack((rudder_fid[None, ...], saliency_fid[None, ...], attn_fid[None,  ...], rat_fid[None,  ...],
#                      dgp_3_fid[None, ...]))
# explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'Our']
# metrics_all = ['ZeroOne', 'Top5', 'Top15', 'Top25', 'Stability']
# save_stab_path = save_path+'figures_best_weight_x_true/model_fid_stab_bar.pdf'
# draw_fid_fig(fid_all, explainer_all, metrics_all, save_stab_path, box_plot=False)

# Fid RL.

env_name = 'multicomp/YouShallNotPassHumans-v0'
max_ep_len = 200
agent_path = './agent-zoo/you-shall-not-pass'
model = load_agent(env_name, agent_type=['zoo','zoo'], agent_path=agent_path)
norm_path = agent_path + '/obs_rms.pkl'
obs_rms = load_from_file(norm_path)
num_trajs = 2000
env = gym.make(env_name)

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
        original_traj = np.load('trajs_exp/youshallnotpasshumans_v0_traj_{}.npz'.format(i))
        orin_reward = original_traj['final_rewards']
        print(orin_reward)
        if k == 0:
            finals_all[i] = orin_reward
        seed = int(original_traj['seeds'])

        if orin_reward == 0:
            orin_reward = -1000
        else:
            orin_reward = 1000
        rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                          original_traj=original_traj, max_ep_len=max_ep_len, importance=None,
                                          render=False, mask_act=False)
        replay_reward_10 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_10, 
                                  render=False, mask_act=True)
        replay_reward_20 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_20,
                                  render=False, mask_act=True)
        replay_reward_30 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_30,
                                  render=False, mask_act=True)

        diff_all_10[k, i] = np.abs(orin_reward-replay_reward_10)
        diff_all_20[k, i] = np.abs(orin_reward-replay_reward_20)
        diff_all_30[k, i] = np.abs(orin_reward-replay_reward_30)
        importance_len_10[k, i] = len(importance_traj_10)
        importance_len_20[k, i] = len(importance_traj_20)
        importance_len_30[k, i] = len(importance_traj_30)

np.savez('fid_baselines.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_50=diff_all_20,
         len_10=importance_len_10, len_30=importance_len_30, len_20=importance_len_20, rewards=finals_all)

print(np.sum(diff_all_10, 1))
print(np.sum(diff_all_30, 1))
print(np.sum(diff_all_20, 1))


# DGP fidelity
diff_all_10 = np.zeros((3, num_trajs))
diff_all_20 = np.zeros((3, num_trajs))
diff_all_30 = np.zeros((3, num_trajs))

importance_len_10 = np.zeros((3, num_trajs))
importance_len_20 = np.zeros((3, num_trajs))
importance_len_30 = np.zeros((3, num_trajs))
finals_all = np.zeros(num_trajs)
exps_all = [dgp_1_sal, dgp_2_sal, dgp_3_sal]
for k in range(3):
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
        original_traj = np.load('trajs_exp/youshallnotpasshumans_v0_traj_{}.npz'.format(i))
        orin_reward = original_traj['final_rewards']
        print(orin_reward)
        if k == 0:
            finals_all[i] = orin_reward
        seed = int(original_traj['seeds'])

        if orin_reward == 0:
            orin_reward = -1000
        else:
            orin_reward = 1000
        rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                          original_traj=original_traj, max_ep_len=max_ep_len, importance=None,
                                          render=False, mask_act=False)
        replay_reward_10 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_10,
                                  render=False, mask_act=True)
        replay_reward_20 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_20,
                                  render=False, mask_act=True)
        replay_reward_30 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
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
print(np.sum(diff_all_30, 1))
print(np.sum(diff_all_20, 1))


# a1 = np.load('exp_results/fid_baselines.npz')['diff_10']
# b1 = np.load('exp_results/fid_dgp.npz')['diff_10']
# diff_10 = np.vstack((a1, b1))
#
# a2 = np.load('exp_results/fid_baselines.npz')['diff_30']
# b2 = np.load('exp_results/fid_dgp.npz')['diff_30']
# diff_30 = np.vstack((a2, b2))
#
# a3 = np.load('exp_results/fid_baselines.npz')['diff_50']
# b3 = np.load('exp_results/fid_dgp.npz')['diff_50']
# diff_50 = np.vstack((a3, b3))
#
# a1 = np.load('exp_results/fid_baselines.npz')['len_10']
# b1 = np.load('exp_results/fid_dgp.npz')['len_10']
# len_10 = np.vstack((a1, b1))
#
# a2 = np.load('exp_results/fid_baselines.npz')['len_30']
# b2 = np.load('exp_results/fid_dgp.npz')['len_30']
# len_30 = np.vstack((a2, b2))
#
# a3 = np.load('exp_results/fid_baselines.npz')['len_50']
# b3 = np.load('exp_results/fid_dgp.npz')['len_50']
# len_50 = np.vstack((a3, b3))

# Reward diff and explanation len figures
# explainer_all = ['Value', 'Rudder', 'Saliency', 'Attention', 'RatNet', 'Our_1', 'Our_2', 'Our_3']
# metrics_all = ['Top5', 'Top15', 'Top25']
#
# diff_all = np.vstack((diff_10[None, ...], diff_30[None, ...], diff_50[None,  ...]))
# draw_fid_fig_t(diff_all, explainer_all, metrics_all, save_path+'rl_fid_diff_bar.pdf', box_plot=False, log_scale=False)
# draw_fid_fig_t(diff_all, explainer_all, metrics_all, save_path+'rl_fid_diff_box.pdf', box_plot=True, log_scale=False)
#
# len_all = np.vstack((len_10[None, ...], len_30[None, ...], len_50[None,  ...]))
# draw_fid_fig_t(len_all, explainer_all, metrics_all, save_path+'rl_fid_len_bar.pdf', box_plot=False, log_scale=False)
# draw_fid_fig_t(len_all, explainer_all, metrics_all, save_path+'rl_fid_len_box.pdf', box_plot=True, log_scale=False)

# eps = 0.001
# rl_fid_10 = compute_rl_fid(diff_10, len_10, diff_max=2000.0, eps=eps)
# rl_fid_30 = compute_rl_fid(diff_30, len_30, diff_max=2000.0, eps=eps)
# rl_fid_50 = compute_rl_fid(diff_50, len_50, diff_max=2000.0, eps=eps)
#
# print(np.mean(rl_fid_10, 1))
# print(np.std(rl_fid_10, 1))
# print(np.mean(rl_fid_30, 1))
# print(np.std(rl_fid_30, 1))
# print(np.mean(rl_fid_50, 1))
# print(np.std(rl_fid_50, 1))
#
#
# explainer_all = ['Value', 'Rudder', 'Saliency', 'Attention', 'RatNet', 'Our_1', 'Our_2', 'Our_3']
# metrics_all = ['Top10', 'Top30', 'Top50']
# rl_fid_all = np.vstack((rl_fid_10[None, ...], rl_fid_30[None, ...], rl_fid_50[None,  ...]))
# draw_fid_fig_t(rl_fid_all, explainer_all, metrics_all, save_path+'rl_fid_bar.pdf', box_plot=False, log_scale=False)
# draw_fid_fig_t(rl_fid_all, explainer_all, metrics_all, save_path+'rl_fid_box.pdf', box_plot=True, log_scale=False)
#
# rl_fid_10 = np.vstack((rl_fid_10[0:5], rl_fid_10[6:]))
# rl_fid_30 = np.vstack((rl_fid_30[0:5], rl_fid_30[6:]))
# rl_fid_50 = np.vstack((rl_fid_50[0:5], rl_fid_50[6:]))
# rl_fid_all = np.vstack((rl_fid_10[None, ...], rl_fid_30[None, ...], rl_fid_50[None,  ...]))
#
# explainer_all = ['Value', 'Rudder', 'Saliency', 'Attention', 'RatNet', 'Our', 'Our_x']
# metrics_all = ['Top5', 'Top15', 'Top25']
# draw_fid_fig_t(rl_fid_all, explainer_all, metrics_all, save_path+'figures_weight_x_true_false/rl_fid_bar_our.pdf',
#                box_plot=False, log_scale=False)
#
# explainer_all = ['Value', 'Rudder', 'Saliency', 'Attention', 'RatNet', 'Our']
# metrics_all = ['Top5', 'Top15', 'Top25']
# draw_fid_fig_t(rl_fid_all[:, :-1, ...], explainer_all, metrics_all, save_path+'figures_best_weight_x_true/rl_fid_bar_our.pdf',
#                box_plot=False, log_scale=False)
#


