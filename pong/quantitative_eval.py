import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import gym, torch
import numpy as np
from pong.utils import rl_fed, NNPolicy
from explainer.quantitative_test import truncate_importance, draw_fid_fig, draw_fid_fig_t, compute_rl_fid

encoder_type = 'CNN'
rnn_cell_type = 'GRU'
save_path = 'exp_results/'
likelihood_type = 'classification'
run_rl_fid = False

# Explainer 1 - Value function.
sal_value = np.load(save_path + 'value_exp.npz')['sal']

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
dgp_1_fid_results = np.load(
    save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_False_0.1_10_8_True_exp.npz')
dgp_1_sal = dgp_1_fid_results['sal']
dgp_1_fid = dgp_1_fid_results['fid']
dgp_1_stab = dgp_1_fid_results['stab']

dgp_2_fid_results = np.load(
    save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_True_0.005_10_8_True_exp.npz')
dgp_2_sal = dgp_2_fid_results['sal']
dgp_2_fid = dgp_2_fid_results['fid']
dgp_2_stab = dgp_2_fid_results['stab']

# Model Fid/Stab figures.
rudder_fid = np.vstack((rudder_fid, rudder_stab[None, ...]))
saliency_fid = np.vstack((saliency_fid, saliency_stab[None, ...]))
attn_fid = np.vstack((attn_fid, attn_stab[None, ...]))
rat_fid = np.vstack((rat_fid, rat_stab[None, ...]))
dgp_1_fid = np.vstack((dgp_1_fid, dgp_1_stab[None, ...]))
dgp_2_fid = np.vstack((dgp_2_fid, dgp_2_stab[None, ...]))
fid_all = np.vstack((rudder_fid[None, ...], saliency_fid[None, ...], attn_fid[None,  ...], rat_fid[None,  ...],
                     dgp_1_fid[None, ...], dgp_2_fid[None, ...]))
fid_all = fid_all[:, 1:, :]
explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'EDGE', 'EDGE_x']
metrics_all = ['Top10', 'Top20', 'Top30', 'Stability']
save_stab_path = save_path+'model_fid_stab_bar.pdf'
draw_fid_fig(fid_all, explainer_all, metrics_all, save_path+'model_fid_stab.pdf', box_plot=False)


# Fid RL.
if run_rl_fid:
    env_name = 'Pong-v0'
    max_ep_len = 200
    agent_path = 'agents/{}/'.format(env_name.lower())
    num_trajs = 1880

    env = gym.make(env_name)
    model = NNPolicy(channels=1, num_actions=env.action_space.n)
    _ = model.try_load(agent_path, checkpoint='*.tar')
    torch.manual_seed(1)

    # Baseline fidelity
    diff_all_10 = np.zeros((7, num_trajs))
    diff_all_20 = np.zeros((7, num_trajs))
    diff_all_30 = np.zeros((7, num_trajs))

    importance_len_10 = np.zeros((7, num_trajs))
    importance_len_20 = np.zeros((7, num_trajs))
    importance_len_30 = np.zeros((7, num_trajs))

    exps_all = [sal_value, rudder_sal, saliency_sal, attn_sal, rat_sal, dgp_1_sal, dgp_2_sal]
    for k in range(7):
        print(k)
        importance = exps_all[k]
        for i in range(num_trajs):
            print(i)
            if k == 2:
                # saliency is equivalent to random on this game.
                importance_traj = np.arange(max_ep_len)
                np.random.shuffle(importance_traj)
            else:
                importance_traj = np.argsort(importance[i,])[::-1]
            importance_traj_10 = truncate_importance(importance_traj, 10)
            importance_traj_20 = truncate_importance(importance_traj, 20)
            importance_traj_30 = truncate_importance(importance_traj, 30)
            original_traj = np.load('trajs_exp/Pong-v0_traj_{}.npz'.format(i))
            orin_reward = original_traj['final_rewards']
            print(orin_reward)
            seed = int(original_traj['seed'])
            if orin_reward == 0:
                orin_reward = -1
            # replay_reward_orin = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
            #                             max_ep_len=max_ep_len, importance=None, render=False, mask_act=False)

            replay_reward_10 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
                                      max_ep_len=max_ep_len, importance=importance_traj_10, render=False, mask_act=True)

            replay_reward_20 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
                                      max_ep_len=max_ep_len, importance=importance_traj_20, render=False, mask_act=True)

            replay_reward_30 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
                                      max_ep_len=max_ep_len, importance=importance_traj_30, render=False, mask_act=True)

            diff_all_10[k, i] = np.abs(orin_reward-replay_reward_10)
            diff_all_20[k, i] = np.abs(orin_reward-replay_reward_20)
            diff_all_30[k, i] = np.abs(orin_reward-replay_reward_30)
            importance_len_10[k, i] = len(importance_traj_10)
            importance_len_20[k, i] = len(importance_traj_20)
            importance_len_30[k, i] = len(importance_traj_30)

    np.savez(save_path+'fid_rl.npz', diff_10=diff_all_10, diff_20=diff_all_20, diff_30=diff_all_30,
             len_10=importance_len_10, len_20=importance_len_20, len_30=importance_len_30)

diff_10 = np.load('exp_results/fid_rl.npz')['diff_10']
diff_20 = np.load('exp_results/fid_rl.npz')['diff_20']
diff_30 = np.load('exp_results/fid_rl.npz')['diff_30']

len_10 = np.load('exp_results/fid_rl.npz')['len_10']
len_20 = np.load('exp_results/fid_rl.npz')['len_20']
len_30 = np.load('exp_results/fid_rl.npz')['len_30']

eps = 0.001
rl_fid_10 = compute_rl_fid(diff_10, len_10, diff_max=2, eps=eps)
rl_fid_20 = compute_rl_fid(diff_20, len_20, diff_max=2, eps=eps)
rl_fid_30 = compute_rl_fid(diff_30, len_30, diff_max=2, eps=eps)

print(np.mean(rl_fid_10, 1))
print(np.std(rl_fid_10, 1))
print(np.mean(rl_fid_20, 1))
print(np.std(rl_fid_20, 1))
print(np.mean(rl_fid_30, 1))
print(np.std(rl_fid_30, 1))

# rl_fid_10[-2, ] is dgp_classification_GRU_100_False_False_False_False_False_False_True_0.005_10_8_True_exp.npz.

rl_fid_10 = np.vstack((rl_fid_10[:6, :], rl_fid_10[-1, :]))
rl_fid_20 = np.vstack((rl_fid_20[:6, :], rl_fid_20[-1, :]))
rl_fid_30 = np.vstack((rl_fid_30[:6, :], rl_fid_30[-1, :]))

rl_fid_10 = np.vstack((rl_fid_10[1:, :], rl_fid_10[0, :]))
rl_fid_20 = np.vstack((rl_fid_20[1:, :], rl_fid_20[0, :]))
rl_fid_30 = np.vstack((rl_fid_30[1:, :], rl_fid_30[0, :]))

explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'EDGE', 'EDGE_x', 'Value']
metrics_all = ['Top10', 'Top20', 'Top30']
rl_fid_all = np.vstack((rl_fid_10[None, ...], rl_fid_20[None, ...], rl_fid_30[None,  ...]))
draw_fid_fig_t(rl_fid_all, explainer_all, metrics_all, save_path+'rl_fid_all.pdf',
               box_plot=False, log_scale=False)

rl_fid_10 = rl_fid_10[0:5]
rl_fid_20 = rl_fid_20[0:5]
rl_fid_30 = rl_fid_30[0:5]

explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'EDGE']
metrics_all = ['Top10', 'Top20', 'Top30']
rl_fid_all = np.vstack((rl_fid_10[None, ...], rl_fid_20[None, ...], rl_fid_30[None,  ...]))
draw_fid_fig_t(rl_fid_all, explainer_all, metrics_all, save_path+'rl_fid.pdf',
               box_plot=False, log_scale=False)
