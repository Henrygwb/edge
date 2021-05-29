import os, sys
from tqdm import tqdm, trange
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import numpy as np
from explainer.quantitative_test import truncate_importance, draw_fid_fig, draw_fid_fig_t, compute_rl_fid

run_rl_fid = False

encoder_type = 'MLP'
rnn_cell_type = 'GRU'
save_path = 'exp_results/'
embed_dim = 3
likelihood_type = 'regression'

# Explainer 1 - Value function.
sal_value = np.load(save_path+'value_exp.npz')['sal']

# Explainer 2 - Rudder.
path = save_path+'rudder_MLP_GRU_3_exp.npz'
rudder_fid_results = np.load(path)
rudder_sal = rudder_fid_results['sal']
rudder_fid = rudder_fid_results['fid']
rudder_stab = rudder_fid_results['stab']

# Explainer 3 - RNN + Saliency.
path = save_path+'saliency_regression_MLP_LSTM_True_3_exp_best.npz'
saliency_fid_results = np.load(path)
saliency_sal = saliency_fid_results['sal']
saliency_fid = saliency_fid_results['fid']
saliency_stab = saliency_fid_results['stab']

# Explainer 4 - AttnRNN.
attention_type = 'tanh'
path = save_path+'attention_regression_MLP_GRU_tanh_3_exp.npz'
attn_fid_results = np.load(path)
attn_sal = attn_fid_results['sal']
attn_fid = attn_fid_results['fid']
attn_stab = attn_fid_results['stab']

# Explainer 5 - RationaleNet.
path = save_path+'rationale_regression_MLP_GRU_3_50_exp.npz'
rat_fid_results = np.load(path)
rat_sal = rat_fid_results['sal']
rat_fid = rat_fid_results['fid']
rat_stab = rat_fid_results['stab']

# # Explainer 6 - DGP.
path = save_path+'dgp_regression_GRU_600_False_False_False_False_False_False_True_0.01_10_16_True_3_exp.npz'
dgp_fid_results = np.load(path)
dgp_sal = dgp_fid_results['sal']
dgp_fid = dgp_fid_results['fid']
dgp_stab = dgp_fid_results['stab']

rudder_fid = np.vstack((rudder_fid, rudder_stab[None, ...]))
saliency_fid = np.vstack((saliency_fid, saliency_stab[None, ...]))
attn_fid = np.vstack((attn_fid, attn_stab[None, ...]))
rat_fid = np.vstack((rat_fid, rat_stab[None, ...]))
dgp_fid = np.vstack((dgp_fid, dgp_stab[None, ...]))

fid_all = np.vstack((rudder_fid[None, ...], saliency_fid[None, ...], attn_fid[None,  ...], rat_fid[None,  ...],
                     dgp_fid[None, ...]))

fid_all = fid_all[:, 1:, :]

explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'EDGE']
metrics_all = ['Top10', 'Top20', 'Top30', 'Stability']
save_fig_path = save_path+'model_fid_stab.pdf'
draw_fid_fig(fid_all, explainer_all, metrics_all, save_fig_path, box_plot=False, log_scale=True)

if run_rl_fid:
    import gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
    from pendulum.utils import rl_fed

    env_name = 'Pendulum-v0'
    max_ep_len = 100
    agent_path = 'models/Pendulum-v0/Pendulum-v0.zip'

    # load ppo2 model
    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them

    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    model = PPO.load(agent_path, custom_objects=custom_objects)
    env = make_vec_env(env_name)

    num_trajs = 4000

    # Baseline fidelity
    diff_all_10 = np.zeros((6, num_trajs))
    diff_all_20 = np.zeros((6, num_trajs))
    diff_all_30 = np.zeros((6, num_trajs))

    importance_len_10 = np.zeros((6, num_trajs))
    importance_len_20 = np.zeros((6, num_trajs))
    importance_len_30 = np.zeros((6, num_trajs))
    finals_all = np.zeros(num_trajs)
    exps_all = [sal_value, rudder_sal, saliency_sal, attn_sal, rat_sal, dgp_sal]

    loader = trange(num_trajs, desc='Loading data')

    for k,importance in enumerate(exps_all):
        print(k)
        for i in trange(num_trajs, desc=f'exp {k}'):
            value = importance[i,0]
            if np.sum(importance[i,:] == value) == importance.shape[1]:
                importance_traj = np.arange(max_ep_len)
                np.random.shuffle(importance_traj)
            else:
                importance_traj = np.argsort(importance[i,])[::-1]
            importance_traj_10 = truncate_importance(importance_traj, 10)
            importance_traj_20 = truncate_importance(importance_traj, 20)
            importance_traj_30 = truncate_importance(importance_traj, 30)
            original_traj = np.load('logs/Pendulum-v0-2021-05-05-14:55:08-4000-episodes/Pendulum-v0_traj_{}.npz'.format(i))
            orin_reward = original_traj['final_rewards']

            if k == 0:
                finals_all[i] = orin_reward
            orin_reward = sum(original_traj['rewards'])
            seed = int(original_traj['seed'])+123456
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

    np.savez(save_path+'fid_rl.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_20=diff_all_20,
             len_10=importance_len_10, len_30=importance_len_30, len_20=importance_len_20, rewards=finals_all)


# Reward diff and explanation len figures

diff_10 = np.load(save_path+'fid_rl.npz')['diff_10']
diff_20 = np.load(save_path+'fid_rl.npz')['diff_20']
diff_30 = np.load(save_path+'fid_rl.npz')['diff_30']
len_10 = np.load(save_path+'fid_rl.npz')['len_10']
len_20 = np.load(save_path+'fid_rl.npz')['len_20']
len_30 = np.load(save_path+'fid_rl.npz')['len_30']

eps = 0.0001
rl_fid_10 = compute_rl_fid(diff_10, len_10, diff_max=diff_10.max(), len_max=100, eps=eps, weight=1)
rl_fid_20 = compute_rl_fid(diff_20, len_20, diff_max=diff_20.max(), len_max=100, eps=eps, weight=1)
rl_fid_30 = compute_rl_fid(diff_30, len_30, diff_max=diff_30.max(), len_max=100, eps=eps, weight=1)

print(np.mean(rl_fid_10, 1))
print(np.std(rl_fid_10, 1))
print(np.mean(rl_fid_20, 1))
print(np.std(rl_fid_20, 1))
print(np.mean(rl_fid_30, 1))
print(np.std(rl_fid_30, 1))

rl_fid_10 = np.vstack((rl_fid_10[1:, :], rl_fid_10[0, :]))
rl_fid_20 = np.vstack((rl_fid_20[1:, :], rl_fid_20[0, :]))
rl_fid_30 = np.vstack((rl_fid_30[1:, :], rl_fid_30[0, :]))

explainer_all = ['Rudder', 'Saliency', 'Attention', 'RatNet', 'EDGE', 'Value']
metrics_all = ['Top10', 'Top20', 'Top30']
rl_fid_all = np.vstack((rl_fid_10[None, ...], rl_fid_20[None, ...], rl_fid_30[None,  ...]))
draw_fid_fig_t(rl_fid_all, explainer_all, metrics_all, save_path+'rl_fid.pdf',
               box_plot=False, log_scale=False)
