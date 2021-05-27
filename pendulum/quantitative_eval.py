import os, sys
from tqdm import tqdm, trange
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import gym
import numpy as np

from explainer.quantitative_test import truncate_importance, draw_fid_fig, draw_stab_fig, draw_fid_fig_t, compute_rl_fid
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

def rl_fed(env, seed, model, original_traj, max_ep_len=1e3, importance=None, render=False, mask_act=False):

    acts_orin = original_traj['actions']
    traj_len = np.count_nonzero(acts_orin)
    start_step = max_ep_len - traj_len

    env.seed(seed)
    env.action_space.seed(seed)

    episode_length, epr, done = 0, 0, False  # bookkeeping
    observation = env.reset()

    state = None
    for i in range(traj_len):
        act, state = model.predict(observation, state=state, deterministic=True)
        if mask_act:
            if start_step+i in importance:
                act = np.random.uniform(-2,2,(1,1))

        observation, reward, done, infos = env.step(act)
        if render: env.render()
        epr += reward[0]
        # save info!
        episode_length += 1
        if done:
            assert reward[0] != 0
            break

    # print('step # {}, reward {:.0f}.'.format(episode_length, epr))
    return epr

encoder_type = 'MLP'
rnn_cell_type = 'GRU'
save_path = 'models/explainer/'
embed_dim = 3
likelihood_type = 'regression'

# Explainer 1 - Value function.
sal_value = np.load('models/explainer/value_exp.npz')['sal']

# Explainer 2 - Rudder.
path = 'models/explainer/rudder_MLP_GRU_3_exp.npz'
rudder_fid_results = np.load(path)
rudder_sal = rudder_fid_results['sal']
rudder_fid = rudder_fid_results['fid']
rudder_stab = rudder_fid_results['stab']

# Explainer 3 - RNN + Saliency.
path = 'models/explainer/saliency_regression_MLP_LSTM_True_3_exp_best.npz'
saliency_fid_results = np.load(path)
saliency_sal = saliency_fid_results['sal']
saliency_fid = saliency_fid_results['fid']
saliency_stab = saliency_fid_results['stab']

# Explainer 4 - AttnRNN.
attention_type = 'tanh'
path = 'models/explainer/attention_regression_MLP_GRU_tanh_3_exp.npz'
attn_fid_results = np.load(path)
attn_sal = attn_fid_results['sal']
attn_fid = attn_fid_results['fid']
attn_stab = attn_fid_results['stab']

# Explainer 5 - RationaleNet.
path = 'models/explainer/rationale_regression_MLP_GRU_3_50_exp.npz'
rat_fid_results = np.load(path)
rat_sal = rat_fid_results['sal']
rat_fid = rat_fid_results['fid']
rat_stab = rat_fid_results['stab']

# # Explainer 6 - DGP.
path = 'models/explainer/dgp_models/dgp_regression_GRU_600_False_False_False_False_False_False_True_1e-05_10_16_True_3_exp.npz'
dgp_fid_results = np.load(path)
dgp_sal = dgp_fid_results['sal']
dgp_fid = dgp_fid_results['fid']
dgp_stab = dgp_fid_results['stab']

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
# trajs_all = [ for i in loader]

# for k,importance in enumerate(exps_all):
#     print(k)
#     for i in trange(num_trajs, desc=f'exp {k}'):
#         value = importance[i,0]
#         if np.sum(importance[i,:] == value) == importance.shape[1]:
#             importance_traj = np.arange(max_ep_len)
#             np.random.shuffle(importance_traj)
#         else:
#             importance_traj = np.argsort(importance[i,])[::-1]
#         importance_traj_10 = truncate_importance(importance_traj, 10)
#         importance_traj_20 = truncate_importance(importance_traj, 20)
#         importance_traj_30 = truncate_importance(importance_traj, 30)
#         original_traj = np.load('logs/Pendulum-v0-2021-05-05-14:55:08-4000-episodes/Pendulum-v0_traj_{}.npz'.format(i))
#         orin_reward = original_traj['final_rewards']

#         if k == 0:
#             finals_all[i] = orin_reward
#         orin_reward = sum(original_traj['rewards'])
#         seed = int(original_traj['seed'])+123456
#         # rl_fed(env=env, seed=seed, model=model, original_traj=original_traj, max_ep_len=max_ep_len, importance=None,
#         #        render=False, mask_act=False)
#         replay_reward_10 = rl_fed(env=env, seed=seed, model=model, 
#                                   original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_10, 
#                                   render=False, mask_act=True)
#         replay_reward_20 = rl_fed(env=env, seed=seed, model=model,
#                                   original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_20,
#                                   render=False, mask_act=True)
#         replay_reward_30 = rl_fed(env=env, seed=seed, model=model,
#                                   original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_30,
#                                   render=False, mask_act=True)

#         diff_all_10[k, i] = np.abs(orin_reward-replay_reward_10)
#         diff_all_20[k, i] = np.abs(orin_reward-replay_reward_20)
#         diff_all_30[k, i] = np.abs(orin_reward-replay_reward_30)
#         importance_len_10[k, i] = len(importance_traj_10)
#         importance_len_20[k, i] = len(importance_traj_20)
#         importance_len_30[k, i] = len(importance_traj_30)

# np.savez('exp_results/fid_baselines.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_20=diff_all_20,
#          len_10=importance_len_10, len_30=importance_len_30, len_20=importance_len_20, rewards=finals_all)

# print(np.sum(diff_all_10, 1))
# print(np.sum(diff_all_20, 1))
# print(np.sum(diff_all_30, 1))


# Reward diff and explanation len figures

explainer_all = ['Value', 'Rudder', 'Sal', 'Att', 'RatNet', 'Our']

diff_10 = np.load('exp_results/fid_baselines.npz')['diff_10']
diff_20 = np.load('exp_results/fid_baselines.npz')['diff_20']
diff_30 = np.load('exp_results/fid_baselines.npz')['diff_30']
len_10 = np.load('exp_results/fid_baselines.npz')['len_10']
len_20 = np.load('exp_results/fid_baselines.npz')['len_20']
len_30 = np.load('exp_results/fid_baselines.npz')['len_30']

eps = 0.0001
diff_max = np.max(diff_10) # 672.9776036475669
rl_fid_10 = compute_rl_fid(diff_10, len_10, diff_max=diff_10.max(), len_max=100, eps=eps, weight=0.5)
rl_fid_20 = compute_rl_fid(diff_20, len_20, diff_max=diff_20.max(), len_max=100, eps=eps, weight=0.5)
rl_fid_30 = compute_rl_fid(diff_30, len_30, diff_max=diff_30.max(), len_max=100, eps=eps, weight=0.5)

print(np.mean(rl_fid_10, 1))
# print(np.std(rl_fid_10, 1))
print(np.mean(rl_fid_20, 1))
# print(np.std(rl_fid_20, 1))
print(np.mean(rl_fid_30, 1))
# print(np.std(rl_fid_30, 1))

print(np.argmin(np.mean(rl_fid_10, 1)))
print(np.argmin(np.mean(rl_fid_20, 1)))
print(np.argmin(np.mean(rl_fid_30, 1)))

print(''.join(['\t\t'] + [explainer+'\t' for explainer in explainer_all]))
print(''.join(['mean(rl_fid_10)\t'] + [f'{x:.3f}\t' for x in np.mean(rl_fid_10, 1)]))
print(''.join(['mean(rl_fid_20)\t'] + [f'{x:.3f}\t' for x in np.mean(rl_fid_20, 1)]))
print(''.join(['mean(rl_fid_30)\t'] + [f'{x:.3f}\t' for x in np.mean(rl_fid_30, 1)]))

metrics_all = ['Top5', 'Top15', 'Top25']
rl_fid_all = np.vstack((rl_fid_10[None, ...], rl_fid_20[None, ...], rl_fid_30[None,  ...]))
draw_fid_fig_t(rl_fid_all, explainer_all, metrics_all, save_path+'rl_fid_bar.pdf', box_plot=False, log_scale=False)

# rl_fid_10 = np.vstack((rl_fid_10[0:5], rl_fid_10[6:]))
# rl_fid_20 = np.vstack((rl_fid_20[0:5], rl_fid_20[6:]))
# rl_fid_30 = np.vstack((rl_fid_30[0:5], rl_fid_30[6:]))
# rl_fid_all = np.vstack((rl_fid_10[None, ...], rl_fid_20[None, ...], rl_fid_30[None,  ...]))

# explainer_all = ['Value', 'Rudder', 'Saliency', 'Attention', 'RatNet', 'Our']
# metrics_all = ['Top5', 'Top15', 'Top25']
# draw_fid_fig_t(rl_fid_all, explainer_all, metrics_all, save_path+'figures_x_l1/rl_fid_bar.pdf',
#                box_plot=False, log_scale=False)

