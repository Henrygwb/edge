import os, sys
# sys.path.append('..')
import gym
# import gym_compete
import numpy as np
from matplotlib import pyplot as plt
# from explainer.gp_utils import VisualizeCovar
from utils import rl_fed, load_agent, load_from_file


def visualize_traj(traj, save_path):
    acts_orin = traj['actions']
    obs_orin = traj['observations']
    traj_len = np.count_nonzero(acts_orin)
    start_step = max_ep_len - traj_len

    for i in range(traj_len):
        obs_tmp = obs_orin[start_step+i]
        plt.axis('off')
        plt.imsave(save_path+str(i+start_step)+'.png', obs_tmp)


def demonstrate_trajs(traj_idx, num_step=10):
    exp_value = np.argsort(sal_value[traj_idx])[::-1]
    exp_rudder = np.argsort(rudder_sal[traj_idx])[::-1]
    exp_sal = np.argsort(saliency_sal[traj_idx])[::-1]
    exp_attn = np.argsort(attn_sal[traj_idx])[::-1]
    exp_rat = np.argsort(rat_sal[traj_idx])[::-1]
    exp_dgp_1 = np.argsort(dgp_1_sal[traj_idx])[::-1]
    exp_dgp_2 = np.argsort(dgp_2_sal[traj_idx])[::-1]

    with open('exp_results/'+str(traj_idx)+'/exp.txt', 'w') as f:
        f.writelines('Value function: \n')
        f.writelines('Most important: \n')
        f.writelines(np.array2string(exp_value[0:num_step]) + '\n')
        f.writelines('Most unimportant: \n')
        f.writelines(np.array2string(exp_value[-num_step:]) + '\n')
        f.writelines('\n')

        f.writelines('Rudder: \n')
        f.writelines('Most important: \n')
        f.writelines(np.array2string(exp_rudder[0:num_step]) + '\n')
        f.writelines('Most unimportant: \n')
        f.writelines(np.array2string(exp_rudder[-num_step:]) + '\n')
        f.writelines('\n')

        f.writelines('Saliency: \n')
        f.writelines('Most important: \n')
        f.writelines(np.array2string(exp_sal[0:num_step]) + '\n')
        f.writelines('Most unimportant: \n')
        f.writelines(np.array2string(exp_sal[-num_step:]) + '\n')
        f.writelines('\n')

        f.writelines('Attention: \n')
        f.writelines('Most important: \n')
        f.writelines(np.array2string(exp_attn[0:num_step]) + '\n')
        f.writelines('Most unimportant: \n')
        f.writelines(np.array2string(exp_attn[-num_step:]) + '\n')
        f.writelines('\n')

        f.writelines('Rationale net: \n')
        f.writelines('Most important: \n')
        f.writelines(np.array2string(exp_rat[0:num_step]) + '\n')
        f.writelines('Most unimportant: \n')
        f.writelines(np.array2string(exp_rat[-num_step:]) + '\n')
        f.writelines('\n')

        f.writelines('DGP 1: \n')
        f.writelines('Most important: \n')
        f.writelines(np.array2string(exp_dgp_1[0:num_step]) + '\n')
        f.writelines('Most unimportant: \n')
        f.writelines(np.array2string(exp_dgp_1[-num_step:]) + '\n')
        f.writelines('\n')

        f.writelines('DGP 2: \n')
        f.writelines('Most important: \n')
        f.writelines(np.array2string(exp_dgp_2[0:num_step]) + '\n')
        f.writelines('Most unimportant: \n')
        f.writelines(np.array2string(exp_dgp_2[-num_step:]) + '\n')
    f.close()

    original_traj = np.load('trajs_exp/Pong-v0_traj_'+str(traj_idx)+'.npz')
    visualize_traj(original_traj, 'exp_results/'+str(traj_idx)+'/')

    return 0


save_path = 'exp_results/'
env_name = 'multicomp/YouShallNotPassHumans-v0'
max_ep_len = 200
agent_path = './agent-zoo/you-shall-not-pass'
model = load_agent(env_name, agent_type=['zoo','zoo'], agent_path=agent_path)
norm_path = agent_path + '/obs_rms.pkl'
obs_rms = load_from_file(norm_path)
env = gym.make(env_name)

# Explainer 1 - Value function.
sal_value = np.load(save_path + 'value_exp.npz')['sal']

# Explainer 2 - Rudder.
rudder_fid_results = np.load(save_path + 'rudder_MLP_GRU_exp.npz')
rudder_sal = rudder_fid_results['sal']

# Explainer 3 - RNN + Saliency.
saliency_fid_results = np.load(save_path + 'saliency_classification_MLP_GRU_True_exp_best.npz')
saliency_sal = saliency_fid_results['sal']

# Explainer 4 - AttnRNN.
attn_fid_results = np.load(save_path + 'attention_classification_MLP_GRU_tanh_exp.npz')
attn_sal = attn_fid_results['sal']

# Explainer 5 - RationaleNet.
rat_fid_results = np.load(save_path + 'rationale_classification_MLP_GRU_exp.npz')
rat_sal = rat_fid_results['sal']

# Explainer 6 - DGP.
dgp_1_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_600_False_False_False_False_False_False_False_0.01_10_16_True_exp.npz')
dgp_1_sal = dgp_1_fid_results['sal']
# traj_covar_1 = dgp_1_fid_results['traj_cova']
# step_covar_1 = dgp_1_fid_results['step_covar']

dgp_2_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_600_False_False_False_False_False_False_True_1e-05_10_16_True_exp.npz')
dgp_2_sal = dgp_2_fid_results['sal']
# traj_covar_2 = dgp_2_fid_results['traj_cova']
# step_covar_2 = dgp_2_fid_results['step_covar']

# Traj important time steps visualization.
# Winning trajs.
# demonstrate_trajs(0)
# demonstrate_trajs(9)
# demonstrate_trajs(11)

# Loss trajs.
# demonstrate_trajs(1)
# demonstrate_trajs(5)
# demonstrate_trajs(18)

# Traj/Time step correlation visualization.
# for i in range(40):
#    VisualizeCovar(step_covar_1[0, i*200:(i+1)*200, i*200:(i+1)*200], save_path+'dgp_1_step_covar_'+str(i)+'.pdf')
# traj_covar_small = np.zeros((40, 40))
# for i in range(40):
#     for j in range(40):
#         traj_covar_small[i, j] = traj_covar_1[0, i*200, j*200]
# VisualizeCovar(traj_covar_small, save_path+'dgp_1_traj_covar.pdf')
# del traj_covar_1
# del step_covar_1

# for i in range(40):
#    VisualizeCovar(step_covar_2[0, i*200:(i+1)*200, i*200:(i+1)*200], save_path+'dgp_2_step_covar_'+str(i)+'.pdf')
# traj_covar_small = np.zeros((40, 40))
# for i in range(40):
#     for j in range(40):
#         traj_covar_small[i, j] = traj_covar_2[0, i*200, j*200]
# VisualizeCovar(traj_covar_small, save_path+'dgp_2_traj_covar.pdf')
# del traj_covar_2
# del step_covar_2

# Launch attack at the most importance time steps: Top 10/30/50.
exps_all = [dgp_1_sal, dgp_2_sal, sal_value, rudder_sal, saliency_sal, attn_sal, rat_sal]
diff_all_10 = np.zeros((8, 1006))
diff_all_30 = np.zeros((8, 1006))
diff_all_50 = np.zeros((8, 1006))
for k in range(8):
    print(k)
    total_traj_num = 0
    importance = exps_all[k]
    for i in range(2000):
        if k == 8:
            importance_traj = np.arange(max_ep_len)
            np.random.shuffle(importance_traj)
        else:
            importance_traj = np.argsort(importance[i, ])[::-1]
        original_traj = np.load('trajs_exp/youshallnotpasshumans_v0_traj_{}.npz'.format(i))
        orin_reward = original_traj['final_rewards']
        if orin_reward == 0:
            continue
        orin_reward = 1000
        seed = int(original_traj['seeds'])
        print('============')
        print(seed)
        replay_reward_10 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj[0:10,],
                                  render=False, mask_act=False)
        print(replay_reward_10)
        replay_reward_30 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj[0:30,],
                                  render=False, mask_act=False)
        print(replay_reward_30)
        replay_reward_50 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj[0:50,],
                                  render=False, mask_act=False)
        print(replay_reward_50)

        diff_all_10[k, total_traj_num] = orin_reward-replay_reward_10
        diff_all_30[k, total_traj_num] = orin_reward-replay_reward_30
        diff_all_50[k, total_traj_num] = orin_reward-replay_reward_50
        total_traj_num += 1

np.savez(save_path+'att_results.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_50=diff_all_50)
att_results = np.load(save_path+'att_results.npz')
diff_10 = att_results['diff_10']
diff_30 = att_results['diff_30']
diff_50 = att_results['diff_50']
total_trajs_num = float(diff_10.shape[1])
for k in range(8):
    print('======================')
    print(str(k))
    win = np.where(diff_10[k, ] == 0)[0].shape[0]
    tie = np.where(diff_10[k, ] == 1000)[0].shape[0]
    print('Win rate 10: %.2f' % (100 * (win / total_trajs_num)))
    print('Non loss rate 10: %.2f' % (100 * ((win+tie)/total_trajs_num)))

    win = np.where(diff_30[k, ] == 0)[0].shape[0]
    tie = np.where(diff_30[k, ] == 1000)[0].shape[0]
    print('Win rate 30: %.2f' % (100 * (win / total_trajs_num)))
    print('Non loss rate 30: %.2f' % (100 * ((win+tie)/total_trajs_num)))

    win = np.where(diff_50[k, ] == 0)[0].shape[0]
    tie = np.where(diff_50[k, ] == 1000)[0].shape[0]
    print('Win rate 50: %.2f' % (100 * (win / total_trajs_num)))
    print('Non loss rate 50: %.2f' % (100 * ((win+tie)/total_trajs_num)))
