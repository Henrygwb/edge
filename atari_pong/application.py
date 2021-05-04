import os, sys
import torch, gym
import numpy as np
from matplotlib import pyplot as plt
from explainer.gp_utils import VisualizeCovar
from atari_pong.utils import rl_fed, NNPolicy


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

# Explainer 1 - Value function.
sal_value = np.load(save_path + 'value_exp.npz')['sal'][0:1880]

# Explainer 2 - Rudder.
rudder_fid_results = np.load(save_path + 'rudder_CNN_GRU_exp.npz')
rudder_sal = rudder_fid_results['sal']

# Explainer 3 - RNN + Saliency.
saliency_fid_results = np.load(save_path + 'saliency_classification_CNN_GRU_True_exp_best.npz')
saliency_sal = saliency_fid_results['sal']

# Explainer 4 - AttnRNN.
attn_fid_results = np.load(save_path + 'attention_classification_CNN_GRU_tanh_exp.npz')
attn_sal = attn_fid_results['sal']

# Explainer 5 - RationaleNet.
rat_fid_results = np.load(save_path + 'rationale_classification_CNN_GRU_exp.npz')
rat_sal = rat_fid_results['sal']

# Explainer 6 - DGP.
dgp_1_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_False_0.1_10_8_True_exp.npz')
dgp_1_sal = dgp_1_fid_results['sal']
# full_covar = dgp_1_fid_results['full_covar']
# traj_covar = dgp_1_fid_results['traj_cova']
# step_covar = dgp_1_fid_results['step_covar']
# VisualizeCovar(full_covar[0], save_path+'dgp_1_full_covar.pdf')
# VisualizeCovar(traj_covar[0], save_path+'dgp_1_traj_covar.pdf')
# VisualizeCovar(step_covar[0], save_path+'dgp_1_step_covar.pdf')
# del full_covar
# del traj_covar
# del step_covar

dgp_2_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_True_0.001_10_8_True_exp.npz')
dgp_2_sal = dgp_2_fid_results['sal']
# full_covar = dgp_2_fid_results['full_covar']
# traj_covar = dgp_2_fid_results['traj_cova']
# step_covar = dgp_2_fid_results['step_covar']
# VisualizeCovar(full_covar[0], save_path+'dgp_2_full_covar.pdf')
# VisualizeCovar(traj_covar[0], save_path+'dgp_2_traj_covar.pdf')
# VisualizeCovar(step_covar[0], save_path+'dgp_2_step_covar.pdf')
# del full_covar
# del traj_covar
# del step_covar

# Traj important time steps visualization.
# Winning trajs.
# demonstrate_trajs(2)
# demonstrate_trajs(33)
# demonstrate_trajs(1000)

# Loss trajs.
# demonstrate_trajs(22)
# demonstrate_trajs(1017)
# demonstrate_trajs(1877)

# Launch attack at the most importance time steps: Top 10/30/50.
env_name = 'Pong-v0'
max_ep_len = 200
agent_path = 'agents/{}/'.format(env_name.lower())
num_trajs = 30

env = gym.make(env_name)
model = NNPolicy(channels=1, num_actions=env.action_space.n)
_ = model.try_load(agent_path, checkpoint='*.tar')
torch.manual_seed(1)

exps_all = [sal_value, rudder_sal, saliency_sal, attn_sal, rat_sal, dgp_1_sal, dgp_2_sal]
diff_all_10 = np.zeros((7, 1671))
diff_all_30 = np.zeros((7, 1671))
diff_all_50 = np.zeros((7, 1671))
total_traj_num = 0
for k in range(7):
    print(k)
    importance = exps_all[k]
    for i in range(1880):
        if k == 2:
            importance_traj = np.arange(max_ep_len)
            np.random.shuffle(importance_traj)
        else:
            importance_traj = np.argsort(importance[i,])[::-1]
        original_traj = np.load('trajs_exp/Pong-v0_traj_{}.npz'.format(i))
        orin_reward = original_traj['final_rewards']
        if orin_reward == 0:
            continue
        total_traj_num += 1
        seed = int(original_traj['seed'])
        # replay_reward_orin = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
        #                             max_ep_len=max_ep_len, importance=None, render=False, mask_act=False)

        replay_reward_10 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
                                  max_ep_len=max_ep_len, importance=importance_traj[0:10,], render=False, mask_act=True)

        replay_reward_30 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
                                  max_ep_len=max_ep_len, importance=importance_traj[0:30,], render=False, mask_act=True)

        replay_reward_50 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
                                  max_ep_len=max_ep_len, importance=importance_traj[0:50,], render=False, mask_act=True)

        diff_all_10[k, i] = np.abs(orin_reward-replay_reward_10)
        diff_all_30[k, i] = np.abs(orin_reward-replay_reward_30)
        diff_all_50[k, i] = np.abs(orin_reward-replay_reward_50)

np.savez(save_path+'att_results.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_50=diff_all_50)



# Patch individual trajs.



# Patch policy.
