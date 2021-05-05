import os, sys
import torch, gym
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from explainer.gp_utils import VisualizeCovar
from atari_pong.utils import rl_fed, NNPolicy, prepro


def run_patch(budget, num_trajs):
    exps_all = [sal_value, rudder_sal, saliency_sal, dgp_1_sal]
    diff_all = np.zeros((4, 209))
    for k in range(4):
        print(k)
        importance = exps_all[k]
        if k == 3:
            correct_trajs_all = []
        num_loss = 0
        for i in range(num_trajs):
            original_traj = np.load('trajs_exp/Pong-v0_traj_{}.npz'.format(i))
            orin_reward = original_traj['final_rewards']
            seed = int(original_traj['seed'])
            if orin_reward == 1:
                continue
            if k == 2:
                importance_traj = np.arange(max_ep_len)
                np.random.shuffle(importance_traj)
                importance_traj = importance_traj[0:10]
            elif k == 3:
                importance_traj = [184, 185, 186, 187, 188, 189, 190, 191, 192, 193, 194]
            else:
                importance_traj = np.argsort(importance[i,])[::-1][0:10]
            j = 0
            if k == 3:
                correct_trajs = []
            for _ in range(budget):
                replay_reward_10, traj = run_patch_traj(env_name=env_name, seed=seed, model=model,
                                                        original_traj=original_traj, max_ep_len=max_ep_len,
                                                        importance=importance_traj, render=False)
                j += replay_reward_10
                if k == 3 and replay_reward_10 != -1:
                    correct_trajs.append(traj)
            if k == 3:
                correct_trajs_all.append(correct_trajs)
            diff_all[k, num_loss] = j
            num_loss += 1
    return diff_all, correct_trajs_all


def run_patch_traj(env_name, seed, model, original_traj, importance, max_ep_len=200, render=False):

    acts_orin = original_traj['actions']
    traj_len = np.count_nonzero(acts_orin)
    start_step = max_ep_len - traj_len

    env = gym.make(env_name)
    env.seed(seed)
    env.env.frameskip = 3

    episode_length, epr, done = 0, 0, False  # bookkeeping
    obs_0 = env.reset()  # get first state
    state = torch.tensor(prepro(obs_0))
    hx, cx = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))
    act_set = np.array([0, 1, 2, 3, 4, 5])
    state_all = []
    action_all = []
    hidden_all = []
    for i in range(traj_len+20):
        if epr != 0:
            break
        # Steps before the important steps reproduce original traj.
        if start_step+i in importance:
            hidden_all.append((hx, cx))
            state_all.append(state)
        value, logit, (hx, cx) = model((Variable(state.view(1, 1, 80, 80)), (hx, cx)))
        hx, cx = Variable(hx.data), Variable(cx.data)
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1)[1].data.numpy()[0]
        # Important steps take suboptimal actions.
        if start_step + i in importance:
            act_set_1 = act_set[act_set!=action]
            action = np.random.choice(act_set_1)
        # Steps after the important steps take optimal actions.
        obs, reward, done, expert_policy = env.step(action)
        state = torch.tensor(prepro(obs))
        if render: env.render()
        epr += reward
        if start_step + i in importance:
            action_all.append(action)
        # save info!
        episode_length += 1

    return epr, (state_all, action_all, hidden_all)


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
# env_name = 'Pong-v0'
# max_ep_len = 200
# agent_path = 'agents/{}/'.format(env_name.lower())
# num_trajs = 30
#
# env = gym.make(env_name)
# model = NNPolicy(channels=1, num_actions=env.action_space.n)
# _ = model.try_load(agent_path, checkpoint='*.tar')
# torch.manual_seed(1)
#
# exps_all = [sal_value, rudder_sal, saliency_sal, attn_sal, rat_sal, dgp_1_sal, dgp_2_sal]
# diff_all_10 = np.zeros((7, 1671))
# diff_all_30 = np.zeros((7, 1671))
# diff_all_50 = np.zeros((7, 1671))
# for k in range(7):
#     print(k)
#     total_traj_num = 0
#     importance = exps_all[k]
#     for i in range(1880):
#         if k == 2:
#             importance_traj = np.arange(max_ep_len)
#             np.random.shuffle(importance_traj)
#         else:
#             importance_traj = np.argsort(importance[i,])[::-1]
#         original_traj = np.load('trajs_exp/Pong-v0_traj_{}.npz'.format(i))
#         orin_reward = original_traj['final_rewards']
#         if orin_reward == 0:
#             continue
#         seed = int(original_traj['seed'])
#         replay_reward_10 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
#                                   max_ep_len=max_ep_len, importance=importance_traj[0:10,], render=False, mask_act=True)
#
#         replay_reward_30 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
#                                   max_ep_len=max_ep_len, importance=importance_traj[0:30,], render=False, mask_act=True)
#
#         replay_reward_50 = rl_fed(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
#                                   max_ep_len=max_ep_len, importance=importance_traj[0:50,], render=False, mask_act=True)
#
#         diff_all_10[k, total_traj_num] = orin_reward-replay_reward_10
#         diff_all_30[k, total_traj_num] = orin_reward-replay_reward_30
#         diff_all_50[k, total_traj_num] = orin_reward-replay_reward_50
#         total_traj_num += 1
#
# np.savez(save_path+'att_results.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_50=diff_all_50)

# Patch individual trajs.
env_name = 'Pong-v0'
max_ep_len = 200
agent_path = 'agents/{}/'.format(env_name.lower())
num_trajs = 30

env = gym.make(env_name)
model = NNPolicy(channels=1, num_actions=env.action_space.n)
_ = model.try_load(agent_path, checkpoint='*.tar')
torch.manual_seed(1)

# budget = 10
# diff_10, trajs_10 = run_patch(budget, 1000)
# np.savez(save_path+'patch_results_10.npz', diff_10=diff_10, trajs_10=trajs_10)
budget = 30
diff_30, trajs_30 = run_patch(budget, 30)
np.savez(save_path+'patch_results_30.npz', diff_30=diff_30, trajs_30=trajs_30)

# Patch policy.

# optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 1e-4}], lr=0.01)
# loss_fn = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     model = model.cuda()
# value, logit, _ = model((Variable(state.view(1, 1, 80, 80)), (hx, cx)))
# loss = loss_fn(logit, actions)
# loss.backward()
# optimizer.step()
