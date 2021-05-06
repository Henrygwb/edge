import os, sys
#sys.path.append('..')
import gym
import gym_compete
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from utils import rl_fed, load_agent, load_from_file


def VisualizeCovar(covariance, save_path):
    plt.figure()
    heat = sns.heatmap(
        covariance,
        cmap="YlGnBu",
        square=True,
        robust=True,
        xticklabels=False,
        yticklabels=False,
    )
    if save_path[-3:] != 'pdf':
        raise TypeError('Output format should be pdf.')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    return 0


def run_patch(budget, model, obs_rms, max_ep_len, num_trajs, num_step):
    exps_all = [sal_value, rudder_sal, saliency_sal, dgp_2_sal]
    tie_all = np.zeros((5, 994))
    win_all = np.zeros((5, 994))
    for k in range(5):
        print(k)
        importance = exps_all[k]
        if k == 3:
            correct_trajs_all = []
        num_loss = 0
        for i in range(num_trajs):
            original_traj = np.load('trajs_exp/youshallnotpasshumans_v0_traj_{}.npz'.format(i))
            orin_reward = original_traj['final_rewards']
            seed = int(original_traj['seed'])
            if orin_reward == 1:
                continue
            if k == 5:
                importance_traj = np.arange(max_ep_len)
                np.random.shuffle(importance_traj)
                importance_traj = importance_traj[0:num_step]
            else:
                importance_traj = np.argsort(importance[i,])[::-1][0:num_step]
            j = 0
            j_1 = 0
            if k == 3:
                correct_trajs = []
            for _ in range(budget):
                replay_reward, traj = run_patch_traj(env, seed, model, obs_rms,
                                                     ['zoo','zoo'], original_traj, max_ep_len,
                                                     importance_traj, exp_agent_id=1, render=False)
                if replay_reward == 0:
                    j += 1
                if replay_reward == 1000:
                    j_1 += 1
                if k == 3 and replay_reward == 1000:
                    correct_trajs.append(traj)
            if k == 3:
                correct_trajs_all.append(correct_trajs)
            tie_all[k, num_loss] = j
            win_all[k, num_loss] = j_1
            num_loss += 1
    return tie_all, win_all, correct_trajs_all


def run_patch_traj(env, seed, model, obs_rms, agent_type, original_traj, max_ep_len, importance,
                   exp_agent_id=1, render=False):

    importance = np.sort(importance)
    values_orin = original_traj['values']
    traj_len = np.count_nonzero(values_orin)
    start_step = max_ep_len - traj_len
    env.seed(seed)

    state_all = []
    action_all = []

    episode_length, epr, done = 0, 0, False
    observation = env.reset()

    for i in range(traj_len):
        actions = []
        for id, obs in enumerate(observation):
            if agent_type[id] == 'zoo':
                if id != exp_agent_id:
                    # fixed opponent agent
                    act, _ = model[id].act(stochastic=False, observation=obs)
                else:
                    if start_step + i > importance[0]:
                        state_all.append(obs)
                    # victim agent we need to explain
                    act, _ = model[id].act(stochastic=False, observation=obs)
                    if start_step + i in importance:
                        # add noise into the action
                        act = act + np.random.rand(act.shape[0]) * 2 - 1
                    if start_step + i > importance[0]:
                        action_all.append(act)
            else:
                obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                act = model[id].step(obs=obs[None, :], deterministic=True)[0][0]
            actions.append(act)

        actions = tuple(actions)
        observation, _, done, infos = env.step(actions)
        reward = infos[exp_agent_id]['reward_remaining']
        episode_length += 1

        if render: env.render()
        if done:
            assert reward != 0
            epr = reward
            break
    # print('step # {}, reward {:.0f}.'.format(episode_length, epr))
    return epr, (state_all, action_all)


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
#traj_covar_1 = dgp_1_fid_results['traj_cova']
#step_covar_1 = dgp_1_fid_results['step_covar']

dgp_2_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_600_False_False_False_False_False_False_True_1e-05_10_16_True_exp.npz')
dgp_2_sal = dgp_2_fid_results['sal']
#traj_covar_2 = dgp_2_fid_results['traj_cova']
#step_covar_2 = dgp_2_fid_results['step_covar']

# Traj important time steps visualization.
# Winning trajs.
# demonstrate_trajs(2)
# demonstrate_trajs(33)
# demonstrate_trajs(1000)

# Loss trajs.
# demonstrate_trajs(22)
# demonstrate_trajs(1017)
# demonstrate_trajs(1877)

# Traj/Time step correlation visualization.
"""
for i in range(40):
   VisualizeCovar(step_covar_1[0, i*200:(i+1)*200, i*200:(i+1)*200], save_path+'dgp_1_step_covar_'+str(i)+'.pdf')
traj_covar_small = np.zeros((40, 40))
for i in range(40):
    for j in range(40):
        traj_covar_small[i, j] = traj_covar_1[0, i*200, j*200]
VisualizeCovar(traj_covar_small, save_path+'dgp_1_traj_covar.pdf')
del traj_covar_1
del step_covar_1
for i in range(40):
   VisualizeCovar(step_covar_2[0, i*200:(i+1)*200, i*200:(i+1)*200], save_path+'dgp_2_step_covar_'+str(i)+'.pdf')
traj_covar_small = np.zeros((40, 40))
for i in range(40):
    for j in range(40):
        traj_covar_small[i, j] = traj_covar_2[0, i*200, j*200]
VisualizeCovar(traj_covar_small, save_path+'dgp_2_traj_covar.pdf')
del traj_covar_2
del step_covar_2
"""
# Launch attack at the most importance time steps: Top 10/30/50.
exps_all = [sal_value, rudder_sal, saliency_sal, attn_sal, rat_sal, dgp_1_sal, dgp_2_sal]
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
        replay_reward_10 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj[0:10,],
                                  render=False, mask_act=False)
        replay_reward_30 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj[0:30,],
                                  render=False, mask_act=False)
        replay_reward_50 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj[0:50,],
                                  render=False, mask_act=False)

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
    tie = np.where(diff_10[k, ] == 1)[0].shape[0]
    print('Win rate 10: %.2f' % (100 * (win / total_trajs_num)))
    print('Non loss rate 10: %.2f' % (100 * ((win+tie)/total_trajs_num)))

    win = np.where(diff_30[k, ] == 0)[0].shape[0]
    tie = np.where(diff_30[k, ] == 1)[0].shape[0]
    print('Win rate 30: %.2f' % (100 * (win / total_trajs_num)))
    print('Non loss rate 30: %.2f' % (100 * ((win+tie)/total_trajs_num)))

    win = np.where(diff_50[k, ] == 0)[0].shape[0]
    tie = np.where(diff_50[k, ] == 1)[0].shape[0]
    print('Win rate 50: %.2f' % (100 * (win / total_trajs_num)))
    print('Non loss rate 50: %.2f' % (100 * ((win+tie)/total_trajs_num)))


# Patch individual trajs.
# budget = 15
# tie_30, win_30, trajs_30 = run_patch(budget, model, obs_rms, max_ep_len, 30, 5)
# np.savez(save_path+'patch_results_30.npz', tie_30=tie_30, win_30=win_30, trajs_30=trajs_30)
# patch_results = np.load(save_path+'patch_results_30.npz')


# Patch policy.

# optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 1e-4}], lr=0.01)
# loss_fn = nn.CrossEntropyLoss()
# if torch.cuda.is_available():
#     model = model.cuda()
# value, logit, _ = model((Variable(state.view(1, 1, 80, 80)), (hx, cx)))
# loss = loss_fn(logit, actions)
# loss.backward()
# optimizer.step()