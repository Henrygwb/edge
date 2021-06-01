import os, sys
# sys.path.append('..')
import gym
import gym_compete
import numpy as np
from matplotlib import pyplot as plt
# from explainer.gp_utils import VisualizeCovar
from utils import rl_fed, load_agent, load_from_file


def run_exploration(budget, importance, num_trajs, num_step=3, fix_importance=True, random_importance=False):
    tie = []
    win = []
    correct_trajs_all = []
    num_loss = 0
    loss_seeds = []
    for i in range(num_trajs):
        original_traj = np.load('trajs_exp/KickAndDefend-v0_traj_{}.npz'.format(i))
        orin_reward = original_traj['final_rewards']
        seed = int(original_traj['seed'])
        if orin_reward == 1:
            continue
        loss_seeds.append(seed)
        # print(num_loss)
        if random_importance:
            importance_traj = np.arange(num_step)
            np.random.shuffle(importance_traj)
        elif fix_importance:
            importance_traj = [184, 185, 186]
        else:
            importance_traj = np.argsort(importance[i,])[::-1][0:num_step]
        j = 0
        j_1 = 0
        correct_trajs = []
        for _ in range(budget):
            replay_reward, traj = run_exploration_traj(env=env, seed=seed, model=model, obs_rms=obs_rms,
                                                       original_traj=original_traj, importance=importance_traj,
                                                       render=False)
            if replay_reward == 0:
                j += 1
            if replay_reward == 1000:
                j_1 += 1
            if replay_reward == 1000 and len(correct_trajs) == 0:
                correct_trajs.append(traj)
        correct_trajs_all.append(correct_trajs)
        tie.append(j)
        win.append(j_1)
        num_loss += 1
    print(num_loss)

    obs_all = []
    acts_all = []
    for trajs in correct_trajs_all:
        for traj in trajs:
            for step in range(len(traj[0])):
                obs_all.append(traj[0][step])
                acts_all.append(traj[1][step])

    obs_all = np.array(obs_all)
    acts_all = np.array(acts_all)

    print(obs_all.shape)
    print(acts_all.shape)

    return np.array(tie), np.array(win), correct_trajs_all, obs_all, acts_all, loss_seeds


def run_exploration_traj(env, seed, model, obs_rms, original_traj, importance, agent_type=['zoo', 'zoo'],
                         exp_agent_id=0, max_ep_len=200, render=False):

    values_orin = original_traj['values']
    traj_len = np.count_nonzero(values_orin)
    start_step = max_ep_len - traj_len
    env.seed(seed)

    episode_length, epr, done = 0, 0, False  # bookkeeping
    observation = env.reset()  # get first state
    for id in range(2):
        if agent_type[id] == 'zoo':
            model[id].reset()
    obs = observation[exp_agent_id]
    state_all = []
    action_all = []
    for i in range(traj_len + 200):
        if start_step + i in importance:
            state_all.append(obs)
        actions = []
        for id, obs in enumerate(observation):
            if agent_type[id] == 'zoo':
                if id != exp_agent_id:
                    # fix the opponent agent.
                    act, _ = model[id].act(stochastic=False, observation=obs)
                else:
                    # victim agent.
                    act, _ = model[id].act(stochastic=False, observation=obs)
                    if start_step + i in importance:
                        # print(start_step + i)
                        # add noise into the action
                        act = act + np.random.rand(act.shape[0]) * 3 - 1
                        action_all.append(act)
            else:
                obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                act = model[id].step(obs=obs[None, :], deterministic=True)[0][0]
            actions.append(act)
        actions = tuple(actions)
        observation, _, done, infos = env.step(actions)
        obs = observation[exp_agent_id]
        reward = infos[exp_agent_id]['reward_remaining']
        episode_length += 1
        if render: env.render()
        episode_length += 1
        if done:
            epr = reward
            break
    return epr, (state_all, action_all)


def run_patch_traj(env, seed, model, obs_rms, obs_dict, act_dict, p, agent_type=['zoo', 'zoo'], exp_agent_id=0,
                   max_ep_len=200, eps=1e-4, render=False, mix_policy=True):

    env.seed(seed)
    in_dict = False

    act_idx = np.random.binomial(1, p)
    episode_length, epr, done = 0, 0, False  # bookkeeping
    observation = env.reset()
    for id in range(2):
        if agent_type[id] == 'zoo':
            model[id].reset()

    for i in range(max_ep_len):
        actions = []
        for id, obs in enumerate(observation):
            if agent_type[id] == 'zoo':
                if id != exp_agent_id:
                    # fixed opponent agent
                    act, _ = model[id].act(stochastic=False, observation=obs)
                else:
                    # victim agent we need to explain
                    # action = acts_orin[start_step+i]
                    act, _ = model[id].act(stochastic=False, observation=obs)
                    state_diff = np.sum(np.abs(obs_dict - obs), 1)
                    if np.min(state_diff) < eps:
                        in_dict = True
                        if mix_policy:
                            idx = np.argmin(state_diff)
                            acts = [act, act_dict[idx]]
                            act = acts[act_idx]
            else:
                obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                act = model[id].step(obs=obs[None, :], deterministic=True)[0][0]

            actions.append(act)

        actions = tuple(actions)
        observation, _, done, infos = env.step(actions)
        reward = infos[exp_agent_id]['reward_remaining']
        if render: env.render()
        episode_length += 1
        if done:
            epr = reward
            break
    # print(episode_length)
    return epr, in_dict


def visualize_traj(traj, save_path):
    values_orin = traj['values']
    obs_orin = traj['observations']
    traj_len = np.count_nonzero(values_orin)
    start_step = max_ep_len - traj_len

    for i in range(traj_len):
        obs_tmp = obs_orin[start_step+i]
        plt.axis('off')
        plt.imsave(save_path+str(i+start_step)+'.png', obs_tmp)
        plt.close()


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
    original_traj = np.load('trajs_exp/KickAndDefend-v0_traj_'+str(traj_idx)+'.npz')
    visualize_traj(original_traj, 'exp_results/'+str(traj_idx)+'/')

    return 0


save_path = 'exp_results/'
env_name = 'multicomp/KickAndDefend-v0'
max_ep_len = 200
agent_path = './agent-zoo/kick-and-defend'
model = load_agent(env_name, agent_type=['zoo','zoo'], agent_path=agent_path)
obs_rms = None
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

# Launch attack at the most importance time steps: Top 10/30/50.
exps_all = [dgp_1_sal, sal_value, rudder_sal, saliency_sal, attn_sal, rat_sal, None]
orin_reward_all = np.zeros((7, 500))
reward_10_all = np.zeros((7, 500))
reward_20_all = np.zeros((7, 500))
reward_30_all = np.zeros((7, 500))
for k in range(7):
    print(k)
    importance = exps_all[k]
    for i in range(500):
        if i % 100 == 0: print(i)
        if k == -1:
            importance_traj = np.arange(max_ep_len)
            np.random.shuffle(importance_traj)
        else:
            importance_traj = np.argsort(importance[i, ])[::-1]
        original_traj = np.load('trajs_exp/KickAndDefend-v0_traj_{}.npz'.format(i))
        seed = int(original_traj['seed']) + 2000
        orin_reward = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                             original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj[0:10,],
                             render=False, exp_agent_id=0, mask_act=False)
       
        reward_10 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj[0:10,],
                                  render=False, exp_agent_id=0, mask_act=True)

        reward_20 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj[0:20,],
                                  render=False, exp_agent_id=0, mask_act=True)

        reward_30 = rl_fed(env=env, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj[0:30,],
                                  render=False, exp_agent_id=0, mask_act=True)
        
        orin_reward_all[k, i] = orin_reward
        reward_10_all[k, i] = reward_10
        reward_20_all[k, i] = reward_20
        reward_30_all[k, i] = reward_30

np.savez(save_path+'att_results.npz', orin_reward=orin_reward_all,
         diff_10=reward_10_all, diff_20=reward_20_all, diff_30=reward_30_all)
       
att_results = np.load(save_path+'att_results.npz')
total_trajs_num = 500
for k in range(7):
    print('======================')
    print(str(k))
    win = np.where(att_results['orin_reward'][k, ] == 1000)[0].shape[0]
    print('Original winning rate: %.2f' % (100 * (win / total_trajs_num)))

    win = np.where(att_results['diff_10'][k, ] == 1000)[0].shape[0]
    print('Win rate 10: %.2f' % (100 * (win / total_trajs_num)))

    win = np.where(att_results['diff_20'][k, ] == 1000)[0].shape[0]
    print('Win rate 20: %.2f' % (100 * (win / total_trajs_num)))

    win = np.where(att_results['diff_30'][k, ] == 1000)[0].shape[0]
    print('Win rate 30: %.2f' % (100 * (win / total_trajs_num)))

# Patch individual trajs and policy.
def patch_trajs_policy(exp_method, sal, budget, num_patch_traj, num_test_traj, num_step, free_test=False, collect_dict=True):
    print(exp_method)
    if collect_dict:
        if exp_method == 'dgp':
            tie, win, trajs_all, obs_dict, acts_dict, loss_seeds = run_exploration(budget, sal, num_patch_traj,
                                                                                   num_step=num_step)
        else:
            tie, win, trajs_all, obs_dict, acts_dict, loss_seeds = run_exploration(budget, sal, num_patch_traj,
                                                                                   num_step=num_step,
                                                                                   fix_importance=False,
                                                                                   random_importance=False)
    else:
        tie = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['tie']
        win = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['win']
        obs_dict = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['obs']
        acts_dict = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['acts']
        loss_seeds = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['seed']
        p = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['p']

    total_trajs_num = float(win.shape[0])
    win_num = np.count_nonzero(win)
    print('Win rate: %.2f' % (100 * (win_num / total_trajs_num)))
    print('Exploration success rate: %.2f' % (100 * (np.mean(win) / budget)))

    if win_num == 0:
        np.savez(save_path + exp_method + '_patch_results_' + str(budget) + '.npz', tie=tie, win=win,
                 obs=obs_dict, acts=acts_dict, seed=loss_seeds)
        print('could not find any winning policy.')
        return 0
    # print(obs_dict.shape)
    # print(acts_dict.shape)
    # print(len(loss_seeds))
    num_seed_trajs = int((len(loss_seeds)/num_patch_traj)*obs_dict.shape[0]) + 1
    loss_seeds_1 = loss_seeds[0:num_seed_trajs]
    obs_dict = obs_dict[0:num_seed_trajs, ]
    acts_dict = acts_dict[0:num_seed_trajs, ]

    # print(len(loss_seeds))
    # print(obs_dict.shape)
    # print(acts_dict.shape)

    # Get the patch prob.
    if collect_dict:
        num_rounds = 0
        num_loss = 0
        for i in range(num_test_traj):
            seed = i + 1000
            r_1, in_dict = run_patch_traj(env, seed, model, obs_rms, obs_dict, acts_dict, p=0, eps=1e-3, render=False,
                                          mix_policy=False)
            if r_1 != 0 and in_dict:
                num_rounds += 1.0
                if r_1 == -1000:
                    num_loss += 1.0
        p = num_loss / (num_rounds + 1e-16)
    print('===')
    print(p)
    print('===')
    if p == 0:
        p = 1
    # p = 1.0
    num_rounds = 0
    results_1 = []
    results_p = []
    for i in range(num_test_traj):
        if i % 100 == 0:
            print(i)
        if i < len(loss_seeds_1) and not free_test:
            seed = int(loss_seeds_1[i])
        else:
            seed = i
        # print('=========')
        r_1, _ = run_patch_traj(env, seed, model, obs_rms, obs_dict, acts_dict, p=0, eps=1e-3, max_ep_len=200,
                                render=False, mix_policy=False)
        # print(r_1)
        # print('----')
        r_p, _ = run_patch_traj(env, seed, model, obs_rms, obs_dict, acts_dict, p=p, eps=1e-5, max_ep_len=200,
                                render=False, mix_policy=True)

        # print(r_p)
        if r_1 != 0 and r_p != 0:
            num_rounds += 1
            results_1.append(r_1)
            results_p.append(r_p)

    results_1 = np.array(results_1)
    results_p = np.array(results_p)

    num_win_1 = np.where(results_1 == 1000)[0].shape[0]
    num_win_p = np.where(results_p == 1000)[0].shape[0]

    win_diff = results_1 - results_p
    num_all_win = np.where(win_diff == 0)[0].shape[0]
    num_1_win_p_loss = np.where(win_diff == 2000)[0].shape[0]
    num_1_loss_p_win = np.where(win_diff == -2000)[0].shape[0]

    print('Testing winning rate of the original model %.2f' % (100 * (num_win_1 / num_rounds)))
    print('Testing winning rate of the patched model %.2f' % (100 * (num_win_p / num_rounds)))
    print('Total Number of games: %d' % num_rounds)
    print('Number of games that original policy wins but patched policy loses: %d' % num_1_win_p_loss)
    print('Number of games that original policy loses but patched policy win: %d' % num_1_loss_p_win)

    np.savez(save_path + exp_method + '_patch_results_' + str(budget) + '.npz', tie=tie, win=win,
             obs=obs_dict, acts=acts_dict, results_1=results_1, results_p=results_p, seed=loss_seeds, p=p)

    return 0


budget = 10
num_patch_traj = 2000
num_test_traj = 500

exp_methods = ['dgp_1', 'value', 'rudder', 'attention', 'rationale', 'saliency']
sals = [dgp_1_sal,  sal_value, rudder_sal, attn_sal, rat_sal, saliency_sal]

for k in range(6):
    patch_trajs_policy(exp_methods[k], sals[k], budget, num_patch_traj, num_test_traj, num_step=10, free_test=True,
                       collect_dict=True)
    # patch_trajs_policy(exp_methods[k], sals[k], budget, num_patch_traj, num_test_traj, num_step=10, free_test=False,
    #                    collect_dict=False)
