import os, sys, tqdm
sys.path.append('..')
import torch, gym
import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from matplotlib import pyplot as plt
from explainer.gp_utils import VisualizeCovar
from atari_pong.utils import rl_fed, NNPolicy, prepro


def run_exploration(budget, importance, num_trajs, num_step=3, fix_importance=True, random_importance=False):
    tie = []
    win = []
    correct_trajs_all = []
    num_loss = 0
    loss_seeds = []
    for i in range(num_trajs):
        original_traj = np.load('trajs_exp/Pong-v0_traj_{}.npz'.format(i))
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
            replay_reward_10, traj = run_exploration_traj(env_name=env_name, seed=seed, model=model,
                                                          original_traj=original_traj, max_ep_len=max_ep_len,
                                                          importance=importance_traj, render=False)
            if replay_reward_10 == 0:
                j += 1
            if replay_reward_10 == 1:
                j_1 += 1
            if replay_reward_10 == 1 and len(correct_trajs) == 0:
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
                obs_all.append(traj[0][step].numpy())
                acts_all.append(traj[1][step])

    obs_all = np.array(obs_all)
    acts_all = np.array(acts_all)

    print(obs_all.shape)
    print(acts_all.shape)

    return np.array(tie), np.array(win), correct_trajs_all, obs_all, acts_all, loss_seeds


def run_exploration_traj(env_name, seed, model, original_traj, importance, max_ep_len=200, render=False):

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
    for i in range(traj_len + 100):
        if epr != 0:
            break
        # Steps before the important steps reproduce original traj.
        if start_step+i in importance:
            state_all.append(state)
        value, logit, (hx, cx) = model((Variable(state.view(1, 1, 80, 80)), (hx, cx)))
        hx, cx = Variable(hx.data), Variable(cx.data)
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1)[1].data.numpy()[0]
        # Important steps take random actions.
        if start_step + i in importance:
            act_set_1 = act_set[act_set!=action]
            action = np.random.choice(act_set_1)
            action_all.append(action)
        # Steps after the important steps take optimal actions.
        obs, reward, done, expert_policy = env.step(action)
        state = torch.tensor(prepro(obs))
        if render: env.render()
        epr += reward
        # save info!
        episode_length += 1
    
    return epr, (state_all, action_all)


def run_patch_traj(env_name, seed, model, obs_dict, act_dict, p, max_ep_len=200, eps=1e-4,
                   render=False, mix_policy=True):

    env = gym.make(env_name)
    env.seed(seed)
    env.env.frameskip = 3
    in_dict = False

    episode_length, epr, done = 0, 0, False  # bookkeeping
    obs_0 = env.reset()  # get first state
    state = torch.tensor(prepro(obs_0))
    hx, cx = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))
    for i in range(max_ep_len):
        if epr != 0:
            break
        value, logit, (hx, cx) = model((Variable(state.view(1, 1, 80, 80)), (hx, cx)))
        hx, cx = Variable(hx.data), Variable(cx.data)
        prob = F.softmax(logit, dim=-1)
        action = prob.max(1)[1].data.numpy()[0]
        # check the lookup table and take the corresponding action if state is similar.
        state_diff = np.sum(np.abs(obs_dict - state.numpy()), (1, 2, 3))
        if np.min(state_diff) < eps:
            in_dict = True
            if mix_policy:
                idx = np.argmin(state_diff)
                actions = [act_dict[idx], action]
                act_idx = np.random.binomial(1, p)
                action = actions[act_idx]
        obs, reward, done, expert_policy = env.step(action)
        state = torch.tensor(prepro(obs))
        if render: env.render()
        epr += reward
        # save info!
        episode_length += 1
    # print(episode_length)
    return epr, in_dict


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
env_name = 'Pong-v0'
max_ep_len = 200
agent_path = 'agents/{}/'.format(env_name.lower())

env = gym.make(env_name)
model = NNPolicy(channels=1, num_actions=env.action_space.n)
_ = model.try_load(agent_path, checkpoint='*.tar')
torch.manual_seed(1)
#
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
# traj_covar = dgp_1_fid_results['traj_cova']
# step_covar = dgp_1_fid_results['step_covar']
#
# dgp_2_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_True_0.001_10_8_True_exp.npz')
# dgp_2_sal = dgp_2_fid_results['sal']
# traj_covar = dgp_2_fid_results['traj_cova']
# step_covar = dgp_2_fid_results['step_covar']

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
# for i in range(40):
#    VisualizeCovar(step_covar[0, i*200:(i+1)*200, i*200:(i+1)*200], save_path+'dgp_1_step_covar_'+str(i)+'.pdf')
# traj_covar_small = np.zeros((40, 40))
# for i in range(40):
#     for j in range(40):
#         traj_covar_small[i, j] = traj_covar[0, i*200, j*200]
# VisualizeCovar(traj_covar_small, save_path+'dgp_1_traj_covar.pdf')
# del traj_covar
# del step_covar

# Launch attack at the most importance time steps: Top 10/30/50.
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
# att_results = np.load(save_path+'att_results.npz')
# diff_10 = att_results['diff_10']
# diff_30 = att_results['diff_30']
# diff_50 = att_results['diff_50']
# total_trajs_num = float(diff_10.shape[1])
# for k in range(7):
#     print('======================')
#     print(str(k))
#     win = np.where(diff_10[k, ] == 0)[0].shape[0]
#     tie = np.where(diff_10[k, ] == 1)[0].shape[0]
#     print('Win rate 10: %.2f' % (100 * (win / total_trajs_num)))
#     print('Non loss rate 10: %.2f' % (100 * ((win+tie)/total_trajs_num)))
#
#     win = np.where(diff_30[k, ] == 0)[0].shape[0]
#     tie = np.where(diff_30[k, ] == 1)[0].shape[0]
#     print('Win rate 30: %.2f' % (100 * (win / total_trajs_num)))
#     print('Non loss rate 30: %.2f' % (100 * ((win+tie)/total_trajs_num)))
#
#     win = np.where(diff_50[k, ] == 0)[0].shape[0]
#     tie = np.where(diff_50[k, ] == 1)[0].shape[0]
#     print('Win rate 50: %.2f' % (100 * (win / total_trajs_num)))
#     print('Non loss rate 50: %.2f' % (100 * ((win+tie)/total_trajs_num)))


# Patch individual trajs and policy.
def patch_trajs_policy(exp_method, sal, budget, num_patch_traj, num_test_traj, free_test=False, collect_dict=True):
    print(exp_method)
    if collect_dict:
        if exp_method == 'dgp':
            tie, win, trajs_all, obs_dict, acts_dict, loss_seeds = run_exploration(budget, sal, num_patch_traj)
        elif exp_method == 'saliency':
            tie, win, trajs_all, obs_dict, acts_dict, loss_seeds = run_exploration(budget, sal, num_patch_traj,
                                                                                   fix_importance=False,
                                                                                   random_importance=True)
        else:
            tie, win, trajs_all, obs_dict, acts_dict, loss_seeds = run_exploration(budget, sal, num_patch_traj,
                                                                                   fix_importance=False,
                                                                                   random_importance=False)
    else:
        tie = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['tie']
        win = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['win']
        obs_dict = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['obs']
        acts_dict = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['acts']
        loss_seeds = np.load(save_path + exp_method + '_patch_results_' + str(budget) + '.npz')['seed']
   
    total_trajs_num = float(win.shape[0])
    win_num = np.count_nonzero(win)
    print('Win rate: %.2f' % (100 * (win_num / total_trajs_num)))
    print('Exploration success rate: %.2f' % (100 * (np.mean(win) / budget)))
   
    # print(obs_dict.shape)
    # print(acts_dict.shape)
    # print(len(loss_seeds)) 
    num_seed_trajs = 22 # int((len(loss_seeds)/num_patch_traj)*num_test_traj)
    loss_seeds = loss_seeds[0:num_seed_trajs]
    obs_dict = obs_dict[0:num_seed_trajs, ]
    acts_dict = acts_dict[0:num_seed_trajs, ]

    # print(len(loss_seeds))
    # print(obs_dict.shape)
    # print(acts_dict.shape)

    # Get the patch prob.
    num_rounds = 0
    num_loss = 0
    for i in range(num_test_traj):
        seed = i + 1000
        r_1, in_dict = run_patch_traj(env_name, seed, model, obs_dict, acts_dict, p=0, max_ep_len=200, eps=1e-3,
                                      render=False, mix_policy=False)
        if r_1 !=0 and in_dict:
            num_rounds += 1.0
            if r_1 == -1:
                num_loss += 1.0
    p = num_loss/num_rounds
    print('===')
    print(p)
    print('===')
    num_rounds = 0
    results_1 = []
    results_p = []
    for i in range(num_test_traj):
        if i % 100 == 0:
            print(i)
        if i < len(loss_seeds) and not free_test:
            seed = int(loss_seeds[i])
        else:
            seed = i
        # print('=========') 
        r_1, _ = run_patch_traj(env_name, seed, model, obs_dict, acts_dict, p=0, max_ep_len=200, eps=1e-3,
                                render=False, mix_policy=False)
        # print(r_1)
        # print('----')
        r_p, _ = run_patch_traj(env_name, seed, model, obs_dict, acts_dict, p=p, max_ep_len=200, eps=1e-5,
                                render=False, mix_policy=True)
        # print(r_p)
        if r_1 != 0 and r_p !=0:
            num_rounds += 1
            results_1.append(r_1)
            results_p.append(r_p)

    results_1 = np.array(results_1)
    results_p = np.array(results_p)

    num_win_1 = np.where(results_1==1)[0].shape[0]
    num_win_p = np.where(results_p==1)[0].shape[0]

    win_diff = results_1 - results_p
    num_all_win = np.where(win_diff==0)[0].shape[0]
    num_1_win_p_loss = np.where(win_diff==2)[0].shape[0]
    num_1_loss_p_win = np.where(win_diff==-2)[0].shape[0]

    print('Testing winning rate of the original model %.2f' % (100 * (num_win_1/num_rounds)))
    print('Testing winning rate of the patched model %.2f' % (100 * (num_win_p/num_rounds)))
    print('Total Number of games: %d' % num_rounds)
    print('Number of games that original policy wins but patched policy loses: %d' % num_1_win_p_loss)
    print('Number of games that original policy loses but patched policy win: %d' % num_1_loss_p_win)
   
    np.savez(save_path+exp_method+'_patch_results_'+str(budget)+'.npz', tie=tie, win=win,
             obs=obs_dict, acts=acts_dict, results_1=results_1, results_p=results_p, seed=loss_seeds, p=p)

    return 0


budget = 10
num_patch_traj = 1880
num_test_traj = 200

exp_methods = ['dgp', 'value', 'rudder', 'attention', 'rationale', 'saliency']
sals = [dgp_1_sal, sal_value, rudder_sal, attn_sal, rat_sal, saliency_sal]

for k in range(0, 6):
    patch_trajs_policy(exp_methods[k], sals[k], budget, num_patch_traj, num_test_traj, free_test=True, collect_dict=False)

















# Deprecated code: Patch model with behavior cloning.
# def run_patch_traj_for_retraining(env_name, seed, model, original_traj, importance, max_ep_len=200, render=False, mask=True):
#     importance = np.sort(importance)
#     acts_orin = original_traj['actions']
#     traj_len = np.count_nonzero(acts_orin)
#     start_step = max_ep_len - traj_len
#
#     env = gym.make(env_name)
#     env.seed(seed)
#     env.env.frameskip = 3
#
#     episode_length, epr, done = 0, 0, False  # bookkeeping
#     obs_0 = env.reset()  # get first state
#     state = torch.tensor(prepro(obs_0))
#     hx, cx = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))
#     act_set = np.array([0, 1, 2, 3, 4, 5])
#     state_all = []
#     action_all = []
#     hidden_all = []
#     for i in range(traj_len + 100):
#         if epr != 0:
#             break
#         # Steps before the important steps reproduce original traj.
#         if start_step+i > importance[0]:
#             hidden_all.append((hx, cx))
#             state_all.append(state)
#         value, logit, (hx, cx) = model((Variable(state.view(1, 1, 80, 80)), (hx, cx)))
#         hx, cx = Variable(hx.data), Variable(cx.data)
#         prob = F.softmax(logit, dim=-1)
#         action = prob.max(1)[1].data.numpy()[0]
#         # Important steps take suboptimal actions.
#         if mask:
#             if start_step + i in importance:
#                 act_set_1 = act_set[act_set!=action]
#                 action = np.random.choice(act_set_1)
#         # Steps after the important steps take optimal actions.
#         obs, reward, done, expert_policy = env.step(action)
#         state = torch.tensor(prepro(obs))
#         if render: env.render()
#         epr += reward
#         if start_step + i > importance[0]:
#             action_all.append(action)
#         # save info!
#         episode_length += 1
#
#     return epr, (state_all, action_all, hidden_all)

# def test(model, test_num=200):
#     num_win = 0
#     for i in range(test_num):
#         print(i)
#         original_traj = np.load('trajs_exp/Pong-v0_traj_{}.npz'.format(i))
#         seed = int(original_traj['seed'])
#         replay_reward_10, _ = run_patch_traj(env_name=env_name, seed=seed, model=model, original_traj=original_traj,
#                                           max_ep_len=max_ep_len, importance=[0, 0], render=False, mask=False)
#         if replay_reward_10 == 1:
#             num_win += 1.0
#     print('Testing winning rate %.2f' % (100 * (num_win/test_num)))
#     return 0
#
#
# def train(model):
#     obs = np.load(save_path+'patch_results_retrain_15.npz')['obs']
#     acts = np.random.randint(0, 6, 489)
#     h = np.load(save_path+'patch_results_retrain_15.npz')['h']
#     c = np.load(save_path+'patch_results_retrain_15.npz')['c']
#
#     obs = torch.tensor(obs, dtype=torch.float32)
#     h = torch.tensor(h, dtype=torch.float32).squeeze(1)
#     c = torch.tensor(c, dtype=torch.float32).squeeze(1)
#     acts = torch.tensor(acts, dtype=torch.long)
#
#     if torch.cuda.is_available():
#         model = model.cuda()
#
#     train_set = torch.utils.data.TensorDataset(obs, acts, h, c)
#     n_epoch = 2
#     batch_size = 20
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
#     optimizer = optim.Adam([{'params': model.parameters(), 'weight_decay': 1e-4}], lr=0.01)
#     loss_fn = nn.CrossEntropyLoss()
#     for epoch in range(n_epoch):
#         model.train()
#         test(model)
#         minibatch_iter = tqdm.tqdm(train_loader, desc=f"(Epoch {epoch}) Minibatch")
#         for data, target, h_x, c_x in minibatch_iter:
#             if torch.cuda.is_available():
#                 data, target, h_x, c_x = data.cuda(), target.cuda(), h_x.cuda(), c_x.cuda()
#             optimizer.zero_grad()
#             _, logit, _ = model((data, (h_x, c_x)))
#             prob = F.softmax(logit, dim=-1)
#             loss = loss_fn(prob, target)
#             loss.backward()
#             optimizer.step()
#             minibatch_iter.set_postfix(loss=loss.item())
#
#     model = model.cpu()
#     state_dict = model.state_dict()
#     torch.save({'model': state_dict, }, save_path+'retrained_model.data')
#     return 0
