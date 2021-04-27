import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import gym, torch
import numpy as np
from atari_pong.utils import rl_fed, NNPolicy, rollout

"""
encoder_type = 'CNN'
rnn_cell_type = 'GRU'
save_path = 'exp_results/'
likelihood_type = 'classification'

# Explainer 1 - Value function.
sal_value = np.load(save_path + 'value_exp.npz')['sal'][0:2120]

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
dgp_1_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_exp.npz')
dgp_1_sal = dgp_1_fid_results['sal']
dgp_1_fid = dgp_1_fid_results['fid']
dgp_1_stab = dgp_1_fid_results['stab']

dgp_2_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_False_0.1_10_8_True_exp.npz')
dgp_2_sal = dgp_2_fid_results['sal']
dgp_2_fid = dgp_2_fid_results['fid']
dgp_2_stab = dgp_2_fid_results['stab']

dgp_3_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_True_0.001_10_8_True_exp.npz')
dgp_3_sal = dgp_3_fid_results['sal']
dgp_3_fid = dgp_3_fid_results['fid']
dgp_3_stab = dgp_3_fid_results['stab']
"""
# Model Fid/Stab figures.

env_name = 'Pong-v0'
max_ep_len = 200

env_name = 'Pong-v0'
agent_path = 'agents/{}/'.format(env_name.lower())
traj_path = 'trajs_test/' + env_name
# traj_path = None
num_traj = 30
max_ep_len = 200

# Load agent, build environment, and play an episode.
env = gym.make(env_name)
model = NNPolicy(channels=1, num_actions=env.action_space.n)
_ = model.try_load(agent_path, checkpoint='*.tar')
torch.manual_seed(1)

# rollout(model, env_name, num_traj=num_traj, max_ep_len=max_ep_len, save_path='trajs_test/'+env_name,render=False)


# for k in range(num_traj):
for k in range(num_traj):
    k=27
    print(k)
    original_traj = np.load('trajs_test/Pong-v0_traj_{}.npz'.format(k))
    replay_orin = []
    print(original_traj['final_rewards'])
    seed = int(original_traj['seed'])
    replay_reward_orin_10 = rl_fed(env_name, seed, original_traj=original_traj, max_ep_len=max_ep_len,
                                   importance=None, num_step=10, render=False, mask_act=False)





# RL fid figures.
env_name = 'Pong-v0'
max_ep_len = 200
traj_path = 'trajs_exp/' + env_name

num_trajs = 1000

# Baseline fidelity
diff_all_10 = np.zeros((5, num_trajs))
diff_all_30 = np.zeros((5, num_trajs))
diff_all_50 = np.zeros((5, num_trajs))
exps_all = [sal_value, rudder_sal, saliency_sal, attn_sal, rat_sal]
for k in range(5):
    print(k)
    importance = exps_all[k]
    for i in range(num_trajs):
        print(i)
        diff_10 = 0
        diff_30 = 0
        diff_50 = 0
        if k == 2:
            importance_traj = np.arange(max_ep_len)
            np.random.shuffle(importance_traj)
        else:
            importance_traj = np.argsort(importance[i,])[::-1]
        original_traj = np.load('trajs_exp/Pong-v0_traj_{}.npz'.format(i))
        replay_orin = []
        replay_orin.append(original_traj['final_rewards'])
        replay_10 = []
        replay_30 = []
        replay_50 = []
        for j in range(2):
            replay_reward_orin_10 = rl_fed(env_name, original_traj=original_traj, max_ep_len=max_ep_len,
                                           importance=importance_traj, num_step=10, render=False, mask_act=False)
            replay_orin.append(replay_reward_orin_10)
        for j in range(2):
            replay_reward_perturbed_10 = rl_fed(env_name, original_traj=original_traj, max_ep_len=max_ep_len,
                                                importance=importance_traj, num_step=10, render=False, mask_act=True)

            replay_10.append(replay_reward_perturbed_10)
        for j in range(2):
            reply_reward_perturbed_30 = rl_fed(env_name, original_traj=original_traj, max_ep_len=max_ep_len,
                                               importance=importance_traj, num_step=30, render=False, mask_act=True)
            replay_30.append(reply_reward_perturbed_30)

        for j in range(2):
            reply_reward_perturbed_50 = rl_fed(env_name, original_traj=original_traj, max_ep_len=max_ep_len,
                                               importance=importance_traj, num_step=50, render=False, mask_act=True)
            replay_50.append(reply_reward_perturbed_50)


        # diff_10 += np.abs(reply_reward_orin_10-reply_reward_perturbed_10)
        # diff_30 += np.abs(reply_reward_orin_30-reply_reward_perturbed_30)
        # diff_50 += np.abs(reply_reward_orin_50-reply_reward_perturbed_50)
        diff_all_10[k, i] = diff_10
        diff_all_30[k, i] = diff_30
        diff_all_50[k, i] = diff_50
np.savez('fid_baselines.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_50=diff_all_50)
print(np.sum(diff_all_10, 1))
print(np.sum(diff_all_30, 1))
print(np.sum(diff_all_50, 1))

# DGP fidelity
diff_all_10 = np.zeros((3, num_trajs))
diff_all_30 = np.zeros((3, num_trajs))
diff_all_50 = np.zeros((3, num_trajs))
exps_all = [dgp_1_sal, dgp_2_sal, dgp_3_sal]
for k in range(3):
    print(k)
    importance = exps_all[k]
    for i in range(num_trajs):
        print(i)
        diff_10 = 0
        diff_30 = 0
        diff_50 = 0
        importance_traj = np.argsort(importance[i,])[::-1]
        original_traj = np.load('trajs_exp/Pong-v0_traj_{}.npz'.format(i))
        for j in range(10):
            reply_reward_orin_10 = rl_fed(env, original_traj=original_traj, max_ep_len=max_ep_len,
                                          importance=importance_traj, num_step=10, render=False, mask_act=False)
            reply_reward_perturbed_10 = rl_fed(env, original_traj=original_traj, max_ep_len=max_ep_len,
                                               importance=importance_traj, num_step=10, render=False, mask_act=True)
            reply_reward_orin_30 = rl_fed(env, original_traj=original_traj, max_ep_len=max_ep_len,
                                          importance=importance_traj, num_step=30, render=False, mask_act=False)
            reply_reward_perturbed_30 = rl_fed(env, original_traj=original_traj, max_ep_len=max_ep_len,
                                               importance=importance_traj, num_step=30, render=False, mask_act=True)
            reply_reward_orin_50 = rl_fed(env, original_traj=original_traj, max_ep_len=max_ep_len,
                                          importance=importance_traj, num_step=50, render=False, mask_act=False)
            reply_reward_perturbed_50 = rl_fed(env, original_traj=original_traj, max_ep_len=max_ep_len,
                                               importance=importance_traj, num_step=50, render=False, mask_act=True)

            diff_10 += np.abs(reply_reward_orin_10-reply_reward_perturbed_10)
            diff_30 += np.abs(reply_reward_orin_30-reply_reward_perturbed_30)
            diff_50 += np.abs(reply_reward_orin_50-reply_reward_perturbed_50)
        diff_all_10[k, i] = diff_10
        diff_all_30[k, i] = diff_30
        diff_all_50[k, i] = diff_50
np.savez('fid_dgp.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_50=diff_all_50)
print(np.sum(diff_all_10, 1))
print(np.sum(diff_all_30, 1))
print(np.sum(diff_all_50, 1))

#
# [5099. 4985. 5162.]
# [5016. 4329. 4950.]
# [4871. 4326. 4590.]
#
# [5111. 5148. 5108. 5086. 5045.]
# [5223. 5156. 5031. 4878. 4868.]
# [5107. 4898. 4774. 4749. 4752.]

a1 = np.load('exp_results/fid_baseline.npz')['diff_10']
b1 = np.load('exp_results/fid_dgp.npz')['diff_10']
c1 = np.vstack((a1, b1))[0:500]

a2 = np.load('exp_results/fid_baseline.npz')['diff_30']
b2 = np.load('exp_results/fid_dgp.npz')['diff_30']
c2 = np.vstack((a2, b2))[0:500]

a3 = np.load('exp_results/fid_baseline.npz')['diff_50']
b3 = np.load('exp_results/fid_dgp.npz')['diff_50']
c3 = np.vstack((a3, b3))[0:500]

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

label_list = []
value_list = []
explainer_list = []
for idx_1, explainer_type in enumerate(['value', 'rudder', 'saliency', 'attention', 'rationale', 'dgp_1', 'dgp_2', 'dgp_3']):
    for idx, metric_type in enumerate(['Top5', 'Top15', 'Top25']):
        for metric in range(c1.shape[1]):
            if idx == 0:
                label_list.append(metric_type)
                explainer_list.append(explainer_type)
                value_list.append(c1[idx_1, metric])
            if idx == 1:
                label_list.append(metric_type)
                explainer_list.append(explainer_type)
                value_list.append(c2[idx_1, metric])
            if idx == 2:
                label_list.append(metric_type)
                explainer_list.append(explainer_type)
                value_list.append(c3[idx_1, metric])


data_pd = pd.DataFrame({'Metric': value_list, 'Label': label_list, 'explainer': explainer_list})
figure = plt.figure(figsize=(20, 6))
ax = sns.boxplot(x="Label", y="Metric", hue="explainer", data=data_pd,
                 hue_order=['rudder', 'saliency', 'attention', 'rationale', 'dgp_1', 'dgp_2', 'dgp_3'])
ax.legend(loc='upper left', bbox_to_anchor=(1, 0.5), prop={'size': 30})
ax.set_ylabel('Metric', fontsize=35)
ax.set_xlabel('')
ax.tick_params(axis='both', which='major', labelsize=35)
pp = PdfPages('fid.pdf')
pp.savefig(figure, bbox_inches='tight')
pp.close()

"""
import numpy as np


importance = [ 38, 193, 190, 110, 182, 187, 188, 184, 178]

def tuncate_importance(importance):
    sorted = np.sort(importance)
    diff = sorted[1:] - sorted[:-1]
    diff_thred = np.percentile(diff, 75)
    idx_selected = np.arange(sorted.shape[0])
    if diff_thred <= 4:
        return sorted
    else:
        for i in range(diff.shape[0]):
            if diff[i] > diff_thred:
                idx_selected = idx_selected[idx_selected>=i+1]
        return sorted[idx_selected]
"""