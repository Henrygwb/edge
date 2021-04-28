import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import gym, torch
import numpy as np
from utils import rl_fed, load_agent, load_from_file
from explainer.quantitative_test import truncate_importance
import gym_compete


encoder_type = 'MLP'
rnn_cell_type = 'GRU'
save_path = 'exp_results/'
likelihood_type = 'classification'

# Explainer 1 - Value function.
sal_value = np.load(save_path + 'value_exp.npz')['sal'][0:2000]

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

# # Explainer 6 - DGP.
# dgp_1_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_exp.npz')
# dgp_1_sal = dgp_1_fid_results['sal']
# dgp_1_fid = dgp_1_fid_results['fid']
# dgp_1_stab = dgp_1_fid_results['stab']
#
# dgp_2_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_False_0.1_10_8_True_exp.npz')
# dgp_2_sal = dgp_2_fid_results['sal']
# dgp_2_fid = dgp_2_fid_results['fid']
# dgp_2_stab = dgp_2_fid_results['stab']
#
# dgp_3_fid_results = np.load(save_path + 'dgp/dgp_classification_GRU_100_False_False_False_False_False_False_True_0.001_10_8_True_exp.npz')
# dgp_3_sal = dgp_3_fid_results['sal']
# dgp_3_fid = dgp_3_fid_results['fid']
# dgp_3_stab = dgp_3_fid_results['stab']

# Model Fid/Stab figures.


# Fid RL.
env_name = 'multicomp/YouShallNotPassHumans-v0'
max_ep_len = 200
agent_path = './agent-zoo/you-shall-not-pass'
model = load_agent(env_name, agent_type=['zoo','zoo'], agent_path=agent_path)
norm_path = agent_path + '/obs_rms.pkl'
obs_rms = load_from_file(norm_path)
num_trajs = 30

# Baseline fidelity
diff_all_10 = np.zeros((5, num_trajs))
diff_all_30 = np.zeros((5, num_trajs))
diff_all_50 = np.zeros((5, num_trajs))

importance_len_10 = np.zeros((5, num_trajs))
importance_len_30 = np.zeros((5, num_trajs))
importance_len_50 = np.zeros((5, num_trajs))

exps_all = [sal_value, rudder_sal, saliency_sal, attn_sal, rat_sal]
for k in range(5):
    print(k)
    importance = exps_all[k]
    for i in range(num_trajs):
        print(i)
        if k == 2:
            importance_traj = np.arange(max_ep_len)
            np.random.shuffle(importance_traj)
        else:
            importance_traj = np.argsort(importance[i,])[::-1]
        importance_traj_10 = truncate_importance(importance_traj, 10)
        importance_traj_30 = truncate_importance(importance_traj, 30)
        importance_traj_50 = truncate_importance(importance_traj, 50)
        original_traj = np.load('trajs_exp/youshallnotpasshumans_v0_traj_{}.npz'.format(i))
        orin_reward = original_traj['final_rewards']
        print(orin_reward)
        seed = int(original_traj['seeds'])

        if orin_reward == 0:
            orin_reward = -1000
        else:
            orin_reward = 1000
        # rl_fed(env_name=env_name, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
        #                                   original_traj=original_traj, max_ep_len=max_ep_len, importance,
        #                                   render=False, mask_act=False)
        replay_reward_10 = rl_fed(env_name=env_name, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_10, 
                                  render=False, mask_act=True)
        replay_reward_30 = rl_fed(env_name=env_name, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_30, 
                                  render=False, mask_act=True)
        replay_reward_50 = rl_fed(env_name=env_name, seed=seed, model=model, obs_rms=obs_rms, agent_type=['zoo','zoo'],
                                  original_traj=original_traj, max_ep_len=max_ep_len, importance=importance_traj_50, 
                                  render=False, mask_act=True)

        diff_all_10[k, i] = np.abs(orin_reward-replay_reward_10)
        diff_all_30[k, i] = np.abs(orin_reward-replay_reward_30)
        diff_all_50[k, i] = np.abs(orin_reward-replay_reward_50)
        importance_len_10[k, i] = len(importance_traj_10)
        importance_len_30[k, i] = len(importance_traj_30)
        importance_len_50[k, i] = len(importance_traj_50)

np.savez('fid_baselines.npz', diff_10=diff_all_10, diff_30=diff_all_30, diff_50=diff_all_50,
         len_10=importance_len_10, len_30=importance_len_30, len_50=importance_len_50)

print(np.sum(diff_all_10, 1))
print(np.sum(diff_all_30, 1))
print(np.sum(diff_all_50, 1))

"""
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

