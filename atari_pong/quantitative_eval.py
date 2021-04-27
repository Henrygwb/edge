import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import gym
import numpy as np
from atari_pong.utils import rl_fed

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

# Model Fid/Stab figures.


# RL fid figures.
env_name = 'Pong-v0'
max_ep_len = 200
traj_path = 'trajs_exp/' + env_name

env = gym.make(env_name)
env.seed(1)
env.env.frameskip = 3
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
