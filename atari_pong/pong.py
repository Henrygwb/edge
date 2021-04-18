import torch
import numpy as np
import gym, os, sys, argparse
from scipy.misc import imresize
# from explainer.DGP_XRL import DGPXRL
from explainer.Rudder_XRL import Rudder
from explainer.RnnAttn_XRL import RnnAttn
from explainer.RnnSaliency_XRL import RnnSaliency
from explainer.RationaleNet_XRL import RationaleNet
from atari_pong.utils import NNPolicy, rollout
sys.path.append('..')

parser = argparse.ArgumentParser()
parser.add_argument("--explainer", type=str, default='rationale')

args = parser.parse_args()

hiddens = [4]
encoder_type = 'CNN'
rnn_cell_type = 'GRU'
n_epoch = 2
save_path = 'exp_model_results/'


def obs_resize(obs):
    obs = obs[:, :, 35:195].mean(-1)
    obs_new = np.zeros((obs.shape[0], obs.shape[1], 80, 80))
    for i in range(obs.shape[0]):
        for j in range(obs.shape[1]):
            obs_new[i, j] = imresize(obs[i, j], (80, 80)).astype(np.float32)/255.
    del obs
    return obs_new


# Setup env, load the target agent, and collect the trajectories.
env_name = 'Pong-v0'
agent_path = 'agents/{}/'.format(env_name.lower())
traj_path = 'trajs/Pong-v0.npz'
# traj_path = None
num_traj = 15

if traj_path is None:
    # Load agent, build environment, and play an episode.
    env = gym.make(env_name)
    env.seed(1)

    model = NNPolicy(channels=1, num_actions=env.action_space.n)
    _ = model.try_load(agent_path, checkpoint='*.tar')
    torch.manual_seed(1)

    history = rollout(model, env, num_traj=num_traj, max_ep_len=200, render=False)

    np.savez_compressed('trajs/'+env_name+'.npz', observations=history['observations'], actions=history['actions']
                        , rewards=history['rewards'], values=history['values'])
else:
    history = np.load(traj_path)

obs = history['observations']
acts = history['actions']
rewards = history['rewards']
values = history['values']

del history

# Preprocess the trajs: change the padding -1 back to 0, resize obs to state, extract the final reward.
acts[acts==-20] = -1
acts = acts + 1
traj_lens = np.count_nonzero(acts, axis=1)
obs[obs==-20] = 0
obs = obs_resize(obs)
rewards[rewards==-20] = 0
final_rewards = rewards[:, -1].astype('int32') # get the final reward of each traj.
final_rewards[final_rewards==-1] = 0
final_rewards[final_rewards==1] = 1

# Get the shared parameters, prepare training/testing data.
num_class = len(np.unique(final_rewards))
seq_len = values.shape[1]
input_dim = obs.shape[3]
n_action = np.max(acts) + 1
train_size = int(obs.shape[0]*0.8)

obs = torch.tensor(obs[..., None, :, :], dtype=torch.float32)
acts = torch.tensor(acts, dtype=torch.long)
rewards = torch.tensor(rewards, dtype=torch.float32)
final_rewards = torch.tensor(final_rewards, dtype=torch.long)

train_set = torch.utils.data.TensorDataset(obs[0:train_size, ...], acts[0:train_size, ...], final_rewards[0:train_size, ...])
test_set = torch.utils.data.TensorDataset(obs[train_size:, ...], acts[train_size:, ...], final_rewards[train_size:, ...])
train_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=2, shuffle=False)

if args.explainer == 'value':
    # Explainer 1 - Value function.
    sal_value = (values - np.min(values, axis=1)[:, None]) / \
                (np.max(values, axis=1)[:, None] - np.min(values, axis=1)[:, None])
    np.savez_compressed(save_path+'value_exp.npz', sal=sal_value)

elif args.explainer == 'rudder':
    # Explainer 2 - Rudder.
    rudder_explainer = Rudder(seq_len=seq_len, input_dim=input_dim, hiddens=hiddens, n_action=n_action,
                              encoder_type=encoder_type)
    rudder_explainer.train(train_loader, n_epoch, save_path=save_path+'rudder_model.data')
    rudder_explainer.test(test_loader)
    rudder_explainer.load(save_path+'rudder_model.data')
    rudder_explainer.test(test_loader)
    sal_rudder = rudder_explainer.get_explanations(obs, acts, final_rewards)
    np.savez_compressed(save_path+'rudder_exp.npz', sal=sal_rudder)

elif args.explainer == 'saliency':
    # Explainer 3 - RNN + Saliency.
    saliency_explainer = RnnSaliency(seq_len, input_dim, 'classification', hiddens, n_action, encoder_type=encoder_type,
                                     num_class=2, rnn_cell_type='GRU', use_input_attention=True, normalize=False)
    saliency_explainer.train(train_loader, n_epoch, save_path=save_path+'saliency_model.data')
    saliency_explainer.test(test_loader)
    saliency_explainer.load(save_path+'saliency_model.data')
    saliency_explainer.test(test_loader)
    for back2rnn in [False, True]:
        sal_g = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='gradient',
                                                    back2rnn=back2rnn)
        sal_ig = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='integrated_gradient',
                                                     back2rnn=back2rnn)
        sal_ig_unified = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='unifintgrad',
                                                             back2rnn=back2rnn)
        sal_sg = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='smoothgrad',
                                                     back2rnn=back2rnn)
        sal_sg_exp = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='expgrad',
                                                         back2rnn=back2rnn)
        sal_sg_var = saliency_explainer.get_explanations(obs, acts, final_rewards, saliency_method='vargrad',
                                                         back2rnn=back2rnn)
        if back2rnn:
            np.savez_compressed(save_path+'saliency_exp_rnn_layer.npz', sal_g=sal_g, sal_ig=sal_ig,
                                sal_ig_unified=sal_ig_unified, sal_sg=sal_sg, sal_sg_exp=sal_sg_exp,
                                sal_sg_var=sal_sg_var)
        else:
            np.savez_compressed(save_path+'saliency_exp_input_layer.npz', sal_g=sal_g, sal_ig=sal_ig,
                                sal_ig_unified=sal_ig_unified, sal_sg=sal_sg, sal_sg_exp=sal_sg_exp,
                                sal_sg_var=sal_sg_var)

elif args.explainer == 'attention':
    # Explainer 4 - AttnRNN.
    attention_explainer = RnnAttn(seq_len, input_dim, 'classification', hiddens, n_action, encoder_type=encoder_type,
                                  num_class=2, attention_type='tanh', normalize=False)
    attention_explainer.train(train_loader, n_epoch, save_path=save_path+'attention_model.data')
    attention_explainer.test(test_loader)
    attention_explainer.load(save_path+'attention_model.data')
    attention_explainer.test(test_loader)
    sal_attention = attention_explainer.get_explanations(obs, acts, final_rewards)
    np.savez_compressed(save_path+'attention_exp.npz', sal=sal_attention)

elif args.explainer == 'rationale':
    # Explainer 5 - RationaleNet.
    rationale_explainer = RationaleNet(seq_len, input_dim, 'regression', hiddens, n_action, encoder_type=encoder_type,
                                       num_class=2, normalize=False)
    rationale_explainer.train(train_loader, n_epoch, save_path=save_path+'rationale_model.data')
    rationale_explainer.test(test_loader)
    rationale_explainer.load(save_path+'rationale_model.data')
    rationale_explainer.test(test_loader)
    sal_rationale = rationale_explainer.get_explanations(obs, acts, final_rewards)
    np.savez_compressed(save_path+'rationale_exp.npz', sal=sal_rationale)

# Explainer 6 - DGP.







