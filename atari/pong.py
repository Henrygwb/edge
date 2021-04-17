import torch
import numpy as np
import gym, os, sys
from explainer.DGP_XRL import DGPXRL
from explainer.Rudder_XRL import Rudder
from explainer.RnnAttn_XRL import RnnAttn
from atari.utils import NNPolicy, rollout
from explainer.RnnSaliency_XRL import RnnSaliency

sys.path.append('..')

env_name = 'Pong-v0'
agent_path = 'agents/{}/'.format(env_name.lower())
traj_path = None,

if not traj_path:
    # Load agent, build environment, and play an episode.
    env = gym.make(env_name)
    env.seed(1)

    model = NNPolicy(channels=1, num_actions=env.action_space.n)
    _ = model.try_load(agent_path, checkpoint='*.tar')
    torch.manual_seed(1)

    history = rollout(model, env, num_traj=1000, max_ep_len=200, render=False)

    np.savez_compressed('trajs/'+env_name+'.npz', history)
else:
    history = np.load(traj_path)

Obs = history['Observations']
acts = history['actions']
rewards = history['rewards']
values = history['values']

del history

x_all =

# Explainer 1 - Value function.
sal_value = values
sal_value = (sal_value - np.min(sal_value, axis=1)[:, None])
sal_value = sal_value / (np.max(sal_value, axis=1)[:, None] - np.min(sal_value, axis=1)[:, None])

seq_len =
input_dim =

hiddens = [4]
encoder_type = 'CNN'
rnn_cell_type = 'GRU'

# Explainer 2 - Rudder.
rudder_explainer = Rudder(seq_len, input_dim, hiddens, encoder_type='CNN', dropout_rate=0.25, rnn_cell_type='GRU',
                          normalize=False)


# Explainer 3 - RNN + Saliency.

# Explainer 4 - AttnRNN.

# Explainer 5 - RationaleNet.

# Explainer 6 - DGP.







