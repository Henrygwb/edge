import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import torch
import gym, argparse
from pong.utils import NNPolicy, rollout

# Setup env, load the target agent, and collect the trajectories.
env_name = 'Pong-v0'
agent_path = 'agents/{}/'.format(env_name.lower())
traj_path = 'trajs_test/' + env_name
max_ep_len = 200
num_traj = 50000

# Load agent, build environment, and play an episode.
env = gym.make(env_name)
env.seed(1)

model = NNPolicy(channels=1, num_actions=env.action_space.n)
_ = model.try_load(agent_path, checkpoint='*.tar')
torch.manual_seed(1)

rollout(model, env, num_traj=num_traj, max_ep_len=max_ep_len, save_path='trajs/'+env_name,render=False)
