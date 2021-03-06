import os, sys
sys.path.append('..')
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import gym, argparse
import timeit
import gym_compete
from mujoco.utils import rollout
from mujoco.render_mujoco import Render_mujoco

# Setup env, load the target agent, and collect the trajectories.
env_name = 'multicomp/YouShallNotPassHumans-v0'
agent_path = './agent-zoo/you-shall-not-pass'
traj_path = 'trajs/' + env_name.split('/')[1]
norm_path = agent_path + '/obs_rms.pkl'
#traj_path = 'trajs/Pong-v0.npz'
num_traj = 30000
max_ep_len = 200
resolution = (533, 300)
render = False

# Load agent, build environment, and play an episode.
env = gym.make(env_name)
if render:
   env = Render_mujoco(env, env_name, None, resolution, 'default')
env.seed(1)
rollout(agent_path, env, num_traj=num_traj, norm_path=norm_path, agent_type=['adv','zoo'],
        max_ep_len=200, save_path=traj_path, render=render)
