import numpy as np
from stable_baselines.common import make_vec_env
from cartpole.utils import rollout

env_name = 'CartPole-v1'
agent_path = 'ppo2_cartpole'
traj_path = 'trajs/' + env_name.split('/')[1]
#traj_path = 'trajs/Pong-v0.npz'
num_traj = 30000
max_ep_len = 200

# Load agent, build environment, and play an episode.
env = make_vec_env(env_name, n_envs=1)
#env = Render_mujoco(env, env_name, None, resolution, 'default')
env.seed(1)
rollout(agent_path, env, num_traj=num_traj, max_ep_len=200, save_path=traj_path, render=False)
