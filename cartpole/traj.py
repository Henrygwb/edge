import numpy as np
from stable_baselines.common import make_vec_env
from utils import rollout
from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO2

env_name = 'CartPole-v1'
agent_path = 'agents/ppo2_cartpole.zip'
traj_path = 'test_trajs/' + env_name
num_traj = 300
max_ep_len = 200

# Load agent, build environment, and play an episode.
env = make_vec_env(env_name, n_envs=1)
env.seed(1)
rollout(agent_path, env, num_traj=num_traj, max_ep_len=200, save_path=traj_path, render=False)


# env = make_vec_env('CartPole-v1', n_envs=4)
#
# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000)
# model.save("ppo2_cartpole")
