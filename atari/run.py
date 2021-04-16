import argparse
import sys
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlopen
from urllib.error import HTTPError
from datetime import datetime

import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.preprocessing import get_action_dim, maybe_transpose, preprocess_obs
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecFrameStack, VecNormalize
from stable_baselines3.common.env_util import make_atari_env
import gym
import yaml
import numpy as np

from gym.envs.classic_control import rendering

BASE_MODEL_URL = 'https://github.com/DLR-RM/rl-trained-agents/raw/d81fcd61cef4599564c859297ea68bacf677db6b/ppo'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir', type=str, required=True, help='Log directory path')
    parser.add_argument('--model_dir', type=str, required=False, help='Model directory path')
    # parser.add_argument('--init_seed', type=int, required=False, default=0, help='Random seed')

    parser.add_argument('--game', type=str, required=True, help='Game to run')
    parser.add_argument('--episodes', type=int, required=False, default=2, help='Number of episodes')
    parser.add_argument('--max_timesteps', type=int, required=False, default=-1, help='Max timesteps per episode')

    parser.add_argument('--render_game', action="store_true", help="Render live game during runs")
    parser.add_argument('--use_pretrained_model', action="store_true", help="Render live game during runs")

    args = parser.parse_args()
    
    # Meddle with arguments
    cur_time = str(datetime.now().strftime('%Y-%m-%d-%H:%M:%S'))
    log_path = Path(args.log_dir) / (f'{args.game}-{cur_time}-{args.episodes}-episodes')
    if args.use_pretrained_model:
        model_path = Path(args.model_dir)
        model_path.mkdir(parents=True, exist_ok=True)
    log_path.mkdir(parents=True, exist_ok=True)
    args.log_dir = str(log_path)

    return args

def get_model(args):
    game = args.game
    model_path = Path(args.model_dir)/f'{game}'
    # Download the model to model-dir if doesn't exist
    if not model_path.exists():
        model_path.mkdir()
        print('Downloading pretrained model...')
        zip_url = f'{BASE_MODEL_URL}/{game}_1/{game}.zip'
        args_yaml_url = f'{BASE_MODEL_URL}/{game}_1/{game}/args.yml'
        config_yaml_url = f'{BASE_MODEL_URL}/{game}_1/{game}/config.yml'
        try:
            zipresp = urlopen(zip_url)
            args_yamlresp = urlopen(args_yaml_url)
            config_yamlresp = urlopen(config_yaml_url)
        except HTTPError as err:
            if err.code == 404:
                print(f'tried {zip_url}')
                print('Model file not found. Make sure it exists at https://github.com/DLR-RM/rl-trained-agents/blob/d81fcd61cef4599564c859297ea68bacf677db6b/ppo/')
                exit()
        except Exception as err:
            print(err)
            exit()
        with open(model_path/f'{game}.zip', 'wb') as f:
            f.write(zipresp.read())
        with open(model_path/f'args.yml', 'wb') as f:
            f.write(args_yamlresp.read())
        with open(model_path/f'config.yml', 'wb') as f:
            f.write(config_yamlresp.read())

    env_kwargs = {}
    with open(model_path/f'args.yml', 'r') as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]

    hyperparams = {}
    with open(model_path/f'config.yml', 'r') as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_wrapper"] is not None:
            hyperparams['env_wrapper'] = loaded_args['env_wrapper'][0]
        if loaded_args['frame_stack'] is not None:
            hyperparams['frame_stack'] = loaded_args['frame_stack']
    
    # Check if we are running python 3.8+
    # we need to patch saved model under python 3.6/3.7 to load them
    newer_python_version = sys.version_info.major == 3 and sys.version_info.minor >= 8

    custom_objects = {}
    if newer_python_version:
        custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
        }
    model = PPO.load(model_path/f'{game}.zip', custom_objects=custom_objects)

    env = gym.make(args.game)
    if hyperparams['env_wrapper'] is not None:
        if "AtariWrapper" in hyperparams['env_wrapper']:
            env = make_atari_env(args.game, n_envs=1)
        else:
            print(f'Unknown wrapper {env_wrapper}')
            exit(1)
    if hyperparams['frame_stack'] is not None:
        print(f"Stacking {hyperparams['frame_stack']} frames")
        env = VecFrameStack(env, n_stack=hyperparams['frame_stack'])
    return model, env

def run_agent(args, model=None, env=None):
    if model is None:
        print('Using random agent')
    else:
        print('Using pretrained model')
    if env is None:
        env = gym.make(args.game)

    max_ep_length = 0
    all_obs, all_acts, all_rewards, all_vals, all_rgbs, n_traj = [], [], [], [], [], []
    state = None
    viewer = rendering.SimpleImageViewer()

    for i_episode in range(args.episodes):
        observation = env.reset()
        t = 0
        cur_obs, cur_acts, cur_rewards, cur_vals, cur_rgbs = [], [], [], [], []
        while True:
            t += 1
            if args.render_game:
                env.render()
            if model is None:
                action = env.action_space.sample()
            else:
                action, state = model.predict(observation, state=state, deterministic=True)
            
            val_obs = maybe_transpose(observation, model.observation_space)
            obs_tensor = th.as_tensor(val_obs).to(model.device)
            with th.no_grad():
                value = model.policy.value_net(model.policy._get_latent(obs_tensor)[1])
            rgb_img = env.render(mode='rgb_array')
            observation, reward, done, info = env.step(action)

            cur_rgbs.append(rgb_img)
            cur_obs.append(observation)
            cur_acts.append(action)
            cur_rewards.append(reward)
            cur_vals.append(value.squeeze().numpy())

            if done or (args.max_timesteps > 0 and t >= args.max_timesteps):
                print("Episode {} finished after {} timesteps".format(i_episode,t))
                max_ep_length = max(len(cur_rewards), max_ep_length)
                all_rewards.append(cur_rewards)
                all_acts.append(cur_acts)
                all_obs.append(cur_obs)
                all_vals.append(cur_vals)
                all_rgbs.append(cur_rgbs)
                n_traj.append(t)
                break
    env.close()
    
    # Pad all arrs
    for all_arr in all_obs, all_acts, all_rewards, all_vals, all_rgbs:
        for arr in all_arr:
            padding_amt = max_ep_length - len(arr)
            elem = arr[-1]
            padding_elem = np.zeros_like(elem)
            arr.extend([np.zeros_like(elem) for _ in range(padding_amt)])

    all_obs = np.array(all_obs).squeeze()
    all_acts = np.array(all_acts).squeeze()
    all_rewards = np.array(all_rewards).squeeze()
    all_vals = np.array(all_vals).squeeze()
    all_rgbs = np.array(all_rgbs).squeeze()

    save_dict = {
        'observations': all_obs,
        'actions': all_acts,
        'rewards': all_rewards,
        'values': all_vals,
        'rgb_visualizations': all_rgbs,
        'n_traj': n_traj
    }

    with open(Path(args.log_dir)/'data.npz', 'wb') as f:
        np.savez_compressed(f, observations=all_obs, actions=all_acts, rewards=all_rewards)

if __name__ == '__main__':
    args = get_args()
    if args.use_pretrained_model:
        model, env = get_model(args)
        run_agent(args, model=model, env=env)
    else:
        run_agent(args)
    print(f'Saved logs in {Path(args.log_dir).absolute()}')