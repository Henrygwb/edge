import argparse
from pathlib import Path
from datetime import datetime
import sys
from zipfile import ZipFile
from urllib.request import urlopen
from urllib.error import HTTPError

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import gym
import yaml

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
    parser.add_argument('--drop_visualizations', action="store_true", help="Do not save rgb game frames. Saves a lot of data space.")

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
            print(f'Unknown wrapper {hyperparams["env_wrapper"]}')
            exit(1)
    if hyperparams['frame_stack'] is not None:
        print(f"Stacking {hyperparams['frame_stack']} frames")
        env = VecFrameStack(env, n_stack=hyperparams['frame_stack'])
    return model, env