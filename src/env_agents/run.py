import argparse
from pathlib import Path
from zipfile import ZipFile
from urllib.request import urlopen
from urllib.error import HTTPError
from datetime import datetime
import yaml

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gym
import numpy as np

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
    log_path = Path(args.log_dir) / (f'{args.game}-{cur_time}')
    # model_path = Path(args.model_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    # model_path.mkdir(parents=True, exist_ok=True)
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
        yaml_url = f'{BASE_MODEL_URL}/{game}_1/{game}/args.yml'
        try:
            zipresp = urlopen(zip_url)
            yamlresp = urlopen(yaml_url)
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
            f.write(yamlresp.read())

    env_kwargs = {}
    with open(model_path/f'args.yml', 'r') as f:
        loaded_args = yaml.load(f, Loader=yaml.UnsafeLoader)  # pytype: disable=module-attr
        if loaded_args["env_kwargs"] is not None:
                env_kwargs = loaded_args["env_kwargs"]
    env = env = make_vec_env(game)
    model = PPO.load(model_path/f'{game}.zip', env=env)


def run_agent(args, model=None):
    env = gym.make(args.game)
    max_ep_length = 0
    all_obs, all_acts, all_rewards = [], [], []
    for i_episode in range(args.episodes):
        observation = env.reset()
        t = 0
        cur_obs, cur_acts, cur_rewards = [], [], []
        while True:
            t += 1
            if args.render_game:
                env.render()
            if model is None:
                action = env.action_space.sample()
            observation, reward, done, info = env.step(action)

            cur_obs.append(observation)
            cur_acts.append(action)
            cur_rewards.append(reward)

            if done or (args.max_timesteps > 0 and t >= args.max_timesteps):
                print("Episode {} finished after {} timesteps".format(i_episode,t))
                max_ep_length = max(len(cur_rewards), max_ep_length)
                all_rewards.append(cur_rewards)
                all_acts.append(cur_acts)
                all_obs.append(cur_obs)
                break
    env.close()
    
    # Pad all arrs
    for all_arr in all_obs, all_acts, all_rewards:
        for arr in all_arr:
            padding_amt = max_ep_length - len(arr)
            elem = arr[-1]
            padding_elem = np.zeros_like(elem)
            arr.extend([np.zeros_like(elem) for _ in range(padding_amt)])

    all_obs = np.array(all_obs)
    all_acts = np.array(all_acts)
    all_rewards = np.array(all_rewards)

    save_dict = {
        'observations': all_obs,
        'actions': all_acts,
        'rewards': all_rewards,
    }

    with open(Path(args.log_dir)/'data.npz', 'wb') as f:
        np.savez_compressed(f, observations=all_obs, actions=all_acts, rewards=all_rewards)

if __name__ == '__main__':
    args = get_args()
    if args.use_pretrained_model:
        model = get_model(args)
    run_agent(args)
    print(f'Saved logs in {Path(args.log_dir).absolute()}')