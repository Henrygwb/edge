from utils import *

from tqdm import trange
import torch
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.env_util import make_vec_env
import gym
import numpy as np

def run_agent(args, model=None, env=None):
    if model is None:
        print('Using random agent')
    else:
        print('Using pretrained model')
    if env is None:
        env = gym.make(args.game)

    max_reward_sum = None
    min_reward_sum = None
    max_ep_length = 0
    all_obs, all_acts, all_rewards, all_vals, all_rgbs, n_traj = [], [], [], [], [], []
    state = None

    for i_episode in trange(args.episodes):
        env.seed(i_episode)
        env.action_space.seed(i_episode)
        observation = env.reset()
        t = 0
        cur_obs, cur_acts, cur_rewards, cur_vals, cur_rgbs = [], [], [], [], []
        done = False
        while not (done or (args.max_timesteps > 0 and t >= args.max_timesteps)):
            t += 1
            if args.render_game:
                env.render()
            if model is None:
                action = [env.action_space.sample()]
                value = torch.zeros(1)
            else:
                action, state = model.predict(observation, state=state, deterministic=True)
                observation_space = model.observation_space
                val_obs = maybe_transpose(observation, observation_space)
                obs_tensor = torch.as_tensor(val_obs).to(model.device)
                with torch.no_grad():
                    value = model.policy.value_net(model.policy._get_latent(obs_tensor)[1])

            if not args.drop_visualizations:
                rgb_img = env.render(mode='rgb_array')
                cur_rgbs.append(rgb_img)

            observation, reward, done, info = env.step(action)

            cur_obs.append(observation)
            cur_acts.append(action)
            cur_rewards.append(reward)
            cur_vals.append(value.squeeze().numpy())

        save_dict = {
            'actions': np.array(cur_acts).squeeze(),
            'rewards': np.array(cur_rewards).squeeze(),
            'observations': np.array(cur_obs).squeeze(),
            'values': np.array(cur_vals).squeeze(),
            'seed': np.array(i_episode),
            'n_traj': np.array(len(cur_acts))
        }
        if len(cur_rgbs) > 0:
            save_dict['rgb_visualizations'] = cur_rgbs
        
        with open(Path(args.log_dir)/f'{i_episode}.npz', 'wb') as f:
            np.savez_compressed(f, **save_dict)
        
        if i_episode == 0:
            max_reward_sum = sum(cur_rewards)
            min_reward_sum = sum(cur_rewards)
        else:
            max_reward_sum = max(sum(cur_rewards), max_reward_sum)
            min_reward_sum = min(sum(cur_rewards), min_reward_sum)
    env.close()

    process_normalized_rewards(max_reward=max_reward_sum, min_reward=min_reward_sum,
                               n_eps=args.episodes, base_path=Path(args.log_dir))

def process_normalized_rewards(max_reward, min_reward, n_eps, base_path):
    for i_episode in range(n_eps):
        path = base_path/f'{i_episode}.npz'
        save_dict = dict(np.load(path))
        reward_sum = save_dict['rewards'].sum()
        normalized_reward = (reward_sum-min_reward)/(max_reward-min_reward)
        save_dict['final_reward'] = np.array(normalized_reward).squeeze()
        np.savez_compressed(path, **save_dict)

if __name__ == '__main__':
    args = get_args()
    if args.use_pretrained_model:
        model, env = get_model(args)
        run_agent(args, model=model, env=env)
    else:
        env = make_vec_env(args.game, n_envs=1)
        run_agent(args, env=env)
    print(f'Saved logs in {Path(args.log_dir).absolute()}')