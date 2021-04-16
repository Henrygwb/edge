from utils import *

import torch as th
from stable_baselines3.common.preprocessing import maybe_transpose
import gym
import numpy as np

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

    for i_episode in range(args.episodes):
        observation = env.reset()
        t = 0
        cur_obs, cur_acts, cur_rewards, cur_vals, cur_rgbs = [], [], [], [], []
        while True:
            t += 1
            if args.render_game:
                env.render()
            if model is None:
                action = [env.action_space.sample()]
                value = th.zeros(1)
            else:
                action, state = model.predict(observation, state=state, deterministic=True)
                observation_space = model.observation_space
                val_obs = maybe_transpose(observation, observation_space)
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
                print("Episode {} finished after {} timesteps.".format(i_episode,t,done))
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
        env = make_atari_env(args.game, n_envs=1)
        env = VecFrameStack(env, n_stack=4)
        run_agent(args, env=env)
    print(f'Saved logs in {Path(args.log_dir).absolute()}')