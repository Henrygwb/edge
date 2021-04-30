import numpy as np
from stable_baselines import PPO2


def rollout(agent_path, env, num_traj, max_ep_len=1e3, save_path=None, render=False, save_obs=False):

    model = PPO2.load(agent_path)
    policy = model.act_model

    max_ep_length = 0
    traj_count = 0

    for i in range(num_traj):
        print('Traj %d out of %d.' %(i, num_traj))
        cur_obs, cur_states, cur_acts, cur_rewards, cur_values = [], [], [], [], []
        env.seed(i)
        env.action_space.seed(i)

        observation = env.reset()

        episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping
        
        while not done and episode_length < max_ep_len:
            episode_length += 1
            action, value, _, _ = policy.step(obs=observation, deterministic=True)
            observation, reward, done, infos = env.step(action)
            if done: 
               assert reward != 0
               epr = reward
            if render: env.render()
            state = None
            if save_obs:
               state = env.render(mode='rgb_array')
            # save info
            cur_obs.append(state)
            cur_states.append(observation[0])
            cur_acts.append(action[0])
            cur_rewards.append(reward[0])
            cur_values.append(value[0])

        if epr != 0:
            print('step %d, reward %.f, value %.4f' % (episode_length, epr, value))
            max_ep_length = max(len(cur_rewards), max_ep_length)

            padding_amt = int(max_ep_len - len(cur_acts))
            elem_obs = cur_obs[-1]
            padding_elem_obs = np.zeros_like(elem_obs)
            for _ in range(padding_amt):
                cur_obs.insert(0, padding_elem_obs)
            elem_states = cur_states[-1]
            padding_elem_states = np.zeros_like(elem_states)
            for _ in range(padding_amt):
                cur_states.insert(0, padding_elem_states)
            assert len(cur_states) == max_ep_len
            
            elem_acts = cur_acts[-1]
            padding_elem_acts = np.ones_like(elem_acts) * -1
            for _ in range(padding_amt):
                cur_acts.insert(0, padding_elem_acts)
            assert len(cur_acts) == max_ep_len

            elem_rewards = cur_rewards[-1]
            padding_elem_rewards = np.zeros_like(elem_rewards)
            for _ in range(padding_amt):
                cur_rewards.insert(0, padding_elem_rewards)

            elem_values = cur_values[-1]
            padding_elem_values = np.zeros_like(elem_values)
            for _ in range(padding_amt):
                cur_values.insert(0, padding_elem_values)

            obs = np.array(cur_obs)
            states = np.array(cur_states)
            acts = np.array(cur_acts)
            rewards = np.array(cur_rewards)
            values = np.array(cur_values)
            
            acts += 1
            # discounted_reward / sum_reward / last reward
            gamma = 0.99
            discounted_rewards = 0
            sum_rewards = 0
            factor = 1.0
            for id in range(padding_amt, len(cur_rewards)):
                sum_rewards += cur_rewards[id]
                discounted_rewards += factor * cur_rewards[id]
                factor *= gamma 
            final_rewards = rewards[-1].astype('float32')  # get the final reward of each traj.

            if not save_obs:
                #print(final_rewards)
                np.savez_compressed(save_path + '_traj_' + str(traj_count) + '.npz',
                                    actions=acts, values=values, states=states, rewards=rewards, final_rewards=final_rewards,
                                    discounted_rewards=discounted_rewards, sum_rewards=sum_rewards, seeds=i)
            else:
                np.savez_compressed(save_path + '_traj_' + str(traj_count) + '.npz', observations=obs,
                                    actions=acts, values=values, states=states, rewards=rewards, final_rewards=final_rewards,
                                    discounted_rewards=discounted_rewards, sum_rewards=sum_rewards, seeds=i)

            traj_count += 1
    np.save(save_path + '_max_length_.npy', max_ep_length)
    np.save(save_path + '_num_traj_.npy', traj_count)


def rl_fed(env, seed, model, original_traj, importance, max_ep_len=1e3, render=False, mask_act=False):

    acts_orin = original_traj['actions']
    traj_len = np.count_nonzero(acts_orin)
    start_step = max_ep_len - traj_len

    env.seed(seed)
    env.action_space.seed(seed)

    episode_length, epr, done = 0, 0, False  # bookkeeping
    observation = env.reset()

    act_set = np.array([0, 1])
    for i in range(traj_len):
        action = acts_orin[start_step+i] - 1
        act, _, _, _ = policy.step(obs=observation, deterministic=True)
        if mask_act:
            if start_step+i in importance:
                act_set_1 = act_set[act_set != act]
                act = np.random.choice(act_set_1)
        observation, reward, done, infos = env.step(act)
        if render: env.render()
        epr += reward
        # save info!
        episode_length += 1
        if done:
            assert reward != 0
            break

    print('step # {}, reward {:.0f}.'.format(episode_length, epr))
    return epr
