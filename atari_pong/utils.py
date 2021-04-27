import gym
import glob
import torch
import numpy as np
import torch.nn as nn
from PIL import Image
from scipy.misc import imresize
import torch.nn.functional as F
from torch.autograd import Variable

prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.


class NNPolicy(torch.nn.Module): # an actor-critic neural network
    def __init__(self, channels, num_actions):
        super(NNPolicy, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.lstm = nn.LSTMCell(32 * 5 * 5, 256)
        self.critic_linear, self.actor_linear = nn.Linear(256, 1), nn.Linear(256, num_actions)

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        x = x.view(-1, 32 * 5 * 5)
        hx, cx = self.lstm(x, (hx, cx))
        return self.critic_linear(hx), self.actor_linear(hx), (hx, cx)

    def try_load(self, save_dir, checkpoint='*.tar'):
        paths = glob.glob(save_dir + checkpoint) ; step = 0
        if len(paths) > 0:
            ckpts = [int(s.split('.')[-2]) for s in paths]
            ix = np.argmax(ckpts) ; step = ckpts[ix]
            self.load_state_dict(torch.load(paths[ix]))
        print("\tno saved models") if step is 0 else print("\tloaded model: {}".format(paths[ix]))
        return step


def rollout(model, env_name, num_traj, max_ep_len=1e3, save_path=None, render=False):

    traj_count = 0
    for i in range(num_traj):
        env = gym.make(env_name)
        env.seed(i)
        env.env.frameskip = 3

        print('Traj %d out of %d.' %(i, num_traj))
        cur_obs, cur_states, cur_acts, cur_rewards, cur_values = [], [], [], [], []
        state = torch.tensor(prepro(env.reset()))  # get first state
        episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping
        hx, cx = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))

        while not done and episode_length < max_ep_len:
            value, logit, (hx, cx) = model((Variable(state.view(1, 1, 80, 80)), (hx, cx)))
            hx, cx = Variable(hx.data), Variable(cx.data)
            prob = F.softmax(logit, dim=-1)

            action = prob.max(1)[1].data  # prob.multinomial().data[0] #
            obs, reward, done, expert_policy = env.step(action.numpy()[0])
            if env.env.game == 'pong':
                done = reward
            if render: env.render()
            state = torch.tensor(prepro(obs))
            epr += reward

            # save info!
            cur_obs.append(obs)
            cur_states.append(state.detach().numpy())
            cur_acts.append(action.numpy()[0])
            cur_rewards.append(reward)
            cur_values.append(value.detach().numpy()[0,0])
            episode_length += 1

        print('step # {}, reward {:.0f}, action {:.0f}, value {:.4f}.'.format(episode_length, epr,
                                                                              action.numpy()[0],
                                                                              value.detach().numpy()[0,0]))
        if epr != 0:

            padding_amt = int(max_ep_len - len(cur_obs))

            elem_obs = cur_obs[-1]
            padding_elem_obs = np.zeros_like(elem_obs)
            for _ in range(padding_amt):
                cur_obs.insert(0, padding_elem_obs)

            elem_states = cur_states[-1]
            padding_elem_states = np.zeros_like(elem_states)
            for _ in range(padding_amt):
                cur_states.insert(0, padding_elem_states)

            elem_acts = cur_acts[-1]
            padding_elem_acts = np.ones_like(elem_acts) * -1
            for _ in range(padding_amt):
                cur_acts.insert(0, padding_elem_acts)

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

            acts = acts + 1
            final_rewards = rewards[-1].astype('int32')  # get the final reward of each traj.
            if final_rewards == -1:
                final_rewards = 0
            elif final_rewards == 1:
                final_rewards = 1
            else:
                final_rewards = 0
                print('None support final_rewards')
            print(final_rewards)
            np.savez_compressed(save_path + '_traj_' + str(traj_count) + '.npz', observations=obs,
                                actions=acts, values=values, states=states, rewards=rewards,
                                final_rewards=final_rewards, seed=i)
            traj_count += 1
        env.close()
    np.save(save_path + '_max_length.npy', max_ep_len)
    np.save(save_path + '_num_traj.npy', traj_count)


def rl_fed(env_name, seed, model, original_traj, importance, max_ep_len=1e3, render=False, mask_act=False):

    acts_orin = original_traj['actions']
    traj_len = np.count_nonzero(acts_orin)
    start_step = max_ep_len - traj_len

    env = gym.make(env_name)
    env.seed(seed)
    env.env.frameskip = 3

    episode_length, epr, done = 0, 0, False  # bookkeeping
    obs_0 = env.reset()  # get first state
    state = torch.tensor(prepro(obs_0))
    hx, cx = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))
    act_set = np.array([0, 1, 2, 3, 4, 5])
    for i in range(traj_len):
        # Steps before the important steps reproduce original traj.
        action = acts_orin[start_step+i] - 1
        value, logit, (hx, cx) = model((Variable(state.view(1, 1, 80, 80)), (hx, cx)))
        hx, cx = Variable(hx.data), Variable(cx.data)
        prob = F.softmax(logit, dim=-1)
        action_model = prob.max(1)[1].data.numpy()[0]
        if mask_act:
            # Important steps take suboptimal actions.
            if start_step+i in importance:
                act_set_1 = act_set[act_set!=action_model]
                action = np.random.choice(act_set_1)
            # Steps after the important steps take optimal actions.
            if start_step+1 > importance[-1]:
                action = action_model
        obs, reward, done, expert_policy = env.step(action)
        state = torch.tensor(prepro(obs))
        if render: env.render()
        epr += reward

        # save info!
        episode_length += 1

    print('step # {}, reward {:.0f}.'.format(episode_length, epr))
    return epr
