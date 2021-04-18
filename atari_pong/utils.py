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


def rollout(model, env, num_traj, max_ep_len=1e3, render=False):
    all_obs, all_acts, all_rewards, all_values = [], [], [], []
    max_ep_length = 0
    for i in range(num_traj):
        print('Traj %d out of %d.' %(i, num_traj))
        cur_obs, cur_acts, cur_rewards, cur_values = [], [], [], []
        state = torch.tensor(prepro(env.reset()))  # get first state
        episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping
        hx, cx = Variable(torch.zeros(1, 256)), Variable(torch.zeros(1, 256))

        while not done and episode_length <= max_ep_len:
            episode_length += 1
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
            cur_acts.append(action.numpy()[0])
            cur_rewards.append(reward)
            cur_values.append(value.detach().numpy()[0,0])

        print('step # {}, reward {:.0f}, action {:.0f}, value {:.4f}.'.format(episode_length, epr,
                                                                               action.numpy()[0],
                                                                               value.detach().numpy()[0,0]))
        if epr != 0:
            all_obs.append(cur_obs)
            all_acts.append(cur_acts)
            all_rewards.append(cur_rewards)
            all_values.append(cur_values)
            max_ep_length = max(len(cur_rewards), max_ep_length)

    for all_arr in all_obs, all_acts, all_rewards, all_values:
        for arr in all_arr:
            padding_amt = max_ep_length - len(arr)
            elem = arr[-1]
            padding_elem = np.ones_like(elem) * -20
            for _ in range(padding_amt):
                arr.insert(0, padding_elem)

    all_obs = np.array(all_obs)
    all_acts = np.array(all_acts)
    all_rewards = np.array(all_rewards)
    all_values = np.array(all_values)
    history = {'observations': all_obs, 'actions': all_acts, 'rewards': all_rewards, 'values': all_values}

    return history
