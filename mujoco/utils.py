import glob
import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
import pickle
import copy
import gym
from PIL import Image
from scipy.misc import imresize
import torch.nn.functional as F
from torch.autograd import Variable
from abc import ABC, abstractmethod

#prepro = lambda img: imresize(img[35:195].mean(2), (80,80)).astype(np.float32).reshape(1,80,80)/255.

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


def rollout(agent_path, env, num_traj, exp_agent_id=1, max_ep_len=1e3, save_path=None, render=False):

    # load agent-0 / agent-1 
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    policy = []
    for i in range(2):
        policy.append(MlpPolicyValue(scope="policy" + str(i), reuse=False,
                                     ob_space=env.observation_space.spaces[i],
                                     ac_space=env.action_space.spaces[i],
                                     hiddens=[64, 64], normalize=True))

    sess.run(tf.variables_initializer(tf.global_variables()))

    for i in range(2):
        param_path = agent_path + '/agent' + str(i + 1) + '_parameters-v1.pkl'
        param = load_from_file(param_pkl_path=param_path)
        setFromFlat(policy[i].get_variables(), param)

    all_obs, all_acts, all_rewards, all_values = [], [], [], []
    max_ep_length = 0
    traj_count = 3164
    for i in range(num_traj):
        print('Traj %d out of %d.' %(i, num_traj))
        cur_obs, cur_states, cur_acts, cur_rewards, cur_values = [], [], [], [], []
        observation = env.reset()

        episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping
        
        while not done and episode_length < max_ep_len:
            episode_length += 1
            actions = []
            values = []

            for i, obs in enumerate(observation):
                act, value = policy[i].act(stochastic=False, observation=obs)
                actions.append(act)
                values.append(value)

            actions = tuple(actions)
            observation, _, done, infos = env.step(actions)
            reward = infos[exp_agent_id]['reward_remaining']
            if done: epr = reward

            #state = env.render(mode='rgb_array')

            # save info
            #cur_obs.append(state)
            cur_states.append(observation[exp_agent_id])
            
            cur_acts.append(actions[exp_agent_id])
            cur_rewards.append(reward)
            cur_values.append(values[exp_agent_id])

        print('step %d, reward %.f, value %.4f' %(episode_length, epr, values[exp_agent_id]))

        if epr != 0:
            max_ep_length = max(len(cur_rewards), max_ep_length)
            padding_amt = int(max_ep_len - len(cur_acts))

            # elem_obs = cur_obs[-1]
            # padding_elem_obs = np.zeros_like(elem_obs)
            # for _ in range(padding_amt):
            #     cur_obs.insert(0, padding_elem_obs)

            elem_states = cur_states[-1]
            padding_elem_states = np.zeros_like(elem_states)
            for _ in range(padding_amt):
                cur_states.insert(0, padding_elem_states)

            elem_acts = cur_acts[-1]
            padding_elem_acts = np.zeors_like(elem_acts)
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

            #obs = np.array(cur_obs)
            states = np.array(cur_states)
            acts = np.array(cur_acts)
            rewards = np.array(cur_rewards)
            values = np.array(cur_values)

            
            final_rewards = rewards[-1].astype('int32')  # get the final reward of each traj.
            if final_rewards == -1000:
                final_rewards = 0
            elif final_rewards == 1000:
                final_rewards = 1
            else:
                final_rewards = 0
                print('None support final_rewards')
            #print(final_rewards)
            np.savez_compressed(save_path + '_traj_' + str(traj_count) + '.npz', 
                                actions=acts, values=values, states=states, rewards=rewards, final_rewards=final_rewards)
            traj_count += 1

    np.save(save_path + '_max_length_.npy', max_ep_length)
    np.save(save_path + '_num_traj_.npy', traj_count)

def load_from_file(param_pkl_path):

    if param_pkl_path.endswith('.pkl'):
       with open(param_pkl_path, 'rb') as f:
            params = pickle.load(f)
    else:
        params = np.load(param_pkl_path)
    return params

def setFromFlat(var_list, flat_params, sess=None):
    shapes = list(map(lambda x: x.get_shape().as_list(), var_list))
    total_size = np.sum([int(np.prod(shape)) for shape in shapes])
    if total_size != flat_params.shape[0]:
        redundant = flat_params.shape[0] - total_size
        flat_params = flat_params[redundant:]
        assert flat_params.shape[0] == total_size, \
            print('Number of variables does not match when loading pretrained victim agents.')
    theta = tf.placeholder(tf.float32, [total_size])
    start = 0
    assigns = []
    for (shape, v) in zip(shapes, var_list):
        size = int(np.prod(shape))
        assigns.append(tf.assign(v, tf.reshape(theta[start:start + size], shape)))
        start += size
    op = tf.group(*assigns)
    if sess == None:
        tf.get_default_session().run(op, {theta: flat_params})
    else:
        sess.run(op, {theta: flat_params})

class Policy(object):
    def reset(self, **kwargs):
        pass

    def act(self, observation):
        # should return act, info
        raise NotImplementedError()

    @property
    def value_flat(self):
        return self.vpred

    @property
    def obs_ph(self):
        return self.observation_ph

    @abstractmethod
    def step(self, obs, state=None, mask=None):
        """
        Returns the policy for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
        raise NotImplementedError

    @abstractmethod
    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability for a single step
        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
        raise NotImplementedError


class RunningMeanStd(object):
    def __init__(self, scope="running", reuse=False, epsilon=1e-2, shape=()):
        with tf.variable_scope(scope, reuse=reuse):
            self._sum = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(0.0),
                name="sum", trainable=False)
            self._sumsq = tf.get_variable(
                dtype=tf.float32,
                shape=shape,
                initializer=tf.constant_initializer(epsilon),
                name="sumsq", trainable=False)
            self._count = tf.get_variable(
                dtype=tf.float32,
                shape=(),
                initializer=tf.constant_initializer(epsilon),
                name="count", trainable=False)
            self.shape = shape

            self.mean = tf.to_float(self._sum / self._count)
            var_est = tf.to_float(self._sumsq / self._count) - tf.square(self.mean)
            self.std = tf.sqrt(tf.maximum(var_est, 1e-2))


def dense(x, size, name, weight_init=None, bias=True):
    w = tf.get_variable(name + "/w", [x.get_shape()[1], size], initializer=weight_init)
    ret = tf.matmul(x, w)
    if bias:
        b = tf.get_variable(name + "/b", [size], initializer=tf.zeros_initializer())
        return ret + b
    else:
        return ret


def switch(condition, if_exp, else_exp):
    x_shape = copy.copy(if_exp.get_shape())
    x = tf.cond(tf.cast(condition, 'bool'),
                lambda: if_exp,
                lambda: else_exp)
    x.set_shape(x_shape)
    return x


class DiagonalGaussian(object):
    def __init__(self, mean, logstd):
        self.mean = mean
        self.logstd = logstd
        self.std = tf.exp(logstd)

    def sample(self):
        return self.mean + self.std * tf.random_normal(tf.shape(self.mean))

    def mode(self):
        return self.mean

    def neglogp(self, x):
        return 0.5 * tf.reduce_sum(tf.square((x - self.mean) / self.std), axis=-1) \
               + 0.5 * np.log(2.0 * np.pi) * tf.cast(tf.shape(x)[-1], tf.float32) \
               + tf.reduce_sum(self.logstd, axis=-1)

    def kl(self, other):
        return tf.reduce_sum(other.logstd - self.logstd + (tf.square(self.std) + tf.square(self.mean - other.mean)) /
                             (2.0 * tf.square(other.std)) - 0.5, axis=-1)

    def entropy(self):
        return tf.reduce_sum(self.logstd + .5 * np.log(2.0 * np.pi * np.e), axis=-1)

class MlpPolicyValue(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens, rate=0.0, convs=[], n_batch_train=1,
                 sess=None, reuse=False, normalize=False):
        self.sess = sess
        self.recurrent = False
        self.normalized = normalize
        self.zero_state = np.zeros(1)
        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box)

            self.observation_ph = tf.placeholder(tf.float32, [None] + list(ob_space.shape), name="observation")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, ac_space.shape[0]], name="taken_action")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz
            
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i + 1)))

            self.vpredz = dense(last_out, 1, "vffinal")[:, 0]

            self.vpred = self.vpredz
            # reverse normalization. because the reward is normalized, reversing it to see the real value.

            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean
            last_out = obz


            # ff activation policy
            ff_out = []
            for i, hid_size in enumerate(hiddens):
                last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i + 1)))
                ff_out.append(last_out)
                last_out = tf.nn.dropout(last_out, rate=rate)

            self.policy_ff_acts = tf.concat(ff_out, axis=-1)

            mean = dense(last_out, ac_space.shape[0], "polfinal")
            logstd = tf.get_variable(name="logstd", shape=[n_batch_train, ac_space.shape[0]],
                                     initializer=tf.zeros_initializer())

            self.pd = DiagonalGaussian(mean, logstd)
            self.proba_distribution = self.pd
            self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())
            self.neglogp = self.proba_distribution.neglogp(self.sampled_action)
            self.policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]

    def make_feed_dict(self, observation, taken_action):
        return {
            self.observation_ph: observation,
            self.taken_action_ph: taken_action
        }

    def act(self, observation, stochastic=True, extra_op=None):
        outputs = [self.sampled_action, self.vpred]
        if extra_op == None:
            a, v = tf.get_default_session().run(outputs, {
                self.observation_ph: observation[None],
                self.stochastic_ph: stochastic})
            return a[0], v[0]
        else:
            outputs.append(self.policy_ff_acts)
            a, v, ex = tf.get_default_session().run(outputs, {
                self.observation_ph: observation[None],
                self.stochastic_ph: stochastic})
            return a[0], {'vpred': v[0]}, ex[0, ]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    @property
    def initial_state(self):
        return None

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        if self.sess==None:
            action, value, neglogp = tf.get_default_session().run([self.sampled_action, self.value_flat, self.neglogp],
                                                   {self.obs_ph: obs, self.stochastic_ph: stochastic})
        else:
            action, value, neglogp = self.sess.run([self.sampled_action, self.value_flat, self.neglogp],
                                              {self.obs_ph: obs, self.stochastic_ph: stochastic})
        return action, value, self.initial_state, neglogp

    def proba_step(self, obs, state=None, mask=None):
        if self.sess==None:
            return tf.get_default_session().run(self.policy_proba, {self.obs_ph: obs})
        else:
            return self.sess.run(self.policy_proba, {self.obs_ph: obs})

    def value(self, obs, state=None, mask=None):
        if self.sess==None:
            return tf.get_default_session().run(self.value_flat, {self.obs_ph: obs})
        else:
            return self.sess.run(self.value_flat, {self.obs_ph: obs})