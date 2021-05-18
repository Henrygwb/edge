import glob
import numpy as np
import tensorflow as tf
import pickle
import copy
import gym
from PIL import Image
from abc import ABC, abstractmethod
from stable_baselines.common.policies import MlpPolicy

def rollout(agent_path, env, env_name, num_traj, agent_type=['zoo','zoo'], norm_path=None, exp_agent_id=1,
            max_ep_len=1e3, save_path=None, render=False, save_obs=False):

    # load agent-0 / agent-1 
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    policy = []
    for i in range(2):
        if agent_type[i] == 'zoo':
           if env_name == 'multicomp/YouShallNotPassHumans-v0':
               policy.append(MlpPolicyValue(scope="policy" + str(i), reuse=False,
                                            ob_space=env.observation_space.spaces[i],
                                            ac_space=env.action_space.spaces[i],
                                            hiddens=[64, 64], normalize=True))
           else:
               policy.append(LSTMPolicy(scope="policy" + str(i), reuse=False,
                                        ob_space=env.observation_space.spaces[i],
                                        ac_space=env.action_space.spaces[i],
                                        hiddens=[128, 128], normalize=True))

        elif agent_type[i] == 'adv':
           policy.append(MlpPolicy(sess, env.observation_space.spaces[i], env.action_space.spaces[i],
                          1, 1, 1, reuse=False))

    sess.run(tf.variables_initializer(tf.global_variables()))
    if norm_path != None:
       obs_rms = load_from_file(norm_path)
    for i in range(2):
        if agent_type[i] == 'zoo':
           param_path = get_zoo_path(env_name, tag=i+1)
           param = load_from_file(param_pkl_path=param_path)
           setFromFlat(policy[i].get_variables(), param)
        else:
           param = load_from_model(agent_path + '/model.pkl')
           adv_agent_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
           setFromFlat(adv_agent_variables, param)

    max_ep_length = 0
    traj_count = 0
    for i in range(num_traj):
        env.seed(i)
        print('Traj %d out of %d.' %(i, num_traj))
        cur_obs, cur_states, cur_acts, cur_rewards, cur_values = [], [], [], [], []
        # reset environment and agent (lstm policy)
        observation = env.reset()
        for id in range(2):
            if agent_type[id] == 'zoo': policy[id].reset()
        
        episode_length, epr, eploss, done = 0, 0, 0, False  # bookkeeping
        while not done and episode_length < max_ep_len:
            episode_length += 1
            actions = []
            values = []
            for id, obs in enumerate(observation):
                if agent_type[id] == 'zoo':
                   act, value = policy[id].act(stochastic=False, observation=obs)
                else:
                   obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                   act = policy[id].step(obs=obs[None, :], deterministic=True)[0][0]
                   value = None
                actions.append(act)
                values.append(value)

            actions = tuple(actions)
            observation, _, done, infos = env.step(actions)
            if render: env.render()
            
            reward = infos[exp_agent_id]['reward_remaining']
            if done: 
               epr = reward
            state = None
            if save_obs:
               state = env.render(mode='rgb_array')
            # save info
            cur_obs.append(state)
            cur_states.append(observation[exp_agent_id])
            cur_acts.append(actions[exp_agent_id])
            cur_rewards.append(reward)
            cur_values.append(values[exp_agent_id])

        print('step %d, reward %.f, value %.4f' %(episode_length, epr, values[exp_agent_id]))

        if epr != 0:
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
            padding_elem_acts = np.zeros_like(elem_acts)
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

            
            final_rewards = rewards[-1].astype('int32')  # get the final reward of each traj.
            if final_rewards == -1000:
                final_rewards = 0
            elif final_rewards == 1000:
                final_rewards = 1
            else:
                final_rewards = 2
                #print('None support final_rewards')
            #print(final_rewards)
            if save_obs:
                np.savez_compressed(save_path + '_traj_' + str(traj_count) + '.npz',
                                    actions=acts, values=values, states=states, rewards=rewards, final_rewards=final_rewards, observations=obs, seed=i)
            else:
                np.savez_compressed(save_path + '_traj_' + str(traj_count) + '.npz',
                                    actions=acts, values=values, states=states, rewards=rewards, final_rewards=final_rewards, seed=i)

            traj_count += 1

    np.save(save_path + '_max_length_.npy', max_ep_length)
    np.save(save_path + '_num_traj_.npy', traj_count)


def load_agent(env_name, agent_type, agent_path):
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    sess = tf.Session(config=tf_config)
    sess.__enter__()

    env = gym.make(env_name)

    policy = []
    for i in range(2):
        if agent_type[i] == 'zoo':
           if env_name == 'multicomp/YouShallNotPassHumans-v0':
               policy.append(MlpPolicyValue(scope="policy" + str(i), reuse=False,
                                            ob_space=env.observation_space.spaces[i],
                                            ac_space=env.action_space.spaces[i],
                                            hiddens=[64, 64], normalize=True))
           else:
               policy.append(LSTMPolicy(scope="policy" + str(i), reuse=False,
                                        ob_space=env.observation_space.spaces[i],
                                        ac_space=env.action_space.spaces[i],
                                        hiddens=[128, 128], normalize=True))

        elif agent_type[i] == 'adv':
           policy.append(MlpPolicy(sess, env.observation_space.spaces[i], env.action_space.spaces[i],
                          1, 1, 1, reuse=False))

    sess.run(tf.variables_initializer(tf.global_variables()))

    for i in range(2):
        if agent_type[i] == 'zoo':
            param_path = get_zoo_path(env_name, tag=i + 1)
            param = load_from_file(param_pkl_path=param_path)
            setFromFlat(policy[i].get_variables(), param)
        else:
            param = load_from_model(agent_path + '/model.pkl')
            adv_agent_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='model')
            setFromFlat(adv_agent_variables, param)
    return policy


def get_zoo_path(env_name, **kwargs):
    if env_name == 'multicomp/YouShallNotPassHumans-v0':
        tag = kwargs.pop('tag', 1)
        return './agent-zoo/you-shall-not-pass/agent%d_parameters-v1.pkl'%tag
    elif env_name == 'multicomp/KickAndDefend-v0':
        tag = kwargs.pop('tag', 1)
        dir_name='kicker'
        if tag is 2:
            dir_name='defender'
        version = kwargs.pop('version', 1)
        return './agent-zoo/kick-and-defend/%s/agent%d_parameters-v%d.pkl'%(dir_name, tag, version)

def rl_fed(env, seed, model, obs_rms, agent_type, original_traj, max_ep_len, importance,
           exp_agent_id=1, render=False, mask_act=False):

    acts_orin = original_traj['actions']
    values_orin = original_traj['values']

    traj_len = np.count_nonzero(values_orin)

    start_step = max_ep_len - traj_len
    env.seed(seed)

    episode_length, epr, done = 0, 0, False  # bookkeeping
    observation = env.reset()

    for id in range(2):
        if agent_type[id] == 'zoo':
            model[id].reset()

    for i in range(traj_len):
        actions = []
        for id, obs in enumerate(observation):
            if agent_type[id] == 'zoo':
                if id != exp_agent_id:
                    # fixed opponent agent
                    act, _ = model[id].act(stochastic=False, observation=obs)
                else:
                    # victim agent we need to explain
                    action = acts_orin[start_step+i]
                    act, _ = model[id].act(stochastic=False, observation=obs)
                    if mask_act:
                        if start_step + i in importance:
                            # add noise into the action
                            act = act + np.random.rand(act.shape[0]) * 2 - 1
#                            act = np.clip(act, env.action_space.spaces[exp_agent_id].low,
#                                        env.action_space.spaces[exp_agent_id].high)
            else:
                obs = np.clip((obs - obs_rms.mean) / np.sqrt(obs_rms.var + 1e-8), -10, 10)
                act = model[id].step(obs=obs[None, :], deterministic=True)[0][0]
            
            actions.append(act)

        actions = tuple(actions)
        observation, _, done, infos = env.step(actions)
        reward = infos[exp_agent_id]['reward_remaining']
        episode_length += 1

        if render: env.render()
        if done: 
            epr = reward
            break
    print('step # {}, reward {:.0f}.'.format(episode_length, epr))
    return epr


def load_from_file(param_pkl_path):

    if param_pkl_path.endswith('.pkl'):
       with open(param_pkl_path, 'rb') as f:
            params = pickle.load(f)
    else:
        params = np.load(param_pkl_path)
    return params


def load_from_model(param_pkl_path):

    if param_pkl_path.endswith('.pkl'):
       with open(param_pkl_path, 'rb') as f:
            params = pickle.load(f)
       policy_param = params[1][0]
       flat_param = []
       for param in policy_param:
           flat_param.append(param.reshape(-1))
       flat_param = np.concatenate(flat_param, axis=0)
    else:
        flat_param = np.load(param_pkl_path, allow_pickle=True)
        if len(flat_param)==3:
            flat_param_1 = []
            for i in flat_param[0]:
                    flat_param_1.append(i)
            flat_param = []
            for param in flat_param_1:
                flat_param.append(param.reshape(-1))
            flat_param = np.concatenate(flat_param, axis=0)
    return flat_param


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

class LSTMPolicy(Policy):
    def __init__(self, scope, *, ob_space, ac_space, hiddens, rate=0.0, n_batch_train=1,
                 n_envs=1, sess=None, reuse=False, normalize=False):
        self.sess = sess
        self.recurrent = True
        self.normalized = normalize
        self.n_envs = n_envs
        with tf.variable_scope(scope, reuse=reuse):
            self.scope = tf.get_variable_scope().name

            assert isinstance(ob_space, gym.spaces.Box)

            self.observation_ph = tf.placeholder(tf.float32, [None, None] + list(ob_space.shape), name="observation")
            self.stochastic_ph = tf.placeholder(tf.bool, (), name="stochastic")
            self.taken_action_ph = tf.placeholder(dtype=tf.float32, shape=[None, None, ac_space.shape[0]], name="taken_action")
            self.dones_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1], name="done_ph")

            if self.normalized:
                if self.normalized != 'ob':
                    self.ret_rms = RunningMeanStd(scope="retfilter")
                self.ob_rms = RunningMeanStd(shape=ob_space.shape, scope="obsfilter")

            obz = self.observation_ph
            if self.normalized:
                obz = tf.clip_by_value((self.observation_ph - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

            last_out = obz

            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)

            self.zero_state = []
            self.state_in_ph = []
            self.state_out = []
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmv_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmv_h"))
            self.initial_state_1 = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2] * (1-self.dones_ph),
                                                                 self.state_in_ph[-1]* (1-self.dones_ph))
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=self.initial_state_1, scope="lstmv")
            self.state_out.append(state_out)

            self.vpredz = tf.contrib.layers.fully_connected(last_out, 1, activation_fn=None)[:, :, 0]
            self.vpred = self.vpredz
            if self.normalized and self.normalized != 'ob':
                self.vpred = self.vpredz * self.ret_rms.std + self.ret_rms.mean  # raw = not standardized

            last_out = obz
            for hidden in hiddens[:-1]:
                last_out = tf.contrib.layers.fully_connected(last_out, hidden)
                self.policy_ff_acts = last_out
                last_out = tf.nn.dropout(last_out, rate=rate)
            cell = tf.contrib.rnn.BasicLSTMCell(hiddens[-1], reuse=reuse)
            size = cell.state_size
            self.zero_state.append(np.zeros(size.c, dtype=np.float32))
            self.zero_state.append(np.zeros(size.h, dtype=np.float32))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.c], name="lstmp_c"))
            self.state_in_ph.append(tf.placeholder(tf.float32, [None, size.h], name="lstmp_h"))
            self.initial_state_1 = tf.contrib.rnn.LSTMStateTuple(self.state_in_ph[-2] * (1-self.dones_ph),
                                                                 self.state_in_ph[-1]* (1-self.dones_ph))
            last_out, state_out = tf.nn.dynamic_rnn(cell, last_out, initial_state=self.initial_state_1, scope="lstmp")
            self.state_out.append(state_out)
            self.mean = tf.contrib.layers.fully_connected(last_out, ac_space.shape[0], activation_fn=None)
            logstd = tf.get_variable(name="logstd", shape=[1, ac_space.shape[0]], initializer=tf.zeros_initializer())

            # self.pd_mean = tf.reshape(self.mean
            if reuse:
                self.pd_mean = tf.reshape(self.mean, (n_batch_train, ac_space.shape[0]))
            else:
                self.pd_mean = tf.reshape(self.mean, (n_envs, ac_space.shape[0]))
            self.pd = DiagonalGaussian(self.pd_mean, logstd)
            self.proba_distribution = self.pd
            self.sampled_action = switch(self.stochastic_ph, self.pd.sample(), self.pd.mode())
            self.neglogp = self.proba_distribution.neglogp(self.sampled_action)
            self.policy_proba = [self.proba_distribution.mean, self.proba_distribution.std]

            self.zero_state = np.array(self.zero_state)
            self.state_in_ph = tuple(self.state_in_ph)
            self.state = self.zero_state
            # add deterministic action
            self.deterministic_action = self.pd.mode()

            for p in self.get_trainable_variables():
                tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, tf.reduce_sum(tf.square(p)))

    def make_feed_dict(self, observation, state_in, taken_action):
        return {
            self.observation_ph: observation,
            self.state_in_ph: list(np.transpose(state_in, (1, 0, 2))),
            self.taken_action_ph: taken_action
        }

    def act(self, observation, stochastic=True, extra_op=None):
        outputs = [self.sampled_action, self.vpred, self.state_out]
        # design for the pre_state
        # notice the zero state
        if extra_op == None:
            a, v, s = tf.get_default_session().run(outputs, {
                self.observation_ph: observation[None, None],
                self.state_in_ph: list(self.state[:, None, :]),
                self.stochastic_ph: stochastic,
                self.dones_ph:np.zeros(self.state[0, None, 0].shape)[:,None]})
        else:
            outputs.append(self.policy_ff_acts)
            a, v, s, ex = tf.get_default_session().run(outputs, {
                self.observation_ph: observation[None, None],
                self.state_in_ph: list(self.state[:, None, :]),
                self.stochastic_ph: stochastic,
                self.dones_ph:np.zeros(self.state[0, None, 0].shape)[:,None]})


        self.state = []
        for x in s:
            self.state.append(x.c[0])
            self.state.append(x.h[0])
        self.state = np.array(self.state)

        # finish checking.
        if extra_op == None:
            return a[0, ],  v[0, 0]
        else:
            return a[0, ], {'vpred': v[0, 0], 'state': self.state}, ex[0, 0, ]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def reset(self):
        self.state = self.zero_state

    @property
    def initial_state(self):
        initial_state_shape = []
        for i in range(4):
            initial_state_shape.append(np.repeat(self.zero_state[i][None,], self.n_envs, axis=0))
        self._initial_state = np.array(initial_state_shape)
        return self._initial_state

    def step(self, obs, state=None, mask=None, deterministic=False):
        stochastic = not deterministic
        if mask is not None:
            mask = np.array(mask)[:, None]
        if self.sess==None:
            action, value, state, neglogp = tf.get_default_session().run([self.sampled_action, self.value_flat, self.state_out, self.neglogp],
                                                                         {self.obs_ph: obs[:, None, :], self.state_in_ph: list(state), self.dones_ph: mask,
                                                                          self.stochastic_ph: stochastic})
        else:
            action, value, state, neglogp = self.sess.run([self.sampled_action, self.value_flat, self.state_out, self.neglogp],
                                                          {self.obs_ph: obs[:, None, :], self.state_in_ph: list(state), self.dones_ph: mask,
                                                           self.stochastic_ph: stochastic})
        value = value[:, 0]
        state_np = []
        for state_tmp in state:
            for state_tmp_1 in state_tmp:
                state_np.append(state_tmp_1)

        return action, value, np.array(state_np), neglogp

    def proba_step(self, obs, state=None, mask=None):
        if mask is not None:
            mask = np.array(mask)[:, None]
        if self.sess==None:
            return tf.get_default_session().run(self.policy_proba, {self.obs_ph: obs, self.state_in_ph: state,
                                                                    self.dones_ph: mask})
        else:
            return self.sess.run(self.policy_proba, {self.obs_ph: obs[:, None, :], self.state_in_ph: list(state),
                                                self.dones_ph: mask})

    def value(self, obs, state=None, mask=None):
        if mask is not None:
            mask = np.array(mask)[:, None]
        if self.sess==None:
            return tf.get_default_session().run(self.value_flat, {self.obs_ph: obs[:, None, :],
                                                                       self.state_in_ph: list(state),
                                                                       self.dones_ph: mask})[:,0]
        else:
            return self.sess.run(self.value_flat, {self.obs_ph: obs[:, None, :],
                                                        self.state_in_ph: list(state),
                                                        self.dones_ph: mask})[:,0]
