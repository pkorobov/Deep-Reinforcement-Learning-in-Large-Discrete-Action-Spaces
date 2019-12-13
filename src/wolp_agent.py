import numpy as np
from ddpg import agent
import knn_search
import gym
import copy

from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise, AdaptiveParamNoiseSpec
from stable_baselines import DDPG

class WolpertingerAgent(DDPG):
    def __init__(self, policy, env, k_ratio=1.0, embeddings=None, gamma=0.99, memory_policy=None, eval_env=None, nb_train_steps=50,
                 nb_rollout_steps=100, nb_eval_steps=100, param_noise=None, action_noise=None,
                 normalize_observations=False, tau=0.001, batch_size=128, param_noise_adaption_interval=50,
                 normalize_returns=False, enable_popart=False, observation_range=(-5., 5.), critic_l2_reg=0.,
                 return_range=(-np.inf, np.inf), actor_lr=1e-4, critic_lr=1e-3, clip_norm=None, reward_scale=1.,
                 render=False, render_eval=False, memory_limit=None, buffer_size=50000, random_exploration=0.0,
                 verbose=0, tensorboard_log=None, _init_setup_model=True, policy_kwargs=None,
                 full_tensorboard_log=False):

            if isinstance(env.action_space, gym.spaces.Discrete):
                n = env.action_space.n
                env_ = copy.deepcopy(env)
                env_.action_space = gym.spaces.Box(np.array([-1.] * n), np.array([1.] * n))

            super(WolpertingerAgent, self).__init__(policy, env_, gamma, memory_policy, eval_env, nb_train_steps,
                 nb_rollout_steps, nb_eval_steps, param_noise, action_noise, normalize_observations, tau,
                 batch_size, param_noise_adaption_interval, normalize_returns, enable_popart, observation_range,
                 critic_l2_reg, return_range, actor_lr, critic_lr, clip_norm, reward_scale,
                 render, render_eval, memory_limit, buffer_size, random_exploration, verbose, tensorboard_log,
                 _init_setup_model, policy_kwargs, full_tensorboard_log)

            self.knn_search = knn_search.KNNSearch(env_.action_space, embeddings)
            self.k = max(1, int(n * k_ratio))

    def _policy(self, obs, apply_noise=True, compute_q=True):
        """
        Get the actions and critic output, from a given observation

        :param obs: ([float] or [int]) the observation
        :param apply_noise: (bool) enable the noise
        :param compute_q: (bool) compute the critic output
        :return: ([float], float) the action and critic value
        """
        obs = np.array(obs).reshape((-1,) + self.observation_space.shape)
        feed_dict = {self.obs_train: obs}
        if self.param_noise is not None and apply_noise:
            actor_tf = self.perturbed_actor_tf
            feed_dict[self.obs_noise] = obs
        else:
            actor_tf = self.actor_tf

        proto_action = self.sess.run(actor_tf, feed_dict=feed_dict)
        proto_action = proto_action.flatten()
        if self.action_noise is not None and apply_noise:
            noise = self.action_noise()
            assert noise.shape == proto_action.shape
            proto_action += noise
        proto_action = np.clip(proto_action, -1, 1)

        actions = self.knn_search.search_point(proto_action, self.k)[0]  # the nearest neighbour actions
        states = np.tile(obs, [len(actions), 1])  # make all the state-action pairs for the critic

        feed_dict = {self.obs_train: states, self.actions: actions}
        q_values = self.sess.run(self.critic_with_actor_tf, feed_dict=feed_dict)

        max_index = np.argmax(q_values)  # find the index of the pair with the maximum value
        action, q_value = actions[max_index], q_values[max_index]
        action = (action + 1) / 2
        action = np.where(action)

        return action, q_value

class WolpertingerPolicy(MlpPolicy):
    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs)

    # акторa смысла трогать вроде нет, критика - есть,
    # но сначала попробую обучать его прямиком на выходе непрерывного актора
    # значит, надо просто засунуть кнн
