import numpy as np
import pyflann
from gym.spaces import Box
from ddpg import agent
import knn_search
import gym
import copy

class WolpertingerAgent(agent.DDPGAgent):

    def __init__(self, env, max_actions=1e6, k_ratio=0.1, embeddings=None):

        if isinstance(env.action_space, gym.spaces.Discrete):
            n = env.action_space.n
            env_ = copy.deepcopy(env)
            env_.action_space = gym.spaces.Box(np.array([0.] * n), np.array([1.] * n))

        super().__init__(env_)
        self.knn_search = knn_search.KNNSearch(env_.action_space, embeddings)
        self.k = max(1, int(max_actions * k_ratio))

    def get_action_space(self):
        return self.knn_search

    def act(self, state):
        proto_action = super().act(state)
        action = self.wolp_action(state, proto_action)
        return action

    def wolp_action(self, state, proto_action):
        # get the proto_action's k nearest neighbors
        actions = self.knn_search.search_point(proto_action, self.k)[0]
        # make all the state, action pairs for the critic
        states = np.tile(state, [len(actions), 1])
        # evaluate each pair through the critic
        actions_evaluation = self.critic_net.evaluate_critic(states, actions)
        # find the index of the pair with the maximum value
        max_index = np.argmax(actions_evaluation)
        # return the best action
        return actions[max_index]
