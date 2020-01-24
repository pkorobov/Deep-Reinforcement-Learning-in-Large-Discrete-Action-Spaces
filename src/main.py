import gym

from stable_baselines import DDPG
from wolp_agent import WolpertingerAgent
from stable_baselines.ddpg.policies import MlpPolicy
from stable_baselines.common.noise import OrnsteinUhlenbeckActionNoise

import numpy as np

env = gym.make('CartPole-v0')

n_actions = env.action_space.n
param_noise = None
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

policy_kwargs = {}#{"layers": [400, 300]}
model = WolpertingerAgent(MlpPolicy, env, verbose=1, param_noise=param_noise,
                          action_noise=action_noise, policy_kwargs=policy_kwargs,
                          render=True)
model.learn(total_timesteps=1000000)

for episode in range(100):
    obs = env.reset()
    for step in range(1000):
        total_reward = 0
        env.render()
        action, states = model.predict(obs)
        action = np.where(action)[0][0]
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    print(total_reward)