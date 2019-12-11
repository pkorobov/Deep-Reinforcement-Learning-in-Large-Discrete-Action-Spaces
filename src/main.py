#!/usr/bin/python3
import gym

import numpy as np
from wolp_agent import *
from ddpg.agent import DDPGAgent


def run(episodes=10000,
        render=False,
        experiment='InvertedPendulum-v2',
        max_actions=10000,
        knn=0.2):

    env = gym.make(experiment)

    print('------------------------------')
    print(env.observation_space)
    print(env.action_space)
    print('------------------------------')

    steps = env.spec.max_episode_steps
    agent = WolpertingerAgent(env, max_actions=max_actions, k_ratio=knn)
    reward_sum = 0

    for ep in range(episodes):

        observation = env.reset()
        total_reward = 0

        print('Episode ', ep, '/', episodes - 1, 'started...', end='')
        for t in range(steps):

            if render:
                env.render()

            action = agent.act(observation)
            prev_observation = observation
            observation, reward, done, info = env.step(action[0] if len(action) == 1 else action)

            episode = {'obs': prev_observation,
                       'action': action,
                       'reward': reward,
                       'obs2': observation,
                       'done': done,
                       't': t}

            agent.observe(episode)

            total_reward += reward

            if done or (t == steps - 1):
                t += 1
                reward_sum += total_reward
                print('Reward:{} Steps:{} t:{} ({}/step) Cur avg={}'.format(total_reward, t,
                                                                            'time_passed', 'round(time_passed / t)',
                                                                            round(reward_sum / (ep + 1))))
                break
    # end of episodes
    print('Run {} episodes in {} seconds and got {} average reward'.format(
        episodes, 'time / 1000', reward_sum / episodes))

if __name__ == '__main__':
    run(render=True)
