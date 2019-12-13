import gym
import numpy as np
from wolp_agent import *
from ddpg.agent import DDPGAgent
import copy


def run(episodes=10000,
        render=False,
        experiment='CartPole-v0',
        max_actions=10000,
        knn=1.0):

    env = gym.make(experiment)

    print('------------------------------')
    print(env.observation_space)
    print(env.action_space)
    print('------------------------------')

    steps = env.spec.max_episode_steps
    agent = WolpertingerAgent(env, max_actions=env.action_space.n, k_ratio=knn)
    reward_sum = 0

    for ep in range(episodes):

        observation = env.reset()
        total_reward = 0

        print('Episode ', ep, '/', episodes - 1, 'started...', end='')
        for t in range(steps):
            if render:
                env.render()

            action = agent.act(observation)
            action_ = np.where(action)[0]

            prev_observation = observation
            observation, reward, done, info = env.step(action_[0] if len(action_) == 1 else action_)

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
                print('Reward:{} Steps:{} Cur avg={}'.format(total_reward, t, round(reward_sum / (ep + 1))))
                break
    print('Run {} episodes and got {} average reward'.format(episodes, reward_sum / episodes))


if __name__ == '__main__':
    run(render=True)
