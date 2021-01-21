import gym
import sys

env = gym.make('Pendulum-v0')
for i in range(20):
    observation = env.reset()
    for t in range(20):
        env.render()

        print('observation before action:')
        print(observation)

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        print('observation after action:')
        print(observation)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
