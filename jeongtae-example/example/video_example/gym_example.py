import gym
from gym import wrappers
from time import time

env_to_wrap = gym.make('CartPole-v0')
env = wrappers.Monitor(env_to_wrap, '~/project/jeongtae-example/example/gym_example', force = True)

env.reset()

for _ in range(1000):
    env.render()
    env.step(env.action_space.sample())

env.close()
env_to_wrap.cloes()
