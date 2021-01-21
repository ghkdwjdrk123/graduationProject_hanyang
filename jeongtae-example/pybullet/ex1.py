import os

import gym
import pybullet_envs
from gym.wrappers import Monitor

from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines import PPO2

env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])

env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)


model = PPO2('MlpPolicy', env)
model.learn(total_timesteps=2000)

log_dir = "/tmp/"
model.save(log_dir + "ppo_halfcheetah")
stats_path = os.path.join(log_dir, "vec_normalize.pkl")
env.save(stats_path)

del model, env

model = PPO2.load(log_dir + "ppo_halfcheetah")

env = DummyVecEnv([lambda: gym.make("HalfCheetahBulletEnv-v0")])
env = VecNormalize.load(stats_path, env)

env.training = False

env.norm_reward = False

env.render()
obs = env.reset()

n = 0

while n < 500000:
    env.render()
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        break
    n += 1

