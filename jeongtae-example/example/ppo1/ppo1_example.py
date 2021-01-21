import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

env = gym.make('Pendulum-v0')

model = PPO1(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo1_pendulum")

del model

model = PPO1.load("ppo1_pendulum")

obs = env.reset()

while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
