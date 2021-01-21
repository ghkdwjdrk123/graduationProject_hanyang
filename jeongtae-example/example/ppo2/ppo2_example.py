import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common import make_vec_env
from stable_baselines import PPO2

env = make_vec_env('Pendulum-v0', n_envs=40)

model = PPO2(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=16000000)
model.save("ppo2_pendulum")

del model

model = PPO2.load("ppo2_pendulum")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
#    env.render(mode = 'rgb_array')

