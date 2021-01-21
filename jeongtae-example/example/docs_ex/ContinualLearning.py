from stable_baselines.common.cmd_util import make_atari_env
from stable_baselines import PPO2

env = make_atari_env('DemonAttackNoFrameskip-v4', num_env=8, seed=0)

model = PPO2('CnnPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode = 'rgb_array')

env.close()

env = make_atari_env('SpaceInvadersNoFrameskip-v4', num_env=8, seed=0)

model.set_env(env)
model.learn(total_timesteps=10000)

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render(mode = 'rgb_array')
env.close()
