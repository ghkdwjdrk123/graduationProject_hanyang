import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines import PPO1

env = gym.make('LunarLander-v2')

model = PPO1(MlpPolicy, env, verbose=1)
model.learn(total_timesteps=25000)
model.save("ppo1_lunarlander")

del model

model = PPO1.load("ppo1_lunarlander")

obs = env.reset()

#If it is alive during (goal) rewards, I think it clear CartPole-v1.
goal = 500
flag = False

# render() is just playing display.
while True:
    for i in range(goal):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()
        if dones:
            env.reset()
            break
        if i == goal-1:
            print("CLEAR!!")
            flag = True
            env.close()
    if flag:
        break
