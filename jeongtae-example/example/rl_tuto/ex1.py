import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv

env = gym.make('CartPole-v1')

#For vectorized enviroments. I guess DummyVecEnv gets function as elements.
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=0)

def evaluate(model, num_episodes=100):
    env = model.get_env()
    all_episode_rewards = []
    for i in range(num_episodes):
        episode_rewards = []
        done = False
        obs = env.reset()
        while not done:
            action, _states = model.predict(obs)
            obs, reward, done, info = env.step(action)
            episode_rewards.append(reward)

        all_episode_rewards.append(sum(episode_rewards))

    mean_episode_reward = np.mean(all_episode_rewards)
    print("Mean reward:", mean_episode_reward, "Num episodes:", num_episodes)

    return mean_episode_reward

mean_reward_before_train = evaluate(model, num_episodes=100)

model.learn(total_timesteps = 10000)

mean_reward = evaluate(model, num_episodes=100)
