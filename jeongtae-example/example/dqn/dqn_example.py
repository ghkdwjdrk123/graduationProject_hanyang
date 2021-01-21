import gym

from stable_baselines import DQN
from stable_baselines.common.evaluation import evaluate_policy

env = gym.make('LunarLander-v2')

model = DQN('MlpPolicy', env, learning_rate=1e-3, prioritized_replay=True, verbose=1)

model.learn(total_timesteps=int(2e5))

model.save("dqn_lunar")

del model

model = DQN.load("dqn_lunar")

mean_reward, n_steps = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

obs = env.reset()

for i in range(1000):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
