import ex1
from stable_baselines import DQN

kwargs = {'double_q': True, 'prioritized_replay': True, 'policy_kwargs': dict(dueling=True)}

dqn_model = DQN('MlpPolicy', 'CartPole-v1', verbose=1, **kwargs)

mean_reward_before_train = ex1.evaluate(dqn_model, num_episodes=100)

dqn_model.learn(total_timesteps=10000, log_interval=10)

mean_reward = ex1.evaluate(dqn_model, num_episodes=100)
