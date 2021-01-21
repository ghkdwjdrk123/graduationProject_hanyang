import gym
import numpy as np
from stable_baselines import PPO2
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.common.vec_env import VecVideoRecorder

def record_video(env_id, video_length=500, prefix='', video_folder='videos/'):
    eval_env = DummyVecEnv([lambda: gym.make(env_id)])
    eval_env = VecVideoRecorder(eval_env, video_folder=video_folder,
                                record_video_trigger=lambda step: step == 0, video_length=video_length,
                                name_prefix=prefix)
    model = PPO2(MlpPolicy, eval_env, verbose=0)
    obs = eval_env.reset()
    for _ in range(video_length):
        action, _ = model.predict(obs)
        obs, _, _, _  = eval_env.step(action)

    eval_env.close()

record_video('CartPole-v1', video_length=500, prefix='ppo2-cartpole')
