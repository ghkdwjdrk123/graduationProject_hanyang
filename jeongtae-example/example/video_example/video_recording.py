import gym

from stable_baselines import DQN
from stable_baselines.common.vec_env import DummyVecEnv, VecVideoRecorder

def train_DQN_and_save(env_name, saved_model_name, train_time):
    env = gym.make(env_name)
    model = DQN("MlpPolicy", env, verbose=1)
    model.learn(train_time)
    model.save(saved_model_name)

    return model

def main(env_name, train_time_steps, video_length, train_newly = False):
    saved_model_name = "DQN-{}-{}".format(env_name, train_time_steps)

    if train_newly:
        model = train_DQN_and_save(env_name, saved_model_name, train_time_steps)
    else:
        model = DQN.load(saved_model_name)
        model.set_env(gym.make(env_name))


    env = DummyVecEnv([lambda: gym.make(env_name)])
    env = VecVideoRecorder( env, video_folder = 'videos/', 
                            record_video_trigger = lambda step : step == 0, 
                            video_lenth = video_length, 
                            name_prefix = saved_model_name)

    obs = env.reset()
    for _ in range(video_length+1):
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        if done:
            break

    env.close()

if __name__ == "__main__":
    main("CartPole-v1", 80000, 500, True)
