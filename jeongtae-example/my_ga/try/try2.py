#   This is code of gym_ga + list_seed. Referenced by https://github.com/MarcoSelvatici/Genetic-Algorithm.git
# 어떻게 policy를 바꿔서 elitism을 넣지? 혹은 이미 들어있나?


import numpy as np
import random
import gym
from gym import wrappers

#   Plot the results
import plotly
import plotly.graph_objs as go

#   Globals
n_generations = 0
plot_data = []
final_games = 10
score_requirement = 50
population_size = 100
generation_limit = 100  #   Max number of generations
steps_limit = 300       #   Max number of steps in a game
sigma = 0.1             #   Noise standard deviation
alpha = 0.0005          #   Learning rate

RNG_SEED = 8
NULL_ACTION = 0

class SharedNoiseTable(object):
    def __init__(self):
        import ctypes, multiprocessing
        seed = 123
        count = 250000000
        self._shared_mem = multiprocessing.Array(ctypes.c_float, count)
        self.noise = np.ctypeslib.as_array(self._shared_mem.get_obj())
        assert self.noise.dtype == np.float32
        self.noise[:] = np.random.RandomState(seed).randn(count)
   
    def get(self, i, dim):
        return self.noise[i:i+dim]

    def sample_index(self, stream, dim):
        return stream.rendint(0, len(self.noise)-dim+1)



def create_plot():
    global plot_data
    global n_generations
    trace = go.Scatter(
             x = np.linspace(0, 1, n_generations),
            y = plot_data,
            mode = 'lines+markers',
            fill = 'tozeroy'
    )
    data = [trace]
    plotly.offline.plot({"data" : data, "layout" : go.Layout(title = "LunarLander_ga")}, filename = "LunarLander_ga_plot")

def genetic_algorithm():
    env = gym.make("LunarLander-v2")
    env.reset()
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n
    print(env.action_space)

    env.seed(RNG_SEED)
#    np.random.seed(RNG_SEED)

    global n_generations
    #   Initial weights
    W = np.zeros((input_size, output_size))

    #   seed list를 random.randint를 통해 제작.
    noise = SharedNoiseTable()
    list_seed = []
    for i in range(generation_limit):
        list_seed.append(random.randint(0, 1000))

    for gen in range(generation_limit):
        #   Keep track of Returns
        R = np.zeros(population_size)

        #   noise
        #   해당 seed에서부터 population_size * input_size * output_size만큼을 읽어와서 N에 저장.
        N = np.empty((population_size, input_size, output_size), float)
        for j in range(population_size):
            for k in range(input_size):
                N[j][k] = noise.get(list_seed[gen] + (j * input_size * output_size) + k * output_size, output_size)


        #   Try every set of new values and keep track of the returns
        for j in range(population_size):
            W_ = W + sigma * N[j]
            R[j] = run_episode(env, W_, False)

        #   Update weights on the basis of the previous runned episodes
        #   Summation of episode_weight * episode_reward
        weighted_weights = np.matmul(N.T, R).T
        new_W = W + alpha / (population_size * sigma) * weighted_weights
        W = new_W

        gen_mean = np.mean(R)

        plot_data.append(gen_mean)
        n_generations += 1

        print("Generation {}, Population Mean: {}".format(gen, gen_mean))
        if gen_mean >= score_requirement:
            break
    print("Running final games")
    for i in range(final_games):
        print("episode {}, score: {}".format(i, run_episode(env, W, True)))
    return


#   gym.make() : 강화학습 환경을 불러온다.
#   env.reset() : 환경을 초기화한다.
#   env.render() : 화면을 출력한다.
#   env.action_space.sample() : 임의의 action을 선택
#   env.step() : 선택한 action을 환경으로 보낸다.
#   이 코드에서 run_episode()는 기존의 gym을 활용한 코드에서의 env.step()을 대체하기 위해
#   따로 만든듯 함.
#   원래는 observation, reward, done, info를 전부 반환하지만 이 코드에서는 reward를 제외한
#   다른 부분들은 직접 관리하므로 episode 실행 이후 그 결과 중에 reward만 반환.
#   혹은 필요가 없거나...?
def run_episode(env, weight, render = False):
    obs = env.reset()
    episode_reward = 0
    done = False
    step = 0
    while not done:
        if(render):
            env.render()
        if(step > steps_limit):
            move = NULL_ACTION
        else:
            action = np.matmul(weight.T, obs)
            move = np.argmax(action)
        obs, reward, done, info = env.step(move)
        step += 1
        episode_reward += reward
    return episode_reward

def main():
    genetic_algorithm()
    create_plot()

if __name__ == "__main__":
    main()
