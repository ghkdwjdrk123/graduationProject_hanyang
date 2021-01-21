#   논문의 elitism만 구현한 것입니다.

import gym
import numpy as np
from gym import wrappers
from os import path
import random

import plotly
import plotly.graph_objs as go

n_generations = 0
plot_data = []
final_games = 10
score_requirement = -0.01
population_size = 30
generation_limit = 1000
steps_limit = 1
sigma = 2
alpha = 0.0005

RNG_SEED = 44
NULL_ACTION = 0

parent_size = 50
mutation_size = 0

def set_obs(env, obs):
    env.reset()
    env.state = obs
    env.last_u = None

def adjust_parameter():
    population = population_size
    parent = int(population/2)
    mutation = int(population/2)

    i = 0
    while parent + mutation != population:
        if i % 2 == 0:
            parent += 1
        else:
            mutation += 1
        i += 1

    return parent, mutation

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
    plotly.offline.plot({"data": data, "layout": go.Layout(title = "Pendulum")}, filename = "Pendulum_plot")

def get_random(x):
    W = 4 * np.random.random_sample(x) + (-2)
    return W

def genetic_algorithm():

    #   env.reset()으로는 pendulum 내에서는 초기 시작값이 매번 달라집니다.
    #   이것을 해결하기 위해 action을 0으로 1번 step을 진행한 후에 반환된 obs값을 고정으로
    #   사용하여 매번 같은 시작점에서 학습되도록 합니다.
    env = gym.make("Pendulum-v0")
    env.reset()
    first_obs, first_reward, first_done, first_info = env.step([0])
    
    env.seed(RNG_SEED)
    np.random.seed(RNG_SEED)

    global n_generations
    
    Next_generation = np.zeros((population_size, steps_limit))
    for i in range(population_size):
        Next_generation[i] = get_random(steps_limit)
    
    for gen in range(generation_limit):
        R = np.zeros(population_size)

#        print("Aftet Next_generation[0] : {}".format(Next_generation[0]))

        #   이거 왜 다를까?
#        print("Run 1 : {}".format(run_episode(env, first_obs, Next_generation[0], False)))
#        print("Run 2 : {}".format(run_episode(env, first_obs, Next_generation[0], False)))


        for j in range(population_size):
            R[j] = run_episode(env, first_obs, Next_generation[j], False)


        #   elitism, mutation 넣기
        Parent = np.zeros((parent_size, steps_limit))

        #   우선 상위 n개 선정. 여기서 n은 parent_size 
        R_ = np.zeros(population_size)
        for i in range(population_size):
            R_[i] = R[i]
        
        for k in range(parent_size):
            pick = np.argmax(R_)
            R_[pick] = -500000
            Parent[k] = Next_generation[pick]
#            print("R[{}] : {}".format(pick, R[pick]))


        #   여기서 전체적으로 난수를 더해줘야하는데, 그 결과값이 -2 ~ 2 사이의 값이어야 함.
        
        Mutation = np.zeros((population_size, steps_limit))

        for i in range(population_size):
            temp = random.randint(0, parent_size - 1)
            Mutation[i] = Parent[temp]
        
        for i in range(population_size - 1):
            for j in range(steps_limit):
                Mutation[i][j] = Mutation[i][j] + sigma * 4 * np.random.random_sample() + (-2)
                if Mutation[i][j] > 2.0:
                    Mutation[i][j] = 2.0
                if Mutation[i][j] < -2.0:
                    Mutation[i][j] = -2.0

        # 마지막 1개는 가장 우수한 개체를 넣어줌. 이것이 elitism.
        for i in range(population_size):
            Next_generation[i] = Mutation[i]
        Next_generation[population_size - 1] = Parent[0]

#        print("Before_Next_generation[0] : {}".format(Next_generation[0]))

#        print("Generation {}, 1th Mean: {}".format(gen, run_episode(env, first_obs, Next_generation[0], False)))

        gen_mean = np.mean(R)

        plot_data.append(gen_mean)
        n_generations += 1

        print("Generation {}, Population Mean: {}".format(gen, gen_mean))
        if gen_mean >= score_requirement:
            break

    print("Running final games")
    for i in range(final_games):
        print("episode {}, score: {}".format(i, run_episode(env,first_obs, Next_generation[i], True)))
    return


#   env.reset()안에서 random을 써버림.

def run_episode(env, first_obs, N, render = False):
    obs = first_obs
    set_obs(env, obs)
    episode_reward = 0
    done = False
    step = 0
    while not done:
        if(render):
            env.render()
        if(step >= steps_limit):
            break
        else:
            obs, reward, done, info = env.step([N[step]])
        step += 1
        episode_reward += reward
    return episode_reward

def main():
    genetic_algorithm()
    create_plot()

if __name__ == "__main__":
    main()
