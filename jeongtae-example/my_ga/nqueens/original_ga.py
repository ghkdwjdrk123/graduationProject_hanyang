import numpy as np
import random

import plotly
import plotly.graph_objs as go

n_generations = 0
plot_data = []
population_size = 10
generation_limit = 10000
n_queen = 20
mutation_prob = 10    #   mutation의 확률 = 1 / mutation_prob
                        #   확률이 너무 낮으면 답을 찾는데 굉장히 오래걸림
                        #   높을수록 값이 많이 흔들림. 하지만 그만큼 답을 찾는 시간이 굉장히 빨라짐.
                        #   근데 n-queen이 커질수록 너무 흔들려서 문제.

SEED = 4

parent_size = 0
crossover_size = 0

def adjust_parameter():
    population = population_size
    parent = int(population/2)
    crossover = int(population/2)

    i = 0
    while parent + crossover != population:
        if i % 2 == 0:
            parent += 1
        else:
            crossover += 1
    return parent, crossover

def set_board(board):
    for i in range(n_queen):
        board[i] = np.random.randint(n_queen)
    return board

def mutation_OX():
    result = False
    p = np.random.randint(mutation_prob)
    if p % mutation_prob == 0:
        result = True
    return result

def fitness(board):
    fitness = 0
    row_col_clashes = abs(len(np.unique(board)) - n_queen)
    fitness -= row_col_clashes

    for i in range(n_queen):
        for j in range(n_queen):
            if i != j:
                dx = abs(i - j)
                dy = abs(board[i] - board[j])
                if dx == dy:
                    fitness -=1
    return fitness

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
    plotly.offline.plot({"data" : data, "layout" : go.Layout(title = "N_Queens")}, filename = "N_Queens_plot")

def genetic_algorithm():
    np.random.seed(SEED)
    global n_generations

    board = np.zeros((population_size, n_queen))
    for i in range(population_size):
        board[i] = set_board(board[i])

    parent_size, crossover_size = adjust_parameter()

    global n_generations

    solutionBoard = np.zeros(n_queen)
    fitnessArray = np.zeros(population_size)
    done = False
    for gen in range(generation_limit):

        for i in range(population_size):
            fit = fitness(board[i])
            fitnessArray[i] = fit
#            print("Generation {} {}th board fitness is {}".format(gen, i, fit))
            if fit == 0:
                solutionBoard = board[i]
                done = True
                break
        
        if done == True:
            plot_data.append(0)
            n_generations += 1
            print("Generation {}, Solution is {}".format(gen, solutionBoard))
            break
        
        gen_mean = np.mean(fitnessArray)

        plot_data.append(gen_mean)
        n_generations += 1

        print("Generation {}, Population Mean: {}".format(gen, gen_mean))

        #   Fitness 계산

        tempFitness = np.zeros(population_size)
        for i in range(population_size):
            tempFitness[i] = fitnessArray[i]

        #   Parent 선정

        parent = np.zeros((parent_size, n_queen))

        for i in range(parent_size):
            pick = np.argmax(tempFitness)
            tempFitness[pick] = -500000
            parent[i] = board[pick]

        #   Crossover

        crossover = np.zeros((crossover_size, n_queen))
        for i in range(crossover_size):
            crossover[i][:int(n_queen/2)] = parent[i][:int(n_queen/2)]
            crossover[i][int(n_queen/2):] = parent[(crossover_size - 1) - i][int(n_queen/2):]

        #   Parent + Crossover

        for i in range(parent_size):
            board[i] = parent[i]
        for i in range(crossover_size):
            board[i + parent_size] = crossover[i]

        #   Mutation
        
        for i in range(population_size):
            if mutation_OX() == True:
                board[i][np.random.randint(n_queen)] = np.random.randint(n_queen)

    return 

def main():
    genetic_algorithm()
    create_plot()

if __name__ == "__main__":
    main()
