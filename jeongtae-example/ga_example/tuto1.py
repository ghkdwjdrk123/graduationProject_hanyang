import numpy as np

def calculate_fitness(solution):
    return np.array(solution).dot(parameters)

parameters = np.array([4, -2, 3.5, 5, -11, -4.7])

## initialize solution pool
## 8x6 list로 항목이 6개인 list가 8줄 
current_solution_pool = [list(np.random.normal(0, 1, 6)) for i in range(0, 8)]

for i in range(0, 20):
    ## 현재 솔루션 중에서 가장 성과가 좋은 것만 남기고 모두 버림
    ## 0, 1, 2, 3 까지만 들고 감.
    new_parents = sorted(current_solution_pool, key=calculate_fitness, reverse=True)[:4]
    print(f"optimal fitness in {i:0>2d} generation: {calculate_fitness(new_parents[0])}")
    
    ## 간단하게 crossovers 세팅
    ## 0번째와 1번째 fitness 좋은 것들끼리 교배
    crossovers = [
            new_parents[0][:3] + new_parents[1][3:],
            new_parents[1][:3] + new_parents[0][3:],
    ]

    ## mutations 세팅
    ## 0번째에 정규분포 더해줌으로써 돌연변이 만듦.
    mutations = [
            list(np.array(new_parents[0]) + np.random.normal(0, 1, 6)),
            list(np.array(new_parents[0]) + np.random.normal(0, 1, 6)),
    ]

    ## 0, 1, 2, 3번째 + 교배 2개 + 돌연변이 2개
    current_solution_pool = new_parents + crossovers + mutations

