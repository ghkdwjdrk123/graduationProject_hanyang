import numpy as np
import time
import random


#   아주 간단한 fitness function.
def calculate_fitness(solution):
    return np.array(solution).dot(parameters)

#   deep-neuroevolution의 es.py에 있는 SharedNoiseTable class를 가져옴.
#   논문에 있는 noise를 일일히 적지 않고 seed가 되는 정수만 list에 저장하는 것을 구현하기 위함.
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
        return self.noise[i:i + dim]

    def sample_index(self, stream, dim):
        return stream.randint(0, len(self.noise) - dim + 1)



if __name__ == '__main__':
    noise = SharedNoiseTable()
    parameters = np.array([4, -2, 3.5, 5, -11, -4.7])

    current_solution_pool = [list(np.random.normal(0, 1, 6)) for i in range(0, 8)]

    #   list_seed는 상위 4개로 진행되는 각 elite들의 seed를 기록하는 list.
    list_seed = []
    for i in range(0, 4):
        list_seed.append([])

    #   상위 4개를 elite로 삼아 진행.

    for i in range(0, 20):
        new_parents = sorted(current_solution_pool, key=calculate_fitness, reverse=True)[:4]
        print(f"Optimal Fitness in {i:0>2d} generation: {calculate_fitness(new_parents[0])}")

        for j in range(0, 4):
            list_seed[j].append(random.randint(0, 1000))
        #    print("elite ", j, " list's", i, " th seed is", list_seed[j][i])
        
        elites = []
        for j in range(0, 4):
            elites.append(list(np.array(new_parents[j]) + noise.get(list_seed[j][i], 6)))

        #  논문의 코드에서는 crossovers는 없음. 추후 추가해보면 어떤 성능차이를 낼지 비교해볼 것.
        """        
        crossovers = [
                new_parents[0][:3] + new_parents[1][3:],
                new_parents[1][:3] + new_parents[0][3:]
        ]

        mutations = [
                list(np.array(new_parents[0]) + np.random.normal(0, 1, 6)),
                list(np.array(new_parents[0]) + np.random.normal(0, 1, 6))
        ]
        """

        current_solution_pool = elites + elites
    
