import random
import numpy as np

input_size = random.randint(0, 100)
output_size = 4

#   list는 곱연산이 안됨. 되는건 반복의 의미로 정수로만 해당.
c = 4

def noise_get(i, o):
    K = range(10000)
    return K[i:i+o]

K = range(1000)

N = []
for k in range(input_size):
    N.append(np.array(list(noise_get(input_size + k*output_size, output_size))))

T = N * c

print(T)
