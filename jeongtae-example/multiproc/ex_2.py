import os

from multiprocessing import Process

def doubler(number):
    result = number *2
    proc = os.getpid()
    print('{0} doubled to {1} by process id: {2}'.format(number, result, proc))

if __name__ == '__main__':
    numbers = [5, 10, 15, 20, 25, 30, 35, 40]
    procs = []

    for index, number in enumerate(numbers):
        proc = Process(target=doubler, args=(number, ))
        procs.append(proc)
        proc.start()

    for proc in procs:
        proc.join()
