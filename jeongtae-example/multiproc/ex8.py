from multiprocessing import Process, Lock
import os

def f(l, i):
    l.acquire()
    try:
        print('hello world', i)
        print('my pid is ', os.getpid())
    finally:
        l.release()

if __name__ == '__main__':
    lock = Lock()

    for num in range(10):
        Process(target=f, args=(lock, num)).start()
