from multiprocessing import Pool
import time

def f(x):
    return x*x

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        result = pool.apply_async(f, (10, ))
        print(result.get(timeout=1))

        print(pool.map(f, range(10)))

        it = pool.imap(f, range(10))
        print(next(it))
        print(next(it))
        print(it.next(timeout=1))

        result = pool.apply_async(time.sleep, (10, ))
        print(result.get(timeout=1))
