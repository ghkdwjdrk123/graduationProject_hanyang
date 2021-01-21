from multiprocessing import Pool, TimeoutError
import time
import os

def f(x):
    return x*x
if __name__ == '__main__':
    # start 4 worder processes
    with Pool(processes=4) as pool:

        # print "[0, 1, 4, ..., 81]"
        print(pool.map(f, range(10)))

        # print same numbers in arbitrary order
        for i in pool.imap_unordered(f, range(10)):
            print(i)

        # evaluate "f(20)" asynchronously
        res = pool.apply_async(f, (20, ))       # runs in *only* one process
        print(res.get(timeout=1))               # print "400"

        # evaluate "os.getpid()" asynchronously
        res = pool.apply_async(os.getpid, ())   # runs in *only* one process
        print(res.get(timeout=1))               # prints the PID of that process

        # launching multiple evaluations asynchronously *may* use more processes
        multiple_results = [pool.apply_async(os.getpid, ()) for i in range(4)]
        print([res.get(timeout=1) for res in multiple_results])

        # make a single worder sleep for 10 secs
        res = pool.apply_async(time.sleep, (10, ))
        try:
            print(res.get(timeout=1))
        except TimeoutError:
            print("We lacked patience and got a multiprocessing.TimeoutError")

        print("For the moment, the pool remains available for more word")

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
