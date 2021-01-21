from multiprocessing import Process, Manager, Lock
import os

def RetSum(Mobj, lock):
    funcList = [1, 2, 3]
    lock.acquire()
    try:
        for x in funcList:
            Mobj[0] += x
    finally:
        lock.release()
        print("PROCESS ID : ", os.getpid(), " Number : ", Mobj[0])

if __name__ == '__main__':
    with Manager() as manager:
        lock = Lock()

        Mobj = manager.list([0 for x in range(3)])
        MpList = []

        for _ in range(3):
            MpList.append(Process(target=RetSum, args=(Mobj, lock)))

        for _ in MpList:
            _.start()

        for _ in MpList:
            _.join()

        print(Mobj)
