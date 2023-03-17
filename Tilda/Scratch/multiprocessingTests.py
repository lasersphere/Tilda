"""
Created on 

@author: simkaufm

Module Description:  just created this to play around with multiprocessing
"""

from multiprocessing import Process, Value, Array


def f(n, a):
    n.value = 3.1415927
    for i in range(len(a)):
        a[i] = -a[i]


if __name__ == '__main__':
    num = Value('d', 0.0)  # shared among processes
    arr = Array('i', range(10))  # shared among processes

    p = Process(target=f, args=(num, arr))
    p.start()
    p.join()

    print(num.value)
    print(arr[:])