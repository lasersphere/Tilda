# -*- coding: utf-8 -*-
"""
Tilda.Tilda.source.Scratch.exampleCpp.CppInterface

Created on 29.10.2021

@author: Patrick Mueller

Python/C++ interface. This example implements matrix multiplication of complex types via a DLL.
"""

import timeit
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

# noinspection PyUnresolvedReferences
import CppInterface as Ci

a, b = None, None


def time_test():
    global a, b
    n = 3  # Number of samples for timeit.
    t_l, t_a, t_c = [], [], []

    dims = np.linspace(7, 201, 50, dtype=int)
    for d in dims:
        print(d)
        a = st.unitary_group.rvs(d)  # A random 300 x 300 unitary matrix.
        b = a.conj().T  # a a*T = 1 if a is unitary.

        # Times:
        t_l.append(timeit.timeit('Ci.list_comprehension_multiplication(a, b)', number=n, globals=globals()) / n)
        t_a.append(timeit.timeit('a @ b', number=n, globals=globals()) / n)
        t_c.append(timeit.timeit('Ci.matrix_multiplication(a, b)', number=n, globals=globals()) / n)

    plt.title('Timeit tests, averages of {}'.format(n))
    plt.plot(dims, t_l, label='list comp.')
    plt.plot(dims, t_a, label='@ (numpy)')
    plt.plot(dims, t_c, label='C++')
    plt.yscale('log')
    plt.xlabel('Dimension')
    plt.ylabel('Time (s)')
    plt.legend()
    plt.show()


time_test()
