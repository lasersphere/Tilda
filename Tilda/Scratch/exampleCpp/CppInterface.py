# -*- coding: utf-8 -*-
"""
Tilda.Tilda.source.Scratch.exampleCpp.CppInterface

Created on 29.10.2021

@author: Patrick Mueller

Python/C++ interface. This example implements matrix multiplication of complex types via a DLL.
"""

import os
import ctypes
import timeit
import numpy as np
import scipy.stats as st


# noinspection PyPep8Naming
class c_complex(ctypes.Structure):  # ctypes does not have a compatible complex type by default :(
    """
    Complex number, compatible with std::complex layout.
    """
    _fields_ = [('real', ctypes.c_double), ('imag', ctypes.c_double)]

    def __init__(self, z):
        """
        :param z: A complex scalar.
        """
        super().__init__()
        self.real = z.real
        self.imag = z.imag

    def to_complex(self):
        """
        Convert to Python complex.

        :returns: A Python-type complex scalar.
        """
        return self.real + 1.j * self.imag


c_int_p = ctypes.POINTER(ctypes.c_int)  # Create pointers for the required C types.
c_complex_p = ctypes.POINTER(c_complex)

dll_path = os.path.dirname(os.path.realpath(__file__))  # The path to this files directory.
dll_name = 'TildaDllExample.dll'
if ctypes.sizeof(ctypes.c_void_p) == 4:  # Check for 32-bit architecture.
    dll_name = dll_name[:-4] + '_x86.dll'
dll = ctypes.CDLL(os.path.join(dll_path, dll_name))


def matrix_multiplication(_a, _b):
    """
    Calculate the matrix product of two complex-valued matrices using a DLL.

    :param _a: The first matrix. Must have shape (n, k).
    :param _b: The second matrix. Must have shape (k, m).
    :returns: The matrix product of a and b. Has shape (n, m).
    """
    shape_a, shape_b = np.asarray(_a.shape), np.asarray(_b.shape)
    shape_a_p = shape_a.ctypes.data_as(c_int_p)
    shape_b_p = shape_b.ctypes.data_as(c_int_p)

    ret = np.zeros(_a.shape[0] * _b.shape[1], dtype=complex)  # Initialize the return array.
    ret_p = ret.ctypes.data_as(c_complex_p)

    _a, _b = _a.flatten('F'), _b.flatten('F')  # use ('C', 'C') for _0, ('C', 'F') for _opt and ('F', 'F') for _eigen.
    a_p = _a.ctypes.data_as(c_complex_p)
    b_p = _b.ctypes.data_as(c_complex_p)

    # dll.matrix_multiplication_0(a_p, b_p, shape_a_p, shape_b_p, ret_p)  # Adjust the line above for this.
    # dll.matrix_multiplication_opt(a_p, b_p, shape_a_p, shape_b_p, ret_p)  # Adjust the line above for this.
    dll.matrix_multiplication_eigen(a_p, b_p, shape_a_p, shape_b_p, ret_p)  # Modify the return array.

    return ret.reshape((shape_a[0], shape_b[1]))  # Reshape the flat return array.


def list_comprehension_multiplication(_a, _b):
    """
    Calculate the matrix product of two complex-valued matrices using list comprehensions.

    :param _a: The first matrix. Must have shape (n, k).
    :param _b: The second matrix. Must have shape (k, m).
    :returns: The matrix product of a and b. Has shape (n, m).
    """
    _a, _b = _a.tolist(), _b.T.tolist()  # Iterating over lists is faster than iterating over numpy arrays.
    dim = len(_a[0])
    return np.array([[sum(ai[k] * bi[k] for k in range(dim)) for bi in _b] for ai in _a])


"""
Timeit results.

n = 5, unitary matrix:
Time list comp.: 1.9614198200000001 s
Time @ (numpy): 0.0010353599999998408 s
Time C++: 0.01785989999999984 s  (Eigen)

n = 5, (n x m, n != m) matrices:
Time list comp.: 1.7768547600000002 s
Time @ (numpy): 0.0007898000000000849 s
Time C++: 0.01628038000000025 s  (Eigen)

P.S. Numpy uses some insane algorithms.
"""
n = 1  # Number of samples for timeit.

a = st.unitary_group.rvs(300)  # A random 300 x 300 unitary matrix.
b = a.conj().T  # a a*T = 1 if a is unitary.

# a = st.norm.rvs(size=(200, 400)) + 2 * st.norm.rvs(size=(200, 400)) * 1j  # Some random (n x m, n != m) matrices.
# b = 2 * st.norm.rvs(size=(400, 300)) + st.norm.rvs(size=(400, 300)) * 1j

# Times:
print('Time list comp.: {} s'.format(timeit.timeit('list_comprehension_multiplication(a, b)',
                                                   number=n, globals=globals()) / n))
print('Time @ (numpy): {} s'.format(timeit.timeit('a @ b', number=n, globals=globals()) / n))
print('Time C++: {} s'.format(timeit.timeit('matrix_multiplication(a, b)', number=n, globals=globals()) / n))

# Results:
print('\nSum list comp.: {} s'.format(np.sum(list_comprehension_multiplication(a, b))))
print('Sum @ (numpy): {} s'.format(np.sum(a @ b)))
print('Sum C++: {} s'.format(np.sum(matrix_multiplication(a, b))))
