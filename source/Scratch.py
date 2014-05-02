'''
Created on 31.03.2014

@author: hammen
'''

import os

from scipy.optimize import curve_fit

from Measurement.SimpleImporter import SimpleImporter
from matplotlib import pyplot as plt


path = "../test/test.txt"
a = SimpleImporter(path)

x, y, err = a.getSingleSpec(0, -1)

p = [1]

print(x)
print(y)
print(err)


def func(x, p):
    return x*p

print(curve_fit(func, x, y, p, err))
