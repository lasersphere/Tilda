"""
Created on 23.03.2014
edited on 18.09.2021

@author: hammen, pamueller
"""

import numpy as np


class Straight(object):
    """
    A straight as lineshape object.
    """

    def __init__(self):
        self.nPar = 2
        self.pb = 0
        self.pm = 1
        self.x_min = None
        self.x_max = None

    def evaluate(self, x, p):
        if self.x_max is None:
            self.x_min = np.min(x)
            self.x_max = np.max(x)
        else:
            self.x_min = np.min([np.min(x), self.x_min])
            self.x_max = np.min([np.max(x), self.x_max])
        return p[self.pb] + x * p[self.pm]
    
    def evaluateE(self, e, freq, col, p):
        return self.evaluate(e, p)

    def leftEdge(self):
        return self.x_min - np.abs(self.x_min - self.x_max) * 0.001

    def rightEdge(self):
        return self.x_max + np.abs(self.x_min - self.x_max) * 0.001

    def getPars(self):
        return [0, 1]

    def getParNames(self):
        return ['b', 'm']

    def getFixed(self):
        return [False, False]
    
    def recalc(self, p):
        pass
    
    def parAssign(self):
        return [('Kepco', [True, True])]
    
    def toPlotE(self, freq, col, p, prec=10000):
        """
        :param freq: The
        :param col: Whether its a collinear spectrum.
        :param p: The parameter list.
        :param prec: The number of data points.
        :returns: (x in V, y in V) values with 'prec' number of points.
        """
        x = np.linspace(self.leftEdge(), self.rightEdge(), prec)
        print('VALUES: ', x, self.evaluate(x, p))
        return x, self.evaluate(x, p)
