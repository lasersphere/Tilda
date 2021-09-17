'''
Created on 23.03.2014

@author: hammen
'''

import numpy as np

class Straight(object):
    '''
    A straight as lineshape object
    '''

    def __init__(self):
        self.nPar = 2
        self.nFreePar = 2
        self.pb = 0
        self.pm = 1
        self.x_min = None
        self.x_max = None
        self.cut_x = {}

    def evaluate(self, x, p):
        if self.x_max is None:
            self.x_max = x
            self.x_min = x
        self.x_min = min(x, self.x_min)
        self.x_max = max(x, self.x_max)
        return p[self.pb] + x*p[self.pm]
    
    def evaluateE(self, e, freq, col, p):
        return self.evaluate(e, p)

    def leftEdge(self):
        return self.x_min - abs((self.x_min - self.x_max) * 0.001)

    def rightEdge(self):
        return self.x_max + abs((self.x_min - self.x_max) * 0.001)

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
    
    def getBlankNames(self):
        return ['b', 'm']
    
    def toPlotE(self, freq, col, p, prec = 10000):
        '''Return ([x/V], [y/V]) values with prec number of points'''
        self.recalc(p)
        return ([x for x in np.linspace(self.leftEdge(), self.rightEdge(), prec)], [self.evaluate(x, p) for x in np.linspace(self.leftEdge(), self.rightEdge(), prec)])