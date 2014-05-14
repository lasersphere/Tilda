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
        self.pb = 0;
        self.pm = 1;
        
    
    def evaluate(self, x, p):
        return p[self.pb] + x*p[self.pm]
    
    def evaluateE(self, e, freq, col, p):
        return self.evaluate(e, p)

    
    def leftEdge(self):
        return -10
    
    
    def rightEdge(self):
        return 10
    
    
    def getPars(self):
        return [0, 1]
    
    
    def getParNames(self):
        return ['b', 'm']
    
    
    def getFixed(self):
        return [False, False]
    
    def recalc(self, p):
        pass
    
    def toPlotE(self, freq, col, p, prec = 10000):
        '''Return ([x/V], [y/V]) values with prec number of points'''
        self.recalc(p)
        return ([x for x in np.linspace(self.leftEdge(), self.rightEdge(), prec)], [self.evaluate(x, p) for x in np.linspace(self.leftEdge(), self.rightEdge(), prec)])