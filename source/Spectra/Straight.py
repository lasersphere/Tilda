'''
Created on 23.03.2014

@author: hammen
'''

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
    
    
    def leftEdge(self):
        return -0.1
    
    
    def rightEdge(self):
        return 0.1
    
    
    def getPars(self):
        return [0, 1]
    
    
    def getParNames(self):
        return ['b', 'm']
    
    
    def getFixed(self):
        return [False, False]
    
    def recalc(self, p):
        pass
                        