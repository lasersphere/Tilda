'''
Created on 23.03.2014

@author: hammen
'''

class Straight(object):
    '''
    classdocs
    '''

    def __init__(self):
        self.nPar = 2
        self.nFreePar = 2
        self.pb = 0;
        self.pm = 1;
        
    
    def evaluate(self, x, p):
        return p[self.pb] + x[0]*p[self.pm]
    
    def leftEdge(self):
        return -0.1
    
    def rightEdge(self):
        return 0.1
    
    def initPars(self):
        pass
                        