'''
Created on 23.03.2014

@author: hammen
'''

class Straight(object):
    '''
    classdocs
    '''


    def __init__(self, params):
        self.pb = 0;
        self.pm = 1;
    
    def evaluate(self, x, p):
        return p[self.pb] + x[0]*p[self.pm]
    
    def nPar(self):
        return 2
    
    def nFreePar(self):
        return 2
    
    def leftEdge(self):
        return -1;
    
    def rightEdge(self):
        return 1;
    
    def initPars(self):
        pass
                        