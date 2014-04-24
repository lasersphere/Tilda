'''
Created on 23.03.2014

@author: hammen
'''

from scipy.special import wofz
import math

class Voigt(object):
    '''
    Implementation of a voigt profile object using the Faddeeva function
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.nPar = 2
        self.nFreePar = 2
        self.psig = 0
        self.pgam = 1
        
    def evaluate(self, x, p):
        return wofz((x[0] + 1j * p[self.pgam])/(p[self.psig] * math.sqrt(2))).real / (p[self.psig] * math.sqrt(2 * math.pi))
    
    def leftEdge(self):
        return -0.1
    
    def rightEdge(self):
        return 0.1
    
    def initPars(self):
        pass