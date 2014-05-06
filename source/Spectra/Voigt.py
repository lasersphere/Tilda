'''
Created on 23.03.2014

@author: hammen
'''

import Physics

class Voigt(object):
    '''
    Implementation of a voigt profile object using the Faddeeva function
    '''

    def __init__(self, iso):
        '''
        Constructor
        '''
        self.iso = iso
        self.nPar = 2
        
        self.pSig = 0
        self.pGam = 1
        self.sig = iso.shape['gau']
        self.gam = iso.shape['lor']
        self.norm = Physics.voigt(0, self.sig, self.gam)
        
    def evaluate(self, x, p):    
        return Physics.voigt(x, p[self.pSig], p[self.pGam]) / self.norm
    
    
    def recalc(self, p):
            self.norm = Physics.voigt(0, self.sig, self.gam)
    
    
    def leftEdge(self):
        return -5 * (self.sig + self.gam)
    
    
    def rightEdge(self):
        return 5 * (self.sig + self.gam)
    
    
    def getPars(self, pos = 0):
        self.pSig = pos
        self.pGam = pos + 1
        
        return [self.iso.shape['gau'], self.iso.shape['lor']]
    
    
    def getParNames(self):
        return ['sigma', 'gamma']
    
    
    def getFixed(self):
        return [self.iso.fixShape['gau'], self.iso.fixShape['lor']]
        