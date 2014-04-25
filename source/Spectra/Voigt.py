'''
Created on 23.03.2014

@author: hammen
'''

import Physics

class Voigt(object):
    '''
    Implementation of a voigt profile object using the Faddeeva function
    '''

    def __init__(self, iso = None):
        '''
        Constructor
        '''
        self.iso = iso
        self.nPar = 2
        
        self.pSig = 0
        self.sig = 10
        self.pGam = 1
        self.gam = 10
        self.renorm([10, 10])
        
    def evaluate(self, x, p):
        if self.sig != p[self.pSig] or self.gam != p[self.pGam]:
            self.renorm(p)
        return Physics.voigt(x[0], p[self.pSig], p[self.pGam]) / self.norm
    
    def leftEdge(self):
        return -5 * (self.sig + self.gam)
    
    def rightEdge(self):
        return 5 * (self.sig + self.gam)
    
    def getPars(self, pos = 0):
        self.pSig = pos
        self.pGam = pos + 1
        
        return [self.iso.gauSig, self.iso.lorGam]
    
    def getParNames(self):
        return ['sigma', 'gamma']
    
    def getFixed(self):
        return [self.iso.fixGau, self.iso.fixLor]
    
    def renorm(self, p):
        self.norm = Physics.voigt(0, p[self.pSig], p[self.pGam])