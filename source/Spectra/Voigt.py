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
        
        if iso is not None:
            self.sig = iso.gauWidth
            self.fSig = iso.fixGauss
            self.gam = iso.lorWidth
            self.fGam = iso.fixLor
            self.renorm(self.initPars())
        
    def evaluate(self, x, p):
        if self.sig != p[self.pSig] or self.gam != p[self.pGam]:
            self.renorm(p)
        return self.norm * Physics.voigt(0, p[self.pSig], p[self.pGam])
    
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
        return [self.fSig, self.fGam]
    
    def renorm(self, p):
        self.norm = Physics.voigt(0, p[self.pSig], p[self.pGam])