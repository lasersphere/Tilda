'''
Created on 23.03.2014

@author: hammen
'''

import Physics

class Voigt(object):
    '''
    Implementation of a voigt profile object using the Faddeeva function
    
    The peak height is normalized to one
    Sigma is the standard deviation of the gaussian
    Gamma is the half width half maximum  # I think gamma is the standard deviation of the lorentz profile! CG
    '''

    def __init__(self, iso):
        '''Initialize'''
        self.iso = iso
        self.nPar = 2
        
        self.pSig = 0
        self.pGam = 1
        self.recalc([iso.shape['gau'], iso.shape['lor']])
    
    
    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x / MHz'''
        return Physics.voigt(x, p[self.pSig], p[self.pGam]) / self.norm
    
    
    def recalc(self, p):
        '''Recalculate the norm factor'''
        self.norm = Physics.voigt(0, p[self.pSig], p[self.pGam])
    
    
    def leftEdge(self, p):
        '''Return the left edge of the spectrum in Mhz'''
        return -5 * (p[self.pSig] + p[self.pGam])
    
    
    def rightEdge(self, p):
        '''Return the right edge of the spectrum in MHz'''
        return 5 * (p[self.pSig] + p[self.pGam])
    
    
    def getPars(self, pos = 0):
        '''Return list of initial parameters and initialize positions'''
        self.pSig = pos
        self.pGam = pos + 1
        
        return [self.iso.shape['gau'], self.iso.shape['lor']]
    
    
    def getParNames(self):
        '''Return list of the parameter names'''
        return ['sigma', 'gamma']
    
    
    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [self.iso.fixShape['gau'], self.iso.fixShape['lor']]
        