'''
Created on 27.01.2017

@author: schmid
'''

import Physics

class Lorentz(object):
    '''
    Implementation of a lorentzian profile object using superposition of single lorentzian
    
    The peak height is normalized to one
    Gamma is the half width half maximum
    '''

    def __init__(self, iso):
        '''Initialize'''
        self.iso = iso
        self.nPar = 1
        self.pGam = 0

        self.recalc([iso.shape.get('lor', iso.shape.get('gamma', 0.0))]) # .get() structure due to naming difference in .getParNames() and shape['']
    
    
    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x / MHz'''

        return Physics.lorentz(x, 0, p[self.pGam]) / self.norm
    
    
    def recalc(self, p):
        '''Recalculate the norm factor'''

        self.norm = Physics.lorentz(0, 0, p[self.pGam])
    
    
    def leftEdge(self, p):
        '''Return the left edge of the spectrum in Mhz'''
        return -5 * p[self.pGam]
    
    
    def rightEdge(self, p):
        '''Return the right edge of the spectrum in MHz'''
        return 5 * p[self.pGam]
    
    
    def getPars(self, pos = 0):
        '''Return list of initial parameters and initialize positions'''
        self.pGam = pos
        
        return [self.iso.shape.get('lor', self.iso.shape.get('gamma', 0.0))] # .get() structure due to naming difference in .getParNames() and shape['']
    
    
    def getParNames(self):
        '''Return list of the parameter names'''
        return ['gamma']
    
    
    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [self.iso.fixShape.get('lor', self.iso.fixShape.get('gamma', False))] # .get() structure due to naming difference in .getParNames() and shape['']
        