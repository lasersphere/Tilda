'''
Created on 24.05.2017

@author: chgorges
'''

from Tilda.PolliFit import Physics


class FanoVoigt(object):
    '''
    Implementation of a voigt profile object using the (real part of the) Faddeeva function
    and a small correction using the imaginary part of the Faddeeva function
    
    The peak height is normalized to one
    Sigma is the standard deviation of the gaussian
    Gamma is the half width half maximum
    '''

    def __init__(self, iso):
        '''Initialize'''
        self.iso = iso
        self.nPar = 3
        
        self.pSig = 0
        self.pGam = 1
        self.dispersivePar = 2
        self.recalc([iso.shape.get('gau', iso.shape.get('sigma', 0.0)),
                     iso.shape.get('lor', iso.shape.get('gamma', 0.0)),
                     iso.shape['dispersive']]) # .get() structure due to naming difference in .getParNames() and shape['']
    
    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x / MHz'''
        return Physics.fanoVoigt(x, p[self.pSig], p[self.pGam], p[self.dispersivePar]) / self.norm


    def recalc(self, p):
        '''Recalculate the norm factor'''
        self.norm = Physics.fanoVoigt(0, p[self.pSig], p[self.pGam], p[self.dispersivePar])


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
        self.dispersivePar = pos + 2

        return [self.iso.shape.get('gau', self.iso.shape.get('sigma', 0.0)),
                self.iso.shape.get('lor', self.iso.shape.get('gamma', 0.0)),
                self.iso.shape['dispersive']] # .get() structure due to naming difference in .getParNames() and shape['']


    def getParNames(self):
        '''Return list of the parameter names'''
        return ['sigma', 'gamma', 'dispersive']


    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [self.iso.fixShape.get('gau', self.iso.fixShape.get('sigma', False)),
                self.iso.fixShape.get('lor', self.iso.fixShape.get('gamma', False)),
                self.iso.fixShape['dispersive']] # .get() structure due to naming difference in .getParNames() and shape['']
