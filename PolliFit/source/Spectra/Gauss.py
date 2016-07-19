'''
Created on 18.07.2014

@author: kaufmann
'''

import numpy as np

import Physics


class Gauss(object):
    '''
    Implementation of a gauss profile object
    
    just for fitting to raw data or so, no iso support and no db support for now.
    '''

    def __init__(self):
        '''Initialize'''
        # self.iso = iso
        self.nPar = 3

        self.pMu = 0
        self.pSig = 1
        self.pAmp = 2
        # self.recalc([iso.shape['mu'], iso.shape['sigma']])
    
    
    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x / MHz'''
        return Physics.gaussian(x, p[self.pMu], p[self.pSig], p[self.pAmp])

    def evaluateE(self, e, freq, col, p):
        return Physics.gaussian(e, p[self.pMu], p[self.pSig], p[self.pAmp])

    def recalc(self, p):
        # '''Recalculate the norm factor'''
        # self.norm = Physics.voigt(0, p[self.pMu], p[self.pSig])
        pass
    
    def leftEdge(self, p):
        '''Return the left edge of the spectrum in Mhz'''
        return -5 * (p[self.pSig] + p[self.pMu])
    
    def rightEdge(self, p):
        '''Return the right edge of the spectrum in MHz'''
        return 5 * (p[self.pSig] + p[self.pMu])

    def getPars(self, pos=0):
        '''Return list of initial parameters and initialize positions'''
        self.pMu = pos
        self.pSig = pos + 1
        self.pAmp = pos + 2
        
        return [40, 20, 50000]

    def parAssign(self):
        return [('Gauss', [True, True, True])]
    
    def getParNames(self):
        '''Return list of the parameter names'''
        return ['mu', 'sigma', 'ampl']

    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [False, False, False]

    def toPlotE(self, freq, col, p, prec=10000):
        '''Return ([x/V], [y/V]) values with prec number of points'''
        self.recalc(p)
        return ([x for x in np.linspace(self.leftEdge(p), self.rightEdge(p), prec)],
                [self.evaluate(x, p) for x in np.linspace(self.leftEdge(p), self.rightEdge(p), prec)])
