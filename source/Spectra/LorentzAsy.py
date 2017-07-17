'''
Created on 30.05.2017

@author: chgorges
'''

import Physics

import numpy as np

class LorentzAsy(object):
    '''
    Implementation of a lorentzian profile object using superposition of single lorentzian
    The width is varied with an exponential function as proposed in
    A. KLOSE, K. MINAMISONO, AND P. F. MANTICA: PHYSICAL REVIEW A 88, 042701 (2013)


    The peak height is normalized to one
    Gamma is the half width half maximum
    '''

    def __init__(self, iso):
        '''Initialize'''
        self.iso = iso
        self.nPar = 2

        self.pGam = 0
        self.asyPar = 1
        self.recalc([iso.shape['lor'], iso.shape['asy']])
    
    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x / MHz'''
        gamma = 2*p[self.pGam]/(1 + np.exp(p[self.asyPar]/1000*x))
        return Physics.lorentz(x, 0, gamma) / self.norm


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
        self.asyPar = pos + 1

        return [self.iso.shape['lor'], self.iso.shape['asy']]


    def getParNames(self):
        '''Return list of the parameter names'''
        return ['gamma', 'asy']


    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [self.iso.fixShape['lor'], self.iso.fixShape['asy']]
