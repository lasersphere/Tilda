"""
Created on 18.07.2014

@author: kaufmann
edited by P. Mueller.
"""

import numpy as np

from Tilda.PolliFit import Physics


class Gauss(object):
    """
    Implementation of a gauss profile object.
    """

    def __init__(self, iso):
        """ Initialize """
        self.iso = iso
        self.nPar = 1
        self.norm = 1.

        self.pSig = 0
        self.recalc([iso.shape.get('gau', iso.shape.get('sigma', 0.0))])
        # .get() structure due to naming difference in .getParNames() and shape['']

    def evaluate(self, x, p):
        """Return the value of the hyperfine structure at point x / MHz"""
        return Physics.gaussian(x, 0, p[self.pSig], 1) / self.norm

    def recalc(self, p):
        """ Recalculate the norm factor """
        self.norm = Physics.gaussian(0, 0, p[self.pSig], 1)
    
    def leftEdge(self, p):
        """ Return the left edge of the spectrum in Mhz """
        return -5 * self.fwhm(p)
    
    def rightEdge(self, p):
        """ Return the right edge of the spectrum in MHz """
        return 5 * self.fwhm(p)

    def getPars(self, pos=0):
        """ Return list of initial parameters and initialize positions """
        self.pSig = pos
        
        return [self.iso.shape.get('gau', self.iso.shape.get('sigma', 0.0))]
    
    def getParNames(self):
        """ Return list of the parameter names """
        return ['sigma']

    def getFixed(self):
        """ Return list of parmeters with their fixed-status """
        return [self.iso.fixShape.get('gau', self.iso.fixShape.get('sigma', False))]

    def fwhm(self, p):
        """ Return the fwhm of the Gauss profile """
        return np.sqrt(8 * np.log(2)) * p[self.pSig]
