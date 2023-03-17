"""
Created on 31.01.2019

@author: pamueller
"""

from Tilda.PolliFit import Physics

from scipy.stats import norm
from scipy.integrate import romb
import numpy as np


class ThermalVoigt(object):
    """
    Implementation of a numerically calculated thermal Voigt profile object
    resulting from probing an accelerated ensemble of initially thermal distributed ions.
    The lineshape originates from a numerical convolution of the Physics.thermalLorentz function
    [Kretzschmar et al., Appl. Phys. B, 79, 623 (2004)] and a gauss distribution with standard deviation 'sigma'
    to account for possible line broadening effects.
    The analytical correct lineshape for the Lorentzian can be calculated with

        order = -1.

    If the analytical solution leads to computational problems
    which was stated in [Kretzschmar et al., Appl. Phys. B, 79, 623 (2004)],
    then the thermal Lorentz can be calculated up to

        order = 4

    (So do not call order 66. May the 4th be with you!).

    The numerical integration is done using Romberg integration.
    A 'precision' parameter is used to control the number (2 ** n_precision + 1) of samples used for integration.
    A precision of 8 seems to be sufficient and yields a relatively fast computation.

    The peak height is normalized to one at the 'center' position.
    """

    def __init__(self, iso):
        """ Initialize """
        self.iso = iso
        self.nPar = 6

        self.pGam = 0
        self.pXi = 1
        self.pSigma = 2

        self.pColDirTrue = 3
        self.pOrder = 4
        self.pPrecision = 5

        self.recalc([iso.shape['gamma'], iso.shape['compression'], iso.shape['sigma'],
                     iso.shape['col'], iso.shape['order'], iso.shape.get('precision', 8)])

    def evaluate(self, x, p):
        """ Return the value of the hyperfine structure at point x / MHz """
        x = np.asarray(x)
        n = 2 ** int(abs(p[self.pPrecision])) + 1
        x_int = np.expand_dims(np.linspace(self.leftEdge(p), self.rightEdge(p), n), axis=0)
        y = norm.pdf(x_int, loc=0, scale=p[self.pSigma]) \
            * Physics.thermalLorentz(np.expand_dims(x, axis=1) - x_int, 0, p[self.pGam],
                                     p[self.pXi], p[self.pColDirTrue], p[self.pOrder])
        int_norm = np.max(y)
        y /= int_norm
        y = romb(y, (self.rightEdge(p) - self.leftEdge(p)) / (n - 1)) * int_norm
        return y / self.norm

    def recalc(self, p):
        """ Recalculate the norm factor """
        self.norm = 1.
        self.norm = float(self.evaluate(np.zeros(1), p))

    def leftEdge(self, p):
        """ Return the left edge of the spectrum in MHz """
        xi_left = 0. if p[self.pColDirTrue] else np.abs(p[self.pXi])
        return -5. * (p[self.pGam] + 2. * np.sqrt(2. * np.log(2.)) * p[self.pSigma] + xi_left)

    def rightEdge(self, p):
        """ Return the right edge of the spectrum in MHz """
        xi_right = np.abs(p[self.pXi]) if p[self.pColDirTrue] else 0.
        return 5. * (p[self.pGam] + 2. * np.sqrt(2. * np.log(2.)) * p[self.pSigma] + xi_right)

    def getPars(self, pos=0):
        """ Return list of initial parameters and initialize positions """
        self.pGam = pos
        self.pXi = pos + 1
        self.pSigma = pos + 2

        self.pColDirTrue = pos + 3
        self.pOrder = pos + 4
        self.pPrecision = pos + 5

        return [self.iso.shape['gamma'], self.iso.shape['compression'], self.iso.shape['sigma'],
                self.iso.shape['col'], self.iso.shape['order'], self.iso.shape.get('precision', 8)]

    def getParNames(self):
        """ Return list of the parameter names """
        return ['gamma', 'compression', 'sigma', 'col', 'order', 'precision']

    def getFixed(self):
        """ Return list of parameters with their fixed-status """
        return [self.iso.fixShape['gamma'], self.iso.fixShape['compression'], self.iso.fixShape['sigma'],
                True, True, True]
