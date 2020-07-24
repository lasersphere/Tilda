"""
Created on 31.01.2019

@author: pamueller
"""

import Physics

from scipy.stats import norm
import numpy as np


class ThermalPseudoVoigt(object):
    """
    Implementation of a thermal Pseudo-Voigt profile object
    resulting from probing an accelerated ensemble of initially thermal distributed ions.
    The Pseudo-Voigt comes from a linear combination of the Physics.thermalLorentz function
    [Kretzschmar et al., Appl. Phys. B, 79, 623 (2004)] and a gauss distribution,
    where the standard deviation depends on the half width at half maximum of the Lorentzian

        sigma = gamma / sqrt(2*ln(2))

    to account for possible line broadening effects.
    The analytical correct lineshape for the Lorentzian can be calculated with

        order = -1.

    If the analytical solution leads to computational problems
    which was stated in [Kretzschmar et al., Appl. Phys. B, 79, 623 (2004)],
    then the thermal Lorentz can be calculated up to

        order = 4

    (So do not call order 66. May the 4th be with you!).

    The center peak height is normalized to one at the 'center' position.
    """

    def __init__(self, iso):
        """ Initialize """
        self.iso = iso
        self.nPar = 5

        self.pGam = 0
        self.pXi = 1
        self.pMixing = 2

        self.pColDirTrue = 3
        self.pOrder = 4

        self.recalc([iso.shape['gamma'], iso.shape['compression'], iso.shape['col'],
                     iso.shape['order'], iso.shape['mixing']])

    def evaluate(self, x, p):
        """ Return the value of the hyperfine structure at point x / MHz """
        sigma = p[self.pGam] / np.sqrt(2. * np.log(2.))
        y = (1. - p[self.pMixing]) \
            * Physics.thermalLorentz(x, 0, p[self.pGam], p[self.pXi], p[self.pColDirTrue], p[self.pOrder])
        y += p[self.pMixing] * norm.pdf(x, loc=0, scale=sigma)
        return y / self.norm

    def recalc(self, p):
        """ Recalculate the norm factor """
        self.norm = 1.
        self.norm = self.evaluate(0., p)

    def leftEdge(self, p):
        """ Return the left edge of the spectrum in MHz """
        xi_left = 0. if p[self.pColDirTrue] else np.abs(p[self.pXi])
        return -5. * (p[self.pGam] + xi_left)

    def rightEdge(self, p):
        """ Return the right edge of the spectrum in MHz """
        xi_right = np.abs(p[self.pXi]) if p[self.pColDirTrue] else 0.
        return 5. * (p[self.pGam] + xi_right)

    def getPars(self, pos=0):
        """ Return list of initial parameters and initialize positions """
        self.pGam = pos
        self.pXi = pos + 1
        self.pMixing = pos + 2

        self.pColDirTrue = pos + 3
        self.pOrder = pos + 4

        return [self.iso.shape['gamma'], self.iso.shape['compression'], self.iso.shape['mixing'],
                self.iso.shape['col'], self.iso.shape['order']]

    def getParNames(self):
        """ Return list of the parameter names """
        return ['gamma', 'compression', 'mixing', 'col', 'order']

    def getFixed(self):
        """ Return list of parameters with their fixed-status """
        return [self.iso.fixShape['gamma'], self.iso.fixShape['compression'], self.iso.fixShape['mixing'], True, True]
