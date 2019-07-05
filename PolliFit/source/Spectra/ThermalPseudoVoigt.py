"""
Created on 31.01.2019

@author: mueller
"""

import Physics

from scipy.stats import norm
import numpy as np


class ThermalPseudoVoigt(object):
    """
    Implementation of a Pseudo-Voigt profile object
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

        order = 4.

    (So do not call order 66. May the 4th be with you!)
    Optionally a second peak can be used, if required.

    The center peak height is normalized to one
    Gamma is the half width at half maximum
    """

    def __init__(self, iso):
        """ Initialize """
        self.iso = iso
        self.nPar = 9

        self.diff_doppl = 1
        self.calc_diff_doppl(iso.shape['laserFreq'], iso.shape['col'])

        self.pGam = 0
        self.pXi = 1
        self.pColDirTrue = 2
        self.pOrder = 3

        self.pMixing = 4

        self.p_n_of_peaks = 5

        self.p_laserFreq = 6

        self.pInt1 = 7
        self.pCenter1 = 8

        self.recalc([iso.shape['gamma'], iso.shape['compression'], iso.shape['col'], iso.shape['order'],
                     iso.shape['mixing'],
                     iso.shape['nOfPeaks'], iso.shape['laserFreq'], iso.shape['int1'], iso.shape['center1']])

    def evaluate(self, x, p):
        """ Return the value of the hyperfine structure at point x / MHz """
        sigma = p[self.pGam]/np.sqrt(2*np.log(2))
        ret = (1 - p[self.pMixing])*Physics.thermalLorentz(x, 0, p[self.pGam], p[self.pXi], p[self.pColDirTrue], p[self.pOrder])
        ret += p[self.pMixing]*norm.pdf(x, loc=0, scale=sigma)
        if p[self.p_n_of_peaks] > 1:
            for _ in range(int(p[self.p_n_of_peaks]) - 1):
                freq1 = p[self.pCenter1]*self.diff_doppl
                ret += (1 - p[self.pMixing])*p[self.pInt1]*Physics.thermalLorentz(x - freq1, 0, p[self.pGam], p[self.pXi], p[self.pColDirTrue], p[self.pOrder])
                ret += p[self.pMixing]*norm.pdf(x - freq1, loc=0, scale=sigma)

        return ret/self.norm

    def recalc(self, p):
        """ Recalculate the norm factor """
        self.norm = (1 - p[self.pMixing])*Physics.thermalLorentz(0, 0, p[self.pGam], p[self.pXi],
                                                                 p[self.pColDirTrue], p[self.pOrder]) \
            + p[self.pMixing]*norm.pdf(0, loc=0, scale=p[self.pGam]/(np.sqrt(2*np.log(2))))

    def leftEdge(self, p):
        """ Return the left edge of the spectrum in MHz """
        return -5*p[self.pGam]

    def rightEdge(self, p):
        """ Return the right edge of the spectrum in MHz """
        return 5*p[self.pGam]

    def getPars(self, pos=0):
        """ Return list of initial parameters and initialize positions """
        self.pGam = pos
        self.pXi = pos + 1
        self.pColDirTrue = pos + 2
        self.pOrder = pos + 3

        self.pMixing = pos + 4

        self.p_n_of_peaks = pos + 5
        self.p_laserFreq = pos + 6

        self.pInt1 = pos + 7
        self.pCenter1 = pos + 8
        return [self.iso.shape['gamma'], self.iso.shape['compression'], self.iso.shape['col'],
                self.iso.shape['order'], self.iso.shape['mixing'],
                self.iso.shape['nOfPeaks'], self.iso.shape['laserFreq'],
                self.iso.shape['int1'], self.iso.shape['center1']]

    def getParNames(self):
        """ Return list of the parameter names """
        return ['gamma', 'compression', 'col', 'order', 'mixing',
                'nOfPeaks', 'laserFreq', 'int1', 'center1']

    def getFixed(self):
        """ Return list of parmeters with their fixed-status """
        return [self.iso.fixShape['gamma'], self.iso.fixShape['compression'], self.iso.fixShape['col'],
                self.iso.fixShape['order'], self.iso.fixShape['mixing'],
                self.iso.fixShape['nOfPeaks'], self.iso.fixShape['laserFreq'],
                self.iso.fixShape['int1'], self.iso.fixShape['center1']]

    def calc_diff_doppl(self, laser_freq, col):
        """ Calculate the differential doppler factor for this shape and store it in self.diff_doppl """
        if laser_freq is not None:
            center_velocity = Physics.invRelDoppler(laser_freq, self.iso.freq + self.iso.center)
            center_velocity = - center_velocity if col else center_velocity
            center_volts = Physics.relEnergy(center_velocity, self.iso.mass * Physics.u) / Physics.qe
            self.diff_doppl = Physics.diffDoppler(laser_freq, center_volts, self.iso.mass, real=True)
