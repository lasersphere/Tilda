'''
Created on 31.01.2019

@author: mueller
'''

import Physics

from scipy.stats import norm


class ThermalPseudoVoigt(object):
    '''
    Implementation of a lorentzian profile object using superposition of single lorentzian

    The peak height is normalized to one
    Gamma is the half width half maximum
    '''

    def __init__(self, iso):
        """Initialize"""
        self.iso = iso
        self.nPar = 10

        self.diff_doppl = 1
        self.calc_diff_doppl(iso.shape['laserFreq'], iso.shape['col'])

        self.pGam = 0
        self.pXi = 1
        self.pColDirTrue = 2
        self.pOrder = 3

        self.pMixing = 4
        self.pSigma = 5

        self.p_n_of_peaks = 6

        self.p_laserFreq = 7

        self.pInt1 = 8
        self.pCenter1 = 9

        self.recalc([iso.shape['gamma'], iso.shape['compression'], iso.shape['col'], iso.shape['order'],
                     iso.shape['mixing'], iso.shape['sigma'],
                     iso.shape['nOfPeaks'], iso.shape['laserFreq'], iso.shape['int1'], iso.shape['center1']])

    def evaluate(self, x, p):
        """Return the value of the hyperfine structure at point x / MHz"""
        ret = (1 - p[self.pMixing])*Physics.thermalLorentz(x, 0, p[self.pGam], p[self.pXi], p[self.pColDirTrue], p[self.pOrder])
        ret += p[self.pMixing]*norm.pdf(x, loc=0, scale=p[self.pSigma])
        if p[self.p_n_of_peaks] > 1:
            for _ in range(int(p[self.p_n_of_peaks]) - 1):
                freq1 = p[self.pCenter1]*self.diff_doppl
                ret += (1 - p[self.pMixing])*p[self.pInt1]*Physics.thermalLorentz(x - freq1, 0, p[self.pGam], p[self.pXi], p[self.pColDirTrue], p[self.pOrder])
                ret += p[self.pMixing]*norm.pdf(x - freq1, loc=0, scale=p[self.pSigma])

        return ret/self.norm

    def recalc(self, p):
        """Recalculate the norm factor"""
        self.norm = (1 - p[self.pMixing])*Physics.thermalLorentz(0, 0, p[self.pGam], p[self.pXi],
                                                                 p[self.pColDirTrue], p[self.pOrder]) \
            + p[self.pMixing]*norm.pdf(0, loc=0, scale=p[self.pSigma])

    def leftEdge(self, p):
        """Return the left edge of the spectrum in Mhz"""
        return -5*p[self.pGam]

    def rightEdge(self, p):
        """Return the right edge of the spectrum in MHz"""
        return 5*p[self.pGam]

    def getPars(self, pos=0):
        """Return list of initial parameters and initialize positions"""
        self.pGam = pos
        self.pXi = pos + 1
        self.pColDirTrue = pos + 2
        self.pOrder = pos + 3

        self.pMixing = pos + 4
        self.pSigma = pos + 5

        self.p_n_of_peaks = pos + 6
        self.p_laserFreq = pos + 7

        self.pInt1 = pos + 8
        self.pCenter1 = pos + 9
        return [self.iso.shape['gamma'], self.iso.shape['compression'], self.iso.shape['col'],
                self.iso.shape['order'], self.iso.shape['mixing'], self.iso.shape['sigma'],
                self.iso.shape['nOfPeaks'], self.iso.shape['laserFreq'],
                self.iso.shape['int1'], self.iso.shape['center1']]

    def getParNames(self):
        """Return list of the parameter names"""
        return ['gamma', 'compression', 'col', 'order', 'mixing', 'sigma', 'nOfPeaks', 'laserFreq',
                'int1', 'center1']

    def getFixed(self):
        """Return list of parmeters with their fixed-status"""
        return [self.iso.fixShape['gamma'], self.iso.fixShape['compression'], self.iso.fixShape['col'],
                self.iso.fixShape['order'], self.iso.fixShape['mixing'], self.iso.fixShape['sigma'],
                self.iso.fixShape['nOfPeaks'], self.iso.fixShape['laserFreq'],
                self.iso.fixShape['int1'], self.iso.fixShape['center1']]

    def calc_diff_doppl(self, laser_freq, col):
        """ calculate the differential doppler factor for this shape and store it in self.diff_doppl """
        if laser_freq is not None:
            center_velocity = Physics.invRelDoppler(laser_freq, self.iso.freq + self.iso.center)
            center_velocity = - center_velocity if col else center_velocity
            center_volts = Physics.relEnergy(center_velocity, self.iso.mass * Physics.u) / Physics.qe
            self.diff_doppl = Physics.diffDoppler(laser_freq, center_volts, self.iso.mass, real=True)
