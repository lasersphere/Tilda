'''
Created on 23.02.2017

@author: Kaufmann
'''

import Physics


class AsymmetricVoigt(object):
    """
    Implementation of a voigt profile object using the Faddeeva function
    overlapped with a second voigt in distance centerAsym and
     relative intensity IntAsym to the main peak


    The peak height is normalized to one
    Sigma is the standard deviation of the gaussian
    Gamma is the half width half maximum

    line needs to have:
    'gau', 'lor', 'centerAsym', 'IntAsym'
    """

    def __init__(self, iso):
        '''Initialize'''
        self.iso = iso
        self.nPar = 4

        self.norm = 1
        self.asym_norm = 1

        self.pSig = 0
        self.pGam = 1
        self.p_asym_center = 2  # position of the center of the 2nd voigt -> asymmetric peak
        self.p_asym_int = 3  # position of the relative intensity in the asymmetric peak
        self.recalc([iso.shape['gau'], iso.shape['lor'], iso.shape['centerAsym'], iso.shape['IntAsym']])
    
    def evaluate(self, x, p):
        """ Return the value of the hyperfine structure at point x / MHz """
        ret = Physics.voigt(x, p[self.pSig], p[self.pGam]) / self.norm

        laser_freq = 851200725.9994  # TODO not  a fixed value!
        col = True  # TODO

        center_velocity = Physics.invRelDoppler(laser_freq,
                                              self.iso.freq + self.iso.center)
        center_velocity = - center_velocity if col else center_velocity
        center_volts = Physics.relEnergy(
            center_velocity, self.iso.mass * Physics.u) / Physics.qe

        diff_doppl_MHz = Physics.diffDoppler(
            self.iso.freq + self.iso.center,
            center_volts, self.iso.mass)
        side_peak_freq = p[self.p_asym_center] * diff_doppl_MHz

        ret += Physics.voigt(x - side_peak_freq, p[self.pSig], p[self.pGam]) / self.asym_norm

        return ret
    
    def recalc(self, p):
        """Recalculate the norm factor"""
        self.norm = Physics.voigt(0, p[self.pSig], p[self.pGam])
        self.asym_norm = self.norm / p[self.p_asym_int]

    def leftEdge(self, p):
        """Return the left edge of the spectrum in Mhz"""
        return -5 * (p[self.pSig] + p[self.pGam])
    
    def rightEdge(self, p):
        """Return the right edge of the spectrum in MHz"""
        return 5 * (p[self.pSig] + p[self.pGam])
    
    def getPars(self, pos = 0):
        """Return list of initial parameters and initialize positions"""
        self.pSig = pos
        self.pGam = pos + 1
        self.p_asym_center = pos + 2
        self.p_asym_int = pos + 3

        return [self.iso.shape['gau'], self.iso.shape['lor'],
                self.iso.shape['centerAsym'], self.iso.shape['IntAsym']]
    
    def getParNames(self):
        '''Return list of the parameter names'''
        return ['sigma', 'gamma', 'centerAsym', 'IntAsym']
    
    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [self.iso.fixShape['gau'], self.iso.fixShape['lor'],
                self.iso.fixShape['centerAsym'], self.iso.fixShape['IntAsym']]
