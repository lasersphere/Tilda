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
     distance will be assumed to be in eV if the laserFreq is a float.
     if the distance is desired in frequency, use 'laserFreq': None


    The peak height is normalized to one
    Sigma is the standard deviation of the gaussian
    Gamma is the half width half maximum

    line needs to have:
    'gau', 'lor', 'centerAsym', 'IntAsym', 'laserFreq', 'col', 'nOfPeaks'

    optional:
    'offsetSlope' (for an linear offset added to the voigt)

    """

    def __init__(self, iso):
        """Initialize"""
        self.iso = iso
        self.nPar = 7

        self.norm = 1
        self.asym_norm = 1  # norming facotr for all side peaks
        self.diff_doppl = 1
        self.calc_diff_doppl(iso.shape['laserFreq'], iso.shape['col'])

        self.pSig = 0
        self.pGam = 1
        self.p_asym_center = 2  # position of the center of the 2nd voigt -> asymmetric peak
        self.p_asym_int = 3  # position of the relative intensity in the asymmetric peak
        self.p_n_of_peaks = 4  # position of the number of peaks
        self.p_laserFreq = 5  # not really a parameter but is needed to calculate the differential doppler factor
        #  to determine the distance to the center in eV instead of MHz
        self.p_col = 6  # needed for eV instead of MHz center dif
        self.recalc([iso.shape.get('gau', iso.shape.get('sigma', 0.0)),
                     iso.shape.get('lor', iso.shape.get('gamma', 0.0)), iso.shape['centerAsym'],
                     iso.shape['IntAsym'], iso.shape['nOfPeaks'],
                     iso.shape['laserFreq'], iso.shape['col']]) # .get() structure due to naming difference in .getParNames() and shape['']

    def evaluate(self, x, p):
        """ Return the value of the hyperfine structure at point x / MHz """
        ret = Physics.voigt(x, p[self.pSig], p[self.pGam]) / self.norm

        for peak_num in range(p[self.p_n_of_peaks]):
            side_peak_freq = p[self.p_asym_center] * self.diff_doppl * (peak_num + 1)

            ret += Physics.voigt(x - side_peak_freq, p[self.pSig], p[self.pGam]) / (self.asym_norm * (2 ** peak_num))

        return ret

    def recalc(self, p):
        """Recalculate the norm factor"""
        self.norm = Physics.voigt(0, p[self.pSig], p[self.pGam])
        self.asym_norm = self.norm / p[self.p_asym_int]

    def leftEdge(self, p):
        """Return the left edge of the spectrum in Mhz"""
        return -10 * (p[self.pSig] + p[self.pGam])

    def rightEdge(self, p):
        """Return the right edge of the spectrum in MHz"""
        return 10 * (p[self.pSig] + p[self.pGam])

    def getPars(self, pos=0):
        """Return list of initial parameters and initialize positions"""
        self.pSig = pos
        self.pGam = pos + 1
        self.p_asym_center = pos + 2
        self.p_asym_int = pos + 3
        self.p_n_of_peaks = pos + 4
        self.p_laserFreq = pos + 5
        self.p_col = pos + 6

        return [self.iso.shape.get('gau', self.iso.shape.get('sigma', 0.0)),
                self.iso.shape.get('lor', self.iso.shape.get('gamma', 0.0)),
                self.iso.shape['centerAsym'], self.iso.shape['IntAsym'],
                self.iso.shape['nOfPeaks'], self.iso.shape['laserFreq'], self.iso.shape['col']] # .get() structure due to naming difference in .getParNames() and shape['']

    def getParNames(self):
        """Return list of the parameter names"""
        return ['sigma', 'gamma', 'centerAsym', 'IntAsym', 'nOfPeaks', 'laserFreq', 'col']

    def getFixed(self):
        """Return list of parmeters with their fixed-status"""
        return [self.iso.fixShape.get('gau', self.iso.fixShape.get('sigma', False)),
                self.iso.fixShape.get('lor', self.iso.fixShape.get('gamma', False)),
                self.iso.fixShape['centerAsym'], self.iso.fixShape['IntAsym'],
                self.iso.fixShape['nOfPeaks'], True, True] # .get() structure due to naming difference in .getParNames() and shape['']

    def calc_diff_doppl(self, laser_freq, col):
        """ calculate the differential doppler factor for this shape and store it in self.diff_doppl """
        if laser_freq is not None:
            center_velocity = Physics.invRelDoppler(laser_freq, self.iso.freq + self.iso.center)
            center_velocity = - center_velocity if col else center_velocity
            center_volts = Physics.relEnergy(center_velocity, self.iso.mass * Physics.u) / Physics.qe
            self.diff_doppl = Physics.diffDoppler(laser_freq, center_volts, self.iso.mass, real=True)
