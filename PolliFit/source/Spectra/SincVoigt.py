'''
Created on 20.07.18

@author: Imgram
'''

import Physics


class SincVoigt(object):
    """
    Implementation of a dip voigt profile object using the Faddeeva function
    with a big peak in distance centerPeak relative to dip center and
     relative intensity IntPeak to the main peak; dip function will be sin x²/x²


    The peak height is normalized to one
    Sigma is the standard deviation of the gaussian
    Gamma is the half width half maximum

    line needs to have:
    'transitTimeDip', 'sigmaPeak', 'gammaPeak', 'centerPeak', 'IntPeak'

    optional:
    'offsetSlope' (for an linear offset added to the voigt)

    """

    def __init__(self, iso):
        '''Initialize'''
        self.iso = iso
        self.nPar = 5

        self.norm = 1
        self.peak_norm = 1  # norming facotr for dip

        self.pTransitDip = 0
        self.pSigPeak = 1
        self.pGamPeak = 2
        self.p_peak_center = 3  # position of the center of the 2nd voigt -> Big peak
        self.p_peak_int = 4 # position of the relative intensity in the big peak
        self.recalc([iso.shape['transitTimeDip'], iso.shape['sigmaPeak'], iso.shape['gammaPeak'], iso.shape['centerPeak'],
                     iso.shape['IntPeak']])

    def evaluate(self, x, p):
        """ Return the value of the hyperfine structure at point x / MHz """
        ret = Physics.voigt(x - p[self.p_peak_center], p[self.pSigPeak], p[self.pGamPeak]) / self.peak_norm
        ret -= Physics.transit(x, p[self.pTransitDip]) / self.norm

        return ret
    
    def recalc(self, p):
        """Recalculate the norm factor"""
        self.norm = Physics.transit(0, p[self.pTransitDip])
        self.peak_norm = Physics.voigt(0, p[self.pSigPeak], p[self.pGamPeak]) / p[self.p_peak_int]

    def leftEdge(self, p):
        """Return the left edge of the spectrum in Mhz"""
        return -10 * (p[self.pSigPeak] + p[self.pGamPeak])
    
    def rightEdge(self, p):
        """Return the right edge of the spectrum in MHz"""
        return 10 * (p[self.pSigPeak] + p[self.pGamPeak])
    
    def getPars(self, pos = 0):
        """Return list of initial parameters and initialize positions"""
        self.pTransitDip = pos
        self.pSigPeak = pos +1
        self.pGamPeak = pos + 2
        self.p_peak_center = pos + 3
        self.p_peak_int = pos + 4

        return [self.iso.shape['transitTimeDip'], self.iso.shape['sigmaPeak'], self.iso.shape['gammaPeak'],
                self.iso.shape['centerPeak'], self.iso.shape['IntPeak']]
    
    def getParNames(self):
        '''Return list of the parameter names'''
        return ['transitTimeDip', 'sigmaPeak', 'gammaPeak', 'centerPeak', 'IntPeak']
    
    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [self.iso.fixShape['transitTimeDip'], self.iso.fixShape['sigmaPeak'],
                self.iso.fixShape['gammaPeak'], self.iso.fixShape['centerPeak'], self.iso.fixShape['IntPeak']]


if __name__ == "__main__":
    from DummyIsotope import DummyIsotope
    from Spectra.FullSpec import FullSpec
    import MPLPlotter as mpl

    shape = {'name': 'SincVoigt', 'sigmaPeak': 5, 'gammaPeak': 5, 'transitTimeDip': 1e-6,
             'centerPeak': 10.0, 'IntPeak': 0, 'laserFreq': None, 'col': False, 'offsetSlope': 0, 'offset': 0}
    fixShape = {'sigmaPeak': False, 'gammaPeak': False, 'transitTimeDip': False,
                'centerPeak': False, 'IntPeak': False, 'laserFreq': True, 'col': True}
    line_list = ['tester', 850344226.10401, 3, 2,
                 str(shape),  # str for better compatibility with DummyIsotope
                 str(fixShape),
                 0
                 ]
    # mass_dbl, mass_d_dbl, I_dbl, center_dbl,
    # Al_dbl, Bl_dbl, Au_dbl, Bu_dbl,
    # fixedArat_bool, fixedBrat_bool, intScale_dbl, fixedInt_int(0/1),
    # relInt_str_list, m_str, fixedAl_int(0/1), fixedBl_int(0/1),
    # fixedAu_int(0/1), fixedBu_int(0/1)
    # use default settings -> 60Ni
    iso_list = [59.9307859, 5e-07, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0, 0, -2000.0, 0,
                None, None, 0, 0,
                0, 0]

    iso = DummyIsotope(True, 'tester', line_list=line_list, iso_list=iso_list)
    spec = FullSpec(iso)
    to_plot = spec.toPlot(spec.getPars())
    mpl.plot(to_plot)
    mpl.show(True)




