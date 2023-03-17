'''
Created on 20.07.18

@author: Imgram
'''

from Tilda.PolliFit import Physics, MPLPlotter as mpl


class DipVoigt(object):
    """
    Implementation of a dip voigt profile object using the Faddeeva function
    with a big peak in distance centerPeak relative to dip center and
     relative intensity IntPeak to the main peak; separate sigPeak & gamPeak
     distance will be assumed to be in MHz.


    The peak height is normalized to one
    Sigma is the standard deviation of the gaussian
    Gamma is the half width half maximum

    line needs to have:
    'sigmaDip', 'gammaDip', 'sigmaPeak', 'gammaPeak', 'centerPeak', 'IntPeak'

    optional:
    'offsetSlope' (for an linear offset added to the voigt)

    """

    def __init__(self, iso):
        '''Initialize'''
        self.iso = iso
        self.nPar = 6

        self.norm = 1
        self.peak_norm = 1  # norming facotr for dip

        self.pSigDip = 0
        self.pGamDip = 1
        self.pSigPeak = 2
        self.pGamPeak = 3
        self.p_peak_center = 4  # position of the center of the 2nd voigt -> Big peak
        self.p_peak_int = 5  # position of the relative intensity in the big peak
        self.recalc([iso.shape['sigmaDip'], iso.shape['gammaDip'], iso.shape['sigmaPeak'], iso.shape['gammaPeak'], iso.shape['centerPeak'],
                     iso.shape['IntPeak']])

    def evaluate(self, x, p):
        """ Return the value of the hyperfine structure at point x / MHz """
        ret = Physics.voigt(x - p[self.p_peak_center], p[self.pSigPeak], p[self.pGamPeak]) / self.peak_norm
        ret -= Physics.voigt(x, p[self.pSigDip], p[self.pGamDip]) / self.norm

        return ret
    
    def recalc(self, p):
        """Recalculate the norm factor"""
        self.norm = Physics.voigt(0, p[self.pSigDip], p[self.pGamDip])
        self.peak_norm = Physics.voigt(0, p[self.pSigPeak], p[self.pGamPeak]) / p[self.p_peak_int]

    def leftEdge(self, p):
        """Return the left edge of the spectrum in Mhz"""
        return -10 * (p[self.pSigPeak] + p[self.pGamPeak])
    
    def rightEdge(self, p):
        """Return the right edge of the spectrum in MHz"""
        return 10 * (p[self.pSigPeak] + p[self.pGamPeak])
    
    def getPars(self, pos = 0):
        """Return list of initial parameters and initialize positions"""
        self.pSigDip = pos
        self.pGamDip = pos + 1
        self.pSigPeak = pos + 2
        self.pGamPeak = pos + 3
        self.p_peak_center = pos + 4
        self.p_peak_int = pos + 5

        return [self.iso.shape['sigmaDip'], self.iso.shape['gammaDip'], self.iso.shape['sigmaPeak'], self.iso.shape['gammaPeak'],
                self.iso.shape['centerPeak'], self.iso.shape['IntPeak']]
    
    def getParNames(self):
        '''Return list of the parameter names'''
        return ['sigmaDip', 'gammaDip', 'sigmaPeak', 'gammaPeak', 'centerPeak', 'IntPeak']
    
    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [self.iso.fixShape['sigmaDip'], self.iso.fixShape['gammaDip'], self.iso.fixShape['sigmaPeak'],
                self.iso.fixShape['gammaPeak'], self.iso.fixShape['centerPeak'], self.iso.fixShape['IntPeak']]


if __name__ == "__main__":
    from Tilda.PolliFit.DummyIsotope import DummyIsotope
    from Tilda.PolliFit.Spectra.FullSpec import FullSpec

    shape = {'name': 'DipVoigt', 'gauDip': 0.00001, 'lorDip': 5, 'gauPeak': 5, 'lorPeak': 5,
             'centerPeak': 10.0, 'IntPeak': 0.1, 'laserFreq': None, 'col': False, 'offsetSlope': 0, 'offset': 0}
    fixShape = {'gauDip': False, 'lorDip': False, 'gauPeak': False, 'lorPeak': False,
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
                0, 0, 2000.0, 0,
                None, None, 0, 0,
                0, 0]

    iso = DummyIsotope(True, 'tester', line_list=line_list, iso_list=iso_list)
    spec = FullSpec(iso)
    to_plot = spec.toPlot(spec.getPars())
    mpl.plot(to_plot)
    mpl.show(True)




