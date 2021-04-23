"""
Created on 23.03.2014

@author: hammen, pamueller (HyperfineN)
"""

import Physics


class Hyperfine(object):
    """
    A single hyperfine spectrum built from an Isotope object
    """

    def __init__(self, iso, shape):
        """
        Initialize values, calculate number of transitions and initial line positions
        """
        print("Creating hyperfine structure for", iso.name)
        self.iso = iso
        self.shape = shape

        self.fixA = iso.fixArat
        self.fixB = iso.fixBrat
        self.fixedAl = iso.fixedAl
        self.fixedBl = iso.fixedBl
        self.fixedAu = iso.fixedAu
        self.fixedBu = iso.fixedBu
        self.fixInt = iso.fixInt
        self.center = iso.center
        
        self.trans = Physics.HFTrans(self.iso.I, self.iso.Jl, self.iso.Ju)
        self.hfInt = Physics.HFInt(self.iso.I, self.iso.Jl, self.iso.Ju, self.trans)

        Au = iso.Au * iso.Al if self.fixA and not self.fixedAu else iso.Au
        Bu = iso.Bu * iso.Bl if self.fixB and not self.fixedBu else iso.Bu
        self.lineSplit = Physics.HFLineSplit(iso.Al, iso.Bl, Au, Bu, self.trans)

        self.nPar = 5 + 2 * len(self.trans) if self.iso.shape['name'] == 'LorentzQI' else 5 + len(self.trans)

        self.pCenter = 0
        self.pAl = 1
        self.pBl = 2
        self.pAu = 3
        self.pBu = 4
        self.pInt = 5
        self.pIntCross = 5 + len(self.trans)

    def evaluate(self, x, p):
        """Return the value of the hyperfine structure at point x/MHz"""   
        rx = x - p[self.pCenter]
        if self.iso.shape['name'] != 'LorentzQI':
            return sum(i * self.shape.evaluate(rx - j, p) for i, j in zip(self.intens, self.lineSplit))
        else:
            return sum(i * self.shape.evaluate(rx - j, p) for i, j in zip(self.intens, self.lineSplit)) \
                   + sum(i * self.shape.evaluateQI(rx - j, p, j, self.lineSplit) for i, j in zip(self.IntCross, self.lineSplit))

    def evaluateE(self, e, freq, col, p):
        """Return the value of the hyperfine structure at point e/eV"""
        v = Physics.relVelocity(Physics.qe * e, self.iso.mass * Physics.u)
        v = -v if col else v

        f = Physics.relDoppler(freq, v) - self.iso.freq
        
        return self.evaluate(f, p)
    
    def recalc(self, p):
        """Recalculate upper A and B factors, line splittings and intensities"""
        #Use upper factors as ratio if fixed, else directly
        Au = p[self.pAu] * p[self.pAl] if self.fixA else p[self.pAu]
        Bu = p[self.pBu] * p[self.pBl] if self.fixB else p[self.pBu]

        self.lineSplit = Physics.HFLineSplit(p[self.pAl], p[self.pBl], Au, Bu, self.trans)
        self.intens = self.buildInt(p)
        self.IntCross = self.buildIntCross(p)

    def getPars(self, pos=0, int_f=1):
        """
        Return list of initial parameters and initialize positions
        :param pos:
        :param int_f: possibility to scale the intensities with a factor (used when offset parameter is guessed)
        :return:
        """
        self.pCenter = pos
        self.pAl = pos + 1
        self.pBl = pos + 2
        self.pAu = pos + 3
        self.pBu = pos + 4        
        self.pInt = pos + 5
        self.pIntCross = pos + 5 + len(self.trans)

        if self.fixInt:
            if self.iso.shape['name'] == 'LorentzQI':
                if len(self.iso.relInt) != 2 * len(self.trans):
                    print("List of relative intensities has to consist of", len(self.trans), "elements! \n"
                                                                    "Using RACAH coefficients instead.... \n"
                                                                    "The sign of some CrossIntensities could be wrong!")
                    ret = [x for x in Physics.HFInt(self.iso.I, self.iso.Jl, self.iso.Ju, self.trans)]
                    div = ret[0]
                    for i in range(0, len(ret)):
                        ret[i] = ret[i] / div
                    ret[0] *= self.iso.intScale*int_f
                    self.IntCross = [g * (- 0.2) for g in ret] if self.iso.shape['name'] == 'LorentzQI' else []
                else:
                    ret = [i / self.iso.relInt[0] for i in self.iso.relInt[:len(self.trans)]]
                    self.IntCross = [i / self.iso.relInt[0] for i in self.iso.relInt[len(self.trans):]]
                    ret[0] *= self.iso.intScale*int_f
            else:
                self.IntCross = []
                if len(self.iso.relInt) != len(self.trans):
                    print("List of relative intensities has to consist of", len(self.trans), "elements! \n"
                                                                    "Using RACAH coefficients instead....")
                    ret = [x for x in Physics.HFInt(self.iso.I, self.iso.Jl, self.iso.Ju, self.trans)]
                    div = ret[0]
                    for i in range(0, len(ret)):
                        ret[i] = ret[i]/div
                    ret[0] *= self.iso.intScale*int_f
                else:
                    ret = [i / self.iso.relInt[0] for i in self.iso.relInt]
                    ret[0] *= self.iso.intScale*int_f
        else:
            ret = [self.iso.intScale*int_f * x for x in Physics.HFInt(self.iso.I, self.iso.Jl, self.iso.Ju, self.trans)]
            self.IntCross = [g * (- 0.2) for g in ret] if self.iso.shape['name'] == 'LorentzQI' else []

        # faktor -0.2 is the mean ratio to Int

        return ([self.iso.center, self.iso.Al, self.iso.Bl, self.iso.Au, self.iso.Bu]
                + ret + self.IntCross)

    def getParNames(self):
        """Return list of the parameter names"""
        if self.iso.shape['name'] == 'LorentzQI':
            return ['center', 'Al', 'Bl', 'Au', 'Bu'] + ['Int' + str(x) for x in range(len(self.trans))] +\
                   ['IntCross' + str(x) for x in range(len(self.trans))]
        else:
            return ['center', 'Al', 'Bl', 'Au', 'Bu'] + ['Int' + str(x) for x in range(len(self.trans))]
    
    def getFixed(self):
        """
        Return list of parameters with their fixed-status
        par names see: self.getParNames
        ['center', 'Al', 'Bl', 'Au', 'Bu'] + [... int etc. ...]
        """
        ret = 4 * [True]
        if self.iso.I > 0.1 and self.iso.Jl > 0.1 and not self.fixedAl:
            # Al
            ret[0] = False
        if self.iso.I > 0.6 and self.iso.Jl > 0.6 and not self.fixedBl:
            # Bl
            ret[1] = False
        if self.iso.I > 0.1 and self.iso.Ju > 0.1 and not self.fixA and not self.fixedAu:
            # Au
            ret[2] = False
        if self.iso.I > 0.6 and self.iso.Ju > 0.6 and not self.fixB and not self.fixedBu:
            # Bu
            ret[3] = False
         
        fInt = [False] + (len(self.trans) - 1)*[True] if self.fixInt else len(self.trans)*[False]
        # Check whether any IntX pars were set to fix in the lines db fixShape dictionary?
        fInt = [self.iso.fixShape.get('Int{}'.format(i), fInt[i]) for i in range(len(self.trans))]

        if self.iso.shape['name'] == 'LorentzQI':
            if self.fixInt:
                fCross = len(self.trans) * [True]
            else:
                fCross = len(self.trans) * [False]
        else:
            fCross=[]

        return [False] + ret + fInt + fCross
    
    def buildInt(self, p):
        """If relative intensities are fixed, calculate absolute intensities.
         Else return relevant parameters directly"""
        ret = len(self.trans) * [0]
        for i in range(len(self.trans)):
            if(self.fixInt and i > 0):
                ret[i] = ret[0] * p[self.pInt + i]
            else:
                ret[i] = p[self.pInt + i]
                
        return ret

    def buildIntCross(self,p):
        """If relative intensities are fixed in LorentzQI shape, calculate absolute intensities.
         Else return relevant parameters directly"""
        if self.iso.shape['name'] == 'LorentzQI':
            ret = len(self.trans) * [0]
            for i in range(len(self.trans)):
                if (self.fixInt and i > 0):
                    ret[i] = ret[0] * p[self.pIntCross + i]
                else:
                    ret[i] = p[self.pIntCross + i]
        else:
            ret = []

        return ret

    def leftEdge(self, p):
        """Return the left edge of the spectrum in Mhz"""
        self.recalc(p)
        return p[self.pCenter] + min(self.lineSplit) + self.shape.leftEdge(p)
    
    def rightEdge(self, p):
        """Return the right edge of the spectrum in MHz"""
        self.recalc(p)
        return p[self.pCenter] + max(self.lineSplit) + self.shape.rightEdge(p)

    def leftEdgeE(self, freq, p):
        """Return the left edge of the spectrum in eV"""
        self.recalc(p)
        l = p[self.pCenter] + min(self.lineSplit) + self.shape.leftEdge(p) + self.iso.freq
        v = Physics.invRelDoppler(freq, l)

        return (self.iso.mass * Physics.u * v**2)/2 / Physics.qe
    
    def rightEdgeE(self, freq, p):
        """Return the right edge of the spectrum in eV"""
        self.recalc(p)
        r = p[self.pCenter] + max(self.lineSplit) + self.shape.rightEdge(p) + self.iso.freq
        v = Physics.invRelDoppler(freq, r)

        return (self.iso.mass * Physics.u * v**2)/2 / Physics.qe


class HyperfineN(Hyperfine):
    """
    A single hyperfine spectrum built from an Isotope object where each fundamental peak itself consists of n peaks,
     for example due to more than one velocity class. After changing 'nPeaks' the parameters
     have to be saved to the database and the spectrum reloaded, since changing 'nPeaks' changes the parameter space.
    """

    def __init__(self, iso, shape):
        """
        Initialize values, calculate number of transitions and initial line positions
        """
        super().__init__(iso, shape)
        self.p_n_peaks = 0
        self.n_peaks = 1
        self.nParBase = self.nPar + 1
        self.nPar = self.nParBase

    def set_peaks(self, n):
        n = 1 if n < 1 else int(n)
        if n == self.n_peaks:
            return
        for i in range(n - 1):
            setattr(self, 'pRelCenter{}'.format(i), 1 + i)
            setattr(self, 'pRelInt{}'.format(i), n + i)
        self.n_peaks = n
        self.nPar = self.nParBase + 2 * n

    def evaluate(self, x, p):
        """Return the value of the hyperfine structure at point x/MHz"""
        center_pars = [0.] + [p[getattr(self, 'pRelCenter{}'.format(i))] for i in range(self.n_peaks - 1)]
        int_pars = [1.] + [p[getattr(self, 'pRelInt{}'.format(i))] for i in range(self.n_peaks - 1)]
        return sum(super(HyperfineN, self).evaluate(x - x0, p) * relInt for x0, relInt in zip(center_pars, int_pars))

    def getPars(self, pos=0, int_f=1):
        """Return list of initial parameters and initialize positions"""
        self.p_n_peaks = pos
        self.set_peaks(self.iso.shape.get('nPeaks', 1))
        super_pars = super().getPars(pos + 1 + 2 * (self.n_peaks - 1), int_f)
        for i in range(self.n_peaks - 1):
            setattr(self, 'pRelCenter{}'.format(i), pos + 1 + i)
            setattr(self, 'pRelInt{}'.format(i), pos + self.n_peaks + i)
        center_pars = [self.iso.shape.get('relCenter{}'.format(i), 0.) for i in range(self.n_peaks - 1)]
        int_pars = [self.iso.shape.get('relInt{}'.format(i), 1.) for i in range(self.n_peaks - 1)]
        return [self.n_peaks] + center_pars + int_pars + super_pars

    def getParNames(self):
        """Return list of the parameter names"""
        super_par_names = super().getParNames()
        center_names = ['relCenter{}'.format(i) for i in range(self.n_peaks - 1)]
        int_names = ['relInt{}'.format(i) for i in range(self.n_peaks - 1)]
        return ['nPeaks'] + center_names + int_names + super_par_names

    def getFixed(self):
        """Return list of parameters with their fixed-status"""
        super_fixed = super().getFixed()
        center_fixed = [self.iso.fixShape.get('relCenter{}'.format(i), False) for i in range(self.n_peaks - 1)]
        int_fixed = [self.iso.fixShape.get('relInt{}'.format(i), False) for i in range(self.n_peaks - 1)]
        return [True] + center_fixed + int_fixed + super_fixed

    def leftEdge(self, p):
        """Return the left edge of the spectrum in Mhz"""
        self.recalc(p)
        center_pars = [p[self.pCenter]] + [p[self.pCenter] + p[getattr(self, 'pRelCenter{}'.format(i))]
                                           for i in range(self.n_peaks - 1)]
        return min(center_pars) + min(self.lineSplit) + self.shape.leftEdge(p)

    def rightEdge(self, p):
        """Return the right edge of the spectrum in MHz"""
        self.recalc(p)
        center_pars = [p[self.pCenter]] + [p[self.pCenter] + p[getattr(self, 'pRelCenter{}'.format(i))]
                                           for i in range(self.n_peaks - 1)]
        return max(center_pars) + max(self.lineSplit) + self.shape.rightEdge(p)

    def leftEdgeE(self, freq, p):
        """Return the left edge of the spectrum in eV"""
        self.recalc(p)
        center_pars = [p[self.pCenter]] + [p[self.pCenter] + p[getattr(self, 'pRelCenter{}'.format(i))]
                                           for i in range(self.n_peaks - 1)]
        l = min(center_pars) + min(self.lineSplit) + self.shape.leftEdge(p) + self.iso.freq
        v = Physics.invRelDoppler(freq, l)

        return (self.iso.mass * Physics.u * v ** 2) / 2 / Physics.qe

    def rightEdgeE(self, freq, p):
        """Return the right edge of the spectrum in eV"""
        self.recalc(p)
        center_pars = [p[self.pCenter]] + [p[self.pCenter] + p[getattr(self, 'pRelCenter{}'.format(i))]
                                           for i in range(self.n_peaks - 1)]
        r = max(center_pars) + max(self.lineSplit) + self.shape.rightEdge(p) + self.iso.freq
        v = Physics.invRelDoppler(freq, r)

        return (self.iso.mass * Physics.u * v ** 2) / 2 / Physics.qe
