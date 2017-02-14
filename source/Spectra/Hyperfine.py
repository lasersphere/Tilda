'''
Created on 23.03.2014

@author: hammen
'''

import Physics

class Hyperfine(object):
    '''
    A single hyperfine spectrum built from an Isotope object
    '''

    def __init__(self, iso, shape):
        '''
        Initialize values, calculate number of transitions and initial line positions
        '''
        print("Creating hyperfine structure for", iso.name)
        self.iso = iso
        self.shape = shape

        self.fixA = iso.fixArat
        self.fixB = iso.fixBrat
        self.fixInt = iso.fixInt
        self.center = iso.center
        
        self.trans = Physics.HFTrans(self.iso.I, self.iso.Jl, self.iso.Ju)

        Au = iso.Au * iso.Al if self.fixA else iso.Au
        Bu = iso.Bu * iso.Bl if self.fixB else iso.Bu
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
        '''Return the value of the hyperfine structure at point x/MHz'''   
        rx = x - p[self.pCenter]
        if self.iso.shape['name'] != 'LorentzQI':
            return sum(i * self.shape.evaluate(rx - j, p) for i, j in zip(self.intens, self.lineSplit))
        else:
            return sum(i * self.shape.evaluate(rx - j, p) for i, j in zip(self.intens, self.lineSplit)) \
                   + sum(i * self.shape.evaluateQI(rx - j, p, j, self.lineSplit) for i, j in zip(self.IntCross, self.lineSplit))



    def evaluateE(self, e, freq, col, p):
        '''Return the value of the hyperfine structure at point e/eV'''
        v = Physics.relVelocity(Physics.qe * e, self.iso.mass * Physics.u)
        v = -v if col else v

        f = Physics.relDoppler(freq, v) - self.iso.freq
        
        return self.evaluate(f, p)
    
    
    def recalc(self, p):
        '''Recalculate upper A and B factors, line splittings and intensities'''
        #Use upper factors as ratio if fixed, else directly
        Au = p[self.pAu] * p[self.pAl] if self.fixA else p[self.pAu]
        Bu = p[self.pBu] * p[self.pBl] if self.fixB else p[self.pBu]

        self.lineSplit = Physics.HFLineSplit(p[self.pAl], p[self.pBl], Au, Bu, self.trans)
        self.intens = self.buildInt(p)
        if self.iso.shape['name'] == 'LorentzQI':
            for i in range(len(self.trans)):
                self.IntCross[i] = p[self.pIntCross + i]
        else:
            self.IntCross = []
    
  
    def getPars(self, pos = 0):
        '''Return list of initial parameters and initialize positions'''
        self.pCenter = pos
        self.pAl = pos + 1
        self.pBl = pos + 2
        self.pAu = pos + 3
        self.pBu = pos + 4        
        self.pInt = pos + 5
        self.pIntCross = pos + 5 + len(self.trans)
        
        if self.fixInt:
            if len(self.iso.relInt) != len(self.trans):
                print("List of relative intensities has to consist of", len(self.trans), "elements! \n"
                                                                    "Using RACAH coefficients instead....")
                ret = [x for x in Physics.HFInt(self.iso.I, self.iso.Jl, self.iso.Ju, self.trans)]
                div = ret[0]
                for i in range(0, len(ret)):
                    ret[i] = ret[i]/div
                ret[0] *= self.iso.intScale
            else:
                ret = [i / self.iso.relInt[0] for i in self.iso.relInt]
                ret[0] *= self.iso.intScale
        else:
            ret = [self.iso.intScale * x for x in Physics.HFInt(self.iso.I, self.iso.Jl, self.iso.Ju, self.trans)]

        self.IntCross = [g * (- 0.2) for g in ret] if self.iso.shape['name'] == 'LorentzQI' else []
        # faktor -0.2 is the mean ratio to Int

        return ([self.iso.center, self.iso.Al, self.iso.Bl, self.iso.Au, self.iso.Bu]
                + ret + self.IntCross)
    
    
    def getParNames(self):
        '''Return list of the parameter names'''
        if self.iso.shape['name'] == 'LorentzQI':
            return ['center', 'Al', 'Bl', 'Au', 'Bu',] + ['Int' + str(x) for x in range(len(self.trans))] +\
                   ['IntCross' + str(x) for x in range(len(self.trans))]
        else:
            return ['center', 'Al', 'Bl', 'Au', 'Bu',] + ['Int' + str(x) for x in range(len(self.trans))]
    
    
    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        ret = 4*[True]
        if self.iso.I > 0.1 and self.iso.Jl > 0.1:
            ret[0] = False
        if self.iso.I > 0.6 and self.iso.Jl > 0.6:
            ret[1] = False
        if self.iso.I > 0.1 and self.iso.Ju > 0.1 and not self.fixA:
            ret[2] = False
        if self.iso.I > 0.6 and self.iso.Ju > 0.6 and not self.fixB:
            ret[3] = False
         
        fInt = [False] + (len(self.trans) - 1)*[True] if self.fixInt else len(self.trans)*[False]

        fCross = len(self.trans)*[False] if self.iso.shape['name'] == 'LorentzQI' else []

        return [False] + ret + fInt + fCross
    
    
    def buildInt(self, p):
        '''If relative intensities are fixed, calculate absolute intensities. Else return relevant parameters directly'''
        ret = len(self.trans) * [0]
        for i in range(len(self.trans)):
            if(self.fixInt and i > 0):
                ret[i] = ret[0] * p[self.pInt + i]
            else:
                ret[i] = p[self.pInt + i]
                
        return ret
    
    
    def leftEdge(self, p):
        '''Return the left edge of the spectrum in Mhz'''
        self.recalc(p)
        return p[self.pCenter] + min(self.lineSplit) + self.shape.leftEdge(p)
    
    
    def rightEdge(self, p):
        '''Return the right edge of the spectrum in MHz'''
        self.recalc(p)
        return p[self.pCenter] + max(self.lineSplit) + self.shape.rightEdge(p)
    
    
    def leftEdgeE(self, freq, p):
        '''Return the left edge of the spectrum in eV'''
        self.recalc(p)
        l = p[self.pCenter] + min(self.lineSplit) + self.shape.leftEdge(p) + self.iso.freq
        v = Physics.invRelDoppler(freq, l)

        return (self.iso.mass * Physics.u * v**2)/2 / Physics.qe
    
    
    def rightEdgeE(self, freq, p):
        '''Return the right edge of the spectrum in eV'''
        self.recalc(p)
        r = p[self.pCenter] + max(self.lineSplit) + self.shape.rightEdge(p) + self.iso.freq
        v = Physics.invRelDoppler(freq, r)

        return (self.iso.mass * Physics.u * v**2)/2 / Physics.qe
        


        
    