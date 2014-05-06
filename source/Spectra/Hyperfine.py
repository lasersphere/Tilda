'''
Created on 23.03.2014

@author: hammen
'''

import Physics

class Hyperfine(object):
    '''
    This class 
    '''

    def __init__(self, iso, shape):
        '''
        Constructor
        '''
        print("Creating hyperfine structure for", iso.name)
        self.iso = iso
        self.shape = shape

        self.fixA = iso.fixArat
        self.fixB = iso.fixBrat
        self.fixInt = iso.fixInt
        
        self.Al = iso.Al
        self.Bl = iso.Bl
        self.Au = iso.Au * iso.Al if self.fixA else iso.Au
        self.Bu = iso.Bu * iso.Bl if self.fixB else iso.Bu
        
        self.trans = Physics.HFTrans(self.iso.I, self.iso.Ju, self.iso.Jl)
        self.lineSplit = Physics.HFLineSplit(self.Al, self.Bl, self.Au, self.Bu, self.trans)
        
        self.nPar = 5 + len(self.trans)
        
        self.pCenter = 0
        self.pAl = 1
        self.pBl = 2
        self.pAu = 3
        self.pBu = 4
        self.pInt = 5
        
        
    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x'''    
            
        rx = x - p[self.pCenter]
        
        return sum(i * self.shape.evaluate(rx - j, p) for i, j in zip(self.intens, self.lineSplit))
            

    def evaluateE(self, e, freq, col, p):
        '''Return the intensity of the hyperfine structure for ions with energy e/eV'''
        v = Physics.relVelocity(Physics.qe * e, self.iso.mass * Physics.u)
        v = -v if col else v
        
        f = Physics.relDoppler(freq, v)
        
        return self.evaluate(f, p)
    
    
    def recalc(self, p):
            self.Al = p[self.pAl]
            self.Bl = p[self.pBl]
            #Use upper factors as ratio if fixed, else directly
            self.Au = p[self.pAu] * p[self.pAl] if self.fixA else p[self.pAu]
            self.Bu = p[self.pBu] * p[self.pBl] if self.fixB else p[self.pBu]

            self.lineSplit = Physics.HFLineSplit(self.Al, self.Bl, self.Au, self.Bu, self.trans)
            self.intens = self.buildInt(p)
    
  
    def getPars(self, pos = 0):
        self.pCenter = pos
        self.pAl = pos + 1
        self.pBl = pos + 2
        self.pAu = pos + 3
        self.pBu = pos + 4        
        self.pInt = pos + 5
        
        if self.fixInt:
            if len(self.iso.relInt) != len(self.trans):
                print("List of relative intensities has to consist of", len(self.trans), "elements!")
            ret = [i / self.iso.relInt[0] for i in self.iso.relInt]
            ret[0] *= self.iso.intScale
        else:
            ret = [self.iso.intScale * x for x in Physics.HFInt(self.iso.I, self.iso.Ju, self.iso.Jl, self.trans)]
        return ([self.iso.center, self.iso.Al, self.iso.Bl, self.iso.Au, self.iso.Bu]
                + ret)
    
    
    def getParNames(self):
        return ['center', 'Al', 'Bl', 'Au', 'Bu',] + ['Int' + str(x) for x in range(len(self.trans))]
    
    
    def getFixed(self):
        """Return the fix status for fitting of parameters"""
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

        return [False] + ret + fInt
    
    
    def buildInt(self, p):
        '''If relative intensities are fixed, calculate absolute intensities. Else return relevant parameters directly'''
        ret = len(self.trans) * [0]        

        for i in range(len(self.trans)):
            if(self.fixInt and i > 0):
                ret[i] = ret[0] * p[self.pInt + i]
            else:
                ret[i] = p[self.pInt + i]
                
        return ret
    
    
    def leftEdge(self):
        return min(self.lineSplit) + self.shape.leftEdge()
    
    
    def rightEdge(self):
        return max(self.lineSplit) + self.shape.rightEdge()
        


        
    