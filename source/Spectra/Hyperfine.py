'''
Created on 23.03.2014

@author: hammen
'''

import Physics

class Hyperfine(object):
    '''
    classdocs
    '''

    def __init__(self, iso, shape):
        '''
        Constructor
        '''
        self.iso = iso
        self.shape = shape
        
        self.Al = iso.Al
        self.Bl = iso.Bl
        self.Ar = iso.Ar
        self.Br = iso.Br
        
        self.trans = Physics.HFTrans(self.iso.I, self.iso.Ju, self.iso.Jl)
        self.lineSplit = Physics.HFLineSplit(self.Al, self.Bl, self.Al * self.Ar, self.Bl * self.Br, self.trans)
        
        self.nPar = 5 + len(self.trans)
        
        self.pCenter = 0
        self.pAl = 1
        self.pBl = 2
        self.pAr = 3
        self.pBr = 4    
        self.pInt = 5
        
    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x, recalculate line positions if necessary'''
        if(self.Al != p[self.pAl] or self.Bl != p[self.pBl] or self.Ar != p[self.pAr] or self.Br != p[self.pBr]):
            self.Al = p[self.pAl]
            self.Bl = p[self.pBl]
            self.Ar = p[self.pAr]
            self.Br = p[self.pBr]
            self.lineSplit = Physics.HFLineSplit(self.Al, self.Bl, self.Al * self.Ar, self.Bl * self.Br, self.trans)
            
        rx = x[0] - p[self.pCenter]
        return sum(p[self.pInt + i] * self.shape.evaluate(rx, p) for i in range(len(self.trans)))
  
    def getPars(self, pos = 0):
        self.pCenter = pos
        self.pAl = pos + 1
        self.pBl = pos + 2
        self.pAr = pos + 3
        self.pBr = pos + 4        
        self.pInt = pos + 5
        
        return ([self.iso.shift, self.iso.Al, self.iso.Bl, self.iso.Ar, self.iso.Br]
                + [self.iso.intScale * x for x in Physics.HFInt(self.iso.I, self.iso.Ju, self.iso.Jl, self.trans)])
    
    def getParNames(self):
        return ['center', 'Al', 'Bl', 'Ar', 'Br',] + ['Int' + x for x in range(len(self.trans))]
    
    def getFixed(self):
        return [False] * self.nPar
        
    def leftEdge(self):
        return min(self.lineSplit) + self.shape.leftEdge()
    
    def rightEdge(self):
        return max(self.lineSplit) + self.shape.rightEdge()
        


        
    