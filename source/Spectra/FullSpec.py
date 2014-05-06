'''
Created on 25.04.2014

@author: hammen
'''

from itertools import chain
import importlib

from Spectra.Hyperfine import Hyperfine

class FullSpec(object):
    '''
    classdocs
    '''


    def __init__(self, iso):
        '''
        Constructor
        '''
        self.pOff = 0
        
        shapemod = importlib.import_module('Spectra.' + iso.shape['name'])
        shape = getattr(shapemod, iso.shape['name'])
        
        self.shape = shape(iso)
        
        miso = iso
        self.hyper = []
        self.hN = ['G', 'I', 'I2', 'I3']
        while miso != None:
            self.hyper.append(Hyperfine(iso, self.shape))
            miso = iso.m

        self.nPar = 1 + self.shape.nPar + sum(hf.nPar for hf in self.hyper)
        
        
    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x, recalculate line positions if necessary'''            
        return p[self.pOff] + sum(hf.evaluate(x, p) for hf in self.hyper)
    
    
    def evaluateE(self, e, freq, col, p):
        return p[self.pOff] + sum(hf.evaluateE(e, freq, col, p) for hf in self.hyper)


    def recalc(self, p):
        self.shape.recalc(p)
        for hf in self.hyper:
            hf.recalc(p)
     
  
    def getPars(self, pos = 0):
        self.pOff = pos
        ret = [0]
        pos += 1
        
        ret += self.shape.getPars(pos)
        pos += self.shape.nPar

        for hf in self.hyper:
            ret += hf.getPars(pos)
            pos += hf.nPar
            
        return ret
    
    
    def getParNames(self):
        return (['offset'] + self.shape.getParNames()
                + list(chain(*([self.hN[i] + el for el in hf.getParNames()] for i, hf in enumerate(self.hyper)))))
    
    
    def getFixed(self):
        return [False] + self.shape.getFixed() + list(chain(*[hf.getFixed() for hf in self.hyper]))
        
        
    def leftEdge(self):
        return min(hf.leftEdge() for hf in self.hyper)
    
    
    def rightEdge(self):
        return max(hf.rightEdge() for hf in self.hyper)
    
    
    def leftEdgeE(self):
        pass
    
    
    def rightEdgeE(self):
        pass