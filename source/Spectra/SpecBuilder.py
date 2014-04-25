'''
Created on 25.04.2014

@author: hammen
'''

from itertools import chain

from Spectra.Hyperfine import Hyperfine

class SpecBuilder(object):
    '''
    classdocs
    '''


    def __init__(self, iso, shape):
        '''
        Constructor
        '''
        self.pOff = 0
        self.shape = shape()
        
        miso = iso
        self.hyper = []
        self.hN = ['G', 'I']
        while miso != None:
            self.hyper.append(Hyperfine(iso, shape))
            miso = iso.m
            
        self.nPar = 1 + self.shape.nPar + sum(hf.nPar for hf in self.hyper)
        
    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x, recalculate line positions if necessary'''            
        return p[self.pOff] + sum(hf.evaluate(x, p) for hf in self.hyper)
  
    def getPars(self, pos = 0):
        self.pOff = pos
                
        ret = [0]
        pos, par = self.shape.getPars(pos + 1)
        ret += par
        for hf in self.hyper:
            pos, par = hf.getPars(pos)
            ret += par
            
        return ret
    
    def getParNames(self):
        return (['offset'] + self.shape.getParNames()
                + chain([self.hN[i] + el for el in hf.getParNames()] for i, hf in enumerate(self.hyper)))
    
    def getFixed(self):
        return [False] + self.shape.getFixed() + chain(hf.getFixed() for hf in self.hyper)
        
    def leftEdge(self):
        return min(hf.leftEdge() for hf in self.hyper)
    
    def rightEdge(self):
        return max(hf.rightEdge() for hf in self.hyper)