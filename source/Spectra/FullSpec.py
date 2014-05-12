'''
Created on 25.04.2014

@author: hammen
'''

from itertools import chain
import importlib

import numpy as np

from Spectra.Hyperfine import Hyperfine
import Physics

class FullSpec(object):
    '''
    A full spectrum function consisting of offset + all isotopes in iso with lineshape as declared in iso
    '''


    def __init__(self, iso):
        '''
        Import the shape and initializes reasonable values 
        '''
        shapemod = importlib.import_module('Spectra.' + iso.shape['name'])
        shape = getattr(shapemod, iso.shape['name'])
        self.shape = shape(iso)
        
        self.pOff = 0
        self.hN = ['G', 'I', 'I2', 'I3']    #This is somewhat crude but well enough for now. Some iso.short scheme maybe?
        
        miso = iso
        self.hyper = []
        while miso != None:
            self.hyper.append(Hyperfine(miso, self.shape))
            miso = miso.m

        self.nPar = 1 + self.shape.nPar + sum(hf.nPar for hf in self.hyper)
        
        
    def evaluate(self, x, p):
        '''Return the value of the hyperfine structure at point x / MHz'''            
        return p[self.pOff] + sum(hf.evaluate(x, p) for hf in self.hyper)
    
    
    def evaluateE(self, e, freq, col, p):
        '''Return the value of the hyperfine structure at point e / eV'''
        return p[self.pOff] + sum(hf.evaluateE(e, freq, col, p) for hf in self.hyper)


    def recalc(self, p):
        '''Forward recalc to lower objects'''
        self.shape.recalc(p)
        for hf in self.hyper:
            hf.recalc(p)
     
  
    def getPars(self, pos = 0):
        '''Return list of initial parameters and initialize positions'''
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
        '''Return list of the parameter names'''
        return (['offset'] + self.shape.getParNames()
                + list(chain(*([self.hN[i] + el for el in hf.getParNames()] for i, hf in enumerate(self.hyper)))))
    
    
    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [False] + self.shape.getFixed() + list(chain(*[hf.getFixed() for hf in self.hyper]))
        
        
    def leftEdge(self):
        '''Return the left edge of the spectrum in Mhz'''
        return min(hf.leftEdge() for hf in self.hyper)
    
    
    def rightEdge(self):
        '''Return the right edge of the spectrum in MHz'''
        return max(hf.rightEdge() for hf in self.hyper)
    
    
    def leftEdgeE(self, freq):
        '''Return the left edge of the spectrum in eV'''
        return min(hf.leftEdgeE(freq) for hf in self.hyper)
    
    
    def rightEdgeE(self, freq):
        '''Return the right edge of the spectrum in eV'''
        return max(hf.rightEdgeE(freq) for hf in self.hyper)
    
    def toPlot(self, p, prec = 10000):
        '''Return ([x/Mhz], [y]) values with prec number of points'''
        self.recalc(p)
        return ([x for x in np.linspace(self.leftEdge(), self.rightEdge(), prec)], [self.evaluate(x, p) for x in np.linspace(self.leftEdge(), self.rightEdge(), prec)])
      
    def toPlotE(self, freq, col, p, prec = 10000):
        '''Return ([x/eV], [y]) values with prec number of points'''
        self.recalc(p)
        return ([x for x in np.linspace(self.leftEdgeE(freq), self.rightEdgeE(freq), prec)], [self.evaluateE(x, freq, col, p) for x in np.linspace(self.leftEdgeE(freq), self.rightEdgeE(freq), prec)])