'''
Created on 25.04.2014

@author: hammen
'''

import importlib
from itertools import chain

import numpy as np

from Spectra.Hyperfine import Hyperfine


class FullSpec(object):
    '''
    A full spectrum function consisting of offset + all isotopes in iso with lineshape as declared in iso
    '''


    def __init__(self, iso, iso_m=None):
        '''
        Import the shape and initializes reasonable values 
        '''
        shapemod = importlib.import_module('Spectra.' + iso.shape['name'])
        shape = getattr(shapemod, iso.shape['name'])
        self.shape = shape(iso)
        self.iso = iso
        
        self.pOff = 0
        
        miso = iso
        self.hyper = []
        while miso != None:
            self.hyper.append(Hyperfine(miso, self.shape))
            miso = miso.m
        miso_m = iso_m
        while miso_m!=None:
            self.hyper.append(Hyperfine(miso_m, self.shape))
            miso_m = miso_m.m
        self.nPar = 1 + self.shape.nPar + sum(hf.nPar for hf in self.hyper)
        
        
    def evaluate(self, x, p, ih = -1):
        '''Return the value of the hyperfine structure at point x / MHz'''
        if ih == -1:
            return p[self.pOff] + sum(hf.evaluate(x, p) for hf in self.hyper)
        else:
            return p[self.pOff] + self.hyper[ih].evaluate(x, p)
    
    
    def evaluateE(self, e, freq, col, p, ih = -1):
        '''Return the value of the hyperfine structure at point e / eV'''
        if ih == -1:
            return p[self.pOff] + sum(hf.evaluateE(e, freq, col, p) for hf in self.hyper)
        else:
            return p[self.pOff] + self.hyper[ih].evaluateE(e, freq, col, p)


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
        return (['offset'] + self.shape.getParNames() + list(chain(*([hf.getParNames() for hf in self.hyper]))))
    
    
    def getFixed(self):
        '''Return list of parmeters with their fixed-status'''
        return [False] + self.shape.getFixed() + list(chain(*[hf.getFixed() for hf in self.hyper]))
    
    
    def parAssign(self):
        '''Return [(hf.name, parAssign)], where parAssign is a boolean list indicating relevant parameters'''
        ret = []
        i = 1 + self.shape.nPar
        a = [False] * self.nPar
        a[0:3] = [True] * 3
        for hf in self.hyper:
            assi = list(a)
            assi[i:(i+hf.nPar)] = [True] * hf.nPar
            i += hf.nPar
            
            ret.append((hf.iso.name, assi))
            
        return ret
    
    def toPlot(self, p, prec = 10000):
        '''Return ([x/Mhz], [y]) values with prec number of points'''
        self.recalc(p)
        return ([x for x in np.linspace(self.leftEdge(p), self.rightEdge(p), prec)],
                [self.evaluate(x, p) for x in np.linspace(self.leftEdge(p), self.rightEdge(p), prec)])
      
    def toPlotE(self, freq, col, p, prec = 10000):
        '''Return ([x/eV], [y]) values with prec number of points'''
        self.recalc(p)
        return ([x for x in np.linspace(self.leftEdgeE(freq, p), self.rightEdgeE(freq, p), prec)],
                [self.evaluateE(x, freq, col, p) for x in np.linspace(self.leftEdgeE(freq, p), self.rightEdgeE(freq, p), prec)])
    
           
    def leftEdge(self, p):
        '''Return the left edge of the spectrum in Mhz'''
        return min(hf.leftEdge(p) for hf in self.hyper)
    
    
    def rightEdge(self, p):
        '''Return the right edge of the spectrum in MHz'''
        return max(hf.rightEdge(p) for hf in self.hyper)
    
    
    def leftEdgeE(self, freq, p):
        '''Return the left edge of the spectrum in eV'''
        return min(hf.leftEdgeE(freq, p) for hf in self.hyper)
    
    
    def rightEdgeE(self, freq, p):
        '''Return the right edge of the spectrum in eV'''
        return max(hf.rightEdgeE(freq, p) for hf in self.hyper)