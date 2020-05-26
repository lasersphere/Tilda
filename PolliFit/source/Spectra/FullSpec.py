"""
Created on 25.04.2014

@author: hammen, pamueller (track offsets)
"""

import importlib
from itertools import chain

import numpy as np

from Spectra.Hyperfine import HyperfineN
from Spectra.AsymmetricVoigt import AsymmetricVoigt
import Physics


class FullSpec(object):
    """
    A full spectrum function consisting of offset + all isotopes in iso with lineshape as declared in iso.
     Arbitrary many offsets can be defined in sections of the x-axis.
    """

    def __init__(self, iso, iso_m=None):
        """
        Import the shape and initializes reasonable values 
        """
        # for example: iso.shape['name'] = 'Voigt'
        shapemod = importlib.import_module('Spectra.' + iso.shape['name'])
        shape = getattr(shapemod, iso.shape['name'])
        # leading to: Spectra.Voigt.Voigt(iso)
        self.shape = shape(iso)
        self.iso = iso
        
        self.pOff = 0
        self.p_offset_slope = 1
        self.cut_x = {}

        miso = iso
        self.hyper = []
        while miso != None:
            self.hyper.append(HyperfineN(miso, self.shape))
            miso = miso.m
        miso_m = iso_m
        while miso_m!=None:
            self.hyper.append(HyperfineN(miso_m, self.shape))
            miso_m = miso_m.m
        self.nPar = 2 + self.shape.nPar + sum(hf.nPar for hf in self.hyper)
        
    def evaluate(self, x, p, ih=-1, full=False):
        """Return the value of the hyperfine structure at point x / MHz, ih=index hyperfine"""
        if not full and self.cut_x != {}:
            order = np.argsort(x)
            inverse_order = np.array([int(np.where(order == i)[0]) for i in range(order.size)])
            x_temp = x[order]
            cut_i = np.array([np.argwhere(x_temp < self.cut_x[track]).T[0][-1] + 1
                              if np.min(x_temp) < self.cut_x[track] else -x_temp.size
                              for track in range(len(self.cut_x.keys()))])
            x_ret = np.split(x_temp, cut_i)
            ret = np.concatenate([self.evaluate(x_i, p, ih=ih, full=True) + p[getattr(self, 'pOff{}'.format(track - 1))]
                                  if track > 0 else self.evaluate(x_i, p, ih=ih, full=True)
                                  for track, x_i in enumerate(x_ret)])
            return ret[inverse_order]
        if ih == -1:        
            return p[self.pOff] + x * p[self.p_offset_slope] + sum(hf.evaluate(x, p) for hf in self.hyper)
        else:
            return p[self.pOff] + x * p[self.p_offset_slope] + self.hyper[ih].evaluate(x, p)

    def evaluateE(self, e, freq, col, p, ih=-1):
        """Return the value of the hyperfine structure at point e / eV"""

        # used to be like the call in self.evaluate, but calling hf.evaluateE ...
        # since introducing the linear offset this caused problems, so now the frequency is already converted here

        v = Physics.relVelocity(Physics.qe * e, self.iso.mass * Physics.u)
        v = -v if col else v

        f = Physics.relDoppler(freq, v) - self.iso.freq

        return self.evaluate(f, p, ih)

    def recalc(self, p):
        """Forward recalc to lower objects"""
        self.shape.recalc(p)
        for hf in self.hyper:
            hf.recalc(p)

    def add_track_offsets(self, cut_x, freq, col):
        if self.cut_x != {} or cut_x == {}:
            return
        for track, cut in cut_x.items():
            v = Physics.relVelocity(Physics.qe * cut, self.iso.mass * Physics.u)
            v = -v if col else v
            k = int(max(cut_x.keys())) - track if col else track
            self.cut_x[k] = Physics.relDoppler(freq, v) - self.iso.freq
            setattr(self, 'pOff{}'.format(k), 2 + k)
        self.nPar += len(cut_x.keys())

    def getPars(self, pos=0):
        """Return list of initial parameters and initialize positions"""
        self.pOff = pos
        self.p_offset_slope = pos + 1
        for track in self.cut_x.keys():
            setattr(self, 'pOff{}'.format(track), pos + 2 + track)
        ret = [self.iso.shape.get('offset', 0), self.iso.shape.get('offsetSlope', 0)]
        ret += [self.iso.shape.get('offset{}'.format(track), 0) for track in range(len(self.cut_x.keys()))]
        pos += len(ret)
        ret += self.shape.getPars(pos)
        pos += self.shape.nPar

        for hf in self.hyper:
            ret += hf.getPars(pos)
            pos += hf.nPar
            
        return ret

    def getParNames(self):
        """Return list of the parameter names"""
        return ['offset', 'offsetSlope'] + ['offset{}'.format(track) for track in range(len(self.cut_x.keys()))]\
            + self.shape.getParNames() + list(chain(*([hf.getParNames() for hf in self.hyper])))
    
    def getFixed(self):
        """Return list of parmeters with their fixed-status"""
        return [self.iso.fixShape.get('offset', False), self.iso.fixShape.get('offsetSlope', True)]\
            + [self.iso.fixShape.get('offset{}'.format(track), False) for track in range(len(self.cut_x.keys()))]\
            + self.shape.getFixed() + list(chain(*[hf.getFixed() for hf in self.hyper]))

    def parAssign(self):
        '''Return [(hf.name, parAssign)], where parAssign is a boolean list indicating relevant parameters'''
        # TODO: This is stupid, why would we want to hardcode which parameters are relevant?
        #  I'd say just output them all! Or is there anything speaking against that? (Felix May2020)
        ret = []
        i = 2 + self.shape.nPar  # 2 for offset, offsetSlope
        a = [True] * self.nPar  # must be False if below code is to be used:
        # if isinstance(self.shape, AsymmetricVoigt):
        #     a[0:6] = [True] * 6  # 'offset', 'offsetSlope', 'sigma', 'gamma', 'centerAsym', 'IntAsym'
        # else:
        #     a[0:4] = [True] * 4  # 'offset', 'offsetSlope', 'sigma', 'gamma'  for normal voigt
        for hf in self.hyper:
            assi = list(a)
            assi[i:(i+hf.nPar)] = [True] * hf.nPar
            i += hf.nPar
            
            ret.append((hf.iso.name, assi))
            
        return ret
    
    def toPlot(self, p, prec=10000):
        """Return ([x/Mhz], [y]) values with prec number of points"""
        self.recalc(p)
        x = np.linspace(self.leftEdge(p), self.rightEdge(p), prec)
        return x, self.evaluate(x, p)
      
    def toPlotE(self, freq, col, p, prec=10000):
        """Return ([x/eV], [y]) values with prec number of points"""
        self.recalc(p)
        x = np.linspace(self.leftEdgeE(freq, p), self.rightEdgeE(freq, p), prec)
        return x, self.evaluateE(x, freq, col, p)

    def leftEdge(self, p):
        """Return the left edge of the spectrum in Mhz"""
        return min(hf.leftEdge(p) for hf in self.hyper)

    def rightEdge(self, p):
        """Return the right edge of the spectrum in MHz"""
        return max(hf.rightEdge(p) for hf in self.hyper)

    def leftEdgeE(self, freq, p):
        """Return the left edge of the spectrum in eV"""
        return min(hf.leftEdgeE(freq, p) for hf in self.hyper)

    def rightEdgeE(self, freq, p):
        """Return the right edge of the spectrum in eV"""
        return max(hf.rightEdgeE(freq, p) for hf in self.hyper)
