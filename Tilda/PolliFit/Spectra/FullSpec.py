"""
Created on 25.04.2014

@author: hammen, pamueller (track offsets)
"""

import importlib
from itertools import chain

import numpy as np

from Tilda.PolliFit.Spectra.Hyperfine import HyperfineN
from Tilda.PolliFit import Physics


class FullSpec(object):
    """
    A full spectrum function consisting of offset + all isotopes in iso with lineshape as declared in iso.
     Arbitrary many offsets can be defined in sections of the x-axis.
    """

    def __init__(self, iso, iso_m=None, guess_offset=False):
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
        self.cut_x = {}  # Dictionary of frequencies to split the x-axis for unique offsets in each track.
        # The keys determine the order in which the offsets are applied along the x-axis.

        self.guess_offset = guess_offset  # either False (offset from line db) or (meas.cts, st) for guessing

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
        x = np.asarray(x)
        if x.size == 0:
            return np.array([])
        if not full and self.cut_x != {}:  # The x-axis is only split if explicitly stated and cuts are available.
            order = np.argsort(x)  # Find ascending order of x
            inverse_order = np.array([int(np.where(order == i)[0])
                                      for i in range(order.size)])  # Invert the order for later.
            x_temp = x[order]  # Temporarily store a sorted x-axis.
            cut_i = np.array([np.argwhere(x_temp < self.cut_x[track]).T[0][-1] + 1
                              if np.min(x_temp) < self.cut_x[track] else 0
                              for track in range(len(self.cut_x.keys()))])  # Find the indices where to cut the x-axis.
            # If every x-value is larger than a position where to cut the x-axis, the index 0 is stored.
            x_ret = np.split(x_temp, cut_i)  # Split the x-axis at the given indices.
            # The indices 0 and x_temp.size result in empty arrays in front or behind all x-values, respectively.
            ret = np.concatenate([self.evaluate(x_i, p, ih=ih, full=True) + p[getattr(self, 'pOff{}'.format(track - 1))]
                                  if track > 0 else self.evaluate(x_i, p, ih=ih, full=True)
                                  for track, x_i in enumerate(x_ret)])
            # The spectrum is calculated for each interval of the split x-axis
            # and individual offsets are added relative to the absolute offset of the first interval.
            # The separate y-values get concatenated to a single array, matching the order of x_temp.
            return ret[inverse_order]  # The original order of the x-axis is restored for the y-values.
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

    def guess_offset_par(self, meas_cts_st):
        """
        Replaces the offset from lines db with a guess based on the cts data.
        It seems to be better to overestimate the offset.
        :param meas_cts_st: tuple: (SPFitter.meas, SPFitter.st)
        :return:
        """
        meas_cts = meas_cts_st[0]  # list of cts-arrays of shape (nOfPmts, nOfSteps) per track from SPFitter.meas.
        scalers = meas_cts_st[1][0]  # describing of scalers to be used, e.g.: [1, 2, -4] = scaler 1 + scaler 2 - scaler4
        tracks = meas_cts_st[1][1]  # tracks to be used; -1 for all tracks

        off_slope_per_tr = np.zeros((len(meas_cts)))
        for tr, arr in enumerate(meas_cts):
            if tr == tracks or tracks == -1:
                off_slope_per_sc = np.zeros((len(scalers)))
                for pos, sc in enumerate(scalers):
                    if sc != 0:
                        sign = np.sign(sc)
                    else:
                        sign = 1
                    cts = meas_cts[tr][sign*sc]  # sign used to make negative scalers positive here
                    low = cts[0]  # first value in cts
                    high = cts[-1]  # last value in cts
                    off_slope_per_sc[pos] = np.array([sign*(low+high)/2])  # include sign of scaler
                off_slope_per_tr[tr] = np.sum(off_slope_per_sc, axis=0)  # sum over all scalers
        # use the maximum offset and slope from all tracks as a guess
        offset = np.max(off_slope_per_tr, axis=0)
        return offset

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
        """
        Return list of initial parameters and initialize positions
        :param pos: parameter position to start from
        :return: list of initial parameters
        """
        self.pOff = pos
        self.p_offset_slope = pos + 1
        for track in self.cut_x.keys():
            setattr(self, 'pOff{}'.format(track), pos + 2 + track)
        if self.guess_offset:
            # must be of shape (meas.cts, st) if not False
            off = self.guess_offset_par(self.guess_offset)
            off_scale = off/self.iso.shape.get('offset', 1)  # factor how much we changed the offset vs db value
            ret = [off, self.iso.shape.get('offsetSlope', 0)]
        else:
            # don't guess the offset but load it normally from db
            off_scale = 1  # no scaling of the db offset
            ret = [self.iso.shape.get('offset', 0), self.iso.shape.get('offsetSlope', 0)]
        ret += [self.iso.shape.get('offset{}'.format(track), 0) for track in range(len(self.cut_x.keys()))]
        pos += len(ret)
        ret += self.shape.getPars(pos)
        pos += self.shape.nPar

        for hf in self.hyper:
            ret += hf.getPars(pos, int_f=off_scale)  # scale the intensity by the same factor as the offset
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
