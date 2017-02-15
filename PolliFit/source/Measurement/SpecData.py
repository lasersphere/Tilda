'''
Created on 29.03.2014

@author: hammen
'''
import itertools as it
from datetime import datetime

import numpy as np


class SpecData(object):
    '''
    This object contains a general spectrum with multiple tracks and multiple scalers
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.path = None  # str, path of the file
        self.type = None  # str, isotope name
        self.line = None  # str, lineVar
        self.date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')  #
        self.nrLoops = [1]  # list of integers, len=nrTracks, holds how often the track was repeated
        self.nrTracks = 1  # int, number of tracks in scan.
        self.nrScalers = []  # list, len = nrTracks, holds the number of scaler for each track
        self.accVolt = None  # float, acceleration voltage of the ions
        self.laserFreq = 0  # float, fundamental laser frequency in MHz in the laser lab system
        self.col = False  # bool, collinear = True, anticollinear = False
        self.dwell = 0  # float or list of lists, depending on importer
        
        self.offset = None  # float, measured offset pre scan, take mean if multiple ones measured
        self.lineMult = None  # float, applied_voltage = (DAC_voltage * lineMult + lineOffset) * voltDivRatio
        self.lineOffset = None  # float, offset of the DAC at 0V set
        self.voltDivRatio = None  # dict, {'accVolt': , 'offset'
        
        #Data is organized as list of tracks containing arrays with information
        self.x = []
        self.cts = []
        self.err = []

    def getSingleSpec(self, scaler, track):
        '''Return a tuple with (volt, cts, err) of the specified scaler and track. -1 for all tracks'''
        if track == -1:
            return ( [i for i in it.chain(*self.x)],
                     [i for i in it.chain(*(t[scaler] for t in self.cts))],
                     [i for i in it.chain(*(t[scaler] for t in self.err))] )
        else:
            return (self.x[track], self.cts[track][scaler], self.err[track][scaler])
    
    def getArithSpec(self, scaler, track_index):
        '''Same as getSingleSpec, but scaler is of type [+i, -j, +k], resulting in s[i]-s[j]+s[k]'''
        l = self.getNrSteps(track_index)
        flatc = np.zeros((l,))
        flate = np.zeros((l,))
        if isinstance(self.nrScalers, list):
            if track_index == -1:
                nrScalers = self.nrScalers[0]
            else:
                nrScalers = self.nrScalers[track_index]
        else:
            nrScalers = self.nrScalers

        for s in scaler:
            if nrScalers >= np.abs(s):
                flatx, c, e = self.getSingleSpec(abs(s), track_index)
                flatc = flatc + np.copysign(1, s) * c
                flate = flate + np.square(e)
            else:
                pass
        flate = np.sqrt(flate)
        return (flatx, flatc, flate)
        
    def getNrSteps(self, track):
        if track == -1:
            return sum(map(len, self.x))
        else:
            return len(self.x[track])

    def _normalizeTracks(self):
        '''Check whether a different number of loops was used for the different tracks and correct'''
        maxLoops = max(self.nrLoops)
        for i in range(0, self.nrTracks):
            if self.nrLoops[i] < maxLoops:
                self._multScalerCounts(i, maxLoops / self.nrLoops[i])
    
    def _multScalerCounts(self, scaler, mult):
        '''Multiply counts and error of a specific scaler by mult, according to error propagation'''
        self.cts[scaler] *= mult
        self.err[scaler] *= mult
        
    def deadtimeCorrect(self, scaler, track):
        for i, cts in enumerate(self.cts[track][scaler]):
            self.cts[track][scaler][i] = (cts*(self.nrLoops[track]*self.dwell)) / (
                1-(cts*(self.nrLoops[track]*self.dwell))*1.65e-8
            )/((self.nrLoops[track]*self.dwell))
        
    
# test = SpecData()
# print(test.date)
