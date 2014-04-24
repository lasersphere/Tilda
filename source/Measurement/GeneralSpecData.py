'''
Created on 29.03.2014

@author: hammen
'''

import time

import numpy as np

import Measurement.SingleSpecData as SingleSpec

class GeneralSpecData(object):
    '''
    This object contains a general spectrum with multiple tracks and multiple scalers
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.time = 0
        self.nrLoops = []
        self.x = np.array()
        self.counts = np.array()
        self.err = np.array()
        
        
    def getSingleSpec(self, scaler, track):
        if track == -1:
            return SingleSpec(self.x.flatten(), self.counts[scaler].flatten(), self.err[scaler].flatten())
        else:
            return SingleSpec(self.x[track], self.counts[scaler][track], self.err[scaler][track])
    

    def normalizeTracks(self):
        '''check whether a different number of loops was used for the different tracks and correct'''
        maxLoops = max(self.nrLoops)
        for i in range(0, self.nrTracks):
            if self.nrLoops[i] < maxLoops:
                self._multScalerCounts(i, maxLoops / self.nrLoops[i])
    

    def _multScalerCounts(self, scaler, mult):        
        '''multiply counts and error of a specific scaler by mult, according to error propagation'''
        self.counts[scaler] *= mult
        self.err[scaler] *= mult
    
    