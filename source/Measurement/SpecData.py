'''
Created on 29.03.2014

@author: hammen
'''

import numpy as np

class SpecData(object):
    '''
    This object contains a general spectrum with multiple tracks and multiple scalers
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.time = 0
        self.nrLoops = [1]
        self.nrTracks = 1
        self.nrScalers = 1
        self.type = None
        self.line = None
        self.path = None
        #self.x = np.array()
        #self.cts = np.array()
        #self.err = np.array()
        self.accVolt = None
        self.laser = None
        self.col = None
        
        
    def getSingleSpec(self, scaler, track):
        '''Return a tuple with (volt, cts, err) of the specified scaler and track. -1 for all tracks'''        
        if track == -1:
            return (self.x.flatten(), self.cts[scaler].flatten(), self.err[scaler].flatten())
        else:
            return (self.x[track], self.cts[scaler][track], self.err[scaler][track])
    
    def getArithSpec(self, scaler, track):
        '''Same as getSingleSpec, but scaler is of type [+i, -j, +k], resulting in s[i]-s[j]+s[k]'''
        aCts = np.zeros((self.nrTracks,))
        
        for i in scaler:
            for i, t in enumerate(self.cts[abs(i)]):
                
                tCts = t * np.copysign(1, i)
            
        

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
    
    