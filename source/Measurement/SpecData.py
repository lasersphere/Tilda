'''
Created on 29.03.2014

@author: hammen
'''
import itertools as it

import numpy as np

class SpecData(object):
    '''
    This object contains a general spectrum with multiple tracks and multiple scalers
    '''

    def __init__(self):
        '''
        Constructor
        '''
        self.path = None
        self.type = None
        self.line = None
        self.time = 0
        self.nrLoops = [1]
        self.nrTracks = 1
        self.nrScalers = 1
        self.accVolt = None
        self.laserFreq = None
        self.col = None
        
        #Data is organized as list of tracks containing arrays with information
        #self.x = [np.array((steps,))]
        #self.cts = [np.array((scaler, steps))]
        #self.err = [np.array((scaler, steps))]
        
        
    def getSingleSpec(self, scaler, track):
        '''Return a tuple with (volt, cts, err) of the specified scaler and track. -1 for all tracks'''        
        if track == -1:
            return ( [i for i in it.chain(*self.x)], [i for i in it.chain(*(t[scaler] for t in self.cts))], [i for i in it.chain(*(t[scaler] for t in self.err))] )
        else:
            return (self.x[track], self.cts[track][scaler], self.err[track][scaler])
    
    
    def getArithSpec(self, scaler, track):
        '''Same as getSingleSpec, but scaler is of type [+i, -j, +k], resulting in s[i]-s[j]+s[k]'''
        l = self.getNrSteps(track)
        flatc = np.zeros((l,))
        flate = np.zeros((l,))
        
        for s in scaler:
            flatx, c, e = self.getSingleSpec(abs(s), track)
            flatc = flatc + np.copysign(1, s) * c
            flate = flate + np.square(e)
            
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
    
    