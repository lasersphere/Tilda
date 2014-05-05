'''
Created on 02.05.2014

@author: hammen
'''

import numpy as np
from scipy.optimize import curve_fit

import Physics
import Experiment as Exp

class SPFitter(object):
    '''This class encapsulates the scipi.optimize.curve_fit routine for Pollifit'''


    def __init__(self, spec, file, track):
        '''Initialize and prepare'''
        self.spec = spec
        self.data = file.getSingleSpec(*track)
        
        self.accVolt = Exp.getAccVolt(file.time)
        self.laser = Exp.getLaserFreq(file.time)
        self.col = Exp.dirColTrue(file.time) 
        
        self.par = spec.getPars()
        print(self.par)
        self.fix = spec.getFixed()
        print(self.fix)
        self.npar = spec.getParNames()
        print(self.npar)
        
        self.oldp = None
        self.pcov = None
        self.rchi = None
        
        
    def fit(self):
        '''Fit the free parameters of spec to data'''
        truncp = [p for p, f in zip(self.par, self.fix) if f == False]
        
        popt, self.pcov = curve_fit(self.evaluate, self.data[0], self.data[1], truncp, self.data[2])        
        self.untrunc(popt)
        self.rchi = self.calcRchi()
       
        
    def calcRchi(self):
        '''calculate the reduced chi square'''
        return sum(x**2 for x in self.calcRes()) / self.calcNdef
    
    
    def calcNdef(self):
        '''calculate number of degrees of freedom'''
        return (len(self.data[0] - sum(self.fix)))
    
    
    def calcRes(self):
        '''Calculate the residuals of the current parameter set'''
        res = np.zeros(len(self.data[0]))
        
        valgen = (self.spec.evaluateE(e, self.laser, self.col, self.par) for e in self.data[0])
        
        for i, (dat, val) in enumerate(zip(self.data[1], valgen)):
            res[i] = (dat - val)
        
        return res
    
    
    def calcErr(self):
        err = []
        j = 0
        for f in self.fix:
            if not f:
                err.append(self.pcov[j][j]**2)
                j += 1
            else:
                err.append(0)
                
        return err
    
    def untrunc(self, p):
        '''Copy the free parameters to their places in the full parameter set'''
        j = 0
        for i, f in enumerate(self.fix):
            if not f:
                self.par[i] = p[j]
                j += 1
    
    
    def evaluate(self, x, *p):
        '''This functions masks the fixed parameters and adds Experimental values'''
        e = self.accVolt - x
        self.untrunc([i for i in p])
        
        if p != self.oldp:
            self.spec.recalc(self.par)
        
        self.spec.evaluateE(e, self.laser, self.col, self.par)
        

        
        