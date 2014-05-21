'''
Created on 02.05.2014

@author: hammen
'''

import numpy as np
from scipy.optimize import curve_fit

class SPFitter(object):
    '''This class encapsulates the scipi.optimize.curve_fit routine'''


    def __init__(self, spec, meas, st):
        '''Initialize and prepare'''
        print('Initializing fit of S:', st[0], ', T:', st[1])
        self.spec = spec
        self.meas = meas
        self.st = st
        self.data = meas.getSingleSpec(*st)
        
        self.par = spec.getPars()
        self.fix = spec.getFixed()
        self.npar = spec.getParNames()
        self.pard = None #Will contain the parameter errors after fitting
        
        self.oldp = None
        self.pcov = None
        self.rchi = None
        
        
    def fit(self):
        '''
        Fit the free parameters of spec to data
        
        Calls evaluateE of spec
        As curve_fit can't fix parameters, they are manually truncated and reinjected
        Curve_fit expects standard deviations as weights
        '''
        print("Starting fit")
        
        truncp = [p for p, f in zip(self.par, self.fix) if f == False]
        popt, self.pcov = curve_fit(self.evaluateE, self.data[0], self.data[1], truncp, self.data[2], False)
        self.untrunc(popt)
        
        self.rchi = self.calcRchi()
        
        print('Done:')
        print('rChi^2' + '\t' + str(self.rchi))
        
        self.err = [np.sqrt(self.pcov[j][j]) for j in range(self.pcov.shape[0])]
        errit = iter(self.err)
        self.pard = [0 if f else next(errit) for f in self.fix]

        for n, x, e in zip(self.npar, self.par, self.pard):
            print(str(n) + '\t' +  str(x) + '\t' + '+-' + '\t' + str(e))
   
        
    def calcRchi(self):
        '''Calculate the reduced chi square'''
        return sum(x**2/e**2 for x, e in zip(self.calcRes(), self.data[2])) / self.calcNdef()
    
    
    def calcNdef(self):
        '''Calculate number of degrees of freedom'''
        return (len(self.data[0]) - (len(self.fix) - sum(self.fix)))
    
    
    def calcRes(self):
        '''Calculate the residuals of the current parameter set'''
        res = np.zeros(len(self.data[0]))
        
        valgen = (self.spec.evaluateE(e, self.meas.laserFreq, self.meas.col, self.par) for e in self.data[0])
        
        for i, (dat, val) in enumerate(zip(self.data[1], valgen)):
            res[i] = (dat - val)
        
        return res

    
    def untrunc(self, p):
        '''Copy the free parameters to their places in the full parameter set'''
        ip = iter(p)
        for i, f in enumerate(self.fix):
            if not f:
                self.par[i] = next(ip)
    
    def evaluate(self, x, *p):
        '''
        Encapsulate evaluate of spec
        
        Call recalc on parameter change
        Unpack the array of x-values curve_fit tends to call and return the list of results
        '''
        if p != self.oldp:
            self.untrunc([i for i in p])
            self.spec.recalc(self.par)
            self.oldp = p
        
        return [self.spec.evaluate(sx, self.par) for sx in x]


    
    def evaluateE(self, x, *p):
        '''Encapsulate evaluateE of spec'''
        if p != self.oldp:
            self.untrunc([i for i in p])
            self.spec.recalc(self.par)
            self.oldp = p
        
        return [self.spec.evaluateE(sx, self.meas.laserFreq, self.meas.col, self.par) for sx in x]
        

    def result(self):
        '''Return a list of result-tuples (name, pardict)'''
        ret =  []
        for p in self.spec.parAssign():
            name = p[0]
            npar = [x for x, f in zip(self.spec.getBlankNames(), p[1]) if f == True]
            par = [x for x, f in zip(self.par, p[1]) if f == True]
            err = [x for x, f in zip(self.err, p[1]) if f == True]
            fix = [x for x, f in zip(self.fix, p[1]) if f == True]
            pardict = dict(zip(npar, zip(par, err, fix)))
            ret.append((name, pardict, fix))
            
        return ret
            
        
        