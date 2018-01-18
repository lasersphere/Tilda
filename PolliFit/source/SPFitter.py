'''
Created on 02.05.2014

@author: hammen
'''

import numpy as np
from scipy.optimize import curve_fit
from scipy import version

import TildaTools as TiTs
from Measurement.SpecData import SpecDataXAxisUnits


class SPFitter(object):
    '''This class encapsulates the scipi.optimize.curve_fit routine'''


    def __init__(self, spec, meas, st):
        '''Initialize and prepare'''
        print('Initializing fit of S:', st[0], ', T:', st[1])
        self.spec = spec
        self.meas = meas
        self.st = st
        self.data = meas.getArithSpec(*st)

        self.par = spec.getPars()
        self.oldpar = list(self.par)
        self.fix = spec.getFixed()
        self.npar = spec.getParNames()
        self.pard = None  # Will contain the parameter errors after fitting

        try:
            # will fail if no software gates available: -> not time resolved...
            run_gates_width, del_list, iso_mid_tof = TiTs.calc_db_pars_from_software_gate(
                self.meas.softw_gates[0])  # track 0
            # run_gates_width, del_list, iso_mid_tof = 0, 0, 0
            self.par += run_gates_width, del_list, iso_mid_tof
            self.fix += True, True, True
            self.npar += 'softwGatesWidth', 'softwGatesDelayList', 'midTof'
        except Exception as e:
            # fail on purpose, just to check if software gates exist
            pass
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
        self.data = self.meas.getArithSpec(*self.st)  # needed if data was regated in between.

        self.oldpar = list(self.par)

        truncp = [p for p, f in zip(self.par, self.fix) if not f or isinstance(f, list)]
        boundl = ()
        boundu = ()
        try:
            for i in range(len(self.par)):
                # print(self.par[i])
                if isinstance(self.fix[i], list):
                    # -> bounds are explicitly given as a list
                    boundl += self.fix[i][0],
                    boundu += self.fix[i][1],
                else:
                    # check if it is fixed and matches one of the given restrictions:
                    if not self.fix[i]:
                        if self.npar[i][:3] == 'Int':
                            if self.par[i] > 0:
                                boundl += 0,
                                boundu += +np.inf,
                            else:
                                boundl += -np.inf,
                                boundu += 0,
                        elif self.npar[i] in ['sigma', 'gamma']:
                            boundl += 0,
                            boundu += +np.inf,
                        else:
                            boundl += -np.inf,
                            boundu += +np.inf,
        except Exception as e:
            print('error in fit: %s' % e)
        bounds = (boundl, boundu)
        scipy_version = int(version.version.split('.')[1])
        if scipy_version >= 17:
            popt, self.pcov = curve_fit(self.evaluateE, self.data[0], self.data[1],
                                        truncp, self.data[2], False, bounds=bounds)
        else:  # bounds not included before version 0.17.0
            popt, self.pcov = curve_fit(self.evaluateE, self.data[0], self.data[1],
                                        truncp, self.data[2], False)
        self.untrunc(popt)
        
        self.rchi = self.calcRchi()
        
        print('Done:')
        print('rChi^2' + '\t' + str(self.rchi))
        
        err = [np.sqrt(self.pcov[j][j]) for j in range(self.pcov.shape[0])]
        errit = iter(err)
        self.pard = [0 if f and not isinstance(f, list) else next(errit) for f in self.fix]

        for n, x, e in zip(self.npar, self.par, self.pard):
            print(str(n) + '\t' + str(x) + '\t' + '+-' + '\t' + str(e))
   
        
    def calcRchi(self):
        '''Calculate the reduced chi square'''
        return sum(x**2/e**2 for x, e in zip(self.calcRes(), self.data[2])) / self.calcNdef()
    
    
    def calcNdef(self):
        '''Calculate number of degrees of freedom'''
        # if bounds are given instead of boolean, write False to fixed bool list.
        fixed_bool_list = [f if isinstance(f, bool) else False for f in self.fix]
        fixed_sum = sum(fixed_bool_list)
        return (len(self.data[0]) - (len(self.fix) - fixed_sum))
    
    
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
            if not f or isinstance(f, list):
                self.par[i] = next(ip)
                
        return 
    
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
            npar = [x for x, f in zip(self.npar, p[1]) if f == True]
            par = [x for x, f in zip(self.par, p[1]) if f == True]
            err = [x for x, f in zip(self.pard, p[1]) if f == True]
            fix = [x for x, f in zip(self.fix, p[1]) if f == True]
            pardict = dict(zip(npar, zip(par, err, fix)))
            ret.append((name, pardict, fix))
            
        return ret
            
    def reset(self):
        '''Reset parameters to the values before optimization'''
        self.par = list(self.oldpar)
        
    
    def setPar(self, i, par):
        '''Set parameter with name to value par'''
        self.par[i] = par

            
    def setFix(self, i, val):
        self.fix[i] = val

        