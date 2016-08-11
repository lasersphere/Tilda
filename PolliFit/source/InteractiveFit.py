'''
Created on 27.05.2014

@author: hammen

InteractiveFit is an interactive fitting wrapper that can be used to easily determine starting parameters.
Import the module in IPython and load() a file. The spectrum as well as the datapoints will be plotted.
printPars() is used to plot starting values, setPar() to change them, fit() to try and fit, reset() to
revert to the values before the last fit try.
'''
import ast
import os
import sqlite3

import MPLPlotter as plot
import Measurement.MeasLoad as MeasLoad
from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from Spectra.Straight import Straight


class InteractiveFit(object):
    def __init__(self, file, db, run, block=True, x_as_voltage=True, softw_gates_trs=None):
        self.fitter_iso = None
        self.fitter_m = None
        plot.ion()
        plot.clear()
        con = sqlite3.connect(db)
        cur = con.cursor()
        print('Starting InteractiveFit.')
        cur.execute('''SELECT filePath FROM Files WHERE file = ?''', (file,))
        
        try:
            path = os.path.join(os.path.dirname(db), cur.fetchall()[0][0])
        except:
            raise Exception(str(file) + " not found in DB")
        
        print('Loading file', path)
        
        cur.execute('''SELECT isoVar, lineVar, scaler, track FROM Runs WHERE run = ?''', (run,))
        var = cur.fetchall()[0]
        st = (ast.literal_eval(var[2]), ast.literal_eval(var[3]))
        linevar = var[1]

        if softw_gates_trs is None:  # if no software gates provided check db
            try:  # check if there are software gates available in database
                cur.execute('''SELECT softwGates FROM Runs WHERE run = ?''', (run,))
                soft_var = cur.fetchall()[0]
                softw_gates_trs_db = ast.literal_eval(soft_var[0])
                if isinstance(softw_gates_trs_db, list):
                    softw_gates_trs = softw_gates_trs_db
            except Exception as e:
                print('error while trying to extract the software Gates from Runs: ', e)
                print('will use gates from file')

        meas = MeasLoad.load(path, db, x_as_voltage=x_as_voltage, softw_gates=softw_gates_trs)
        if meas.type == 'Kepco':  # keep this for all other fileformats than .xml
            spec = Straight()
            spec.evaluate(meas.x[0][-1], (0, 1))
        else:
            try:
                # if the measurment is an .xml file it will have a self.seq_type
                if meas.seq_type == 'kepco':
                    spec = Straight()
                    spec.evaluate(meas.x[0][-1], (0, 1))
                else:
                    iso = DBIsotope(db, meas.type, lineVar=linevar)
                    if var[0] == '_m':
                        iso_m = DBIsotope(db, meas.type, var[0], var[1])
                        spec = FullSpec(iso, iso_m)
                        spec_iso = FullSpec(iso)
                        spec_m = FullSpec(iso_m)
                        self.fitter_iso = SPFitter(spec_iso, meas, st)
                        self.fitter_m = SPFitter(spec_m, meas, st)
                        plot.plotFit(self.fitter_iso, color='-b', plot_residuals=False)
                        plot.plotFit(self.fitter_m, color='-g', plot_residuals=False)
                    else:
                        spec = FullSpec(iso)
            except:
                iso = DBIsotope(db, meas.type, lineVar=linevar)
                if var[0] == '_m':
                    iso_m = DBIsotope(db, meas.type, var[0], var[1])
                    spec = FullSpec(iso, iso_m)
                    spec_iso = FullSpec(iso)
                    spec_m = FullSpec(iso_m)
                    self.fitter_iso = SPFitter(spec_iso, meas, st)
                    self.fitter_m = SPFitter(spec_m, meas, st)
                    plot.plotFit(self.fitter_iso, color='-b', plot_residuals=False)
                    plot.plotFit(self.fitter_m, color='-g', plot_residuals=False)
                else:
                    spec = FullSpec(iso)
        self.fitter = SPFitter(spec, meas, st)
        plot.plotFit(self.fitter)
        plot.show(block)
        self.printPars()
        
        
    def printPars(self):
        print('Current parameters:')
        for n, p, f in zip(self.fitter.npar, self.fitter.par, self.fitter.fix):
            print(n + '\t' + str(p) + '\t' + str(f))
            
            
    def getPars(self):
        return zip(self.fitter.npar, self.fitter.par, self.fitter.fix)
            
    def fit(self):
        self.fitter.fit()
        pars = self.fitter.par
        plot.clear()
        if self.fitter_m is not None:
            self.fitter_iso.par = pars[0:len(self.fitter_iso.par)]
            self.fitter_m.par = pars[0:3] + pars[len(self.fitter_iso.par):]
            plot.plotFit(self.fitter_iso, color='-b', plot_residuals=False)
            plot.plotFit(self.fitter_m, color='-g', plot_residuals=False)
        plot.plotFit(self.fitter)
        plot.show()
        
    def reset(self):
        self.fitter.reset()
        pars = self.fitter.par
        plot.clear()
        if self.fitter_m is not None:
            self.fitter_iso.par = pars[0:len(self.fitter_iso.par)]
            self.fitter_m.par = pars[0:3] + pars[len(self.fitter_iso.par):]
            plot.plotFit(self.fitter_iso, color='-b', plot_residuals=False)
            plot.plotFit(self.fitter_m, color='-g', plot_residuals=False)
        plot.plotFit(self.fitter)
        plot.show()
        
    def setPar(self, i, par):
        self.fitter.setPar(i, par)
        pars = self.fitter.par
        plot.clear()
        if self.fitter_m is not None:
            self.fitter_iso.par = pars[0:len(self.fitter_iso.par)]
            self.fitter_m.par = pars[0:3] + pars[len(self.fitter_iso.par):]
            plot.plotFit(self.fitter_iso, color='-b', plot_residuals=False)
            plot.plotFit(self.fitter_m, color='-g', plot_residuals=False)
        plot.plotFit(self.fitter)
        plot.show()
        
    def setFix(self, i, val):
        self.fitter.setFix(i, val)
    
    def setPars(self, par):
        self.fitter.par = par
        pars = self.fitter.par
        plot.clear()
        if self.fitter_m is not None:
            self.fitter_iso.par = pars[0:len(self.fitter_iso.par)]
            self.fitter_m.par = pars[0:3] + pars[len(self.fitter_iso.par):]
            plot.plotFit(self.fitter_iso, color='-b', plot_residuals=False)
            plot.plotFit(self.fitter_m, color='-g', plot_residuals=False)
        plot.plotFit(self.fitter)
        plot.show()
    
