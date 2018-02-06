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

import MPLPlotter as plot
import Measurement.MeasLoad as MeasLoad
import TildaTools as TiTs
from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from Spectra.Straight import Straight


class InteractiveFit(object):
    def __init__(self, file, db, run, block=True, x_as_voltage=True, softw_gates_trs=None, fontSize=10):
        self.fitter_iso = None
        self.fitter_m = None
        self.fontSize =fontSize
        plot.ion()
        plot.clear()
        print('Starting InteractiveFit.')
        var = TiTs.select_from_db(db, 'filePath', 'Files', [['file'], [file]], caller_name=__name__)
        if var:
            path = os.path.join(os.path.dirname(db), var[0][0])
        else:
            print(str(file) + " not found in DB")
        
        print('Loading file', path)
        
        var = TiTs.select_from_db(db, 'isoVar, lineVar, scaler, track', 'Runs', [['run'], [run]], caller_name=__name__)
        if var:
            st = (ast.literal_eval(var[0][2]), ast.literal_eval(var[0][3]))
            linevar = var[0][1]
        else:
            print('Run cannot be selected!')
        if softw_gates_trs is None:  # # if no software gate provided pass on run and db via software gates
            softw_gates_trs = (db, run)

        meas = MeasLoad.load(path, db, x_as_voltage=x_as_voltage, softw_gates=softw_gates_trs)
        if meas.type == 'Kepco':  # keep this for all other fileformats than .xml
            spec = Straight()
            spec.evaluate(meas.x[0][-1], (0, 1))
            self.fitter = SPFitter(spec, meas, st)
            plot.plotFit(self.fitter, color='-r', fontsize_ticks=self.fontSize)
        else:
            try:
                # if the measurment is an .xml file it will have a self.seq_type
                if meas.seq_type == 'kepco':
                    spec = Straight()
                    spec.evaluate(meas.x[0][-1], (0, 1))
                    self.fitter = SPFitter(spec, meas, st)
                    plot.plotFit(self.fitter, color='-r', fontsize_ticks=self.fontSize)
                else:
                    iso = DBIsotope(db, meas.type, lineVar=linevar)
                    if var[0][0] == '_m':
                        iso_m = DBIsotope(db, meas.type, var[0][0], var[0][1])
                        spec = FullSpec(iso, iso_m)
                        spec_iso = FullSpec(iso)
                        spec_m = FullSpec(iso_m)
                        self.fitter_iso = SPFitter(spec_iso, meas, st)
                        self.fitter_m = SPFitter(spec_m, meas, st)
                        plot.plotFit(self.fitter_iso, color='-r', plot_residuals=False,
                                     fontsize_ticks=self.fontSize, plot_data=False, add_label=' gs')
                        plot.plotFit(self.fitter_m, color='-g', plot_residuals=False,
                                     fontsize_ticks=self.fontSize, plot_data=False, add_label=' m')
                        self.fitter = SPFitter(spec, meas, st)
                        plot.plotFit(self.fitter, color='-b', fontsize_ticks=self.fontSize,
                                     add_label=' gs+m', plot_side_peaks=False)
                    else:
                        spec = FullSpec(iso)
                        self.fitter = SPFitter(spec, meas, st)
                        plot.plotFit(self.fitter, color='-r', fontsize_ticks=self.fontSize)
            except:  # for mcp data etc
                iso = DBIsotope(db, meas.type, lineVar=linevar)
                if var[0][0] == '_m':
                    iso_m = DBIsotope(db, meas.type, var[0][0], var[0][1])
                    spec = FullSpec(iso, iso_m)
                    spec_iso = FullSpec(iso)
                    spec_m = FullSpec(iso_m)
                    self.fitter_iso = SPFitter(spec_iso, meas, st)
                    self.fitter_m = SPFitter(spec_m, meas, st)
                    plot.plotFit(self.fitter_iso, color='-r', plot_residuals=False,
                                 fontsize_ticks=self.fontSize, plot_data=False, add_label=' gs')
                    plot.plotFit(self.fitter_m, color='-g', plot_residuals=False,
                                 fontsize_ticks=self.fontSize, plot_data=False, add_label=' m')
                    self.fitter = SPFitter(spec, meas, st)
                    plot.plotFit(self.fitter, color='-b', fontsize_ticks=self.fontSize,
                                 add_label=' gs+m', plot_side_peaks=False)
                else:
                    spec = FullSpec(iso)
                    self.fitter = SPFitter(spec, meas, st)
                    plot.plotFit(self.fitter, color='-r', fontsize_ticks=self.fontSize)
        self.num_of_common_vals = 0
        if not isinstance(self.fitter.spec, Straight):
            self.num_of_common_vals = self.fitter.spec.shape.nPar + 2  # number of common parameters useful if isotope
            #  is being used -> comes from the number of parameters the shape needs
            #  e.g. (Voigt:2) + offset + offsetSlope = 4
        plot.show(block)
        self.printPars()
        
    def printPars(self):
        print('Current parameters:')
        for n, p, f in zip(self.fitter.npar, self.fitter.par, self.fitter.fix):
            print(n + '\t' + str(p) + '\t' + str(f))
            
    def getPars(self):
        return zip(self.fitter.npar, self.fitter.par, self.fitter.fix)
            
    def fit(self, show=True):
        self.fitter.fit()
        pars = self.fitter.par
        plot.clear()
        if self.fitter_m is not None:
            self.fitter_iso.par = pars[0:len(self.fitter_iso.par)]
            self.fitter_m.par = pars[0:self.num_of_common_vals] + pars[len(self.fitter_iso.par):]
            plot.plotFit(self.fitter_iso, color='-r', plot_residuals=False,
                         fontsize_ticks=self.fontSize, plot_data=False, add_label=' gs')
            plot.plotFit(self.fitter_m, color='-g', plot_residuals=False,
                         fontsize_ticks=self.fontSize, plot_data=False, add_label=' m')
            plot.plotFit(self.fitter, color='-b', fontsize_ticks=self.fontSize,
                         add_label=' gs+m', plot_side_peaks=False)
        else:
            plot.plotFit(self.fitter, color='-r', fontsize_ticks=self.fontSize)
        plot.show(show)
        
    def reset(self):
        self.fitter.reset()
        pars = self.fitter.par
        plot.clear()
        if self.fitter_m is not None:
            self.fitter_iso.par = pars[0:len(self.fitter_iso.par)]
            self.fitter_m.par = pars[0:self.num_of_common_vals] + pars[len(self.fitter_iso.par):]
            plot.plotFit(self.fitter_iso, color='-r', plot_residuals=False,
                         fontsize_ticks=self.fontSize, plot_data=False, add_label=' gs')
            plot.plotFit(self.fitter_m, color='-g', plot_residuals=False,
                         fontsize_ticks=self.fontSize, plot_data=False, add_label=' m')
            plot.plotFit(self.fitter, color='-b', fontsize_ticks=self.fontSize,
                         add_label=' gs+m', plot_side_peaks=False)
        else:
            plot.plotFit(self.fitter, color='-r', fontsize_ticks=self.fontSize)
        plot.show()
        
    def setPar(self, i, par):
        self.fitter.setPar(i, par)
        if self.fitter.npar[i] in ['softwGatesWidth', 'softwGatesDelayList', 'midTof']:
            # one of the gate parameter was changed -> gate data again
            # then data needs also to be gated again.
            gates_tr0 = TiTs.calc_soft_gates_from_db_pars(self.fitter.par[-3], self.fitter.par[-2], self.fitter.par[-1])
            softw_gate_all_tr = [gates_tr0 for each in self.fitter.meas.cts]
            self.fitter.meas.softw_gates = softw_gate_all_tr
            self.fitter.meas = TiTs.gate_specdata(self.fitter.meas)
        pars = self.fitter.par
        plot.clear()
        if self.fitter_m is not None:
            self.fitter_iso.par = pars[0:len(self.fitter_iso.par)]
            self.fitter_m.par = pars[0:self.num_of_common_vals] + pars[len(self.fitter_iso.par):]
            plot.plotFit(self.fitter_iso, color='-r', plot_residuals=False,
                         fontsize_ticks=self.fontSize, plot_data=False, add_label=' gs')
            plot.plotFit(self.fitter_m, color='-g', plot_residuals=False,
                         fontsize_ticks=self.fontSize, plot_data=False, add_label=' m')
            plot.plotFit(self.fitter, color='-b', fontsize_ticks=self.fontSize,
                         add_label=' gs+m', plot_side_peaks=False)
        else:
            plot.plotFit(self.fitter, color='-r', fontsize_ticks=self.fontSize)
        plot.show()
        
    def setFix(self, i, val):
        self.fitter.setFix(i, val)
    
    def setPars(self, par):
        self.fitter.par = par
        pars = self.fitter.par
        plot.clear()
        if self.fitter_m is not None:
            self.fitter_iso.par = pars[0:len(self.fitter_iso.par)]
            self.fitter_m.par = pars[0:self.num_of_common_vals] + pars[len(self.fitter_iso.par):]
            plot.plotFit(self.fitter_iso, color='-r', plot_residuals=False,
                         fontsize_ticks=self.fontSize, plot_data=False, add_label=' gs')
            plot.plotFit(self.fitter_m, color='-g', plot_residuals=False,
                         fontsize_ticks=self.fontSize, plot_data=False, add_label=' m')
            plot.plotFit(self.fitter, color='-b', fontsize_ticks=self.fontSize,
                         add_label=' gs+m', plot_side_peaks=False)
        else:
            plot.plotFit(self.fitter, color='-r', fontsize_ticks=self.fontSize)
        plot.show()

    def save_fig_to(self, path):
        plot.save(path)
        print('interactive fit saved_to:', path)

