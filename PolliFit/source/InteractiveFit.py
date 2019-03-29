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
import TildaTools as TiTs
from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from Spectra.Straight import Straight


class InteractiveFit(object):
    def __init__(self, file, db, run, block=True, x_as_voltage=True,
                 softw_gates_trs=None, fontSize=10, clear_plot=True, data_fmt='k.',
                 plot_in_freq=True, save_plot=False):
        self.fitter_iso = None
        self.fitter_m = None
        self.fontSize =fontSize
        self.plot_in_freq = plot_in_freq
        self.save_plot = save_plot
        self.save_path = os.path.join(os.path.normpath(os.path.dirname(db)), "saved_plots")
        self.data_fmt = data_fmt

        plot.ion()
        if clear_plot:
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
            # st: tuple of PMTs and tracks from selected run
            st = (ast.literal_eval(var[0][2]), ast.literal_eval(var[0][3]))
            linevar = var[0][1]
        else:
            print('Run cannot be selected!')
        if softw_gates_trs is None:  # # if no software gate provided pass on run and db via software gates
            softw_gates_trs = (db, run)

        # Import Measurement from file using Importer
        meas = MeasLoad.load(path, db, x_as_voltage=x_as_voltage, softw_gates=softw_gates_trs)
        if meas.type == 'Kepco':  # keep this for all other fileformats than .xml
            spec = Straight()
            spec.evaluate(meas.x[0][-1], (0, 1))
            self.fitter = SPFitter(spec, meas, st)
        else:
            try:
                # if the measurment is an .xml file it will have a self.seq_type
                if meas.seq_type == 'kepco':
                    spec = Straight()
                    spec.evaluate(meas.x[0][-1], (0, 1))
                    self.fitter = SPFitter(spec, meas, st)
                else:
                    # get isotope information
                    iso = DBIsotope(db, meas.type, lineVar=linevar)
                    if var[0][0] == '_m' or var[0][0] == '_m1' or var[0][0] == '_m2':
                        iso_m = DBIsotope(db, meas.type, var[0][0], var[0][1])
                        spec = FullSpec(iso, iso_m)  # get the full spectrum function
                        spec_iso = FullSpec(iso)
                        spec_m = FullSpec(iso_m)
                        self.fitter_iso = SPFitter(spec_iso, meas, st)
                        self.fitter_m = SPFitter(spec_m, meas, st)
                        self.fitter = SPFitter(spec, meas, st)
                    else:
                        spec = FullSpec(iso)
                        self.fitter = SPFitter(spec, meas, st)
            except:  # for mcp data etc
                iso = DBIsotope(db, meas.type, lineVar=linevar)
                if var[0][0] == '_m':
                    iso_m = DBIsotope(db, meas.type, var[0][0], var[0][1])
                    spec = FullSpec(iso, iso_m)
                    spec_iso = FullSpec(iso)
                    spec_m = FullSpec(iso_m)
                    self.fitter_iso = SPFitter(spec_iso, meas, st)
                    self.fitter_m = SPFitter(spec_m, meas, st)
                    self.fitter = SPFitter(spec, meas, st)
                else:
                    spec = FullSpec(iso)
                    self.fitter = SPFitter(spec, meas, st)
        if not isinstance(spec, Straight):
            self.num_of_common_vals = self.fitter.spec.shape.nPar + 2  # number of common parameters useful if isotope
            #  is being used -> comes from the number of parameters the shape needs
            #  e.g. (Voigt:2) + offset + offsetSlope = 4
        else:
            self.num_of_common_vals = 2  # offset + slope
        self.printPars()
        self.plot_fit(True)

    def printPars(self):
        print('Current parameters:')
        for n, p, f in zip(self.fitter.npar, self.fitter.par, self.fitter.fix):
            print(n + '\t' + str(p) + '\t' + str(f))
            
    def getPars(self):
        return zip(self.fitter.npar, self.fitter.par, self.fitter.fix)

    def getParsE(self):
        return self.fitter.parsToE()

    def fit(self, show=True, clear_plot=True, data_fmt='k.'):
        self.printPars()
        self.fitter.fit()
        self.plot_fit(clear_plot=clear_plot, show=show, save_plt=self.save_plot)

    def reset(self):
        self.fitter.reset()
        self.plot_fit(True)

    def setPar(self, i, par):
        if self.plot_in_freq:
            self.fitter.setPar(i, par)
        else:
            self.fitter.setParE(i, par)
        if self.fitter.npar[i] in ['softwGatesWidth', 'softwGatesDelayList', 'midTof']:
            # one of the gate parameter was changed -> gate data again
            # then data needs also to be gated again.
            gates_tr0 = TiTs.calc_soft_gates_from_db_pars(self.fitter.par[-3], self.fitter.par[-2], self.fitter.par[-1])
            softw_gate_all_tr = [gates_tr0 for each in self.fitter.meas.cts]
            self.fitter.meas.softw_gates = softw_gate_all_tr
            self.fitter.meas = TiTs.gate_specdata(self.fitter.meas)

        self.plot_fit(clear_plot=True)

    def setFix(self, i, val):
        self.fitter.setFix(i, val)
    
    def setPars(self, par):
        self.fitter.par = par
        self.plot_fit(clear_plot=True)

    def save_fig_to(self, path):
        plot.save(path)
        print('interactive fit saved_to:', path)

    def parsToDB(self, db):

        parsName = self.fitter.npar
        parsValue = self.fitter.par
        parsFix = self.fitter.fix
        indexCenter = parsName.index('center')
        # Split at 'center' since this marks the border between "Lines" pars & "Isotopes" pars

        # Save Lines pars (Pars 0 until center)
        shape = dict(zip(parsName[:indexCenter], parsValue[:indexCenter]))
        shapeFix = dict(zip(parsName[:indexCenter], parsFix[:indexCenter]))
        lineVar = self.fitter.spec.iso.lineVar
        lineName = self.fitter.spec.iso.shape['name']
        shape.update({'name': lineName})
        print(lineName)
        print(shape, shapeFix, lineVar)
        try:
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute('''UPDATE Lines SET shape = ?, fixShape = ? WHERE lineVar = ?''',
                        (str(shape), str(shapeFix), str(lineVar)))
            con.commit()
            con.close()
            print("Saved pars in Lines!")
        except Exception as e:
            print("error: No database connection possible. No line pars have been saved!")

    def plot_fit(self, clear_plot, show=True, save_plt=False):
        """
        function to encapsulate the plot.plotFit(..) call
        :param clear_plot: bool, True for clearing plot in advance
        :param show: bool, default: True, True for showing plot and block
        :param save_plt: bool, default: False, will store the plot to self.save_path... if wanted
        :return: None
        """
        pars = self.fitter.par
        save_path = ''
        if save_plt:
            save_path = self.save_path
        if clear_plot:
            plot.clear()
        if self.fitter_m is not None:
            if self.fitter.meas.seq_type == 'trs':  # needed in next step since self.fitter_iso.par has 3 pars for trs meas appended:
                num_of_trs_pars = 3  # SoftwGatesWidth, SoftwGatesDelayList, midTof
                trs_pars = pars[-3:]
            else:
                num_of_trs_pars = 0
                trs_pars = []
            self.fitter_iso.par = pars[0:len(self.fitter_iso.par) - num_of_trs_pars] + trs_pars
            self.fitter_m.par = pars[0:self.num_of_common_vals] + pars[len(self.fitter_iso.par) - num_of_trs_pars:]
            plot.plotFit(self.fitter_iso, color='-r', plot_residuals=False,
                         fontsize_ticks=self.fontSize, plot_data=False, add_label='_gs',
                         x_in_freq=self.plot_in_freq, data_fmt=self.data_fmt,
                         save_plot=save_plt, save_path=save_path)
            plot.plotFit(self.fitter_m, color='-g', plot_residuals=False,
                         fontsize_ticks=self.fontSize, plot_data=False, add_label='_m',
                         x_in_freq=self.plot_in_freq, data_fmt=self.data_fmt,
                         save_plot=save_plt, save_path=save_path)
            plot.plotFit(self.fitter, color='-b', fontsize_ticks=self.fontSize,
                         add_label='_gs+m', plot_side_peaks=False,
                         x_in_freq=self.plot_in_freq, data_fmt=self.data_fmt,
                         save_plot=save_plt, save_path=save_path)
        else:
            plot.plotFit(self.fitter, color='-r', fontsize_ticks=self.fontSize,
                         x_in_freq=self.plot_in_freq, data_fmt=self.data_fmt,
                         save_plot=save_plt, save_path=save_path)
        if show:
            plot.show(show)



        #Save Isotopes pars (Hyperfine Pars; from center to end of Pars) ;; CURRENTLY NOT INTENDED TO WORK!!
        # hfs = zip(parsName[indexCenter:], parsValue[indexCenter:])
        #
        # relInt = Physics.HFInt(f.iso.I, f.iso.Jl, f.iso.Ju, hfs.trans)
        # print(hfs['Int0'] / relInt[0])
        # print(hfs)



