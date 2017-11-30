'''
Created on 15.05.2014

@author: hammen
'''

import ast
import os
import sqlite3
import sys
import traceback

import MPLPlotter as plot
from DBIsotope import DBIsotope
from Measurement import MeasLoad
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from Spectra.Straight import Straight


def batchFit(fileList, db, run='Run0', x_as_voltage=True, softw_gates_trs=None):
    '''Fit fileList with run and write results to db'''
    print("BatchFit started")
    print("Opening DB:", db)
    
    oldPath = os.getcwd()
    projectPath, dbname = os.path.split(db)
    os.chdir(projectPath)
    
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    
    cur.execute('''SELECT isoVar, lineVar, scaler, track FROM Runs WHERE run = ?''', (run,))
    var = cur.fetchall()[0]
    st = (ast.literal_eval(var[2]), ast.literal_eval(var[3]))
    if softw_gates_trs is None:  # if no software gate provided pass on run and db via software gates
        softw_gates_trs = (db, run)

    print("Go for", run, "with IsoVar = \"" + var[0] + "\" and LineVar = \"" + var[1] + "\"")
    
    errcount = 0
    files_with_error = []
    fits = []
    for file in fileList:
        try:
            fits.append(singleFit(file, st, dbname, run, var, cur, x_as_voltage, softw_gates_trs))
        except:
            errcount += 1
            print("Error working on file", file, ":", sys.exc_info()[1])
            traceback.print_tb(sys.exc_info()[2])
            files_with_error.append(file)


    con.commit()
    con.close()
    os.chdir(oldPath)

    print("BatchFit finished,", errcount, "errors occured")
    return fits, files_with_error


def singleFit(file, st, db, run, var, cur, x_as_voltage=True, softw_gates_trs=None):
    '''Fit st of file, using run. Save result to db and picture of spectrum to folder'''
    fitter_iso = None
    fitter_m = None

    print('-----------------------------------')
    print("Fitting", file)
    cur.execute('''SELECT filePath FROM Files WHERE file = ?''', (file,))
    
    try:
        path = cur.fetchall()[0][0]
    except:
        raise Exception(str(file) + " not found in DB")

    meas = MeasLoad.load(path, db, x_as_voltage=x_as_voltage, softw_gates=softw_gates_trs)
    if meas.type == 'Kepco':
        print('Fitting Straight!')
        spec = Straight()
    else:
        try:
            # if the measurment is an .xml file it will have a seq_type
            if meas.seq_type == 'kepco':
                spec = Straight()
                spec.evaluate(meas.x[0][-1], (0, 1))
            else:
                iso = DBIsotope(db, meas.type, lineVar=var[1])
                if var[0] == '_m':
                    iso_m = DBIsotope(db, meas.type, var[0], var[1])
                    spec = FullSpec(iso, iso_m)
                    spec_iso = FullSpec(iso)
                    spec_m = FullSpec(iso_m)
                    fitter_iso = SPFitter(spec_iso, meas, st)
                    fitter_m = SPFitter(spec_m, meas, st)
                    # plot.plotFit(fitter_iso, color='-b', plot_residuals=False)
                    # plot.plotFit(fitter_m, color='-g', plot_residuals=False)
                else:
                    spec = FullSpec(iso)
        except:
            iso = DBIsotope(db, meas.type, lineVar=var[1])
            if var[0] == '_m':
                iso_m = DBIsotope(db, meas.type, var[0], var[1])
                spec = FullSpec(iso, iso_m)
                spec_iso = FullSpec(iso)
                spec_m = FullSpec(iso_m)
                fitter_iso = SPFitter(spec_iso, meas, st)
                fitter_m = SPFitter(spec_m, meas, st)
                # plot.plotFit(fitter_iso, color='-b', plot_residuals=False)
                # plot.plotFit(fitter_m, color='-g', plot_residuals=False)
            else:
                spec = FullSpec(iso)
    fit = SPFitter(spec, meas, st)
    fit.fit()
    
    #Create and save graph
    fig = os.path.splitext(path)[0] + run + '.png'
    pars = fit.par
    num_of_common_vals = fit.spec.shape.nPar + 2  # number of common parameters useful if isotope
    #  is being used -> comes from the number of parameters the shape needs e.g. (Voigt:2) + offset + offsetSlope = 4
    if fitter_m is not None:
        fitter_iso.par = pars[0:len(fitter_iso.par)]
        fitter_m.par = pars[0:num_of_common_vals] + pars[len(fitter_iso.par):]
        plot.plotFit(fitter_iso, color='-r', plot_residuals=False, plot_data=False, add_label=' gs')
        plot.plotFit(fitter_m, color='-g', plot_residuals=False, plot_data=False, add_label=' m')
        plot.plotFit(fit, color='-b', add_label=' gs+m', plot_side_peaks=False)
    else:
        plot.plotFit(fit)
    plot.save(fig)
    plot.clear()
    
    result = fit.result()
    
    for r in result:
        #Only one unique result, according to PRIMARY KEY, thanks to INSERT OR REPLACE
        cur.execute('''INSERT OR REPLACE INTO FitRes (file, iso, run, rChi, pars) 
        VALUES (?, ?, ?, ?, ?)''', (file, r[0], run, fit.rchi, repr(r[1])))
        
    
    print("Finished fitting", file)
    return fit

