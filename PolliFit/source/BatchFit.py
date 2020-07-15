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


def batchFit(fileList, db, run='Run0', x_as_voltage=True, softw_gates_trs=None, guess_offset=False, save_file_as='.png', save_to_folder=None):
    '''Fit fileList with run and write results to db'''
    print("BatchFit started")
    print("Opening DB:", db)

    # change directory into db folder
    oldPath = os.getcwd()
    projectPath, dbname = os.path.split(db)
    os.chdir(projectPath)

    # connect to db and create cursor
    con = sqlite3.connect(dbname)
    cur = con.cursor()

    # extract isoVariant, lineVariant, active scalers and tracks from run in database
    # store scalers and tracks as non-string tuple of list of numbers and number
    cur.execute('''SELECT isoVar, lineVar, scaler, track FROM Runs WHERE run = ?''', (run,))
    var = cur.fetchall()[0]
    st = (ast.literal_eval(var[2]), ast.literal_eval(var[3]))  # tuple of (scaler and track)
    # TODO: The following is very inelegant since we swap None to (db,run) and 'File' to None. Would be better to have None meaning load from file all the time
    if softw_gates_trs is None:  # if no software gate provided pass on run and db via software gates
        softw_gates_trs = (db, run)
    elif softw_gates_trs is 'File':
        softw_gates_trs = None  # when passed on with 'None' the gates will be read from file in XMLImporter

    print("Go for", run, "with IsoVar = \"" + var[0] + "\" and LineVar = \"" + var[1] + "\"")
    
    errcount = 0
    files_with_error = []
    fits = []
    for file in fileList:
        try:
            fits.append(singleFit(file, st, dbname, run, var, cur, x_as_voltage, softw_gates_trs, guess_offset,
                                  save_file_as=save_file_as, save_to_folder=save_to_folder))
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


def singleFit(file, st, db, run, var, cur, x_as_voltage=True, softw_gates_trs=None, guess_offset=False, save_file_as='.png', save_to_folder=None):
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
    if guess_offset:
        # if true, pass on the counts data and info about tracks and scaler as a basis for guessing the offset
        guess_offset = (meas.cts, st)
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
                    spec = FullSpec(iso, iso_m, guess_offset)
                    spec_iso = FullSpec(iso, guess_offset=guess_offset)
                    spec_m = FullSpec(iso_m, guess_offset=guess_offset)
                    fitter_iso = SPFitter(spec_iso, meas, st)
                    fitter_m = SPFitter(spec_m, meas, st)
                    # plot.plotFit(fitter_iso, color='-b', plot_residuals=False)
                    # plot.plotFit(fitter_m, color='-g', plot_residuals=False)
                else:
                    spec = FullSpec(iso, guess_offset=guess_offset)
        except:
            iso = DBIsotope(db, meas.type, lineVar=var[1])
            if var[0] == '_m':
                iso_m = DBIsotope(db, meas.type, var[0], var[1])
                spec = FullSpec(iso, iso_m, guess_offset)
                spec_iso = FullSpec(iso, guess_offset=guess_offset)
                spec_m = FullSpec(iso_m, guess_offset=guess_offset)
                fitter_iso = SPFitter(spec_iso, meas, st)
                fitter_m = SPFitter(spec_m, meas, st)
                # plot.plotFit(fitter_iso, color='-b', plot_residuals=False)
                # plot.plotFit(fitter_m, color='-g', plot_residuals=False)
            else:
                spec = FullSpec(iso, guess_offset=guess_offset)
    fit = SPFitter(spec, meas, st)
    fit.fit()
    
    #Create and save graph
    pars = fit.par
    num_of_common_vals = 0
    if not isinstance(spec, Straight):
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

    if save_to_folder is not None:
        # specific target folder given. use this.
        fig = save_to_folder + os.path.splitext(os.path.split(path)[1])[0] + run + save_file_as
    else:
        fig = os.path.splitext(path)[0] + run + save_file_as
    plot.save(fig)
    #plot.show()  # If this is un-commented each plot will be shown in batchfit. Only do this for debugging purposes.
    plot.clear()
    
    result = fit.result()
    
    for r in result:
        #Only one unique result, according to PRIMARY KEY, thanks to INSERT OR REPLACE
        cur.execute('''INSERT OR REPLACE INTO FitRes (file, iso, run, rChi, pars) 
        VALUES (?, ?, ?, ?, ?)''', (file, r[0], run, fit.rchi, repr(r[1])))
        
    
    print("Finished fitting", file)
    return fit

