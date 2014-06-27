'''
Created on 15.05.2014

@author: hammen
'''

import sqlite3
import os
import sys
import traceback
import ast

from Measurement import MeasLoad

from DBIsotope import DBIsotope
from Spectra.Straight import Straight
from Spectra.FullSpec import FullSpec

from SPFitter import SPFitter

import MPLPlotter as plot


def batchFit(fileList, db, run = 'Run0'):
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
    
    
    print("Go for", run, "with IsoVar = \"" + var[0] + "\" and LineVar = \"" + var[1] + "\"")
    
    errcount = 0
    
    for file in fileList:
        try:
            singleFit(file, st, dbname, run, var, cur)
        except:
            errcount += 1
            print("Error working on file", file, ":", sys.exc_info()[1])
            traceback.print_tb(sys.exc_info()[2])
            
    os.chdir(oldPath)
    con.commit()
    con.close()
    
    print("BatchFit finished,", errcount, "errors occured")
    
        
def singleFit(file, st, db, run, var, cur):
    '''Fit st of file, using run. Save result to db and picture of spectrum to folder'''
    print('-----------------------------------')
    print("Fitting", file)
    cur.execute('''SELECT filePath FROM Files WHERE file = ?''', (file,))
    
    try:
        path = cur.fetchall()[0][0]
    except:
        raise Exception(str(file) + " not found in DB")

    meas = MeasLoad.load(path, db)
    if meas.type == 'Kepco':
        spec = Straight()
    else:
        iso = DBIsotope(db, meas.type, var[0], var[1])
        spec = FullSpec(iso)
        meas.deadtimeCorrect(st[0][0],st[1])

    fit = SPFitter(spec, meas, st)
    fit.fit()
    
    #Create and save graph
    fig = os.path.splitext(path)[0] + run + '.png'
    plot.plotFit(fit)
    plot.save(fig)
    plot.clear()
    
    result = fit.result()
    
    for r in result:
        #Only one unique result, according to PRIMARY KEY, thanks to INSERT OR REPLACE
        cur.execute('''INSERT OR REPLACE INTO FitRes (file, iso, run, rChi, pars) 
        VALUES (?, ?, ?, ?, ?)''', (file, r[0], run, fit.rchi, repr(r[1])))
        
    
    print("Finished fitting", file)

