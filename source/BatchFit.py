'''
Created on 15.05.2014

@author: hammen
'''

import sqlite3
import os
import sys
import traceback

from Measurement import MeasLoad

from DBIsotope import DBIsotope
from Spectra.Straight import Straight
from Spectra.FullSpec import FullSpec

from SPFitter import SPFitter

import MPLPlotter as plot


def batchFit(fileList, st, projectPath, db, run = 'Run0'):
    '''Fit scaler/track st of fileList and write results to db'''
    print("BatchFit started")
    print("Opening DB:", db)
    
    oldPath = os.getcwd()
    os.chdir(projectPath)
    
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT isoVar, lineVar FROM Runs WHERE run = ?''', (run,))
    var = cur.fetchall()[0]
    
    print("Go for", run, "with IsoVar = \"" + var[0] + "\" and LineVar = \"" + var[1] + "\"")
    
    errcount = 0
    
    for file in fileList:
        try:
            singleFit(file, st, db, run, var, cur)
        except:
            errcount += 1
            print("Error working on file", file, ":", sys.exc_info()[1])
            traceback.print_tb(sys.exc_info()[2])
            
    os.chdir(oldPath)
    con.commit()
    con.close()
    
    print("BatchFit finished,", errcount, "errors occured")
    
        
def singleFit(file, st, db, run, var, cur):
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
        iso = DBIsotope(meas.type, meas.line, db, var[0], var[1])
        spec = FullSpec(iso)

    fit = SPFitter(spec, meas, st)
    fit.fit()
    
    #Create and save graph
    fig = os.path.splitext(path)[0] + run + 'S' + str(st[0]) + 'T' + str(st[1]) + '.jpg'
    plot.plotFit(fit)
    plot.save(fig)
    plot.clear()
    
    result = fit.result()
    
    for r in result:
        #Only one unique result, according to PRIMARY KEY, thanks to INSERT OR REPLACE
        cur.execute('''INSERT OR REPLACE INTO Results (file, iso, run, sctr, rChi, pars) 
        VALUES (?, ?, ?, ?, ?, ?)''', (file, r[0], run, repr(st), fit.rchi, repr(r[1])))
        
    
    print("Finished fitting", file)
