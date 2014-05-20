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


def batchFit(fileList, st, projectPath, anadb, isodb, run = 'Run0'):
    '''Fit scaler/track st of fileList and write results to db'''
    print("BatchFit started")
    print("Opening DB:", anadb)
    
    oldPath = os.getcwd()
    os.chdir(projectPath)
    
    con = sqlite3.connect(anadb)
    cur = con.cursor()
    createTables(cur)
    con.commit()
    
    cur.execute('''SELECT IsoVar, LineVar FROM Runs WHERE Run = ?''', (run,))
    var = cur.fetchall()[0]
    
    print("Go for", run, "with IsoVar = \"" + var[0] + "\" and LineVar = \"" + var[1] + "\"")
    
    errcount = 0
    
    for file in fileList:
        try:
            singleFit(file, st, anadb, isodb, run, var, cur)
        except:
            errcount += 1
            print("Error working on file", file, ":", sys.exc_info()[1])
            traceback.print_tb(sys.exc_info()[2])
            
    os.chdir(oldPath)
    con.commit()
    con.close()
    
    print("BatchFit finished,", errcount, "errors occured")
    
        
def singleFit(file, st, anadb, isodb, run, var, cur):
    print('-----------------------------------')
    print("Fitting", file)
    cur.execute('''SELECT FilePath FROM Files WHERE File = ?''', (file,))
    
    try:
        path = cur.fetchall()[0][0]
    except:
        raise Exception(str(file) + " not found in DB")

    meas = MeasLoad.load(path, anadb)
    if meas.type == 'Kepco':
        spec = Straight()
    else:
        iso = DBIsotope(meas.type + var[0], meas.line + var[1], isodb)
        spec = FullSpec(iso)

    fit = SPFitter(spec, meas, st)
    fit.fit()
    
    #Create and save graph
    fig = os.path.splitext(path)[0] + '.pdf'
    plot.plotFit(fit)
    plot.save(fig)
    
    result = fit.result()
    
    for r in result:
        #Only one unique result, according to PRIMARY KEY, thanks to INSERT OR REPLACE
        cur.execute('''INSERT OR REPLACE INTO Results (File, Iso, Run, Scaler, Track, rChi, pars) 
        VALUES (?, ?, ?, ?, ?, ?, ?)''', (file, r[0], run, st[0], st[1], fit.rchi, repr(r[1])))
        
    
    print("Finished fitting", file)


def createTables(cur):
    '''Create necessary tables if they do not exist'''
    cur.execute('''CREATE TABLE IF NOT EXISTS Runs (
    Run TEXT PRIMARY KEY NOT NULL,
    LineVar TEXT,
    IsoVar TEXT
    )''')
    
    #insert Run0 as default if not available
    cur.execute('''SELECT Run FROM Runs WHERE Run = "Run0"''')
    if len(cur.fetchall()) == 0:
        cur.execute('''INSERT INTO Runs VALUES ("Run0", "", "")''')

    #Primary Key is necessary for unique results, allowing INSERT OR REPLACE
    cur.execute('''CREATE TABLE IF NOT EXISTS Results (
    File TEXT NOT NULL,
    Iso TEXT NOT NULL,
    Run TEXT NOT NULL,
    Scaler INT NOT NULL,
    Track INT NOT NULL,
    rChi REAL,
    pars TEXT,
    PRIMARY KEY (File, Iso, Run, Scaler, Track)
    )''')
    


if __name__ == '__main__':
    path = "V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/"

    batchFit(['KepcoScan_PCI.txt'], (0, -1), path, 'anaDB.sqlite', 'calciumD1.sqlite')