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


def batchFit(fileList, db, run = 'Run0'):
    print("BatchFit started")
    print("Opening DB:", db)
    
    con = sqlite3.connect(db)
    cur = con.cursor()
    createTables(cur)
    con.commit()
    
    cur.execute('''SELECT IsoVar, LineVar FROM Runs WHERE Run = ?''', (run,))
    var = cur.fetchall()[0]
    
    print("Go for", run, "with IsoVar = \"" + var[0] + "\" and LineVar = \"" + var[1] + "\"")
    
    os.chdir(os.path.dirname(db))

    for file in fileList:
        try:
            singleFit(file, var, cur)
        except:
            print("Error working on file", file, ":", sys.exc_info()[1])
            traceback.print_tb(sys.exc_info()[2])
            
    con.close()
    
    print("BatchFit finished")
    
        
def singleFit(file, run, var, cur):
    print("Fitting", file)
    cur.execute('''SELECT FilePath FROM Files WHERE File = ?''', (file,))
    
    try:
        path = cur.fetchall()[0][0]
    except:
        raise Exception(str(file) + " not found in DB")

    meas = MeasLoad.load(path)
    if meas.type == 'Kepco':
        spec = Straight()
    else:
        iso = DBIsotope(meas.type + var[0], meas.line + var[1], '../test/iso.sqlite')
        spec = FullSpec(iso)

    fit = SPFitter(spec, meas, (0, -1))
    fit.fit()
    
    plot.plotFit(fit)
    
    fig = os.path.splitext(path)[0] + '.pdf'
    plot.save(fig)
    
    '''INSERT OR REPLACE INTO Results (File, Run, rChi, pars, fix) VALUES (?, ?, ?, ?, ?)''', (file, run, fit.rchi, buildPars(fit), buildFix(fit))
    
    print("Finished fitting", file)


def createTables(cur):
    cur.execute('''CREATE TABLE IF NOT EXISTS Runs (
    Run TEXT PRIMARY KEY NOT NULL,
    LineVar TEXT,
    IsoVar TEXT
    )''')
    
    #insert Run0 as default if not available
    cur.execute('''SELECT Run FROM Runs WHERE Run = "Run0"''')
    if len(cur.fetchall()) == 0:
        cur.execute('''INSERT INTO Runs VALUES ("Run0", "", "")''')

    cur.execute('''CREATE TABLE IF NOT EXISTS Results
    File TEXT NOT NULL,
    Run TEXT NOT NULL,
    rChi REAL,
    pars TEXT,
    fix TEXT
    PRIMARY KEY (File, Run)
    ''')
    
    cur.execute('''CREATE UNIQUE INDEX result ON Results(File, Run)''')


if __name__ == '__main__':
    path = "V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/"
    db = 'AnaDB.sqlite'
    
    batchFit(['KepcoScan_PCI.txt'], os.path.join(path, db))