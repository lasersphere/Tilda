'''
Created on 21.05.2014

@author: hammen

The Tools module contains simple helper methods like createDB() to create a new empty database with the
right structure, isoPlot() to plot the spectrum of an isotope and centerplot() to plot acceleration
voltages depending on laser frequencies.

'''

import sqlite3
import os
import sys
import traceback

import numpy as np

import Measurement.MeasLoad as Meas
from DBIsotope import DBIsotope
from Spectra.FullSpec import FullSpec
import Physics
import MPLPlotter as plot
import matplotlib.pyplot as plt

def isoPlot(iso, line, db, isovar = '', linevar = ''):
    '''plot isotope iso'''
    iso = DBIsotope(db, iso, isovar, linevar)
    
    spec =  FullSpec(iso)
    
    print(spec.getPars())
    
    plot.plot(spec.toPlot(spec.getPars()))
    plot.show()


def centerPlot(isoL, line, db, width = 1e6):
    '''Plot kinetic energy / eV, under which isotopes in isoL are on resonace depending on laser frequency up to width MHz away'''
    isos = [DBIsotope(iso, line, db) for iso in isoL]
    
    res = 100
    fx = np.linspace(isos[0].freq - width, isos[0].freq + width, res)
    wnx = Physics.wavenumber(fx)
    
    y = np.zeros((len(isos), len(fx)))
    for i, iso in enumerate(isos):
        for j, x in enumerate(fx):
            v = Physics.invRelDoppler(x, iso.freq + iso.center)
            y[i][j] = (iso.mass * Physics.u * v**2)/2 / Physics.qe
    
    fig = plt.figure(1, (8, 8))
    fig.patch.set_facecolor('white')
    
    for i in y:
        plt.plot(wnx, i, '-')
    
    plt.xlabel("Laser wavenumber / cm^-1")
    plt.ylabel("Ion energy on resonance / eV")
    plt.axvline(Physics.wavenumber(isos[0].freq), 0, 20000, color = 'k')
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.show()
    

def crawl(db, crawl = '.', rec = True):
    '''Crawl the path and add all measurement files to the database, recursively if requested'''
    projectPath, dbname = os.path.split(db)
    print("Crawling", projectPath)
    oldPath = os.getcwd()
    
    os.chdir(projectPath)
    _insertFolder(crawl, rec, dbname)
    os.chdir(oldPath)
    
    print("Done")
    

def _insertFolder(path, rec, db):
    (p, d, f) = next(os.walk(path))
        
    if rec:
        for _d in d:
            _insertFolder(os.path.join(p, _d), rec, db)
    
    for _f in f:
        _insertFile(os.path.join(p, _f), db)
        
def _insertFile(f, db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT (1) FROM Files WHERE file = ?''', (os.path.basename(f),))
    if len(cur.fetchall()) != 0:
        print('Skipped', f, ': already in db.')
        return
    
    if not Meas.check(os.path.splitext(f)[1]):
        print('Skipped', f, ': not importable.')
        return
    
    try:
        cur.execute('''INSERT INTO Files (file, filePath) VALUES (?, ?)''', (os.path.basename(f), f))
        con.commit()
        spec = Meas.load(f, db, True)
        spec.export(db)  
    except:
        print("Error working on file", f, ":", sys.exc_info()[1])
        traceback.print_tb(sys.exc_info()[2])
        
    con.close() 


def createDB(db):
    '''Initialize a new database. Does not alter existing tables.'''
    print('Initializing db', db)
    con = sqlite3.connect(db)
    
    #Isotopes
    con.execute('''CREATE TABLE IF NOT EXISTS Isotopes (
    iso TEXT PRIMARY KEY  NOT NULL,
    mass FLOAT,
    mass_d FLOAT,
    I FLOAT,
    center FLOAT,
    Al FLOAT DEFAULT 0,
    Bl FLOAT DEFAULT 0,
    Au FLOAT DEFAULT 0,
    Bu FLOAT DEFAULT 0,
    fixedArat BOOL DEFAULT 0,
    fixedBrat BOOL DEFAULT 0,
    intScale DOUBLE DEFAULT 1,
    fixedInt BOOL DEFAULT 0,
    relInt TEXT,
    m TEXT
    )''')
    
    #Lines
    con.execute('''CREATE TABLE IF NOT EXISTS Lines (
    lineVar TEXT PRIMARY KEY  NOT NULL ,
    reference TEXT,
    refRun TEXT,
    frequency FLOAT,
    Jl FLOAT,
    Ju FLOAT,
    shape TEXT,
    fixShape TEXT,
    charge INT,
    FOREIGN KEY (reference) REFERENCES Isotopes (iso)
    FOREIGN KEY (refRun) REFERENCES Runs (run)
    )''')
    
    #Files
    con.execute('''CREATE TABLE IF NOT EXISTS Files (
    file TEXT PRIMARY KEY NOT NULL,
    filePath TEXT UNIQUE NOT NULL,
    date DATE,
    type TEXT,
    line TEXT,
    offset FLOAT,
    accVolt FLOAT,
    laserFreq FLOAT,
    colDirTrue BOOL,
    voltDivRatio FLOAT,
    lineMult FLOAT,
    lineOffset FLOAT,
    FOREIGN KEY (type) REFERENCES Isotopes (iso),
    FOREIGN KEY (line) REFERENCES Lines (line)
    )''')
    
    #Runs
    con.execute('''CREATE TABLE IF NOT EXISTS Runs (
    run TEXT PRIMARY KEY NOT NULL,
    lineVar TEXT DEFAULT "",
    isoVar TEXT DEFAULT "",
    scaler TEXT,
    track TEXT
    )''')
    
    con.execute('''INSERT OR IGNORE INTO Runs VALUES ("Run0", "", "", "[0]", "-1")''')
    
    #Fit results
    con.execute('''CREATE TABLE IF NOT EXISTS FitRes (
    file TEXT NOT NULL,
    iso TEXT NOT NULL,
    run TEXT NOT NULL,
    rChi FLOAT,
    pars TEXT,
    PRIMARY KEY (file, iso, run),
    FOREIGN KEY (file) REFERENCES Files (file),
    FOREIGN KEY (run) REFERENCES Runs (run)
    )''')
    
    #Combined results (averaged from fit)
    con.execute('''CREATE TABLE IF NOT EXISTS Combined (
    iso TEXT NOT NULL,
    parname TEXT,
    run TEXT,
    config TEXT DEFAULT "[]",
    final BOOL DEFAULT 0,
    rChi FLOAT,
    val FLOAT,
    statErr FLOAT,
    statErrForm TEXT DEFAULT err,
    systErr FLOAT,
    systErrForm TEXT DEFAULT 0,
    PRIMARY KEY (iso, parname, run)
    FOREIGN KEY (run) REFERENCES Runs (run)
    )''')

    con.close()
