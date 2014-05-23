'''
Created on 21.05.2014

@author: hammen
'''

import sqlite3
import os
import sys
import traceback

import Measurement.MeasLoad as Meas

def crawl(db, crawl = '.', rec = True):
    '''Crawl the path and add all measurement files to the database, recursively if requested'''
    projectPath, dbname = os.path.split(db)
    print("Crawling", projectPath)
    oldPath = os.getcwd()
    
    os.chdir(projectPath)
    insertFolder(crawl, rec, dbname)
    os.chdir(oldPath)
    
    print("Done")
    

def insertFolder(path, rec, db):
    (p, d, f) = next(os.walk(path))
        
    if rec:
        for _d in d:
            insertFolder(os.path.join(p, _d), rec, db)
    
    for _f in f:
        insertFile(_f, db)
        
def insertFile(f, db):
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
        con.execute('''INSERT INTO Files (file, filePath) VALUES (?, ?)'''(os.path.basename(f), f))
        spec = Meas.load(f, db)
        spec.export(db)  
    except:
        print("Error working on file", f, ":", sys.exc_info()[1])
        traceback.print_tb(sys.exc_info()[2])
        
    con.close() 


def createDB(db):
    '''Initialize a new database. Does not alter existing tables.'''
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
    line TEXT PRIMARY KEY  NOT NULL ,
    reference TEXT,
    frequency FLOAT,
    Jl FLOAT,
    Ju FLOAT,
    shape TEXT,
    fixShape TEXT,
    charge INT,
    FOREIGN KEY (reference) REFERENCES Isotopes (iso)
    )''')
    
    #Files
    con.execute('''CREATE TABLE IF NOT EXISTS Files (
    file TEXT PRIMARY KEY ONT NULL,
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
    isoVar TEXT DEFAULT ""
    )''')
    
    #Fit results
    con.execute('''CREATE TABLE IF NOT EXISTS FitRes (
    file TEXT NOT NULL,
    iso TEXT NOT NULL,
    run TEXT NOT NULL,
    sctr TEXT NOT NULL,
    rChi FLOAT,
    pars TEXT,
    PRIMARY KEY (file, iso, run, sctr),
    FOREIGN KEY (file) REFERENCES Files (file),
    FOREIGN KEY (run) REFERENCES Runs (run)
    )''')
    
    #Combined results (averaged from fit)
    con.execute('''CREATE TABLE IF NOT EXISTS Combined (
    iso TEXT NOT NULL,
    parname TEXT,
    config TEXT DEFAULT "",
    run TEXT NOT NULL,
    sctr TEXT,
    final BOOL DEFAULT 0,
    rChi FLOAT,
    val FLOAT,
    statErr FLOAT,
    statErrForm TEXT DEFAULT err,
    systErr FLOAT,
    systErrForm TEXT DEFAULT 0,
    PRIMARY KEY (iso, parname, run, sctr)
    FOREIGN KEY (run) REFERENCES Runs (run)
    )''')

    con.close()
