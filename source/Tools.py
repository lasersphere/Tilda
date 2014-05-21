'''
Created on 21.05.2014

@author: hammen
'''

import sqlite3

def createDB(path):
    '''Initialize a new database. Be careful when using with existing DBs'''
    con = sqlite3.connect(path)
    con.execute('''PRAGMA foreign_keys''')
    cur = con.cursor()
    
    
    #Isotopes
    cur.execute('''CREATE TABLE IF NOT EXISTS Isotopes (
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
    cur.execute('''CREATE TABLE IF NOT EXISTS Lines (
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
    cur.execute('''CREATE TABLE IF NOT EXISTS Files (
    file TEXT PRIMARY KEY,
    filePath TEXT UNIQUE,
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
    cur.execute('''CREATE TABLE IF NOT EXISTS Runs (
    run TEXT PRIMARY KEY NOT NULL,
    lineVar TEXT DEFAULT "",
    isoVar TEXT DEFAULT ""
    )''')
    
    #Fit results
    cur.execute('''CREATE TABLE IF NOT EXISTS FitRes (
    file TEXT NOT NULL,
    iso TEXT NOT NULL,
    run TEXT NOT NULL,
    scaler INT NOT NULL,
    track INT NOT NULL,
    rChi FLOAT,
    pars TEXT,
    PRIMARY KEY (File, Iso, Run, Scaler, Track),
    FOREIGN KEY (file) REFERENCES Files (file),
    FOREIGN KEY (run) REFERENCES Runs (run)
    )''')
    
    #combined results (averaged from fit)
    cur.execute('''CREATE TABLE IF NOT EXISTS Combined (
    iso TEXT NOT NULL,
    name TEXT,
    run TEXT NOT NULL,
    scaler INT NOT NULL,
    track INT NOT NULL,
    rChi FLOAT,
    par FLOAT,
    err FLOAT,
    usederr TEXT,
    PRIMARY KEY (iso, name)
    FOREIGN KEY (run) REFERENCES Runs (run)
    )''')
    
    con.commit()
    con.close()