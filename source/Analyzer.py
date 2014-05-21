'''
Created on 21.05.2014

@author: hammen
'''

import sqlite3

import numpy as np

    
def extract(iso, run, st, par, db, fileList = ''):
    ''''''
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT File, pars FROM Results WHERE Iso = ? AND Run = ? AND Scaler = ? AND Track = ?''', (iso, run, st[0], st[1]))
    fits = cur.fetchall()
    
    if fileList:
        fits = [f for f in fits if f[0] in fileList]
        
    fits = [eval(f[1]) for f in fits]
    vals = [f[par][0] for f in fits]
    errs = [f[par][1] for f in fits]
    
    con.close()
    return (vals, errs)
    
    
def weightedAverage(vals, errs):
    '''Return (weighted average, propagated error, rChi^2'''
    weights = 1 / np.square(errs)
    average = sum(vals * weights) / sum(weights)
    errorprop = 1 / sum(weights)
    rChi = 1 / (len(vals) - 1) * sum(np.square(vals - average) * weights)
    
    return (average, errorprop, rChi)

def compress():
    pass
    
def createTable(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''CREATE TABLE IF NOT EXISTS Results (
    Iso TEXT NOT NULL,
    Par TEXT NOT NULL,
    Run TEXT NOT NULL,
    Scaler INT NOT NULL,
    Track INT NOT NULL,
    rChi REAL,
    pars TEXT,
    PRIMARY KEY (File, Iso, Run, Scaler, Track)
    )''')
    
    
    con.commit()
    con.close()
    