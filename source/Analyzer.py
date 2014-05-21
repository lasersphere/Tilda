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
    
    cur.execute('''SELECT file, pars FROM Results WHERE iso = ? AND run = ? AND scaler = ? AND track = ?''', (iso, run, st[0], st[1]))
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
    