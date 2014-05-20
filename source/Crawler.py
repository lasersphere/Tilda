'''
Created on 14.05.2014

@author: hammen
'''

import sqlite3
import os
import sys
import traceback

import Measurement.MeasLoad as Meas

def crawl(path, add = True, rec = True, db = 'AnaDB.sqlite'):
    '''Crawl the path and add all measurement files to the database, recursively if requested'''
    
    print("Crawling", path)
    end = ['.tld', '.mcp', '.txt']
    
    con = sqlite3.connect(os.path.join(path, db))
    cur = con.cursor()

    #Clear table
    if not add:
        cur.execute('''DROP TABLE IF EXISTS Files''')
    
    #Create table if new file or cleared
    cur.execute('''CREATE TABLE IF NOT EXISTS Files (
    file TEXT,
    filePath TEXT,
    date DATE,
    type TEXT,
    line TEXT,
    offset REAL,
    accVolt REAL,
    laserFreq REAL,
    colDirTrue BOOL,
    voltDivRatio REAL,
    lineMult REAL,
    lineOffset REAL
    )''')

    oldPath = os.getcwd()
    os.chdir(path)
    insertFiles('.', rec, cur, end)
    os.chdir(oldPath)
    
    con.commit()
    con.close()
    print("Done")
    
    
def insertFiles(path, rec, cur, end):
    (p, d, f) = next(os.walk(path))
        
    if rec:
        for _d in d:
            insertFiles(os.path.join(p, _d), rec, cur, end)
    
    for _f in f:
        if os.path.splitext(_f)[1] in end:
            print("Adding", os.path.join(p, _f))
            cur.execute('''INSERT INTO Files (file, filePath) VALUES (?, ?)''', (_f, os.path.join(p, _f)))
            
            
def loadCrawl(db = 'AnaDB.sqlite'):
    ''''''
    print("crawling db", db)
    con = sqlite3.connect(os.path.join(path, db))
    cur = con.cursor()
    
    cur.execute('''SELECT filePath FROM Files''')
    files = cur.fetchall()
    
    
    oldPath = os.getcwd()
    print("Working directory is", os.path.dirname(path))
    os.chdir(os.path.dirname(path))
    
    errcount = 0
    for (file,) in files:
        try:
            spec = Meas.load(file, db)
            cur.execute('''UPDATE Files SET
            date = ?, type = ?, line = ?, offset = ?, accVolt = ?, laserFreq = ?, colDirTrue = ?, voltDivRatio = ?, lineMult = ?, lineOffset = ?
            WHERE filePath = ?''',
            (spec.date, spec.type, spec.line, spec.offset, spec.accVolt, spec.laserFreq, spec.colDirTrue, spec.voltDivRatio, spec.lineMult, spec.lineOffset))
        except:
            errcount += 1
            print("Error working on file", file, ":", sys.exc_info()[1])
            traceback.print_tb(sys.exc_info()[2])
        
    os.chdir(oldPath)
    
    con.commit()
    con.close()
    print("loadCrawl done,", errcount, "errors occured")
    
if __name__ == '__main__':
    path = "V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/"
    #crawl(path, False)
    loadCrawl(os.path.join(path, 'AnaDB.sqlite'))