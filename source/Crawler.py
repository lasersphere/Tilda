'''
Created on 14.05.2014

@author: hammen
'''

import sqlite3
import os
import sys
import traceback

import Measurement.MeasLoad as Meas

def crawl(db, add = True, crawl = '.', rec = True):
    '''Crawl the path and add all measurement files to the database, recursively if requested'''
    end = ['.tld', '.mcp', '.txt']
    
    print("Crawling", path)
    projectPath, dbname = os.path.split(db)
    
    oldPath = os.getcwd()
    os.chdir(projectPath)

    con = sqlite3.connect(dbname)
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
    

    insertFiles(crawl, rec, cur, end)
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
            
            
def loadCrawl(db):
    '''Crawl the Files table, load the files and write data to DB'''
    print("crawling db", db)
    
    projectPath, dbname = os.path.split(db)
    oldPath = os.getcwd()
    os.chdir(projectPath)
    con = sqlite3.connect(dbname)
    cur = con.cursor()
    
    cur.execute('''SELECT filePath FROM Files''')
    files = cur.fetchall()
    
    errcount = 0
    for (file,) in files:
        try:
            spec = Meas.load(file, db)
            cur.execute('''UPDATE Files SET
            date = ?, type = ?, line = ?, offset = ?, accVolt = ?, laserFreq = ?, colDirTrue = ?, voltDivRatio = ?, lineMult = ?, lineOffset = ?
            WHERE filePath = ?''',
            (spec.date, spec.type, spec.line, spec.offset, spec.accVolt, spec.laserFreq, spec.colDirTrue, spec.voltDivRatio, spec.lineMult, spec.lineOffset, file))
        except:
            errcount += 1
            print("Error working on file", file, ":", sys.exc_info()[1])
            traceback.print_tb(sys.exc_info()[2])
        
    os.chdir(oldPath)
    
    con.commit()
    con.close()
    print("loadCrawl done,", errcount, "errors occured")
    
if __name__ == '__main__':
    path = "V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/AnaDB.sqlite"
    #crawl(path, False)
    loadCrawl(path)