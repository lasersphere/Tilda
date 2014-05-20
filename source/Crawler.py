'''
Created on 14.05.2014

@author: hammen
'''

import sqlite3
import os

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
    File TEXT,
    FilePath TEXT,
    date DATE,
    type TEXT,
    line TEXT
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
            cur.execute('''INSERT INTO Files (File, FilePath) VALUES (?, ?)''', (_f, os.path.join(p, _f)))
            
            
def loadCrawl(db = 'AnaDB.sqlite'):
    ''''''
    
    
    
    
if __name__ == '__main__':
    path = "V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/"
    crawl(path, False)