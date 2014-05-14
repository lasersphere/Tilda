'''
Created on 14.05.2014

@author: hammen
'''

import sqlite3
import os

def crawl(path, add = True, rec = False):
    '''Crawl the path and add all measurement files to the database, recursively if requested'''
    print("Crawling", path)
    print(os.getcwd())
    end = ['.tld', '.mcp', '.txt']
    
    con = sqlite3.connect('AnaDB.sqlite')
    cur = con.cursor()

    #Clear table
    if not add:
        cur.execute('''DROP TABLE IF EXISTS Files''')
    
    #Create table if new file or cleared
    cur.execute('''CREATE TABLE IF NOT EXISTS Files (
    File TEXT, FilePath TEXT, date TEXT, type TEXT)''')
    
    
    insertFiles(path, rec, cur, end)
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
            cur.execute('''INSERT INTO Files VALUES (?, ?, ?, ?)''', _f, (os.path.join(p, _f), None, None))

if __name__ == '__main__':
    path = "V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/Daten/"
    crawl(path)