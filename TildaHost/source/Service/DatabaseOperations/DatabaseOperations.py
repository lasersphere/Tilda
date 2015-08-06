"""

Created on '06.08.2015'

@author:'simkaufm'

"""

import Tools as pollitools
import Service.draftScanParameters as drftScPars

import sqlite3

def createTildaDB(db):
    """
    will create an sqlite db suited for Tilda.
    Will delete all files in Files if db already exists
    :param db: str, path to database
    """
    pollitools.createDB(db)
    formEmptyDBtoTildaDB(db)

def formEmptyDBtoTildaDB(db):
    """
    will drop the Files Tbale in db and create a new one suited for Tilda.
    :param db: str, path to database
    """
    con = sqlite3.connect(db)

    con.execute('''DROP TABLE Files''')
    con.execute('''CREATE TABLE IF NOT EXISTS Files (
    file TEXT PRIMARY KEY NOT NULL,
    filePath TEXT UNIQUE NOT NULL,
    date DATE,
    isotope TEXT,
    type TEXT,
    line TEXT,
    accVolt FLOAT,
    laserFreq FLOAT,
    voltDivRatio FLOAT,
    lineMult FLOAT,
    lineOffset FLOAT,
    trackPars TEXT,
    FOREIGN KEY (type) REFERENCES Isotopes (iso),
    FOREIGN KEY (line) REFERENCES Lines (line)
    )
    ''')

    con.close()

def addTrackDictToDb(db, file, nOfTrack, trackDict):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(''' SELECT trackPars FROM Files Where file = ?''', (file,))
    trackPars = None
    if trackPars == None:
        trackPars = {}
        trackPars['track' + str(nOfTrack)] = trackDict
        cur.execute('''UPDATE Files SET trackPars = ? WHERE file = ?''', (str(trackPars), file))
        con.commit()
    else:
        print('gibbet schon')
    con.close()

bdpath = 'D:\\Testi.sqlite'
createTildaDB(bdpath)
addTrackDictToDb(bdpath, 'blub.txt', 0, drftScPars.draftTrackPars)