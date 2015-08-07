"""

Created on '06.08.2015'

@author:'simkaufm'

"""

import Tools as pollitools
import Service.draftScanParameters as drftScPars

import sqlite3
import ast
import os

def createTildaDB(db):
    """
    will create an sqlite db suited for Tilda.
    Existing tables should be ok.
    :param db: str, path to database
    """
    if not os.path.isfile(db):
        pollitools.createDB(db)
        formPolliFitDBtoTildaDB(db)

def formPolliFitDBtoTildaDB(db):
    """
    will try to convert the Pollifit Database to the Tilda Standard.
    Some information might get lost in the Table Files.
    :param db: str, path to database
    """
    con = sqlite3.connect(db)

    con.executescript('''
    ALter TABLE Files RENAME TO Files_orig;
    CREATE TABLE Files ( file TEXT PRIMARY KEY NOT NULL,
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
    );
    INSERT INTO Files(file, filePath, date, isotope, line, accVolt, laserFreq, voltDivRatio, lineMult, lineOffset)
    SELECT file, filePath, date, type, line, accVolt, laserFreq, voltDivRatio, lineMult, lineOffset FROM Files_orig;
    DROP TABLE Files_orig
    ''')

    con.close()

def addTrackDictToDb(db, file, nOfTrack, trackDict):
    """
    Add a Dictionary containing all infos of a Track to the existing(or not) Trackdictionary
    in the column trackPars of the chosen file. This overwrites the selected Track.
    """
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(''' SELECT trackPars FROM Files Where file = ?''', (file,))
    trackPars = cur.fetchone()[0]
    if trackPars == None:
        trackPars = {}
        trackPars['track' + str(nOfTrack)] = trackDict
    else:
        trackPars = ast.literal_eval(trackPars)
        trackPars['track' + str(nOfTrack)] = trackDict
    cur.execute('''UPDATE Files SET trackPars = ? WHERE file = ?''', (str(trackPars), file))
    con.commit()
    con.close()

bdpath = 'D:\\Workspace\\PyCharm\\Tilda\\PolliFit\\test\\Project\\tildaDB.sqlite'
projectpath = os.path.split(bdpath)[0]
# createTildaDB(bdpath)
# pollitools._insertFile('Data/testTilda.xml', bdpath)
addTrackDictToDb(bdpath, 'testTilda.xml', 0, drftScPars.draftTrackPars)
addTrackDictToDb(bdpath, 'testTilda.xml', 1, drftScPars.draftTrackPars)
addTrackDictToDb(bdpath, 'testTilda.xml', 3, drftScPars.draftTrackPars)
pollitools.crawl(bdpath, 'Data')

