"""

Created on '06.08.2015'

@author:'simkaufm'

"""

import Tools as pollitools

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
