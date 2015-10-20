"""

Created on '06.08.2015'

@author:'simkaufm'

"""

import sqlite3

import Tools as pollitools


def createTildaDB(db):
    """
    will create an sqlite db suited for Tilda.
    :param db: str, path to database
    """
    pollitools.createDB(db)
    formPolliFitDBtoTildaDB(db)

def formPolliFitDBtoTildaDB(db):
    """
    will try to convert the Pollifit Database to the Tilda Standard.
    Some information might get lost in the Table Files.
    :param db: str, path to database
    """
    con = sqlite3.connect(db)

    con.execute('''CREATE TABLE IF NOT EXISTS ScanPars (
    iso TEXT,
    type TEXT,
    track INT,
    dacStartRegister18Bit INT,
    dacStepSize18Bit INT,
    nOfSteps INT,
    postAccOffsetVoltControl INT,
    postAccOffsetVolt INT,
    nOfScans INT,
    dwellTime10ns INT,
    activePmtList TEXT,
    colDirTrue INT
    )''')

    con.close()

def addTrackDictToDb(db, scandict):
    """
    Add a Dictionary containing all infos of a Track to the existing(or not) Trackdictionary
    in the column trackPars of the chosen file. This overwrites the selected Track.
    """
    isod = scandict['isotopeData']
    trackd = scandict['activeTrackPar']
    piped = scandict['pipeInternals']
    iso = isod['isotope']
    nOfTrack = piped['activeTrackNumber']
    sctype = isod['type']
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(''' SELECT iso FROM ScanPars Where iso = ? AND track = ?''', (iso, nOfTrack,))
    if cur.fetchone() is None:
        cur.execute('''INSERT INTO ScanPars (iso, type, track) VALUES (?, ?, ?)''', (iso, sctype, nOfTrack))
    cur.execute('''UPDATE ScanPars
            SET dacStartRegister18Bit = ?,
            dacStepSize18Bit = ?,
            nOfSteps = ?,
            postAccOffsetVoltControl = ?,
            postAccOffsetVolt = ?,
            nOfScans = ?,
            dwellTime10ns = ?,
            activePmtList = ?,
            colDirTrue = ?
             Where iso = ? AND track = ?''',
                (trackd['dacStartRegister18Bit'],
                 trackd['dacStepSize18Bit'],
                 trackd['nOfSteps'],
                 trackd['postAccOffsetVoltControl'],
                 trackd['postAccOffsetVolt'],
                 trackd['nOfScans'],
                 trackd['dwellTime10ns'],
                 str(trackd['activePmtList']),
                 trackd['colDirTrue'],
                 iso, nOfTrack))
    con.commit()
    con.close()

# bdpath = 'D:\\Workspace\\PyCharm\\Tilda\\PolliFit\\test\\Project\\tildaDB.sqlite'
# projectpath = os.path.split(bdpath)[0]
# # createTildaDB(bdpath)
# # # pollitools._insertFile('Data/testTilda.xml', bdpath)
# addTrackDictToDb(bdpath, drftScPars.draftScanDict)
# drftScPars.draftScanDict['pipeInternals']['activeTrackNumber'] = 1
# addTrackDictToDb(bdpath, drftScPars.draftScanDict)
# # addTrackDictToDb(bdpath, 'testTilda.xml', 3, drftScPars.draftTrackPars)
# # pollitools.crawl(bdpath, 'Data')

