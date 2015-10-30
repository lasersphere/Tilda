"""

Created on '06.08.2015'

@author:'simkaufm'

"""

import sqlite3

import Tools as pollitools
import Service.VoltageConversions.VoltageConversions as VCon

def createTildaDB(db):
    """
    will create an sqlite db suited for Tilda.
    :param db: str, path to database
    """
    pollitools.createDB(db)
    form_pollifit_db_to_tilda_db(db)

def form_pollifit_db_to_tilda_db(db):
    """
    will convert a PolliFit Database to the Tilda Standard.
    :param db: str, path to database
    """
    con = sqlite3.connect(db)

    con.execute('''CREATE TABLE IF NOT EXISTS ScanPars (
    iso TEXT PRIMARY KEY NOT NULL,
    type TEXT NOT NULL,
    track INT,
    dacStartVolt FLOAT,
    dacStopVolt FLOAT,
    dacStepSizeVolt FLOAT,
    invertScan TEXT,
    nOfSteps INT,
    nOfScans INT,
    postAccOffsetVoltControl INT,
    postAccOffsetVolt INT,
    activePmtList TEXT,
    colDirTrue TEXT,
    sequencerDict TEXT,
    waitForKepco25nsTicks INT,
    waitAfterReset25nsTicks INT
    )''')

    con.close()

def add_track_dict_to_db(db, scandict):
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
    cur.execute(''' SELECT iso FROM ScanPars Where iso = ? AND type = ? AND track = ?''', (iso, sctype, nOfTrack,))
    if cur.fetchone() is None:
        cur.execute('''INSERT INTO ScanPars (iso, type, track) VALUES (?, ?, ?)''', (iso, sctype, nOfTrack))
    cur.execute('''UPDATE ScanPars
            SET dacStartVolt = ?,
            dacStopVolt = ?,
            dacStepSizeVolt = ?,
            invertScan = ? ,
            nOfSteps = ?,
            nOfScans = ?,
            postAccOffsetVoltControl = ?,
            postAccOffsetVolt = ?,
            activePmtList = ?,
            colDirTrue = ?,
            sequencerDict = ?,
            waitForKepco25nsTicks = ?,
            waitAfterReset25nsTicks = ?
             Where iso = ? AND track = ?''',
                (VCon.get_voltage_from_18bit(trackd['dacStartRegister18Bit']),
                 VCon.get_voltage_from_18bit(
                     trackd['dacStartRegister18Bit'] + trackd['dacStepSize18Bit'] * trackd['nOfSteps']),
                 VCon.get_stepsize_in_volt_from_18bit(trackd['dacStepSize18Bit']),
                 str(trackd['invertScan']),
                 trackd['nOfSteps'],
                 trackd['nOfScans'],
                 trackd['postAccOffsetVoltControl'],
                 trackd['postAccOffsetVolt'],
                 str(trackd['activePmtList']),
                 str(trackd['colDirTrue']),
                 trackd['dwellTime10ns'],
                 trackd['waitForKepco25nsTicks'],
                 trackd['waitAfterReset25nsTicks'],
                 iso, nOfTrack))
    con.commit()
    con.close()



# bdpath = 'D:\\Workspace\\PyCharm\\Tilda\\PolliFit\\test\\Project\\tildaDB.sqlite'
# projectpath = os.path.split(bdpath)[0]
# # createTildaDB(bdpath)
# # # pollitools._insertFile('Data/testTilda.xml', bdpath)
# add_track_dict_to_db(bdpath, drftScPars.draftScanDict)
# drftScPars.draftScanDict['pipeInternals']['activeTrackNumber'] = 1
# add_track_dict_to_db(bdpath, drftScPars.draftScanDict)
# # add_track_dict_to_db(bdpath, 'testTilda.xml', 3, drftScPars.draftTrackPars)
# # pollitools.crawl(bdpath, 'Data')

