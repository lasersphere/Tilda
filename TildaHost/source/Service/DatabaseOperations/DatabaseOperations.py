"""

Created on '06.08.2015'

@author:'simkaufm'

"""

import sqlite3
import ast

import Tools as PolliTools
import Service.VoltageConversions.VoltageConversions as VCon
import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as Dft


def createTildaDB(db):
    """
    will create an sqlite db suited for Tilda.
    :param db: str, path to database
    """
    PolliTools.createDB(db)
    form_pollifit_db_to_tilda_db(db)


def form_pollifit_db_to_tilda_db(db):
    """
    will convert a PolliFit Database to the Tilda Standard.
    :param db: str, path to database
    """
    con = sqlite3.connect(db)

    con.execute('''CREATE TABLE IF NOT EXISTS ScanPars (
    iso TEXT NOT NULL,
    type TEXT NOT NULL,
    track INT,
    accVolt FLOAT,
    laserFreq FLOAT,
    dacStartVolt FLOAT,
    dacStopVolt FLOAT,
    dacStepSizeVolt FLOAT,
    invertScan TEXT,
    nOfSteps INT,
    nOfScans INT,
    postAccOffsetVoltControl INT,
    postAccOffsetVolt FLOAT,
    activePmtList TEXT,
    colDirTrue TEXT,
    sequencerDict TEXT,
    waitForKepco25nsTicks INT,
    waitAfterReset25nsTicks INT,
    measureVoltPars Text,
    UNIQUE (iso, type, track)
    )''')

    con.close()


def add_track_dict_to_db(db, scandict, overwrite=True):
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
    try:
        stop_volt = VCon.get_voltage_from_18bit(
            trackd['dacStartRegister18Bit'] + trackd['dacStepSize18Bit'] * trackd['nOfSteps'])
    except TypeError:
        stop_volt = None
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(''' SELECT iso FROM ScanPars WHERE iso = ? AND type = ? AND track = ?''', (iso, sctype, nOfTrack,))
    if cur.fetchone() is None:
        cur.execute('''INSERT INTO ScanPars (iso, type, track) VALUES (?, ?, ?)''', (iso, sctype, nOfTrack,))
    if overwrite:
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
                waitAfterReset25nsTicks = ?,
                measureVoltPars = ?,
                accVolt = ?,
                laserFreq = ?
                 WHERE iso = ? AND type = ? AND track = ?''',
                    (
                        VCon.get_voltage_from_18bit(trackd['dacStartRegister18Bit']),
                        stop_volt,
                        VCon.get_stepsize_in_volt_from_18bit(trackd['dacStepSize18Bit']),
                        str(trackd['invertScan']),
                        trackd['nOfSteps'],
                        trackd['nOfScans'],
                        trackd['postAccOffsetVoltControl'],
                        trackd['postAccOffsetVolt'],
                        str(trackd['activePmtList']),
                        str(trackd['colDirTrue']),
                        str(SdOp.sequencer_dict_from_track_dict(trackd, sctype)),
                        trackd['waitForKepco25nsTicks'],
                        trackd['waitAfterReset25nsTicks'],
                        str(scandict['measureVoltPars']),
                        str(isod['accVolt']),
                        isod['laserFreq'],
                        iso, sctype, nOfTrack)
                    )
        con.commit()
    con.close()


def check_for_existing_isos(db, sctype):
    """ given a path to a database and a sequencer type this will
    return a list of strings with all isotopes in the db for this sequencer type """
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(' SELECT iso FROM ScanPars WHERE type = ?', (sctype,))
    isos = cur.fetchall()
    if len(isos):
        isos = [iso[0] for iso in isos]
    else:
        isos = []
    con.close()
    return isos


def extract_track_dict_from_db(database_path_str, iso, sctype, tracknum):
    """ for a given database, isotope, scan type and tracknumber, this will return a complete scandictionary """
    scand = SdOp.init_empty_scan_dict(sctype)
    scand['isotopeData']['isotope'] = iso
    scand['isotopeData']['type'] = sctype
    scand['track' + str(tracknum)] = scand.pop('activeTrackPar')
    con = sqlite3.connect(database_path_str)
    cur = con.cursor()
    cur.execute(
        '''
        SELECT     dacStartVolt, dacStepSizeVolt, invertScan,
         nOfSteps, nOfScans, postAccOffsetVoltControl,
          postAccOffsetVolt, activePmtList, colDirTrue,
           sequencerDict, waitForKepco25nsTicks, waitAfterReset25nsTicks,
           measureVoltPars, accVolt, laserFreq
        FROM ScanPars WHERE iso = ? AND type = ? AND track = ?
        ''', (iso, sctype, tracknum,)
    )
    data = cur.fetchone()
    data = list(data)
    scand['isotopeData']['laserFreq'] = data.pop(-1)
    scand['isotopeData']['accVolt'] = data.pop(-1)
    scand['measureVoltPars'] = SdOp.merge_dicts(scand['measureVoltPars'], ast.literal_eval(data.pop(-1)))
    scand['track' + str(tracknum)] = db_track_values_to_trackdict(data, scand['track' + str(tracknum)])
    con.close()

    return scand


def db_track_values_to_trackdict(data, track_dict):
    """ given a data list containing (dacStartVolt, dacStepSizeVolt, invertScan,
         nOfSteps, nOfScans, postAccOffsetVoltControl,
          postAccOffsetVolt, activePmtList, colDirTrue,
           sequencerDict, waitForKepco25nsTicks, waitAfterReset25nsTicks,
            measureVoltPars, accVolt, laserFreq) from the database, this
            converts all values to a useable track dictionary."""
    dict_keys_list = ['dacStartRegister18Bit', 'dacStepSize18Bit', 'invertScan',
                      'nOfSteps', 'nOfScans', 'postAccOffsetVoltControl',
                      'postAccOffsetVolt', 'activePmtList', 'colDirTrue',
                      'sequencerDict', 'waitForKepco25nsTicks', 'waitAfterReset25nsTicks']
    conversion_list = ['VCon.get_18bit_from_voltage(%s)', 'VCon.get_18bit_stepsize(%s)', '%s',
                       '%s', '%s', '%s',
                       '%s', '%s', '%s',
                       '%s', '%s', '%s']
    # step 1: get rid of unwanted strings in data:
    data = list(ast.literal_eval(j) if isinstance(j, str) else j for i, j in enumerate(data))
    # step 2: convert all values according to the conversion list
    data = [eval(conversion_list[i] % j) for i, j in enumerate(data)]
    # step 3: create dictionary from keys as in dict_keys_list, with values from data
    newdict = {j: data[i] for i, j in enumerate(dict_keys_list)}
    # step 4: pop the sequencer dict and merge it with the track_dict
    track_dict = SdOp.merge_dicts(track_dict, newdict.pop('sequencerDict'))
    # step 5: merge with remaining
    track_dict = SdOp.merge_dicts(track_dict, newdict)
    return track_dict


# db = 'D:\\blub\\blub.sqlite'
# # createTildaDB(db)
# # scand = SdOp.init_empty_scan_dict()
# # scand['isotopeData']['isotope'] = '40Ca'
# # scand['isotopeData']['type'] = 'cs'
# # scand['pipeInternals']['activeTrackNumber'] = 0
# # add_track_dict_to_db(db, Dft.draftScanDict)
# print(extract_track_dict_from_db(db, '44Ca', 'cs', 0))