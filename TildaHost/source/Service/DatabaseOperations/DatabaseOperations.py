"""

Created on '06.08.2015'

@author:'simkaufm'

"""

import sqlite3
import ast
import logging
from copy import copy

import Tools as PolliTools
import Service.VoltageConversions.VoltageConversions as VCon
import Service.Scan.ScanDictionaryOperations as SdOp
from Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TiTs


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
    triggerDict TEXT,
    waitForKepco25nsTicks INT,
    waitAfterReset25nsTicks INT,
    measureVoltPars Text,
    UNIQUE (iso, type, track)
    )''')

    con.close()


def add_scan_dict_to_db(db, scandict, n_of_track, track_key='track0', overwrite=True):
    """
    Write the contents of scandict to the database in the table ScanPars.
    Only the selected track parameters will be written to the db.
    if (iso, type, n_of_track) not yet in database, they will be created.
    """
    isod = scandict['isotopeData']
    if scandict.get(track_key) is None:
        scandict[track_key] = SdOp.init_empty_scan_dict()['track0']
    trackd = scandict[track_key]
    trigger_dict = trackd.get('trigger', {})
    trig_name = trigger_dict.get('type').name
    trigger_dict['type'] = trig_name  # string is better for sql sto
    iso = isod['isotope']
    sctype = isod['type']
    try:
        stop_volt = VCon.get_voltage_from_18bit(
            trackd['dacStartRegister18Bit'] + trackd['dacStepSize18Bit'] * trackd['nOfSteps'])
    except TypeError:
        stop_volt = None
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(''' SELECT iso FROM ScanPars WHERE iso = ? AND type = ? AND track = ?''', (iso, sctype, n_of_track,))
    if cur.fetchone() is None:
        cur.execute('''INSERT INTO ScanPars (iso, type, track) VALUES (?, ?, ?)''', (iso, sctype, n_of_track,))
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
                triggerDict = ?,
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
                        str(trigger_dict),
                        trackd['waitForKepco25nsTicks'],
                        trackd['waitAfterReset25nsTicks'],
                        str(scandict['measureVoltPars']),
                        str(isod['accVolt']),
                        isod['laserFreq'],
                        iso, sctype, n_of_track)
                    )
        con.commit()
    con.close()


def check_for_existing_isos(db, sctype):
    """ given a path to a database and a sequencer type this will
    return a list of strings with all isotopes in the db for this sequencer type """
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(' SELECT iso FROM ScanPars WHERE type = ? AND track = 0', (sctype,))
    isos = cur.fetchall()
    if len(isos):
        isos = [iso[0] for iso in isos]
    else:
        isos = []
    con.close()
    return isos


def add_new_iso(db, iso, seq_type):
    """ write an empty isotope dictionary of a given scantype to the database """
    if iso is '':
        return None
    if iso in check_for_existing_isos(db, seq_type):
        logging.info('isotope ' + iso + ' (' + seq_type + ')' + ' already created, will not be added')
        return None
    scand = SdOp.init_empty_scan_dict(type_str=seq_type, load_default_vals=True)
    scand['isotopeData']['isotope'] = iso
    scand['isotopeData']['type'] = seq_type
    scand['pipeInternals']['activeTrackNumber'] = 0
    add_scan_dict_to_db(db, scand, 0, track_key='track0')
    logging.debug('added ' + iso + ' (' + seq_type + ') to database')
    return iso


def extract_track_dict_from_db(database_path_str, iso, sctype, tracknum):
    """ for a given database, isotope, scan type and tracknumber, this will return a complete scandictionary """
    scand = SdOp.init_empty_scan_dict(sctype)
    scand['isotopeData']['isotope'] = iso
    scand['isotopeData']['type'] = sctype
    scand['track' + str(tracknum)] = scand.pop('track0')
    con = sqlite3.connect(database_path_str)
    cur = con.cursor()
    cur.execute(
        '''
        SELECT     dacStartVolt, dacStepSizeVolt, invertScan,
         nOfSteps, nOfScans, postAccOffsetVoltControl,
          postAccOffsetVolt, activePmtList, colDirTrue,
           sequencerDict, waitForKepco25nsTicks, waitAfterReset25nsTicks,
           triggerDict,
           measureVoltPars, accVolt, laserFreq
        FROM ScanPars WHERE iso = ? AND type = ? AND track = ?
        ''', (iso, sctype, tracknum,)
    )
    data = cur.fetchone()
    if data is None:
        return None
    data = list(data)
    scand['isotopeData']['laserFreq'] = data.pop(-1)
    scand['isotopeData']['accVolt'] = data.pop(-1)
    scand['measureVoltPars'] = SdOp.merge_dicts(scand['measureVoltPars'], ast.literal_eval(data.pop(-1)))
    scand['track' + str(tracknum)]['trigger'] = ast.literal_eval(data.pop(-1))
    scand['track' + str(tracknum)]['trigger']['type'] = getattr(TiTs, scand['track' + str(tracknum)]['trigger']['type'])
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


def get_number_of_tracks_in_db(db, iso, sctype):
    """ return the number of tracks for the given isotope and sctype """
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''
        SELECT track FROM ScanPars WHERE iso = ? AND type = ?
        ''', (iso, sctype,))
    data = cur.fetchall()
    return len(data)


def extract_all_tracks_from_db(db, iso, sctype):
    """ will return one scandict which contains all tracks for the given isotope.
     naming is 'track0':{ ... } , 'track1':{ ... } etc.
     general infos like measureVoltPars will be merged, latter (higher tracknum) will overwrite """
    n_o_tracks = get_number_of_tracks_in_db(db, iso, sctype)
    scand = {}
    for i in range(n_o_tracks):
        scand = SdOp.merge_dicts(scand, extract_track_dict_from_db(db, iso, sctype, i))
    scand['isotopeData']['nOfTracks'] = n_o_tracks
    return scand


# db = 'D:\\blub\\blub.sqlite'
# # # createTildaDB(db)
# # # scand = SdOp.init_empty_scan_dict()
# # # scand['isotopeData']['isotope'] = '40Ca'
# # # scand['isotopeData']['type'] = 'cs'
# # # scand['pipeInternals']['activeTrackNumber'] = 0
# # # add_scan_dict_to_db(db, Dft.draftScanDict)
# # print(extract_track_dict_from_db(db, '44Ca', 'cs', 0))
# print(extract_all_tracks_from_db(db, '40Ca', 'cs'))