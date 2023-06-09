"""

Created on '06.08.2015'

@author:'simkaufm'

"""

import ast
import logging
import sqlite3
from copy import deepcopy

import Tilda.Service.Scan.ScanDictionaryOperations as SdOp
import Tilda.Service.VoltageConversions.VoltageConversions as VCon
from Tilda.PolliFit import Tools as PolliTools
from Tilda.PolliFit.TildaTools import clean_pre_post_scan_set_point_in_scan_dict
from Tilda.Driver.DataAcquisitionFpga.TriggerTypes import TriggerTypes as TriTypes


def createTildaDB(db):
    """
    will create an sqlite db suited for Tilda.
    :param db: str, path to database
    """
    PolliTools.createDB(db)
    form_pollifit_db_to_tilda_db(db)
    check_for_missing_columns_scan_pars(db)


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
    waitForKepco1us INT,
    waitAfterReset1us INT,
    measureVoltPars TEXT,
    pulsePattern TEXT,
    triton TEXT,
    outbits TEXT,
    scanDevDict TEXT,
    UNIQUE (iso, type, track)
    )''')

    con.close()


def check_for_missing_columns_scan_pars(db):
    """ will check if all columns for the scan pars table are available and add those which are not """
    target_cols = [
        (1, 'iso', 'TEXT NOT NULL'),
        (2, 'type', 'TEXT NOT NULL'),
        (3, 'track', 'INT'),
        (4, 'accVolt', 'FLOAT'),
        (5, 'laserFreq', 'FLOAT'),
        (6, 'dacStartVolt', 'FLOAT'),
        (7, 'dacStopVolt', 'FLOAT'),
        (8, 'dacStepSizeVolt', 'FLOAT'),
        (9, 'invertScan', 'TEXT'),
        (10, 'nOfSteps', 'INT'),
        (11, 'nOfScans', 'INT'),
        (12, 'postAccOffsetVoltControl', 'INT'),
        (13, 'postAccOffsetVolt', 'FLOAT'),
        (14, 'activePmtList', 'TEXT'),
        (15, 'colDirTrue', 'TEXT'),
        (16, 'sequencerDict', 'TEXT'),
        (17, 'triggerDict', 'TEXT'),
        (18, 'waitForKepco1us', 'INT'),
        (19, 'waitAfterReset1us', 'INT'),
        (20, 'measureVoltPars', 'TEXT'),
        (21, 'pulsePattern', 'TEXT'),
        (22, 'triton', 'TEXT'),
        (23, 'outbits', 'TEXT'),
        (24, 'scanDevDict', 'TEXT'),
        (25, 'sql', 'TEXT')
    ]
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(''' PRAGMA TABLE_INFO(ScanPars)''')
    cols = cur.fetchall()
    cols_name_flat = [each[1] for each in cols]
    for each in target_cols:
        if each[1] not in cols_name_flat:
            logging.info('column %s in ScanPars Table was not yet in db, adding now.' % each[1])
            cur.execute(''' ALTER TABLE ScanPars ADD COLUMN '%s' '%s' ''' % (each[1], each[2]))
    con.commit()
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
    for triggers, dicts in trigger_dict.items():
        trig_name = trigger_dict[triggers].get('type').name
        trigger_dict[triggers]['type'] = trig_name  # string is better for sql sto
    iso = isod['isotope']
    sctype = isod['type']
    dac_start18b = trackd.get('dacStartRegister18Bit', 0)  # backwards comp.
    dac_step_size18b = trackd.get('dacStepSize18Bit', 0)  # backwards comp.
    try:
        stop = VCon.get_voltage_from_bits(
            dac_start18b + dac_step_size18b * trackd['nOfSteps'])  # backwards comp.
    except TypeError:
        stop = None
    start = VCon.get_voltage_from_bits(dac_start18b)  # backwards comp.
    step_size = VCon.get_voltage_from_bits(dac_step_size18b)  # backwards comp.

    scan_dev_dict = trackd.get('scanDevice', {})
    if scan_dev_dict != {}:  # overwrite from scandev.
        start = scan_dev_dict['start']
        stop = scan_dev_dict['stop']
        step_size = scan_dev_dict['stepSize']

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
                waitForKepco1us = ?,
                waitAfterReset1us = ?,
                measureVoltPars = ?,
                accVolt = ?,
                laserFreq = ?,
                pulsePattern = ?,
                triton = ?,
                sql = ?,
                outbits = ?,
                scanDevDict = ?
                WHERE iso = ? AND type = ? AND track = ?''',
                    (
                        start,
                        stop,
                        step_size,
                        str(trackd['invertScan']),
                        trackd['nOfSteps'],
                        trackd['nOfScans'],
                        trackd['postAccOffsetVoltControl'],
                        trackd['postAccOffsetVolt'],
                        str(trackd['activePmtList']),
                        str(trackd['colDirTrue']),
                        str(SdOp.sequencer_dict_from_track_dict(trackd, sctype)),
                        str(trigger_dict),
                        trackd['waitForKepco1us'],
                        trackd['waitAfterReset1us'],
                        str(trackd['measureVoltPars']),
                        str(isod['accVolt']),
                        isod['laserFreq'],
                        str(trackd['pulsePattern']),
                        str(trackd['triton']),
                        str(trackd['sql']),
                        str(trackd['outbits']),
                        str(scan_dev_dict),
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


def get_iso_settings(db, iso):
    """
    get the settings of the desired isotope. None, if not existing yet.
    settings are: mass, mass_d, I, center, Al, Bl, Au, Bu, fixedArat, fixedBrat, intScale, fixedInt, relInt, m, midTof
    return: (cols_list, vals_list)
    """
    logging.info('getting settings of iso: %s from db' % iso)
    cols = ['iso', 'mass', 'mass_d', 'I', 'center', 'Al', 'Bl', 'Au', 'Bu',
            'fixedArat', 'fixedBrat', 'intScale', 'fixedInt', 'relInt', 'm', 'midTof']
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(' SELECT mass, mass_d, I, center, Al, Bl, Au, Bu,'
                ' fixedArat, fixedBrat, intScale, fixedInt, relInt, m, midTof FROM Isotopes WHERE iso = ? ',
                (iso,))
    ret = cur.fetchall()
    con.close()
    if len(ret):
        ret = ret[0]
        ret = list(ret)
        ret.insert(0, iso)
    else:
        return None
    return cols, ret


def update_iso_settings(db, iso, settings_list):
    """
    update or create the settings for the desired isotope in the db
    :param db: str, path of db
    :param iso: str, name of iso existing / not existing
    :param settings_list: list, contains:
        mass, mass_d, I, center, Al, Bl, Au, Bu, fixedArat, fixedBrat, intScale, fixedInt, relInt, m, midTof
    :return: None
    """
    iso_exist = get_iso_settings(db, iso) is not None
    con = sqlite3.connect(db)
    cur = con.cursor()
    if iso_exist:
        cur.execute(''' UPDATE Isotopes SET mass = ?, mass_d = ?, I = ?, center = ?, Al = ?, Bl = ?, Au = ?, Bu = ?,
fixedArat = ?, fixedBrat = ?, intScale = ?, fixedInt = ?, relInt = ?, m = ?, midTof = ? WHERE iso = ?''',
            (settings_list[0], settings_list[1], settings_list[2], settings_list[3],
             settings_list[4], settings_list[5], settings_list[6], settings_list[7],
             settings_list[8], settings_list[9], settings_list[10], settings_list[11],
             settings_list[12], settings_list[13], settings_list[14], iso))
    else:
        cur.execute(
            '''INSERT INTO Isotopes (iso, mass, mass_d, I, center, Al, Bl, Au, Bu,
fixedArat, fixedBrat, intScale, fixedInt, relInt, m, midTof) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (iso, settings_list[0], settings_list[1], settings_list[2], settings_list[3],
             settings_list[4], settings_list[5], settings_list[6], settings_list[7],
             settings_list[8], settings_list[9], settings_list[10], settings_list[11],
             settings_list[12], settings_list[13], settings_list[14]))
    con.commit()
    con.close()


def add_new_iso(db, iso, seq_type, exisiting_iso=None):
    """ write an empty isotope dictionary of a given scantype to the database """
    if iso == '':
        return None
    if iso in check_for_existing_isos(db, seq_type):
        logging.info('isotope ' + iso + ' (' + seq_type + ')' + ' already created, will not be added')
        return None
    if exisiting_iso is None:
        scand = SdOp.init_empty_scan_dict(type_str=seq_type, load_default_vals=True)
        scand['isotopeData']['isotope'] = iso
        scand['isotopeData']['type'] = seq_type
        scand['pipeInternals']['activeTrackNumber'] = 0
        add_scan_dict_to_db(db, scand, 0, track_key='track0')
        logging.debug('added ' + iso + ' (' + seq_type + ') to database')
    else:
        logging.info('adding new iso %s and copying from %s ' % (iso, exisiting_iso))
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''CREATE TEMPORARY TABLE tmp AS SELECT * FROM ScanPars WHERE iso = ?''', (exisiting_iso,))
        cur.execute('''UPDATE tmp SET iso = ?''', (iso,))
        cur.execute('''INSERT INTO ScanPars SELECT * FROM tmp''')
        cur.execute('''DROP TABLE tmp''')
        con.commit()
        con.close()
    return iso


def extract_track_dict_from_db(database_path_str, iso, sctype, tracknum):
    """ for a given database, isotope, scan type and tracknumber, this will return a complete scandictionary """
    scand = SdOp.init_empty_scan_dict(sctype)
    scand['isotopeData']['isotope'] = iso
    scand['isotopeData']['type'] = sctype
    selected_tr_name = 'track' + str(tracknum)
    scand[selected_tr_name] = scand.pop('track0')
    con = sqlite3.connect(database_path_str)
    cur = con.cursor()
    cur.execute(
        '''
        SELECT  dacStartVolt, dacStepSizeVolt, dacStopVolt, invertScan,
         nOfSteps, nOfScans, postAccOffsetVoltControl,
          postAccOffsetVolt, activePmtList, colDirTrue,
           sequencerDict, waitForKepco1us, waitAfterReset1us,
           triggerDict,
           measureVoltPars, accVolt, laserFreq, pulsePattern, triton, outbits, scanDevDict, sql
        FROM ScanPars WHERE iso = ? AND type = ? AND track = ?
        ''', (iso, sctype, tracknum,)
    )
    data = cur.fetchone()
    if data is None:
        return None
    data = list(data)
    sql = data.pop(-1)
    scand[selected_tr_name]['sql'] = ast.literal_eval(sql) if sql is not None else {}
    scan_dev_dict = data.pop(-1)
    scand[selected_tr_name]['scanDevice'] = ast.literal_eval(scan_dev_dict) if scan_dev_dict is not None else {}
    scand[selected_tr_name] = clean_pre_post_scan_set_point_in_scan_dict(scand[selected_tr_name])
    dac_start = data.pop(0)  # float
    dac_stepsize = data.pop(0)  # float
    dac_stop = data.pop(0)  # float
    if scand[selected_tr_name]['scanDevice'] == {}:
        # for backwards compatibility
        scand[selected_tr_name]['scanDevice'] = deepcopy(SdOp.DftSc.draft_scan_device)
        scand[selected_tr_name]['scanDevice']['name'] = 'unknown'
        scand[selected_tr_name]['scanDevice']['start'] = dac_start
        scand[selected_tr_name]['scanDevice']['stop'] = dac_stop
        scand[selected_tr_name]['scanDevice']['stepSize'] = dac_stepsize
    outbits = data.pop(-1)
    scand[selected_tr_name]['outbits'] = ast.literal_eval(outbits) if outbits is not None else {}
    triton = data.pop(-1)
    scand[selected_tr_name]['triton'] = ast.literal_eval(triton) if triton is not None else {}
    scand[selected_tr_name]['pulsePattern'] = ast.literal_eval(data.pop(-1))
    scand['isotopeData']['laserFreq'] = data.pop(-1)
    scand['isotopeData']['accVolt'] = data.pop(-1)
    scand[selected_tr_name]['measureVoltPars'] = SdOp.merge_dicts(
        scand[selected_tr_name]['measureVoltPars'], ast.literal_eval(data.pop(-1)))
    scand[selected_tr_name]['trigger'] = ast.literal_eval(data.pop(-1))
    if 'type' in scand[selected_tr_name]['trigger']:
        scand[selected_tr_name]['trigger']['type'] = getattr(
            TriTypes, scand[selected_tr_name]['trigger']['type'])
    else:
        for triggers, trig_dicts in scand[selected_tr_name]['trigger'].items():
            trig_dicts['type'] = getattr(TriTypes, trig_dicts['type'])
    scand[selected_tr_name] = db_track_values_to_trackdict(data, scand[selected_tr_name])
    print('from db:', scand[selected_tr_name], dac_start, dac_stepsize, dac_stop)
    con.close()

    return scand


def db_track_values_to_trackdict(data, track_dict):
    """ given a data dict containing (dacStartVolt, dacStepSizeVolt, invertScan,
         nOfSteps, nOfScans, postAccOffsetVoltControl,
          postAccOffsetVolt, activePmtList, colDirTrue,
           sequencerDict, waitForKepco1us, waitAfterReset1us,
            accVolt, laserFreq) from the database, this
            converts all values to a useable track dictionary."""
    dict_keys_list = ['invertScan',
                      'nOfSteps', 'nOfScans', 'postAccOffsetVoltControl',
                      'postAccOffsetVolt', 'activePmtList', 'colDirTrue',
                      'sequencerDict', 'waitForKepco1us', 'waitAfterReset1us']
    conversion_list = ['%s',
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
     general infos like will be merged, latter (higher tracknum) will overwrite """
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

if __name__ == '__main__':
    import os
    workdir = 'C:\\TildaDebugging\\Test_17_04_10'
    db = os.path.join(workdir, 'Test_17_04_10.sqlite')
    # print(get_software_gates_from_db(db, '60_Ni', 'wide_gate'))
    # print(get_iso_settings(db, '6230_Ni'))
    add_new_iso(db, 'new2', 'csdummy', 'new')
