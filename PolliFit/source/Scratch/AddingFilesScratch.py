"""
Created on 

@author: simkaufm

Module Description:
"""

import os
import sqlite3
from copy import deepcopy

import numpy as np

import Measurement.MeasLoad as Ml
import Service.FileOperations.FolderAndFileHandling as FileHandl
from Measurement.MCPImporter import MCPImporter
from Measurement.XMLImporter import XMLImporter

mcp_file = os.path.normpath('E:\\Workspace\\AddedTestFiles\\60Ni_no_protonTrigger_Run019.mcp')

desired_xml_saving_path = 'E:\\Workspace\\AddedTestFiles\\test2.xml'
example_xml_file = 'E:/Workspace/AddedTestFiles/neu_trsdummy_141.xml'
# Tools.createDB('E:\\Workspace\\AddedTestFiles\\database.sqlite')
db = 'E:\\Workspace\\AddedTestFiles\\database.sqlite'
# Tools.crawl(db)
files = ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run035.mcp', '60Ni_no_protonTrigger_Run036.mcp',
         '60Ni_no_protonTrigger_Run037.mcp', '60Ni_no_protonTrigger_Run044.mcp', '60Ni_no_protonTrigger_Run045.mcp',
         '60Ni_no_protonTrigger_Run053.mcp', '60Ni_no_protonTrigger_Run054.mcp', '60Ni_no_protonTrigger_Run055.mcp',
         '60Ni_no_protonTrigger_Run058.mcp', '60Ni_no_protonTrigger_Run059.mcp', '60Ni_no_protonTrigger_Run060.mcp',
         '60Ni_no_protonTrigger_Run067.mcp', '60Ni_no_protonTrigger_Run068.mcp', '60Ni_no_protonTrigger_Run72.mcp',
         '60Ni_no_protonTrigger_Run076.mcp', '60Ni_no_protonTrigger_Run077.mcp', '60Ni_no_protonTrigger_Run078.mcp',
         '60Ni_no_protonTrigger_Run082.mcp', 'neu_trsdummy_141.xml']
print(mcp_file)
# meas = MCPImporter(mcp_file)
meas = XMLImporter(example_xml_file)

to_add = [(1, Ml.load(os.path.join(os.path.dirname(desired_xml_saving_path), file), db, raw=True)) for file in
          files[1:]]

dacs = []
for tr_ind in range(meas.nrTracks):
    dac_start = meas.x[tr_ind][0]
    dac_stopp = meas.x[tr_ind][0]
    dac_stepsize = meas.x[tr_ind][1] - meas.x[tr_ind][1]
    dacs.append((dac_start, dac_stopp, dac_stepsize))
# just extract needed values, if not available anyhow.
# * needed & available in file, ! needed, but not be available in file, - only needed in db not relevant for adding
# accVolt*, laserFreq(*)!, colDirTrue*, line-, type*, voltDivRatio-, lineMult-, lineOffset-, offset*
meas2 = MCPImporter(mcp_file)
meas_xml = XMLImporter(example_xml_file)


def get_laserfreq_from_db(db, measurement):
    """
    this will connect to the database and read the laserfrequency for the file in measurement.file
    :param db: str, path to .slite db
    :param measurement: specdata, as of XMLImporter or MCPImporter etc.
    :return: float, laserfrequency
    """
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''SELECT laserFreq FROM Files WHERE file = ?''', (measurement.file,))
    data = cur.fetchall()
    con.close()
    if data:
        laserfreq = data[0][0]
        return laserfreq
    else:
        return 0.0


cts_mcp = meas.cts
cts_tilda = meas_xml.cts


# measures = [(-1, meas2), (-1, meas2)]


def add_specdata(parent_specdata, add_spec_list):
    """
    this will add or subtract the counts in the specdata object to the given parent specdata.
    Will only do that, if the x-axis are equal within 10 ** -5
    :param parent_specdata: specdata, original file on which data will be added.
    :param add_spec_list: list,
    of tuples [(int as multiplikation factor (e.g. +/- 1), specdata which will be added), ..]
    :return: specdata, added file.
    """
    added_files = [parent_specdata.file]
    offsets = [parent_specdata.offset]
    accvolts = [parent_specdata.accVolt]
    for add_meas in add_spec_list:
        for tr_ind, tr in enumerate(parent_specdata.cts):
            try:
                # check if the x-axis of the two specdata are equal:
                if np.allclose(parent_specdata.x[tr_ind], add_meas[1].x[tr_ind], rtol=1 ** -5):
                    if tr_ind == 0:
                        added_files.append((add_meas[0], add_meas[1].file))
                        offsets.append(add_meas[1].offset)
                        accvolts.append(add_meas[1].accVolt)
                    time_res_zf = check_if_attr_exists(add_meas[1], 'time_res_zf', [[]] * add_meas[1].nrTracks)[tr_ind]
                    if len(time_res_zf):  # add the time spectrum (zero free) if it exists
                        appended_arr = np.append(parent_specdata.time_res_zf[tr_ind], time_res_zf)
                        # sort by 'sc', 'step', 'time' (no cts):
                        sorted_arr = np.sort(appended_arr, order=['sc', 'step', 'time'])
                        # find all elements that occur twice:
                        unique_arr, unique_inds, uniq_cts = np.unique(sorted_arr[['sc', 'step', 'time']],
                                                                      return_index=True, return_counts=True)
                        sum_ind = unique_inds[np.where(uniq_cts == 2)]  # only take indexes of double occuring items
                        # use indices of all twice occuring elements to add the counts of those:
                        sum_cts = sorted_arr[sum_ind]['cts'] + sorted_arr[sum_ind + 1]['cts']
                        np.put(sorted_arr['cts'], sum_ind, sum_cts)
                        # delete all remaining items:
                        parent_specdata.time_res_zf[tr_ind] = np.delete(sorted_arr, sum_ind + 1, axis=0)
                    for sc_ind, sc in enumerate(tr):
                        parent_specdata.cts[tr_ind][sc_ind] += add_meas[0] * add_meas[1].cts[tr_ind][sc_ind]

                else:
                    print('warning, file: %s does not have the'
                          ' same x-axis as the parent file: %s,'
                          ' will not add!' % (parent_specdata.file, add_meas[1].file))
            except Exception as e:
                print('warning, file: %s does not have the'
                      ' same x-axis as the parent file: %s,'
                      ' will not add!' % (parent_specdata.file, add_meas[1].file))
        # needs to be converted like this:
        parent_specdata.cts[tr_ind] = np.array(parent_specdata.cts[tr_ind])
        # I Don't know why this is not in this format anyhow.
    parent_specdata.offset = np.mean(offsets)
    parent_specdata.accVolt = np.mean(accvolts)
    return parent_specdata, added_files


def check_if_attr_exists(parent_to_check_from, par, return_val_if_not):
    """ use par as attribute of parent_to_check_from and return the result.
     If this is not a valid arg, return return_val_if_not """
    ret = return_val_if_not
    try:
        ret = getattr(parent_to_check_from, par)
    except Exception as e:
        ret = return_val_if_not
    finally:
        return ret


def create_scan_dict_from_spec_data(specdata, desired_xml_saving_path, database_path=None):
    """
    create a scan_dict according to the values in the specdata.
    Mostly foreseen for adding two or more files and therefore a scan dict is required.

    :param specdata: specdata, as of XMLImprter or MCPImporter etc.
    :param database_path: str, or None to find the laserfreq to the given specdata
    :return: dict, scandict
    """
    if database_path is None:  # prefer laserfreq from db, if existant
        laserfreq = specdata.laserFreq
    else:
        laserfreq = get_laserfreq_from_db(database_path, specdata)

    draftIsotopePars = {
        'version': check_if_attr_exists(specdata, 'version', 'unknown'),
        'type': check_if_attr_exists(specdata, 'seq_type', 'cs') if specdata.type != 'Kepco' else 'Kepco',
        'isotope': specdata.type,
        'nOfTracks': specdata.nrTracks,
        'accVolt': specdata.accVolt,
        'laserFreq': laserfreq,
        'isotopeStartTime': specdata.date
    }
    tracks = {}
    for tr_ind in range(specdata.nrTracks):
        tracks['track%s' % tr_ind] = {
            'dacStepSize18Bit': check_if_attr_exists(
                specdata, 'x_dac', specdata.x)[tr_ind][1] - check_if_attr_exists(
                specdata, 'x_dac', specdata.x)[tr_ind][0],
            'dacStartRegister18Bit': check_if_attr_exists(specdata, 'x_dac', specdata.x)[tr_ind][-1],
            'dacStartVoltage': specdata.x[tr_ind][0],
            'dacStopVoltage': specdata.x[tr_ind][-1],
            'dacStepsizeVoltage': specdata.x[tr_ind][1] - specdata.x[tr_ind][0],
            'nOfSteps': specdata.getNrSteps(tr_ind),
            'nOfScans': specdata.nrScans[tr_ind],
            'nOfCompletedSteps': specdata.nrScans[tr_ind] * specdata.getNrSteps(tr_ind),
            'invertScan': check_if_attr_exists(specdata, 'invert_scan', [False] * specdata.nrTracks)[tr_ind],
            'postAccOffsetVoltControl': specdata.post_acc_offset_volt_control, 'postAccOffsetVolt': specdata.offset,
            'waitForKepco25nsTicks': check_if_attr_exists(specdata, 'wait_after_reset_25ns', [-1] * specdata.nrTracks)[
                tr_ind],
            'waitAfterReset25nsTicks': check_if_attr_exists(specdata, 'wait_for_kepco_25ns', [-1] * specdata.nrTracks)[
                tr_ind],
            'activePmtList': check_if_attr_exists(specdata, 'activePMTlist', False)[tr_ind] if
            check_if_attr_exists(specdata, 'activePMTlist', []) else
            check_if_attr_exists(specdata, 'active_pmt_list', [])[tr_ind],
            'colDirTrue': specdata.col,
            'dwellTime10ns': specdata.dwell,
            'workingTime': check_if_attr_exists(specdata, 'working_time', [None] * specdata.nrTracks)[tr_ind],
            'nOfBins': len(check_if_attr_exists(specdata, 't', [[0]] * specdata.nrTracks)[tr_ind]),
            'softBinWidth_ns': check_if_attr_exists(specdata, 'softBinWidth_ns', [0] * specdata.nrTracks)[tr_ind],
            'nOfBunches': 1,
            'softwGates': check_if_attr_exists(
                specdata, 'softw_gates', [[] * specdata.nrScalers[tr_ind]] * specdata.nrTracks)[tr_ind],
            'trigger': {'type': 'no_trigger'}
        }
    draftMeasureVoltPars_singl = {'measVoltPulseLength25ns': -1, 'measVoltTimeout10ns': -1,
                                  'dmms': {}, 'switchBoxSettleTimeS': -1}
    pre_scan_dmms = {'unknown_dmm': {'assignment': 'offset', 'preScanRead': deepcopy(specdata.offset)},
                     'unknown_dmm_1': {'assignment': 'accVolt', 'preScanRead': deepcopy(specdata.accVolt)},
                     }
    draftMeasureVoltPars = {'preScan': deepcopy(draftMeasureVoltPars_singl),
                            'duringScan': deepcopy(draftMeasureVoltPars_singl)}
    draftMeasureVoltPars['preScan']['dmms'] = pre_scan_dmms
    draftPipeInternals = {
        'curVoltInd': 0,
        'activeTrackNumber': (0, 'track0'),
        'workingDirectory': os.path.dirname(desired_xml_saving_path),
        'activeXmlFilePath': desired_xml_saving_path
    }
    draftScanDict = {'isotopeData': draftIsotopePars,
                     'pipeInternals': draftPipeInternals,
                     'measureVoltPars': draftMeasureVoltPars
                     }
    draftScanDict.update(tracks)  # add the tracks
    return draftScanDict


# add files
par_spec, added_files = add_specdata(meas, to_add)

# create empty xml file
scan_dict = create_scan_dict_from_spec_data(meas, desired_xml_saving_path, db)
scan_dict['isotopeData']['addedFiles'] = added_files
FileHandl.createXmlFileOneIsotope(scan_dict, filename=desired_xml_saving_path)
# call savespecdata in filehandl (will expect to have a .time_res)
FileHandl.save_spec_data(meas, scan_dict)
# MPLPlotter.plot(meas.getArithSpec([4], -1))
# # MPLPlotter.show(True)
meas_import = XMLImporter(desired_xml_saving_path)
# print('after import: offset: %s, accVolt: %s' % (meas_import.offset, meas_import.accVolt))
#
# # print(create_scan_dict_from_spec_data(meas))
# # print(create_scan_dict_from_spec_data(meas_xml))
# MPLPlotter.plot(meas_import.getArithSpec([0], -1))
# MPLPlotter.show(True)

from Interface.LiveDataPlottingUi.LiveDataPlottingUi import TRSLivePlotWindowUi
from PyQt5 import QtWidgets
import sys

app = QtWidgets.QApplication(sys.argv)
mwin = TRSLivePlotWindowUi(desired_xml_saving_path, subscribe_as_live_plot=False)
mwin.new_data(meas_import)
app.exec()
