"""
Created on 21.01.2015

@author: skaufmann
"""

import ast
from datetime import datetime as dt

import numpy as np

import Service.Scan.ScanDictionaryOperations as SdOp


def split_32b_data(int32b_data):
    """
    seperate header, header_index and payload from each other
    :param int32b_data:
    :return: tuple, (first_header, second_header, header_index, payload)
    """
    header_length = 8
    first_header = int32b_data >> (32 - int(header_length/2))
    second_header = int32b_data >> (32 - header_length) & ((2 ** 4) - 1)
    header_index = (int32b_data >> (32 - header_length - 1)) & 1
    payload = int32b_data & ((2 ** 23) - 1)
    return first_header, second_header, header_index, payload


def trs_sum(element, act_volt_ind, sum_array, active_pmt_list=range(8)):
    """
    Add new Scaler event on previous acquired ones. Treat each scaler seperatly.
    :return: np.array, sum
    """
    timestamp = element['payload']
    pmts_with_event = (element['firstHeader'] << 4) + element['secondHeader']  # glue header back together
    for ind, val in enumerate(active_pmt_list):
        if pmts_with_event & (2 ** val):
            sum_array[act_volt_ind, timestamp, ind] += 1  # timestamp equals index in timeArray
    return sum_array


def eval_str_vals_in_dict(dicti):
    """
    function to convert the values of a dictionary to int, float or list, if it is possible
    """
    for key, val in dicti.items():
        try:
            dicti[key] = ast.literal_eval(val)
        except Exception as e:
            print(e, val, type(val))
    return dicti


def add_working_time_to_track_dict(trackdict):
    """adds the timestamp to the working time of the track"""
    time = str(dt.now().strftime("%Y-%m-%d %H:%M:%S"))
    if 'workingTime' in trackdict:
        if trackdict['workingTime'] is None:
            worktime = []
        else:
            worktime = trackdict['workingTime']
    else:
        worktime = []
    worktime.append(time)
    trackdict.update(workingTime=worktime)
    return trackdict


def convert_scandict_v104_to_v106(scandict):
    """
    converts a scandictionary created in Version 1.04 to the new format as it should be in v1.06
    was needed for working with the collected .raw data from 29.07.2015.
    """
    # trackdft = draft_scan_dict['activeTrackPar']
    track = scandict['activeTrackPar']
    trackrenamelist = [('start', 'dacStartRegister18Bit'),
                       ('stepSize', 'dacStepSize18Bit'),
                       ('heinzingerOffsetVolt', 'postAccOffsetVolt'),
                       ('heinzingerControl', 'postAccOffsetVoltControl'),
                       ('dwellTime', 'dwellTime10ns')]
    track['workingTime'] = ['unknown']
    track['colDirTrue'] = scandict['isotopeData']['colDirTrue']
    scandict['isotopeData']['isotopeStartTime'] = scandict['isotopeData']['datetime']
    scandict['measureVoltPars'] = \
        {k: v for (k, v) in track.items() if k in ['measVoltTimeout10ns', 'measVoltPulseLength25ns']}

    scandict['isotopeData'].pop('colDirTrue')
    scandict['isotopeData'].pop('datetime')
    [track.pop(k) for k in ['measVoltTimeout10ns', 'measVoltPulseLength25ns', 'VoltOrScaler', 'measureOffset']]
    for oldkey, newkey in trackrenamelist:
        track[newkey] = track.pop(oldkey)
    scandict['isotopeData']['version'] = 1.06
    return scandict


def create_x_axis_from_scand_dict(scand):
    """
    uses a track dictionary to create the x axis, starting with dacStartRegister18Bit,
    length is nOfSteps and stepsize is dacStepSize18Bit
    """
    arr = []
    tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(scand)
    for tr in track_num_list:
        trackd = scand['track' + str(tr)]
        dac_start_18bit = trackd['dacStartRegister18Bit']
        dac_stepsize_18bit = trackd['dacStepSize18Bit']
        n_of_steps = trackd['nOfSteps']
        dac_stop_18bit = dac_start_18bit + (dac_stepsize_18bit * n_of_steps)
        x = np.arange(dac_start_18bit, dac_stop_18bit, dac_stepsize_18bit)
        arr.append(x)
    return arr


def create_default_scaler_array_from_scandict(scand, dft_val=0):
    """
    create empty ScalerArray, size is determined by the activeTrackPar in the scan dictionary
    """
    arr = []
    tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(scand)
    for tr in track_num_list:
        trackd = scand['track' + str(tr)]
        n_of_steps = trackd['nOfSteps']
        n_of_scaler = len(trackd['activePmtList'])
        arr.append(np.full((n_of_scaler, n_of_steps), dft_val, dtype=np.uint32))
    return arr


def create_default_volt_array_from_scandict(scand, dft_val=(2 ** 30)):
    """
    create Default Voltage array, with default values in dft_val
    (2 ** 30) is chosen, because this is an default value which is not reachable by the DAC
    """
    arr = []
    tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(scand)
    for tr in track_num_list:
        trackd = scand['track' + str(tr)]
        n_of_steps = trackd['nOfSteps']
        arr.append(np.full((n_of_steps,), dft_val, dtype=np.uint32))
    return arr


def add_header_to23_bit(bit23, firstheader, secondheader, indexheader):
    """
    enter a 32 bit header and the other header without their shift.
    So for firstheader = 3 (0011) only enter 3.
    """
    sh_firstheader = firstheader << 28
    sh_secondheader = secondheader << 24
    sh_indexheader = indexheader << 23
    result = sh_firstheader + sh_secondheader + sh_indexheader + bit23
    return result
