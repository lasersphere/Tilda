"""
Created on 21.01.2015

@author: skaufmann
"""

import ast
from datetime import datetime as dt

import numpy as np

import Service.Scan.ScanDictionaryOperations as SdOp
import Service.VoltageConversions.VoltageConversions as VCon


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


def add_working_time_to_track_dict(trackdict, reset=False):
    """
    adds the timestamp to the working time of the track.
    :param reset: bool, True if 'workingTime' should be cleared
    :return: trackdict
    """
    time = str(dt.now().strftime("%Y-%m-%d %H:%M:%S"))
    if 'workingTime' in trackdict and not reset:
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
    # trackdft = draft_scan_dict['track0']
    track = scandict['track0']
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


def create_x_axis_from_scand_dict(scand, as_voltage=False):
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
        if as_voltage:
            f = np.vectorize(VCon.get_voltage_from_18bit)
            x = f(x)
        arr.append(x)
    return arr


def create_time_axis_from_scan_dict(scand, binwidth_ns=10, delay_ns=0):
    """
    will create an time axis for each track in scand.
    Delay can be set to reasonable value, default is 0.
    """
    arr = []
    tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(scand)
    for tr in track_num_list:
        trackd = scand['track' + str(tr)]
        bins = trackd['nOfBins']
        if delay_ns == 'auto':
            try:
                delay_ns = trackd['trigger']['trigDelay10ns'] * 10
            except Exception as e:
                print('while creating a time axis, thie exception occured: ', e)
                delay_ns = 0
        x = np.arange(delay_ns, bins * binwidth_ns + delay_ns, binwidth_ns)
        arr.append(x)
    return arr


def create_default_scaler_array_from_scandict(scand, dft_val=0):
    """
    create empty ScalerArray, size is determined by the track0 in the scan dictionary
    """
    arr = []
    tracks, track_num_list = SdOp.get_number_of_tracks_in_scan_dict(scand)
    for tr in track_num_list:
        trackd = scand['track' + str(tr)]
        n_of_steps = trackd['nOfSteps']
        n_of_scaler = len(trackd['activePmtList'])
        n_of_bins = trackd.get('nOfBins', False)
        if n_of_bins:
            arr.append(np.full((n_of_scaler, n_of_steps, n_of_bins), dft_val, dtype=np.uint32))
        else:
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


def find_closest_value_in_arr(arr, search_val):
    """
    goes through an array and finds the nearest value to search_val
    :return: ind, found_val, abs(found_val - search_val)
    """
    ind, found_val = min(enumerate(arr), key=lambda i: abs(float(i[1]) - search_val))
    return ind, found_val, abs(found_val - search_val)


def gate_all_data(pipeData, data, time_array, volt_array):
    """
    this will find the indices of the gates in the time_array/voltarray
    and projectize it on both axis (sum between those indices).
    values of gates must be stored as:
        pipeData[tr_name]['softwGates'] = [[v_min_pmt0, v_max_pmt_0, t_min_pmt_0, t_max_pmt_0],
        [v_min_pmt1, v_max_pmt_1, t_min_pmt_1, t_max_pmt_1], ... ]
    if gates are not stored properly, the whole scan range will be used as gates.
    """
    tracks, tr_list = SdOp.get_number_of_tracks_in_scan_dict(pipeData)
    ret = []
    for tr_ind, tr_num in enumerate(tr_list):
        tr_name = 'track%s' % tr_num
        gates_tr = []
        pmts = len(pipeData[tr_name]['activePmtList'])
        t_proj_tr = np.zeros((pmts, len(time_array[tr_ind])), dtype=np.uint32)
        v_proj_tr = np.zeros((pmts, len(volt_array[tr_ind])), dtype=np.uint32)
        try:
            gates_val_lists = pipeData[tr_name]['softwGates']  # list of list for each pmt.
            for gates_val_list in gates_val_lists:
                v_min, v_max = sorted((gates_val_list[0], gates_val_list[1]))
                v_min_ind, v_min, vdif = find_closest_value_in_arr(volt_array[tr_ind], v_min)
                v_max_ind, v_max, vdif = find_closest_value_in_arr(volt_array[tr_ind], v_max)

                t_min, t_max = sorted((gates_val_list[2], gates_val_list[3]))
                t_min_ind, t_min, tdif = find_closest_value_in_arr(time_array[tr_ind], t_min)
                t_max_ind, t_max, tdif = find_closest_value_in_arr(time_array[tr_ind], t_max)
                gates_tr.append([v_min_ind, v_max_ind, t_min_ind, t_max_ind])  # indices in data array
        except Exception as e:  # if gates_tr are messud up, use full scan range as gates_tr:
            v_min = round(volt_array[tr_ind][0], 5)
            v_max = round(volt_array[tr_ind][-1], 5)
            t_min = time_array[tr_ind][0]
            t_max = time_array[tr_ind][-1]
            gates_val_list = [v_min, v_max, t_min, t_max]
            pipeData[tr_name]['softwGates'] = [gates_val_list for pmt in pipeData[tr_name]['activePmtList']]
            gates_pmt = [0, len(volt_array[tr_ind]) - 1, 0, len(time_array[tr_ind]) - 1]
            for pmt in pipeData[tr_name]['activePmtList']:
                gates_tr.append(gates_pmt)
        finally:
            for pmt_ind, pmt_gate in enumerate(gates_tr):
                t_proj_xdata = np.sum(data[tr_ind][pmt_ind][pmt_gate[0]:pmt_gate[1] + 1, :], axis=0)
                v_proj_ydata = np.sum(data[tr_ind][pmt_ind][:, pmt_gate[2]:pmt_gate[3] + 1], axis=1)
                v_proj_tr[pmt_ind] = v_proj_ydata
                t_proj_tr[pmt_ind] = t_proj_xdata
            ret.append([v_proj_tr, t_proj_tr])
    return ret