"""
Created on 21.01.2015

@author: skaufmann
"""

import ast
import logging
from copy import deepcopy
from datetime import datetime as dt

import numpy as np

import Service.VoltageConversions.VoltageConversions as VCon
import TildaTools
from TildaTools import gate_one_track
from Measurement.SpecData import SpecDataXAxisUnits as Units


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
            logging.error('error while converting %s of type %s to a string, error is: %s'
                          % (val, type(val), e), exc_info=True)
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

    not supported anymore 20.06.2017
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
    uses a track dictionary to create the x axis, starting with a voltage,
    length is nOfSteps and stepsize is stepsize
    """
    try:
        arr = []
        tracks, track_num_list = TildaTools.get_number_of_tracks_in_scan_dict(scand)
        for tr in track_num_list:
            trackd = scand['track' + str(tr)]
            scan_dev_d = trackd.get('scanDevice', None)
            if scan_dev_d is None:
                dac_start_18bit = trackd['dacStartRegister18Bit']  # backwards comp.
                dac_stepsize_18bit = trackd['dacStepSize18Bit']  # backwards comp.
                n_of_steps = trackd['nOfSteps']  # backwards comp.
                dac_stop_18bit = dac_start_18bit + (dac_stepsize_18bit * n_of_steps)  # backwards comp.
                x = np.arange(dac_start_18bit, dac_stop_18bit, dac_stepsize_18bit)  # backwards comp.
                if as_voltage:
                    f = np.vectorize(VCon.get_voltage_from_bits)  # backwards comp.
                    x = f(x)
            else:
                start = scan_dev_d['start']
                stop = scan_dev_d['stop']
                step_size = scan_dev_d['stepSize']
                unit_name = scan_dev_d['stepUnitName']
                x = np.arange(start, stop, step_size)
                if not as_voltage and (unit_name == Units.line_volts.name or unit_name == Units.total_volts.name):
                    # leave it now like this for kepco scans etc.
                    f = np.vectorize(VCon.get_bits_from_voltage)
                    x = f(x)
            arr.append(x)
        return arr
    except Exception as e:
        logging.error('Exception while creating the x axis from scandict: ' + str(e))


def create_time_axis_from_scan_dict(scand, rebinning=False, binwidth_ns=10, delay_ns=0):
    """
    will create an time axis for each track in scand.
    if rebinning is True, time axis will be shortened according to the binwidth.
    The bin width can be entered explicit in function call, or will be taken from
    the 'softBinWidth_ns' entry in each track.
    Delay can be set to reasonable value, default is 0.
    """
    try:
        arr = []
        tracks, track_num_list = TildaTools.get_number_of_tracks_in_scan_dict(scand)
        for tr in track_num_list:
            trackd = scand['track' + str(tr)]
            bins = trackd['nOfBins']
            # if binwidth_ns is None:
            #     binwidth_ns = 10
            if rebinning:
                if trackd.get('softBinWidth_ns'):
                    binwidth_ns = trackd.get('softBinWidth_ns')
                    bins = bins // (binwidth_ns / 10)
            if delay_ns == 'auto':
                try:
                    delay_ns = trackd['trigger']['trigDelay10ns'] * 10
                except Exception as e:
                    logging.error(
                        'while creating a time axis, this exception occured: %s' % e)
                    delay_ns = 0
            x = np.arange(delay_ns, bins * binwidth_ns + delay_ns, binwidth_ns)
            arr.append(x)
        return arr
    except Exception as e:
        logging.error('Exception while creating the time axis: ' + str(e))


def create_default_scaler_array_from_scandict(scand, dft_val=0, data_type=np.uint32):
    """
    create empty ScalerArray, size is determined by the track0 in the scan dictionary
    """
    try:
        arr = []
        tracks, track_num_list = TildaTools.get_number_of_tracks_in_scan_dict(scand)
        for tr in track_num_list:
            trackd = scand['track' + str(tr)]
            n_of_steps = trackd['nOfSteps']
            n_of_scaler = len(trackd['activePmtList'])
            n_of_bins = trackd.get('nOfBins', False)
            if n_of_bins:
                arr.append(np.full((n_of_scaler, n_of_steps, n_of_bins), dft_val, dtype=data_type))
            else:
                arr.append(np.full((n_of_scaler, n_of_steps), dft_val, dtype=data_type))
        return arr
    except Exception as e:
        logging.error('Exception while creating default scaler array,'
                      '\n should be ignored in tilda passive.\n' + str(e))
        return None


def create_default_volt_array_from_scandict(scand, dft_val=(2 ** 30)):
    """
    create Default Voltage array, with default values in dft_val
    (2 ** 30) is chosen, because this is an default value which is not reachable by the DAC
    """
    try:
        arr = []
        tracks, track_num_list = TildaTools.get_number_of_tracks_in_scan_dict(scand)
        for tr in track_num_list:
            trackd = scand['track' + str(tr)]
            n_of_steps = trackd['nOfSteps']
            arr.append(np.full((n_of_steps,), dft_val, dtype=np.uint32))
        return arr
    except Exception as e:
        logging.error('Exception while creating default volt array,'
                      '\n should be ignored in tilda passive.\n' + str(e))
        return None


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


def gate_all_data(pipeData, data, time_array, volt_array):
    """
    this will find the indices of the gates in the time_array/voltarray
    and projectize it on both axis (sum between those indices).
    values of gates must be stored as:
        pipeData[tr_name]['softwGates'] = [[v_min_pmt0, v_max_pmt_0, t_min_pmt_0, t_max_pmt_0],
        [v_min_pmt1, v_max_pmt_1, t_min_pmt_1, t_max_pmt_1], ... ]
    if gates are not stored properly, the whole scan range will be used as gates.
    """
    tracks, tr_list = TildaTools.get_number_of_tracks_in_scan_dict(pipeData)
    ret = []
    for tr_ind, tr_num in enumerate(tr_list):
        ret = gate_one_track(tr_ind, tr_num, pipeData, data, time_array, volt_array, ret)
    return ret


def time_rebin_all_data_slow(full_data, bins_to_combine):
    """
    similiar to time_rebin_all_data, but much slower.
    Maybe delete this function later.
    """
    newdata = []
    for tr_ind, tr_data in enumerate(full_data):
        shape = tr_data.shape
        new_shape = (shape[0], shape[1], shape[2] // bins_to_combine)
        newdata.append(np.zeros(new_shape, dtype=np.uint32))
        for pmt_ind, pmt_data in enumerate(tr_data):
            for volt_ind, volt_time_arr in enumerate(pmt_data):
                for time_ind, cts_at_time in enumerate(volt_time_arr):
                    new_ind = time_ind // bins_to_combine
                    if new_ind < new_shape[-1]:
                        newdata[tr_ind][pmt_ind][volt_ind][new_ind] += cts_at_time
    return newdata


def time_rebin_all_data(full_data, scan_dict):
    """
    use this function to perform a rebinning on the time axis.
    This means, alle bins within "bins_to_combine" will be summed up.
    length of the output array for each voltage step will be:
        original length // bins_to_combine
    therefore some values in the end migth be dropped.
    e.g. 10 // 3 = 3 -> last bin is ignored.
    :param full_data: full time resolved scaler array, with all tracks
    :param bins_to_combine: int, number of 10 ns bins that will be combined
    :return: rebinned full_data
    """
    newdata = []
    tracks, tr_list = TildaTools.get_number_of_tracks_in_scan_dict(scan_dict)
    for tr_ind, tr_data in enumerate(full_data):
        newdata = rebin_single_track(tr_ind, tr_data, tr_list, newdata, scan_dict)
    return newdata


def rebin_single_track(tr_ind, tr_data, tr_list, return_data, scan_dict):
    bin_width_10ns = scan_dict['track' + str(tr_list[tr_ind])]['softBinWidth_ns']
    bins_to_combine = int(bin_width_10ns / 10)
    bin_ind = np.arange(0, tr_data.shape[-1] // bins_to_combine * bins_to_combine, bins_to_combine)
    new_tr_data = deepcopy(tr_data)
    for reps in range(bins_to_combine - 1):
        new_tr_data += np.roll(tr_data, -(reps + 1), axis=2)
    return_data.append(new_tr_data[:, :, bin_ind])
    return return_data


def time_rebin_all_spec_data(full_data, software_bin_width_ns, track=-1):
    """
    use this function to perform a rebinning on the time axis.
    This means, alle bins within "bins_to_combine" will be summed up.
    length of the output array for each voltage step will be:
        original length // bins_to_combine
    therefore some values in the end migth be dropped.
    e.g. 10 // 3 = 3 -> last bin is ignored.
    :param full_data: specdata of type, XMLImporter
    :param software_bin_width_ns, list, software bin width for all tracks
    :return: rebinned full_data
    """
    bins = deepcopy(software_bin_width_ns)
    newdata = deepcopy(full_data)
    if track == -1:
        # newdata.time_res = []
        for tr_ind, tr_data in enumerate(full_data.time_res):
            newdata.time_res[tr_ind] = rebin_single_track_spec_data(tr_data, [], bins[tr_ind])
            newdata.t[tr_ind] = time_axis_rebin(tr_ind, full_data.t, bins[tr_ind])
            pmts, steps, bin_nums = newdata.time_res[tr_ind].shape
            newdata.t_proj[tr_ind] = np.zeros((pmts, bin_nums))
            full_data.softBinWidth_ns[tr_ind] = bins[tr_ind]
            newdata.softBinWidth_ns[tr_ind] = bins[tr_ind]
    else:
        tr_ind = track
        tr_data = newdata.time_res[tr_ind]
        newdata.time_res[tr_ind] = rebin_single_track_spec_data(tr_data, [], bins[tr_ind])
        newdata.t[tr_ind] = time_axis_rebin(tr_ind, full_data.t, bins[tr_ind])
        pmts, steps, bin_nums = newdata.time_res[tr_ind].shape
        newdata.t_proj[tr_ind] = np.zeros((pmts, bin_nums))
        full_data.softBinWidth_ns[tr_ind] = bins[tr_ind]
        newdata.softBinWidth_ns[tr_ind] = bins[tr_ind]
    return newdata


def rebin_single_track_spec_data(tr_data, return_data, bin_width_10ns):
    """
    by rolling the array as often as there are bins to combine and summing every roll,
    the desired indices will hold the sum.
    """
    start_t = dt.now()
    bins_to_combine = int(bin_width_10ns / 10)
    scaler, steps, time_bins = tr_data.shape
    total_elements = scaler * steps * time_bins
    n_time_bins = time_bins // bins_to_combine
    max_bin = n_time_bins * bins_to_combine
    # time_bins // n_time_bins might have a remainder
    # -> time axis of the tr_data cannot be split into the given number (/width) of time bins without a remainder
    # cut off remaining bins.
    tr_data = tr_data[:, :, :max_bin]  # cut of last time steps, if it max_bin is not equal the end
    # expand the matrix along the time axis with the given number of bins and take the sum over this axis:
    try:
        tr_data = tr_data.reshape(scaler, steps, n_time_bins, time_bins // n_time_bins).sum(3)
    except Exception:
        pass
    # old:
    # bins_to_combine = int(bin_width_10ns / 10)
    # bin_ind = np.arange(0, tr_data.shape[-1] // bins_to_combine * bins_to_combine, bins_to_combine)
    # new_tr_data = deepcopy(tr_data)
    # for reps in range(bins_to_combine - 1):  # this means i have to hold the array ten times or so in ram :(
    #     new_tr_data += np.roll(tr_data, -(reps + 1), axis=2)
    # return_data = new_tr_data[:, :, bin_ind]

    elapsed = dt.now() - start_t
    # logging.debug('done with rebinning of track with %s elements or %.1f MB after %.1f ms:'
    #               % (total_elements, total_elements * 32 / 8 * 10 ** -6, elapsed.total_seconds() * 1000))
    return tr_data


def time_axis_rebin(tr_ind, original_t_axis_10ns_res, softw_bin_width):
    """
    this will reduce the time resolution of the original time axis of the given track index
    """
    delay_ns = original_t_axis_10ns_res[tr_ind][0]
    bins_before = original_t_axis_10ns_res[tr_ind].size
    bins = bins_before // (softw_bin_width / 10)
    in_mu_s = isinstance(delay_ns, float)
    div_by = 1000 if in_mu_s else 1
    delay_ns = delay_ns * div_by  # only then it really is in ns
    t_axis_tr = np.arange(delay_ns, bins * softw_bin_width + delay_ns, softw_bin_width) / div_by
    return t_axis_tr