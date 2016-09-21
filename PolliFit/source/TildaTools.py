"""
Created on 

@author: simkaufm

Module Description: Tools related closely to Tilda
"""
import ast
import logging
import os
from copy import deepcopy

import numpy as np
from lxml import etree as ET


def merge_dicts(d1, d2):
    """ given two dicts, merge them into a new dict as a shallow copy """
    new = d1.copy()
    new.update(d2)
    return new


def numpy_array_from_string(string, shape, datatytpe=np.uint32):
    """
    converts a text array saved in an lxml.etree.Element
    using the function xmlWriteToTrack back into a numpy array
    :param string: str, array
    :param shape: int, or tuple of int, the shape of the output array
    :return: numpy array containing the desired values
    """
    string = string.replace('\\n', '').replace('[', '').replace(']', '').replace('  ', ' ')
    if '(' in string:  # its a zero free dataformat
        string = string.replace('(', '').replace(')', '').replace(',', ' ')
        from_string = np.fromstring(string, dtype=np.uint32, sep=' ')
        result = np.zeros(from_string.size//4, dtype=[('sc', 'u2'), ('step', 'u4'), ('time', 'u4'), ('cts', 'u4')])
        result['sc'] = from_string[0::4]
        result['step'] = from_string[1::4]
        result['time'] = from_string[2::4]
        result['cts'] = from_string[3::4]
    else:
        string = string.replace('nan', '0').replace('.', '')  # if dot or nan in file
        result = np.fromstring(string, dtype=datatytpe, sep=' ')
        result = result.reshape(shape)
        # the following function would be ideal,
        # but this yields errors even for the examples coming with the function
        # result = np.genfromtxt(StringIO(string), delimiter=",", dtype=[('mystring', 'S4')])
    return result


def get_track_names(scan_dict):
    """
    create a list of all track names in the scan dictionary, sorted by its tracknumber
    :return: ['track0', 'track1', ...]
    """
    track_list = [key for key, val in scan_dict.items() if 'track' in str(key)]
    return sorted(track_list)


def eval_str_vals_in_dict(dicti):
    """
    function to convert the values of a dictionary to int, float or list, if it is possible
    """
    for key, val in dicti.items():
        try:
            dicti[key] = ast.literal_eval(val)
        except Exception as e:
            if key == 'trigger':
                val['type'] = val['type'].replace('TriggerTypes.', '')
                # val = val.replace("<TriggerTypes.", "\'")
                # val = val.replace(">", "\'")
                # dicti[key] = ast.literal_eval(val)
            else:
                print(e, val, type(val))
    return dicti


def load_xml(filename):
    """
    loads an .xml file and returns it as an lxml.etree.Element
    :return:lxml.etree.Element, Element of loaded File
    """
    parser = ET.XMLParser(huge_tree=True)
    tree = ET.parse(filename, parser)
    elem = tree.getroot()
    return elem


def save_xml(root_Ele, path, pretty=True):
    """
    Convert a Root lxml Element into an ElementTree and save it to a file
    """
    logging.debug('saving .xml file: %s' % path)
    np.set_printoptions(threshold=np.nan)
    tree = ET.ElementTree(root_Ele)
    tree.write(path, pretty_print=pretty)


def xml_get_dict_from_ele(element):
    """
    Converts an lxml Element into a python dictionary
    """
    return element.tag, dict(map(xml_get_dict_from_ele, element)) or element.text


def get_all_tracks_of_xml_in_one_dict(xml_file):
    """
    reads from an xml file and returns all tracks with its parameters
    :return: dict, {'track0': {...}, 'track1':{...}, ...}
    """
    xml_etree = load_xml(xml_file)
    trackd = {}
    tracks = xml_etree.find('tracks')
    for t in tracks:
        trackd[str(t.tag)] = (xml_get_dict_from_ele(xml_etree)[1]['tracks'][str(t.tag)]['header'])
    for key, val in trackd.items():
        trackd[key] = eval_str_vals_in_dict(trackd[key])
    return trackd


def xml_get_data_from_track(
        root_ele, n_of_track, data_type, data_shape, datatytpe=np.uint32, direct_parent_ele_str='data', default_val=0):
    """
    Get Data From Track
    :param root_ele:  lxml.etree.Element, root of the xml tree
    :param n_of_track: int, which Track should be written to
    :param data_type: str, valid: 'setOffset, 'measuredOffset', 'dwellTime10ns', 'nOfmeasuredSteps',
     'nOfclompetedLoops', 'voltArray', 'timeArray', 'scalerArray'
    :param returnType: int or tuple of int, shape of the numpy array, 0 if output in textfrom is desired
    :return: Text
    """
    if root_ele is None:  # return an
        return np.full(data_shape, default_val, dtype=datatytpe)
    else:
        try:
            actTrack = root_ele.find('tracks').find('track' + str(n_of_track)).find(direct_parent_ele_str)
            dataText = actTrack.find(str(data_type)).text
            data_numpy = numpy_array_from_string(dataText, data_shape, datatytpe)
            return data_numpy
        except Exception as e:
            print('error while searching ' + str(data_type) + ' in track' + str(n_of_track) + ' in ' + str(root_ele))
            print('error is: ', e)
            return None


def scan_dict_from_xml_file(xml_file_name, scan_dict=None):
    """
    creates a Scandictionary for the standard Tilda operations
    values are gained from the loaded xmlFile
    :return: dict, xmlEtree, Scandictionary gained from the xml file, and xmlEtree element.
    """
    if scan_dict is None:
        scan_dict = {}
    xml_etree = load_xml(xml_file_name)
    trackdict = get_all_tracks_of_xml_in_one_dict(xml_file_name)
    scan_dict = merge_dicts(scan_dict, trackdict)
    isotopedict = xml_get_dict_from_ele(xml_etree)[1]['header']
    for key, val in isotopedict.items():
        if key in ['accVolt', 'laserFreq', 'nOfTracks']:
            isotopedict[key] = ast.literal_eval(val)
    scan_dict['isotopeData'] = isotopedict
    scan_dict['pipeInternals'] = {}
    scan_dict['pipeInternals']['workingDirectory'] = os.path.split(os.path.split(xml_file_name)[0])[0]
    scan_dict['pipeInternals']['curVoltInd'] = 0
    scan_dict['pipeInternals']['activeTrackNumber'] = 'None'
    scan_dict['pipeInternals']['activeXmlFilePath'] = xml_file_name
    scan_dict['measureVoltPars'] = xml_get_dict_from_ele(xml_etree)[1]['measureVoltPars']
    return scan_dict, xml_etree


def gate_one_track(tr_ind, tr_num, scan_dict, data, time_array, volt_array, ret):
    """
    Function to gate all data of one track by applying the software gates given
     in scan_dict[tr_name]['softwGates'] as a list of all pmts

    :param tr_ind: int, indice of the track
    :param tr_num: int, number of thr track
    :param scan_dict: dict, as a full scandictionary of a scan
    :param data: np.array, time resolved data array, with all tracks and pmts
    :param time_array: list of np. arrays, time structure [ [0, 10,...,], []] for each track
    :param volt_array: list of np. arrays, voltage axis fr each track
    :param ret: list, on what the return will be appended.
    :return: list, [[v_proj_tr0,t_proj_tr0], [v_proj_tr1,t_proj_tr1],...]
    """

    tr_name = 'track%s' % tr_num
    gates_tr = []
    pmts = len(scan_dict[tr_name]['activePmtList'])
    t_proj_tr = np.zeros((pmts, len(time_array[tr_ind])), dtype=np.uint32)
    v_proj_tr = np.zeros((pmts, len(volt_array[tr_ind])), dtype=np.uint32)
    try:
        gates_val_lists = scan_dict[tr_name]['softwGates']  # list of list for each pmt.
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
        scan_dict[tr_name]['softwGates'] = [gates_val_list for pmt in scan_dict[tr_name]['activePmtList']]
        gates_pmt = [0, len(volt_array[tr_ind]) - 1, 0, len(time_array[tr_ind]) - 1]
        for pmt in scan_dict[tr_name]['activePmtList']:
            gates_tr.append(gates_pmt)
    finally:
        for pmt_ind, pmt_gate in enumerate(gates_tr):
            try:
                t_proj_xdata = np.sum(data[tr_ind][pmt_ind][pmt_gate[0]:pmt_gate[1] + 1, :], axis=0)
                v_proj_ydata = np.sum(data[tr_ind][pmt_ind][:, pmt_gate[2]:pmt_gate[3] + 1], axis=1)
                v_proj_tr[pmt_ind] = v_proj_ydata
                t_proj_tr[pmt_ind] = t_proj_xdata
            except Exception as e:
                print('bla: ', e)
        ret.append([v_proj_tr, t_proj_tr])
    return ret


def gate_specdata(spec_data):
    """
    function to gate spec_data with the softw_gates list in the spec_data itself.
    gate will be applied on spec_data.time_res and
     the time projection will be written to spec_data.t_proj
     the voltage projection will be written to spec_data.cts
    :param spec_data: spec_data
    :return: spec_data
    """
    # get indices of the values first
    compare_arr = [spec_data.x, spec_data.x, spec_data.t, spec_data.t]
    softw_gates_ind = [
        [[find_closest_value_in_arr(compare_arr[lim_ind][tr_ind], lim)[0] for lim_ind, lim in enumerate(gates_pmt)]
         for gates_pmt in gates_tr]
        for tr_ind, gates_tr in enumerate(spec_data.softw_gates)]
    spec_data.softw_gates = [[[compare_arr[lim_ind][tr_ind][found_ind] for lim_ind, found_ind in enumerate(gate_ind_pmt)]
                              for gate_ind_pmt in gate_ind_tr]
                             for tr_ind, gate_ind_tr in enumerate(softw_gates_ind)]

    for tr_ind, tr in enumerate(spec_data.cts):
        for pmt_ind, pmt in enumerate(tr):
            v_min_ind = softw_gates_ind[tr_ind][pmt_ind][0]
            v_max_ind = softw_gates_ind[tr_ind][pmt_ind][1] + 1
            t_min_ind = softw_gates_ind[tr_ind][pmt_ind][2]
            t_max_ind = softw_gates_ind[tr_ind][pmt_ind][3] + 1
            t_proj_res = np.sum(spec_data.time_res[tr_ind][pmt_ind][v_min_ind:v_max_ind, :], axis=0)
            v_proj_res = np.sum(spec_data.time_res[tr_ind][pmt_ind][:, t_min_ind:t_max_ind], axis=1)
            spec_data.t_proj[tr_ind][pmt_ind] = t_proj_res
            spec_data.cts[tr_ind][pmt_ind] = v_proj_res
    return spec_data


def zero_free_to_non_zero_free(zf_time_res_arr, dims_sc_step_time_list):
    """
    this will convert the zerofree time_res array ([(sc, step, time, cts), ...]) to
    the classic matrix approach where sc, step, time are located by index and
    cts are filled at the corresponding positions
    :param zf_time_res_arr: list, of numpy arrays for eaach track with the zero free data structure.
    :param dims_sc_step_time_list: list, of dimensions for each track
                [(n_of_scalers_tr, n_of_steps_tr, n_of_bins_tr), ...]

    """
    new_time_res = [np.zeros(dims_sc_step_time_list[tr_ind]) for tr_ind, tr in enumerate(dims_sc_step_time_list)]

    for tr_ind, tr_dims in enumerate(dims_sc_step_time_list):
        sc_arr = zf_time_res_arr[tr_ind]['sc']
        step_arr = zf_time_res_arr[tr_ind]['step']
        time_arr = zf_time_res_arr[tr_ind]['time']
        new_time_res[tr_ind][sc_arr, step_arr, time_arr] = zf_time_res_arr[tr_ind]['cts']
    return new_time_res


def gate_zero_free_specdata(spec_data):
    """
    function to gate spec_data with the softw_gates list in the spec_data itself.
    gate will be applied on spec_data.time_res and
    the time projection will be written to spec_data.t_proj
    the voltage projection will be written to spec_data.cts
    :param spec_data: spec_data (zero free time res)
    :return: spec_data (zero free time res)
    """
    try:
        dimensions = spec_data.get_scaler_step_and_bin_num(-1)
        zf_spec = deepcopy(spec_data)
        zf_spec.time_res = zero_free_to_non_zero_free(spec_data.time_res_zf, dimensions)
        zf_spec = gate_specdata(zf_spec)
    except Exception as e:
        print('error: while gating zero free specdata: %s ' % e)
    return zf_spec

    # alternative solution (currently slower):
    # compare_arr = [spec_data.x, spec_data.x, spec_data.t, spec_data.t]
    # softw_gates_ind = [
    #     [[find_closest_value_in_arr(compare_arr[lim_ind][tr_ind], lim)[0] for lim_ind, lim in enumerate(gates_pmt)]
    #      for gates_pmt in gates_tr]
    #     for tr_ind, gates_tr in enumerate(spec_data.softw_gates)]
    # spec_data.softw_gates = [
    #     [[compare_arr[lim_ind][tr_ind][found_ind] for lim_ind, found_ind in enumerate(gate_ind_pmt)]
    #      for gate_ind_pmt in gate_ind_tr]
    #     for tr_ind, gate_ind_tr in enumerate(softw_gates_ind)]
    #
    # for tr_ind, tr in enumerate(spec_data.cts):
    #     for pmt_ind, pmt in enumerate(tr):
    #         try:
    #             #### some pmts might not have been fired...
    #             v_min_ind = softw_gates_ind[tr_ind][pmt_ind][0]
    #             v_max_ind = softw_gates_ind[tr_ind][pmt_ind][1]
    #             t_min_ind = softw_gates_ind[tr_ind][pmt_ind][2]
    #             t_max_ind = softw_gates_ind[tr_ind][pmt_ind][3]
    #             pmt_arr = spec_data.time_res[tr_ind][spec_data.time_res[tr_ind]['sc'] == pmt_ind]
    #             pmt_arr_t_gated = pmt_arr[pmt_arr['time'] <= t_max_ind]
    #             pmt_arr_t_gated = pmt_arr_t_gated[t_min_ind <= pmt_arr_t_gated['time']]
    #             v_proj_res = [np.sum(pmt_arr_t_gated[pmt_arr_t_gated['step'] == i]['cts'], axis=0) for i in range(pmt.size)]
    #             spec_data.cts[tr_ind][pmt_ind] = v_proj_res
    #
    #             pmt_arr_v_gated = pmt_arr[pmt_arr['step'] <= v_max_ind]
    #             pmt_arr_v_gated = pmt_arr_v_gated[v_min_ind <= pmt_arr_v_gated['step']]
    #             t_proj_res = [np.sum(pmt_arr_v_gated[pmt_arr_v_gated['time'] == t]['cts'], axis=0) for t in
    #                           spec_data.t[tr_ind]]
    #             spec_data.t_proj[tr_ind][pmt_ind] = t_proj_res
    #
    #         except Exception as e:
    #             print(e)
    # return spec_data


def create_x_axis_from_file_dict(scan_dict, as_voltage=True):
    """
    creates an x axis in units of line volts or in dac registers
    """
    x_arr = []
    for tr_ind, tr_name in enumerate(get_track_names(scan_dict)):
        steps = scan_dict[tr_name]['nOfSteps']
        if as_voltage:
            start = scan_dict[tr_name]['dacStartVoltage']
            stop = scan_dict[tr_name]['dacStopVoltage']
            step = scan_dict[tr_name]['dacStepsizeVoltage']
        else:
            start = scan_dict[tr_name]['dacStartRegister18Bit']
            step = scan_dict[tr_name]['dacStepSize18Bit']
            stop = scan_dict[tr_name].get('dacStopRegister18Bit', start + step * (steps - 1))
        x_tr, new_step = np.linspace(start, stop, steps, retstep=True)
        # np.testing.assert_allclose(
        #     new_step, step, rtol=1e-5, err_msg='error while creating x axis from file, stepsizes do not match.')
        # logging.debug('for the new x axis the new stepsize is: %s, the old one was: %s and the difference is: %s'
        #               % (new_step, step, new_step - step))
        x_arr.append(x_tr)
    return x_arr


def create_t_axis_from_file_dict(scan_dict, with_delay=False, bin_width=10, in_mu_s=True):
    """
    will create a time axis for all tracks, resolution is 10ns.
    """
    t_arr = []
    for tr_ind, tr_name in enumerate(get_track_names(scan_dict)):
        if with_delay:
            delay = scan_dict[tr_name]['trigger'].get('trigDelay10ns', 0)
        else:
            delay = 0
        nofbins = scan_dict[tr_name]['nOfBins']
        div_by = 1000 if in_mu_s else 1
        t_tr = np.arange(delay, nofbins * bin_width + delay, bin_width) / div_by
        t_arr.append(t_tr)
    return t_arr


def find_closest_value_in_arr(arr, search_val):
    """
    goes through an array and finds the nearest value to search_val
    :return: ind, found_val, abs(found_val - search_val)
    """
    ind, found_val = min(enumerate(arr), key=lambda i: abs(float(i[1]) - search_val))
    return ind, found_val, abs(found_val - search_val)