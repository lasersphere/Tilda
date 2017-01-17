"""
Created on 

@author: simkaufm

Module Description: Tools related closely to Tilda
"""
import ast
import logging
import os
import sqlite3
from copy import deepcopy

import numpy as np
from lxml import etree as ET

import Physics
from XmlOperations import xmlCreateIsotope, xml_add_meas_volt_pars, xmlAddCompleteTrack


def merge_dicts(d1, d2):
    """ given two dicts, merge them into a new dict as a shallow copy """
    new = d1.copy()
    new.update(d2)
    return new


def numpy_array_from_string(string, shape, datatytpe=np.int32):
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
        string = string.replace('nan', '0')  # if nan in file
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
                print('error while converting val with ast.literal_eval: ',  e, val, type(val), key)
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
    print('xml file is file: %s' % os.path.isfile(xml_file))
    xml_etree = load_xml(xml_file)
    trackd = {}
    tracks = xml_etree.find('tracks')
    for t in tracks:
        trackd[str(t.tag)] = (xml_get_dict_from_ele(xml_etree)[1]['tracks'][str(t.tag)]['header'])
    for key, val in trackd.items():
        trackd[key] = evaluate_strings_in_dict(trackd[key])
    return trackd


def get_meas_volt_dict(xml_etree):
    """ get the dictionary containing all the voltage measurement parameters
     from an xml_etree element loaded from an xml file. """
    meas_volt_pars_dict = xml_get_dict_from_ele(xml_etree)[1].get('measureVoltPars', {})
    evaluate_strings_in_dict(meas_volt_pars_dict)
    for key, val in meas_volt_pars_dict.items():
        print(val)
        if val['dmms'] is None:
            meas_volt_pars_dict[key]['dmms'] = {}
    return meas_volt_pars_dict


def evaluate_strings_in_dict(dict_to_convert):
    """
    function which will convert all values inside a dict using ast.literal_eval -> '1.0' -> 1.0 etc..
    works also for nested dicts
    """
    for key, val in dict_to_convert.items():
        if isinstance(val, str):
            try:
                dict_to_convert[key] = ast.literal_eval(val)
            except Exception as e:
                if '{' in val:
                    # needed for data of version 1.08
                    val = val.replace('\\', '\'').replace('TriggerTypes.', '').replace('<', '\'').replace('>', '\'')
                    val = ast.literal_eval(val)
                if key == 'trigger':
                    val['type'] = val['type'].replace('TriggerTypes.', '')
                else:
                    # print('error while converting val with ast.literal_eval: ', e, val, type(val), key)
                    # if it cannot be converted it is propably a string anyhow.
                    # print('error: %s could not convert key: %s val: %s' % (e, key, val))
                    pass

        if isinstance(dict_to_convert[key], dict):
            dict_to_convert[key] = evaluate_strings_in_dict(dict_to_convert[key])
    return dict_to_convert


def xml_get_data_from_track(
        root_ele, n_of_track, data_type, data_shape, datatytpe=np.int32, direct_parent_ele_str='data', default_val=0):
    """
    Get Data From Track
    :param root_ele:  lxml.etree.Element, root of the xml tree
    :param n_of_track: int, which Track should be written to
    :param data_type: str, valid: 'setOffset, 'measuredOffset', 'dwellTime10ns', 'nOfmeasuredSteps',
     'nOfclompetedLoops', 'voltArray', 'timeArray', 'scalerArray'
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
    scan_dict['measureVoltPars'] = get_meas_volt_dict(xml_etree)
    # watchout, since trigger type is only imported as string...
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
    t_proj_tr = np.zeros((pmts, len(time_array[tr_ind])), dtype=np.int32)
    v_proj_tr = np.zeros((pmts, len(volt_array[tr_ind])), dtype=np.int32)
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


def create_t_axis_from_file_dict(scan_dict, with_delay=True, bin_width=10, in_mu_s=True):
    """
    will create a time axis for all tracks, resolution is 10ns.
    """
    t_arr = []
    for tr_ind, tr_name in enumerate(get_track_names(scan_dict)):
        if with_delay:
            delay = scan_dict[tr_name]['trigger'].get('trigDelay10ns', 0) * bin_width
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


def add_specdata(parent_specdata, add_spec_list, save_dir='', filename='', db=None):
    """
    this will add or subtract the counts in the specdata object to the given parent specdata.
    Will only do that, if the x-axis are equal within 10 ** -5
    :param db: str, path of sqlite database
    :param filename: str, new filename for the added files, leave blank for automatically
    :param save_dir: str, path of the dir where to save the new file, leave blank for not saving.
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
            # try loop, if the dimensions of cts or x axis of the files do not match
            try:
                # check if the x-axis of the two specdata are equal:
                if np.allclose(parent_specdata.x[tr_ind], add_meas[1].x[tr_ind], rtol=1 ** -5):
                    if tr_ind == 0:
                        added_files.append((add_meas[0], add_meas[1].file))
                        offsets.append(add_meas[1].offset)
                        accvolts.append(add_meas[1].accVolt)
                    parent_specdata.nrScans[tr_ind] += add_meas[1].nrScans[tr_ind]
                    for sc_ind, sc in enumerate(tr):
                        parent_specdata.cts[tr_ind][sc_ind] += add_meas[0] * add_meas[1].cts[tr_ind][sc_ind]
                        parent_specdata.cts[tr_ind][sc_ind] = parent_specdata.cts[tr_ind][sc_ind].astype(np.int32)
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
                else:
                    print('warning, file: %s does not have the'
                          ' same x-axis as the parent file: %s,'
                          ' will not add!' % (parent_specdata.file, add_meas[1].file))
            except Exception as e:
                print('warning, file: %s does not have the'
                      ' same x-axis as the parent file: %s,'
                      ' will not add!' % (parent_specdata.file, add_meas[1].file))
                print('error therefore is: %s' % e)
        # needs to be converted like this:
        parent_specdata.cts[tr_ind] = np.array(parent_specdata.cts[tr_ind])
        # I Don't know why this is not in this format anyhow.
    parent_specdata.offset = np.mean(offsets)
    parent_specdata.accVolt = np.mean(accvolts)
    if save_dir:
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        if not filename:  # automatic filename determination
            filename = nameFile(save_dir, '', 'sum_file', suffix='.xml')
        else:
            filename = os.path.normpath(os.path.join(save_dir, filename + '.xml'))
        # create empty xml file
        scan_dict = create_scan_dict_from_spec_data(
            parent_specdata, filename, db)
        scan_dict['isotopeData']['addedFiles'] = added_files
        createXmlFileOneIsotope(scan_dict, filename=filename)
        # call savespecdata in filehandl (will expect to have a .time_res)
        save_spec_data(parent_specdata, scan_dict)

    return parent_specdata, added_files, filename


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
        laserfreq = specdata.laserFreq  # if existant freq is usually given in 1/cm
    else:
        laserfreq = Physics.wavenumber(get_laserfreq_from_db(database_path, specdata)) / 2

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
            'trigger': {'type': 'no_trigger'},
            'pulsePattern': {'cmdList': []}
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


def nameFile(path, subdir, fileName, prefix='', suffix='.tld'):
    """
    find an unused valid filename.
    :return: str, path/subdir/timestamp_prefix_fileName + suffix
    """
    storagePath = os.path.join(path, subdir)
    if not os.path.exists(storagePath):
        os.makedirs(storagePath)
    filepath = os.path.join(storagePath, prefix + '_' + fileName)
    i = 0
    file = filepath + '_' + str('{0:03d}'.format(i)) + suffix
    if not os.path.isfile(file):
        return filepath + '_' + str('{0:03d}'.format(i)) + suffix
    while os.path.isfile(file):
        i += 1
        file = filepath + '_' + str('{0:03d}'.format(i)) + suffix
    return file


def createXmlFileOneIsotope(scanDict, seq_type=None, filename=None):
    """
    creates an .xml file for one Isotope. Using the Filestructure as stated in OneNote.
    :param scanDict: {'isotopeData', 'track0', 'pipeInternals'}
    :return:str, filename
    """
    isodict = deepcopy(scanDict['isotopeData'])
    meas_volt_dict = deepcopy(scanDict['measureVoltPars'])
    if seq_type is not None:
        isodict['type'] = seq_type
    root = xmlCreateIsotope(isodict)
    xml_add_meas_volt_pars(meas_volt_dict, root)
    if filename is None:
        path = scanDict['pipeInternals']['workingDirectory']
        filename = nameFileXml(isodict, path)
    print('creating .xml File: ' + filename)
    save_xml(root, filename, False)
    return filename


def nameFileXml(isodict, path):
    """
    finds a filename for the xml file in subdirectory 'sums'
    :return:str, filename
    """
    # path = scanDict['pipeInternals']['workingDirectory']
    nIso = isodict['isotope']
    seq_type = isodict['type']
    filename = nIso + '_' + seq_type + '_run'
    subdir = os.path.join(path, 'sums')
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    files = [file if file.endswith('.xml') else '-1....' for file in os.listdir(subdir)]
    if len(files):
        highest_filenum = sorted([int(file[-7:-4]) for file in files])[-1]
        print(files, highest_filenum)
    else:
        highest_filenum = -1
    newpath = os.path.join(subdir, filename + str('{0:03d}'.format(highest_filenum + 1)) + '.xml')
    return newpath


def save_spec_data(spec_data, scan_dict):
    """
    this will write the necessary values of the spec_data to an already existing xml file
    :param scan_dict: dict, containing all scan informations
    :param spec_data: spec_data, as a result from XmlImporter()
    :return:
    """
    try:
        scan_dict = deepcopy(scan_dict)
        try:
            time_res = len(spec_data.time_res) # if there are any values in here, it is a time resolved measurement
        except Exception as e:
            time_res = False
        existing_xml_fil_path = scan_dict['pipeInternals']['activeXmlFilePath']
        root_ele = load_xml(existing_xml_fil_path)
        track_nums, track_num_lis = get_number_of_tracks_in_scan_dict(scan_dict)
        for track_ind, tr_num in enumerate(track_num_lis):
            track_name = 'track' + str(tr_num)
            # only save name of trigger
            scan_dict[track_name]['trigger']['type'] = scan_dict[track_name]['trigger']['type'].name
            if time_res:
                scan_dict[track_name]['softwGates'] = spec_data.softw_gates[track_ind]
                xmlAddCompleteTrack(root_ele, scan_dict, spec_data.time_res_zf[track_ind], track_name)
                xmlAddCompleteTrack(
                    root_ele, scan_dict, spec_data.cts[track_ind], track_name, datatype='voltage_projection',
                    parent_ele_str='projections')
                xmlAddCompleteTrack(
                    root_ele, scan_dict, spec_data.t_proj[track_ind], track_name, datatype='time_projection',
                    parent_ele_str='projections')
            else:  # not time resolved
                scan_dict[track_name]['softwGates'] = []
                xmlAddCompleteTrack(root_ele, scan_dict, spec_data.cts[track_ind], track_name)
        save_xml(root_ele, existing_xml_fil_path, False)
        # now add it to the database:
        db_name = os.path.basename(scan_dict['pipeInternals']['workingDirectory']) + '.sqlite'
        db = scan_dict['pipeInternals']['workingDirectory'] + '\\' + db_name
        if os.path.isfile(db):
            os.chdir(scan_dict['pipeInternals']['workingDirectory'])
            relative_filename = os.path.normpath(
                os.path.join(os.path.split(os.path.dirname(existing_xml_fil_path))[1], os.path.basename(existing_xml_fil_path)))
            # if file is in same folder as db, replace this folder with a dot
            db_dir_name = os.path.split(scan_dict['pipeInternals']['workingDirectory'])[1]
            relative_filename = relative_filename.replace(
                db_dir_name, '.')
            from Tools import _insertFile
            _insertFile(relative_filename, db)
    except Exception as e:
        print('error while saving: ', e)


def get_number_of_tracks_in_scan_dict(scan_dict):
    """
    count the number of tracks in the given dictionary.
    search indicator is 'track' in keys.
    :return: (n_of_tracks, sorted(list_of_track_nums))
    """
    n_of_tracks = 0
    list_of_track_nums = []
    for key, val in scan_dict.items():
        if 'track' in str(key):
            n_of_tracks += 1
            list_of_track_nums.append(int(key[5:]))
    return n_of_tracks, sorted(list_of_track_nums)

if __name__ == '__main__':
    isodi = {'isotope': 'bbb', 'type': 'csdummy'}
    newname = nameFileXml(isodi, 'E:\Workspace\AddedTestFiles')
    print(newname)