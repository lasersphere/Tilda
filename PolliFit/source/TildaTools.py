"""
Created on 

@author: simkaufm

Module Description: Tools related closely to Tilda
"""
import ast
import json
import logging
import os
import sqlite3
import sys
from copy import deepcopy
from copy import copy

from PyQt5 import QtWidgets
from scipy.optimize import curve_fit

import numpy as np
from lxml import etree as ET

import Physics
from XmlOperations import xmlCreateIsotope, xml_add_meas_volt_pars, \
    xmlAddCompleteTrack, xmlFindOrCreateSubElement, xmlWriteDict
from Service.VoltageConversions.VoltageConversions import get_bits_from_voltage


def select_from_db(db, vars_select, var_from, var_where=[], addCond='', caller_name='unknown'):
    """
    connects to database and finds attributes vars_select (string) in table var_from (string)
    with the extra condition
    varwhere[0][i] == varwhere[1][i] (so varwhere = [list, list] is a list...)
    addCond -> e.g. 'ORDER BY date'
    will convert to:

        SELECT vars_select FROM var_from WHERE varwhere[0][i] == varwhere[1][i] ... addCond

    :return
        None, if failure,
        list, with of tuples with values if success [(vars_select0, vars_select1...), (...)]
    """
    sql_cmd = ''
    con = None
    try:
        con = sqlite3.connect(db)
        cur = con.cursor()
        if var_where:
            where = var_where[0][0] + ' = ?'
            list_where_is = [var_where[1][0]]
            var_where[0].remove(var_where[0][0])
            var_where[1].remove(var_where[1][0])
            for i, j in enumerate(var_where[0]):
                where = where + ' and ' + j + ' = ?'
                list_where_is.append(var_where[1][i])
            sql_cmd = str('''SELECT %s FROM %s WHERE %s %s''' % (vars_select, var_from, where, addCond))
            cur.execute(sql_cmd, tuple(list_where_is))
        else:
            sql_cmd = str('''SELECT %s FROM %s %s''' % (vars_select, var_from, addCond))
            cur.execute(sql_cmd, ())
        var = cur.fetchall()
        if var:
            return var
        else:
            print('error, in caller %s:' % caller_name, 'There is no db entry in ', var_from, ' with ', vars_select,
                  ' where ', var_where)
            return None
    except Exception as e:
        print('error in database access while trying to execute:\n', sql_cmd, '\n', e)
        return None
    finally:
        if con is not None:
            con.close()


def merge_dicts(d1, d2):
    """ given two dicts, merge them into a new dict as a shallow copy """
    new = d1.copy()
    new.update(d2)
    return new


def merge_extend_dicts(target_dict, new_dict, overwrite=True, force_overwrite=False):
    """
    This function is used, to merge the content of the second dict into the first.
    Whenever a dict is found inside the dict, a recursive function call is executed, merging the sub-dicts as well.
    :param overwrite: íf True, overwrite conflicting values (e.g. int, float, str, bool...). If False keep the old one.
    :param force_overwrite: if True, empty lists overwrite existing lists.
    :returns None: This function works on the target dict! No new dict is returned!
    """
    for keys, vals in new_dict.items():
        if keys in target_dict:  # key exists already
            is_same = vals == target_dict[keys]
            if not is_same:  # key exists but vals are different
                if type(vals) is dict:  # if its a dict then check this again
                    merge_extend_dicts(target_dict[keys], vals, overwrite)
                elif type(vals) is list:
                    if force_overwrite: # if overwriting existing lists with empty lists is allowed
                        target_dict[keys] = vals
                    else: # if the new list has actually values in it then we want to overwrite the old values
                        if type(target_dict[keys]) is list:
                            if len(vals) > len(target_dict[keys]):
                                target_dict[keys] = vals
                        elif len(vals):
                            target_dict[keys] = vals
                else:  # key exists, but vals are different and can't be combined
                    if overwrite:  # if authorized, overwrite the existing value with the new one
                        target_dict[keys] = vals
            # else: key exists and vals are identical - do nothing
        else:  # key doesn't exist
            target_dict[keys] = new_dict[keys]


# Copyright Ferry Boender, released under the MIT license.
# see here. https://www.electricmonk.nl/log/2017/05/07/merging-two-python-dictionaries-by-deep-updating/
def deepupdate(target, src):
    """
    Deep update target dict with src
    For each k,v in src: if k doesn't exist in target, it is deep copied from
    src to target. Otherwise, if v is a list, target[k] is extended with
    src[k]. If v is a set, target[k] is updated with v, If v is a dict,
    recursively deep-update it.

    Examples:
    >>> t = {'name': 'Ferry', 'hobbies': ['programming', 'sci-fi']}
    >>> deepupdate(t, {'hobbies': ['gaming']})
    >>> print(t)
    {'name': 'Ferry', 'hobbies': ['programming', 'sci-fi', 'gaming']}
    """
    for k, v in src.items():
        if type(v) == list:
            if not k in target:
                target[k] = deepcopy(v)
            else:
                target[k].extend(v)
        elif type(v) == dict:
            if not k in target:
                target[k] = deepcopy(v)
            else:
                deepupdate(target[k], v)
        elif type(v) == set:
            if not k in target:
                target[k] = v.copy()
            else:
                target[k].update(v.copy())
        elif type(v) == np.ndarray:
            if not k in target:
                target[k] = deepcopy(v)
            else:
                target[k] = np.append(target[k], v)
        # TODO: Elif v is None do nothing? Because it might just replace an empty dict with None
        else:
            target[k] = copy(v)


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
        result = np.zeros(from_string.size // 4, dtype=[('sc', 'u2'), ('step', 'u4'), ('time', 'u4'), ('cts', 'u4')])
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
            if key == 'meas_trigger' or key == 'step_trigger' or key == 'scan_trigger':
                val['type'] = val['type'].replace('TriggerTypes.', '')
                # val = val.replace("<TriggerTypes.", "\'")
                # val = val.replace(">", "\'")
                # dicti[key] = ast.literal_eval(val)
            else:
                print('error while converting val with ast.literal_eval: ', e, val, type(val), key)
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
    logging.info('xml file is file: %s' % os.path.isfile(xml_file))
    xml_etree = load_xml(xml_file)
    all_trackd = {}
    tracks = xml_etree.find('tracks')
    for t in tracks:
        all_trackd[str(t.tag)] = (xml_get_dict_from_ele(xml_etree)[1]['tracks'][str(t.tag)]['header'])
    for tr_name, track_d in all_trackd.items():
        all_trackd[tr_name] = evaluate_strings_in_dict(all_trackd[tr_name])
    # make sure no None values exist due to something
    # like <something\> caused by an empty dict when saving.
    all_trackd = replace_none_vals_in_dict(all_trackd, {})
    return all_trackd


def get_meas_volt_dict(xml_etree):
    """
        OUTDATED SINCE VERSION 1.19

    get the dictionary containing all the voltage measurement parameters
     from an xml_etree element loaded from an xml file. """
    meas_volt_pars_dict = xml_get_dict_from_ele(xml_etree)[1].get('measureVoltPars', {})
    evaluate_strings_in_dict(meas_volt_pars_dict)
    for key, val in meas_volt_pars_dict.items():
        if val.get('dmms', None) is None:
            meas_volt_pars_dict[key]['dmms'] = {}
    return meas_volt_pars_dict


def save_dmm_readings_to_xml(file, tr_name, dmms_dict, pre_during_post_scan_str):
    """
    Will save the dmms dicts gotten from the scan_dict to the given xml file.

    :param file: str, path to xml file
    :param tr_name: str, track name
    :param dmms_dict: dict, 'dmms':{...}... as in draft scan pars
    :param pre_during_post_scan_str: str, preScan / duringScan / postScan
    :return: None
    """
    if file:
        root = load_xml(file)
        tracks = xmlFindOrCreateSubElement(root, 'tracks')
        track = xmlFindOrCreateSubElement(tracks, tr_name)
        track_header = xmlFindOrCreateSubElement(track, 'header')
        meas_volt = xmlFindOrCreateSubElement(track_header, 'measureVoltPars')
        pre_during_ele = xmlFindOrCreateSubElement(meas_volt, pre_during_post_scan_str)
        dmms_ele = xmlFindOrCreateSubElement(pre_during_ele, 'dmms')
        for dmm_name, dmm_dict in dmms_dict.items():
            dmm_ele = xmlFindOrCreateSubElement(dmms_ele, dmm_name)
            xmlFindOrCreateSubElement(dmm_ele, 'readings', dmm_dict['readings'])
            logging.debug('saved %s meas of dmm: %s, reading is: %s' % (
                pre_during_post_scan_str, dmm_name, str(dmm_dict['readings'])))
        save_xml(root, file)


def get_triton_dict_from_xml_root(xml_etree):
    """
    OUTDATED SINCE VERSION 1.19
    get the triton dictionary from an exisitng xml file.
    :param xml_etree: lxml.etree.Element, Element of loaded File
    :return: dict,
    {'preScan': {'dummyDev': {'ch1': {'required': 2, 'data': [1,2], 'acquired': 2}, ...}},
     'postScan': ...,
     'duringScan': ...}
    """
    triton_dict = xml_get_dict_from_ele(xml_etree)[1].get('triton', {})
    evaluate_strings_in_dict(triton_dict)
    # print('triton_dict from file: ', triton_dict)
    return triton_dict


def save_triton_to_xml(file, tr_name, triton_dict, pre_during_post_scan_str='preScan'):
    """
    will save the triton log gotten from the triton dict to the given xml file.

    :param file: str, path of the xml file
    :param tr_name: str, track name
    :param triton_dict:  dict, {'dummyDev': {'ch1': {'required': 2, 'data': [], 'acquired': 0}, ...}}
    :param pre_during_post_scan_str: str, preScan / duringScan / postScan
    :return: None
    """
    if file:
        if triton_dict:
            logging.info('triton %s log complete, saving to: %s' % (pre_during_post_scan_str, file))
            logging.debug('saving directly to xml file: ' + str(triton_dict))
            root = load_xml(file)
            tracks = xmlFindOrCreateSubElement(root, 'tracks')
            track = xmlFindOrCreateSubElement(tracks, tr_name)
            track_header = xmlFindOrCreateSubElement(track, 'header')
            triton_ele = xmlFindOrCreateSubElement(track_header, 'triton')
            pre_ele = xmlFindOrCreateSubElement(triton_ele, pre_during_post_scan_str)
            xmlWriteDict(pre_ele, triton_dict)
            save_xml(root, file)


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
                if key == 'meas_trigger' or key == 'step_trigger' or key == 'scan_trigger':
                    val['type'] = val['type'].replace('TriggerTypes.', '')
                else:
                    # print('error while converting val with ast.literal_eval: ', e, val, type(val), key)
                    # if it cannot be converted it is propably a string anyhow.
                    # print('error: %s could not convert key: %s val: %s' % (e, key, val))
                    pass

        if isinstance(dict_to_convert[key], dict):
            dict_to_convert[key] = evaluate_strings_in_dict(dict_to_convert[key])
    return dict_to_convert


def replace_none_vals_in_dict(dict_to_check, replace_val={}):
    """
    this will iterate through the given dict, wich can be nested and
    replace all values that are None with the given replace val.
    :param dict_to_check: dict, can be nested
    :param replace_val: anything
    :return: dict without None vals
    """
    for key, val in dict_to_check.items():
        if isinstance(dict_to_check[key], dict):
            dict_to_check[key] = replace_none_vals_in_dict(dict_to_check[key], replace_val)
        elif val is None:
            dict_to_check[key] = replace_val
    return dict_to_check


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
    try:
        version = float(scan_dict['isotopeData']['version'])
    except ValueError:  # MCP Data which was summed to an combined .xml file has version like 2.0.0.23,
        version = 0.1
        scan_dict['isotopeData']['version'] = '0.1'
    if version <= 1.18:
        # after this version, those infos are stored within each track!
        # kept this for backwards compatibility
        scan_dict['measureVoltPars'] = get_meas_volt_dict(xml_etree)
        scan_dict['triton'] = get_triton_dict_from_xml_root(xml_etree)
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


def gate_specdata(spec_data, full_x_range=True):
    """
    function to gate spec_data with the softw_gates list in the spec_data itself.
    gate will be applied on spec_data.time_res and
     the time projection will be written to spec_data.t_proj
     the voltage projection will be written to spec_data.cts
    :param spec_data: spec_data
    :param full_x_range: bool, True default -> will gate over the full x range
    :return: spec_data
    """
    # logging.debug('gating data now, software gates are: %s' % spec_data.softw_gates)
    # check if enough gates are given for this track
    for tr_ind, gates_tr in enumerate(spec_data.softw_gates):
        tr_scalers = spec_data.nrScalers[tr_ind]
        dif = tr_scalers - len(gates_tr)
        if dif > 0:  # not enough gates defined for this track will add now
            for i in range(dif):
                spec_data.softw_gates[tr_ind].append([float('-inf'), float('inf'), 0, float('inf')])
    if full_x_range:
        for tr_ind, gates_tr in enumerate(spec_data.softw_gates):
            for sc_ind in range(spec_data.nrScalers[tr_ind]):
                spec_data.softw_gates[tr_ind][sc_ind][0] = min(spec_data.x[tr_ind][0], spec_data.x[tr_ind][-1])
                spec_data.softw_gates[tr_ind][sc_ind][1] = max(spec_data.x[tr_ind][0], spec_data.x[tr_ind][-1])
    # get indices of the values first
    compare_arr = [spec_data.x, spec_data.x, spec_data.t, spec_data.t]
    softw_gates_ind = [
        [[find_closest_value_in_arr(compare_arr[lim_ind][tr_ind], lim)[0] for lim_ind, lim in enumerate(gates_pmt)]
         for gates_pmt in gates_tr]
        for tr_ind, gates_tr in enumerate(spec_data.softw_gates)]
    spec_data.softw_gates = [
        [[compare_arr[lim_ind][tr_ind][found_ind] for lim_ind, found_ind in enumerate(gate_ind_pmt)]
         for gate_ind_pmt in gate_ind_tr]
        for tr_ind, gate_ind_tr in enumerate(softw_gates_ind)]
    for tr_ind, tr in enumerate(spec_data.cts):
        for pmt_ind, pmt in enumerate(tr):
            v_min_ind = min(softw_gates_ind[tr_ind][pmt_ind][0], softw_gates_ind[tr_ind][pmt_ind][1])
            v_max_ind = max(softw_gates_ind[tr_ind][pmt_ind][0], softw_gates_ind[tr_ind][pmt_ind][1]) + 1
            t_min_ind = min(softw_gates_ind[tr_ind][pmt_ind][2], softw_gates_ind[tr_ind][pmt_ind][3])
            t_max_ind = max(softw_gates_ind[tr_ind][pmt_ind][2], softw_gates_ind[tr_ind][pmt_ind][3]) + 1
            t_proj_res = np.sum(spec_data.time_res[tr_ind][pmt_ind][v_min_ind:v_max_ind, :], axis=0)
            v_proj_res = np.sum(spec_data.time_res[tr_ind][pmt_ind][:, t_min_ind:t_max_ind], axis=1)
            spec_data.t_proj[tr_ind][pmt_ind] = t_proj_res
            spec_data.cts[tr_ind][pmt_ind] = v_proj_res
            spec_error = deepcopy(v_proj_res) # Errors have to be regated as well
            spec_error[spec_error < 1] = 1
            spec_data.err[tr_ind][pmt_ind] = np.sqrt(spec_error)
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
    new_time_res = [np.zeros(dims_sc_step_time_list[tr_ind], dtype=np.int32)
                    for tr_ind, tr in enumerate(dims_sc_step_time_list)]

    for tr_ind, tr_dims in enumerate(dims_sc_step_time_list):
        sc_arr = zf_time_res_arr[tr_ind]['sc']
        step_arr = zf_time_res_arr[tr_ind]['step']
        time_arr = zf_time_res_arr[tr_ind]['time']
        new_time_res[tr_ind][sc_arr, step_arr, time_arr] = zf_time_res_arr[tr_ind]['cts']
    return new_time_res


def non_zero_free_to_zero_free(non_zf_tr_wise_matrices):
    """
    Convert a trackwise list of numpy arrays which contains all counts and zeros for all pixel without an event
    to a zero free data format like [([(sc, step, time, cts), ...]), ... tr1 ... ]
    :param non_zf_tr_wise_matrices: list of numpy ndarrays with dimensons: [[sc_tr0, step_tr0, bins_tr0], ... tr1 ...]
    :return: list of "zero free data format" one array for each track
            ->  [([(sc, step, time, cts), ...]), ... tr1 ... ]
    """
    result = []

    for tr_ind, tr_dat in enumerate(non_zf_tr_wise_matrices):
        result.append([])
        sc_tr, step_tr, bons_tr = tr_dat.shape
        for sc in range(sc_tr):
            non_z_ind = np.nonzero(tr_dat[sc])
            step_sc_tr, time_sc_tr = non_z_ind
            cts_sc_tr = tr_dat[sc, step_sc_tr, time_sc_tr]
            res_sc_tr = np.zeros(step_sc_tr.size,
                                 dtype=[('sc', 'u2'), ('step', 'u4'), ('time', 'u4'), ('cts', 'u4')])

            res_sc_tr['sc'] = np.full(step_sc_tr.size, sc, dtype=[('sc', 'u2')])
            res_sc_tr['step'] = step_sc_tr
            res_sc_tr['time'] = time_sc_tr
            res_sc_tr['cts'] = cts_sc_tr
            if sc:
                result[tr_ind] = np.append(result[tr_ind], res_sc_tr)
            else:
                result[tr_ind] = res_sc_tr

    return result


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
        # i dont think a copy is necessary.
        # zf_spec = deepcopy(spec_data)
        # zf_spec = spec_data
        spec_data.time_res = zero_free_to_non_zero_free(spec_data.time_res_zf, dimensions)
        spec_data = gate_specdata(spec_data)
    except Exception as e:
        logging.error('error: while gating zero free specdata: %s ' % e, exc_info=True)
    return spec_data

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
    :param as_voltage: bool, True if this should be returned in the native unit of th scan dev (usually volts)
    """
    x_arr = []
    for tr_ind, tr_name in enumerate(get_track_names(scan_dict)):
        steps = scan_dict[tr_name]['nOfSteps']
        sc_dev_d = scan_dict[tr_name].get('scanDevice', {})
        if sc_dev_d == {}:
            if as_voltage:
                start = scan_dict[tr_name]['dacStartVoltage']  # backwards comp.
                stop = scan_dict[tr_name]['dacStopVoltage']  # backwards comp.
                step = scan_dict[tr_name]['dacStepsizeVoltage']  # backwards comp.
            else:
                start = scan_dict[tr_name]['dacStartRegister18Bit']  # backwards comp.
                step = scan_dict[tr_name]['dacStepSize18Bit']  # backwards comp.
                stop = scan_dict[tr_name].get('dacStopRegister18Bit', start + step * (steps - 1))  # backwards comp.
        else:
            start = sc_dev_d['start']
            stop = sc_dev_d['stop']
            step = sc_dev_d['stepSize']
            if not as_voltage:  # use dac register
                start = get_bits_from_voltage(start)
                stop = get_bits_from_voltage(stop)

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
            delay = scan_dict[tr_name]['trigger'].get('trigDelay10ns', 0) * bin_width  # for older versions
            if delay == 0:
                # newer versions -> get delay from meas trigger
                delay = scan_dict[tr_name]['trigger'].get('meas_trigger', {}).get('trigDelay10ns', 0) * bin_width
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
    if np.isinf(search_val):
        search_val = np.max(arr) if search_val > 0 else np.min(arr)
    if isinstance(arr, list):
        arr = np.array(arr)
    ind = (np.abs(arr - search_val)).argmin()
    found_val = arr[ind]
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
                        # add time_res (with zero) matrices if 'time_res' data exists
                        #  this is always the case for time resolved data
                        if check_if_attr_exists(parent_specdata, 'time_res', False) and check_if_attr_exists(
                                add_meas[1], 'time_res', False):
                            try:
                                parent_specdata.time_res[tr_ind][sc_ind] += \
                                    add_meas[0] * add_meas[1].time_res[tr_ind][sc_ind]
                            except Exception as e:
                                print('Timing bins seem not to be the same. '
                                      'Files can not be added in time domain. Error is: %s' % e)
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

    # create the zero free data from the non zero free
    parent_specdata.time_res_zf = non_zero_free_to_zero_free(parent_specdata.time_res)

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
        # take the time from the host file!
        createXmlFileOneIsotope(scan_dict, filename=filename, take_time_now=False)
        # call savespecdata in filehandl (will expect to have a .time_res)
        save_spec_data(parent_specdata, scan_dict)

    return parent_specdata, added_files, filename


def check_if_attr_exists(parent_to_check_from, par, return_val_if_not, and_not_equal_to=None):
    """ use par as attribute of parent_to_check_from and return the result.
     If this is not a valid arg, return return_val_if_not """
    ret = return_val_if_not
    try:
        ret = getattr(parent_to_check_from, par)
        if ret == and_not_equal_to:
            ret = return_val_if_not
    except Exception as e:
        ret = return_val_if_not
    finally:
        return ret


def check_var_type(list_of_vars):
    """
    will go trough a list of vars and check if they are in the right type,
     range and correct them to a standard value if not
    :param list_of_vars: list of tuples, [(var_to_check, var_type, var_range, standard_val), (...) ... ]
    var_range is a list: [min_val, max_val], leave empty for not checking a range
    :return: list, corrected vars
    """
    ret = []
    for var_to_check, var_type, var_range, standard_val in list_of_vars:
        if isinstance(var_to_check, var_type):
            if len(var_range):
                if var_range[0] < var_to_check < var_range[1]:
                    good_val = var_to_check
                else:  # right type and in range
                    good_val = standard_val
            else:  # type ok, range does not matter
                good_val = var_to_check
        else:  # if type is wrong, use stanadrd val !?
            good_val = standard_val
        ret.append(good_val)
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
        # TODO watchout here a doubling is always assumed!
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
            'scanDevice': specdata.scan_dev_dict_tr_wise[tr_ind],
            'nOfSteps': specdata.getNrSteps(tr_ind),
            'nOfScans': specdata.nrScans[tr_ind],
            'nOfCompletedSteps': specdata.nrScans[tr_ind] * specdata.getNrSteps(tr_ind),
            'invertScan': check_if_attr_exists(specdata, 'invert_scan', [False] * specdata.nrTracks)[tr_ind],
            'postAccOffsetVoltControl': specdata.post_acc_offset_volt_control, 'postAccOffsetVolt': specdata.offset,
            'waitForKepco1us': check_if_attr_exists(specdata, 'wait_after_reset_1us', [-1] * specdata.nrTracks)[
                tr_ind],
            'waitAfterReset1us': check_if_attr_exists(specdata, 'wait_for_kepco_1us', [-1] * specdata.nrTracks)[
                tr_ind],
            'activePmtList': check_if_attr_exists(specdata, 'activePMTlist', False)[tr_ind] if
            check_if_attr_exists(specdata, 'activePMTlist', []) else
            check_if_attr_exists(specdata, 'active_pmt_list', [])[tr_ind],
            'colDirTrue': specdata.col,
            'dwellTime10ns': specdata.dwell,
            'workingTime': check_if_attr_exists(specdata, 'working_time', [None] * specdata.nrTracks)[tr_ind],
            'nOfBins': len(check_if_attr_exists(specdata, 't', [[0]] * specdata.nrTracks)[tr_ind]),
            'softBinWidth_ns': check_if_attr_exists(specdata, 'softBinWidth_ns',
                                                    [0] * specdata.nrTracks, [])[tr_ind],
            'nOfBunches': 1,
            'softwGates': check_if_attr_exists(
                specdata, 'softw_gates', [[] * specdata.nrScalers[tr_ind]] * specdata.nrTracks, [])[tr_ind],
            'trigger': check_if_attr_exists(
                specdata, 'trigger', [{'meas_trigger': {'type': 'no_trigger'},
                                       'step_trigger': None,
                                       'scan_trigger': None}] * specdata.nrTracks)[tr_ind],
            'pulsePattern': {'cmdList': [], 'periodicList': [], 'simpleDict': {}},
            'measureVoltPars': specdata.measureVoltPars[tr_ind],
            'triton': specdata.tritonPars[tr_ind]
        }
    draftMeasureVoltPars_singl = {'measVoltPulseLength25ns': -1, 'measVoltTimeout10ns': -1,
                                  'dmms': {}, 'switchBoxSettleTimeS': -1}
    pre_scan_dmms = {'unknown_dmm': {'assignment': 'offset', 'readings': [deepcopy(specdata.offset)]},
                     'unknown_dmm_1': {'assignment': 'accVolt', 'readings': [deepcopy(specdata.accVolt)]},
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


def createXmlFileOneIsotope(scanDict, seq_type=None, filename=None, take_time_now=True):
    """
    creates an .xml file for one Isotope. Using the Filestructure as stated in OneNote.
    :param scanDict: {'isotopeData', 'track0', 'pipeInternals'}
    :return:str, filename
    """
    isodict = deepcopy(scanDict['isotopeData'])
    # meas_volt_dict = deepcopy(scanDict['measureVoltPars'])
    if seq_type is not None:
        isodict['type'] = seq_type
    root = xmlCreateIsotope(isodict, take_time_now=take_time_now)
    # xml_add_meas_volt_pars(meas_volt_dict, root)
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
    # get existing xml files in sums and take tehir name without the ending ".xml":
    files = [file.split('.')[0] if file.endswith('.xml') else '-1....' for file in os.listdir(subdir)]
    if len(files):
        try:
            highest_filenum = sorted(
                [int(file[file.index('_run') + 4:]) for file in files if '_run' in file])[-1]
        except Exception as e:
            print('error finding run number, with _run***.xml (*** should be integers) error is: ', e)
            highest_filenum = -1
    else:
        highest_filenum = -1
    newpath = os.path.join(subdir, filename + str('{0:03d}'.format(highest_filenum + 1)) + '.xml')
    while os.path.isfile(newpath):  # do not overwrite!
        print('error: file already exists! Check your file naming not conflict with '
              '_run***.xml (*** should be integers) filenum is: ', highest_filenum)
        highest_filenum += 1
        newpath = os.path.join(subdir, filename + '_NameErr_' + str('{0:03d}'.format(highest_filenum + 1)) + '.xml')
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
        version = float(scan_dict['isotopeData']['version'])
        # scan_dict = create_scan_dict_from_spec_data(spec_data, scan_dict['pipeInternals']['activeXmlFilePath'])

        # be sure that zero free data is created for storing in the file!
        spec_data.time_res_zf = non_zero_free_to_zero_free(spec_data.time_res)

        try:
            if version > 1.1:
                time_res = len(spec_data.time_res_zf)
            else:
                # if there are any values in here, it is a time resolved measurement
                time_res = len(spec_data.time_res)
        except Exception as e:
            time_res = False
        existing_xml_fil_path = scan_dict['pipeInternals']['activeXmlFilePath']
        root_ele = load_xml(existing_xml_fil_path)
        track_nums, track_num_lis = get_number_of_tracks_in_scan_dict(scan_dict)
        for track_ind, tr_num in enumerate(track_num_lis):
            track_name = 'track' + str(tr_num)
            # only save name of trigger
            if not isinstance(scan_dict[track_name]['trigger'].get('meas_trigger', {}).get('type', {}), str):
                scan_dict[track_name]['trigger']['meas_trigger']['type'] = \
                    scan_dict[track_name]['trigger']['meas_trigger']['type'].name
            if not isinstance(scan_dict[track_name]['trigger'].get('step_trigger', {}).get('type', {}), str):
                scan_dict[track_name]['trigger']['step_trigger']['type'] = \
                    scan_dict[track_name]['trigger']['step_trigger']['type'].name
            if not isinstance(scan_dict[track_name]['trigger'].get('scan_trigger', {}).get('type', {}), str):
                scan_dict[track_name]['trigger']['scan_trigger']['type'] = \
                    scan_dict[track_name]['trigger']['scan_trigger']['type'].name
            if time_res:
                scan_dict[track_name]['softwGates'] = spec_data.softw_gates[track_ind]
                if version > 1.1:
                    xmlAddCompleteTrack(root_ele, scan_dict, spec_data.time_res_zf[track_ind], track_name)
                else:
                    xmlAddCompleteTrack(root_ele, scan_dict, spec_data.time_res[track_ind], track_name)
                xmlAddCompleteTrack(
                    root_ele, scan_dict, spec_data.cts[track_ind], track_name, datatype='voltage_projection',
                    parent_ele_str='projections')
                # removed saving of time projection, because nobody wins anything on this.
                # xmlAddCompleteTrack(
                #     root_ele, scan_dict, spec_data.t_proj[track_ind], track_name, datatype='time_projection',
                #     parent_ele_str='projections')
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
                os.path.join(os.path.split(os.path.dirname(existing_xml_fil_path))[1],
                             os.path.basename(existing_xml_fil_path)))
            # if file is in same folder as db, replace this folder with a dot
            db_dir_name = os.path.split(scan_dict['pipeInternals']['workingDirectory'])[1]
            relative_filename = relative_filename.replace(
                db_dir_name, '.')
            from Tools import _insertFile
            logging.debug('will insert file: %s to database: %s' % (relative_filename, db))
            _insertFile(relative_filename, db)
    except Exception as e:
        logging.error('error while saving: %s' % e, exc_info=True)


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


def get_software_gates_from_db(db, iso, run, track=0):
    """
    get the software gates for a SINGLE TRACK from the database.
    voltages will be gated from -10 to 10 V.
    timings will be calculated like this:
    start_gate_sc0 = mid_tof + delay_sc0 - 0.5 * width
    stopp_gate_sc0 = mid_tof + delay_sc0 + 0.5 * width
    """
    use_db, run_gates_width, run_gates_delay, iso_mid_tof = get_gate_pars_from_db(db, iso, run)
    if use_db == 'file':
        return None
    # if use_db != 'file' and use_db is not None:
    #     if isinstance(ast.literal_eval(use_db[0][0]), list): return ast.literal_eval(use_db[0][0])[track]  # in this case softwGates is the softwGates list
    if iso_mid_tof is None or run_gates_width is None or run_gates_delay is None:
        return None  # return None if failur by getting stuff from db
    else:
        softw_gates_db = calc_soft_gates_from_db_pars(run_gates_width, run_gates_delay, iso_mid_tof)
        print('extracted software gates for isotope: %s and run %s from db %s: ' % (iso, run, os.path.split(db)[1]),
              softw_gates_db)
        return softw_gates_db


def get_gate_pars_from_db(db, iso, run):
    """ will get the use_db, run_gates_width, run_gates_delay, iso_mid_tof from the database """
    use_db = select_from_db(
        db, 'softwGates', 'Runs', [['run'], [run]],
        caller_name='get_software_gates_from_db in DataBaseOperations.py')
    run_gates_width = select_from_db(
        db, 'softwGateWidth', 'Runs', [['run'], [run]],
        caller_name='get_software_gates_from_db in DataBaseOperations.py')
    run_gates_delay = select_from_db(
        db, 'softwGateDelayList', 'Runs', [['run'], [run]],
        caller_name='get_software_gates_from_db in DataBaseOperations.py')
    iso_mid_tof = select_from_db(
        db, 'midTof', 'Isotopes', [['iso'], [iso]],
        caller_name='get_software_gates_from_db in DataBaseOperations.py')
    if iso_mid_tof is None or run_gates_width is None or run_gates_delay is None or iso_mid_tof[0][0] is None or \
            run_gates_width[0][0] is None or run_gates_delay[0][
        0] is None:  # added 3 cases since iso_mid_tof etc. is usually an array and != none (?)
        return None, None, None, None
    else:
        run_gates_width = run_gates_width[0][0]
        run_gates_delay = run_gates_delay[0][0]
        iso_mid_tof = iso_mid_tof[0][0]
        del_list = ast.literal_eval(run_gates_delay)
        return use_db, run_gates_width, del_list, iso_mid_tof


def calc_soft_gates_from_db_pars(run_gates_width, del_list, iso_mid_tof, voltage_gates=[-np.inf, np.inf]):
    """
    calc the software gates for a SINGLE TRACK from the given pars.
    voltages will be gated from -10 to 10 V.
    timings will be calculated like this:
    start_gate_sc0 = mid_tof + delay_sc0 - 0.5 * width
    stopp_gate_sc0 = mid_tof + delay_sc0 + 0.5 * width
    """
    # gates should be applied for all voltages/frequencies
    softw_gates_db = []
    for each_del in del_list:
        softw_gates_db.append(
            [voltage_gates[0], voltage_gates[1],
             iso_mid_tof + each_del - 0.5 * run_gates_width,
             iso_mid_tof + each_del + 0.5 * run_gates_width]
        )
    return softw_gates_db


def calc_db_pars_from_software_gate(softw_gates_single_tr):
    """ inverse function to calc_soft_gates_from_db_pars"""
    run_gates_width = 0
    del_list = []
    iso_mid_tof = 0
    for i, each in enumerate(softw_gates_single_tr):
        t_min, t_max = each[2], each[3]
        if i == 0:  # reference on first scaler one cannot tell on which was referenced before
            run_gates_width = t_max - t_min
            iso_mid_tof = t_min + run_gates_width * 0.5
        del_list.append(t_min + 0.5 * run_gates_width - iso_mid_tof)
    return run_gates_width, del_list, iso_mid_tof


def calc_bunch_width_relative_to_peak_height(spec_data, percentage_of_peak,
                                             show_plt=True, non_consectutive_time_bins_tolerance=1,
                                             save_to_path='', time_around_bunch=(-1, -1), fit_gaussian=True,
                                             additional_time_rChi=0.0):
    """
    This will analyse the time projection of the counts.
    It will get the background, the maximum counts, the timings of the maximum counts
    and where the counts have reached the desired percentage of the maximum peak (above background)
    each value will be given per track and scaler.
    :param spec_data: XMLImporter object, usual xmlobject, t_proj will be handled
    :param percentage_of_peak: float, percent, percentage of the maximum counts above
     background for automatic gate determination
    :param non_consectutive_time_bins_tolerance: int,
    tolerance how many time bins can be below the threshold before this is counted
    as a not connected to the area above threshold anymore.
    :param save_to_path: str, absoulte path were to store the plot as a .png or so
    :param time_around_bunch: tuple, time in us (before, after) bunch in order to display
    zoomed in around this time frame, still all plots will share the same x-axis
    :param fit_gaussian: bool, True in order to fit a gaussian to the time data
    :param additional_time_rChi: float, time in us that will be added/subtracted to the bunch stop/start time
    in order to calculate the rChiSq not on the full time projection but only around the interesting time frame.
    :return: XMLImporter object, ret_dict:
    ret_dict = {'max_counts': max_counts,  #  track wise, scaler wise each of the following
                'backgrounds': backgrounds,
                'bunch_begin_times': bunch_begin_times,
                'max_counts_times': max_counts_times,
                'bunch_end_times': bunch_end_times,
                'bunch_lenght_us': bunch_lenght_us,
                'gaussian_res': gaussian_results  # list, tr_sc wise [x0, sigma, amp, rchisq}
                }
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(spec_data.nrScalers[0], spec_data.nrTracks, sharex='col',
                             gridspec_kw={'hspace': 0.,
                                          'top': 0.95,
                                          'bottom': 0.05,
                                          'wspace': 0.7,
                                          'left': 0.09 / spec_data.nrTracks,
                                          'right': 0.9 - 0.25 / spec_data.nrTracks},
                             figsize=(8 * spec_data.nrTracks, 12 * (spec_data.nrScalers[0] / 4)))
    if spec_data.nrTracks == 1:
        axes = [[ax] for ax in axes]
    backgrounds = []
    max_counts = []
    bunch_begin_times = []
    max_counts_times = []
    max_counts_ind = []
    bunch_end_times = []
    bunch_lenght_us = []
    gaussian_results = []
    # from Measurement.XMLImporter import XMLImporter
    # spec_data = XMLImporter()
    tr = -1
    for tr_t_proj in spec_data.t_proj:
        # print('tr_tproj', tr_t_proj.shape)
        tr += 1
        sc = -1
        max_counts.append(np.max(tr_t_proj, axis=1))  # for all scalers at once
        max_counts_ind.append(np.argmax(tr_t_proj, axis=1))
        # print(max_counts_ind)
        max_counts_times.append(spec_data.t[tr][max_counts_ind[tr]])
        bunch_begin_times += [],
        bunch_end_times += [],
        bunch_lenght_us += [],
        backgrounds += [],
        gaussian_results += [],
        for sc_t_proj in tr_t_proj:
            sc += 1
            # background calc
            smpl_ind = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]  # explicit indice to sample the background
            sc_back_sampl = sc_t_proj[smpl_ind]
            sc_back_mean = np.mean(sc_back_sampl)
            backgrounds[tr] += sc_back_mean,
            # print(sc_back_sampl, sc_back_mean)
            # print(tr, sc, max(sc_t_proj), sc_t_proj)
            cond_min = (max_counts[tr][sc] - sc_back_mean) * (percentage_of_peak / 100) + sc_back_mean
            indice_above_cond = np.where(sc_t_proj >= cond_min)

            # find consecutive indices before and after the maximum counts index
            indice_above_cond_before_max = [val for val in indice_above_cond[0] if val <= max_counts_ind[tr][sc]]
            indice_above_cond_before_max.reverse()
            start_ind = indice_above_cond_before_max[0]
            for i, each in enumerate(indice_above_cond_before_max):
                if i < len(indice_above_cond_before_max) - 1:
                    if np.isclose(each - 1, indice_above_cond_before_max[i + 1],
                                  atol=non_consectutive_time_bins_tolerance):
                        start_ind = each - 2
                    else:
                        # break for loop as soon as indices are not consecutive anymore
                        break
            # print(indice_above_cond_before_max)
            # print('start_ind:', start_ind)
            # print(max_counts_ind[tr][sc])
            # print(indice_above_cond[0])
            indice_above_cond_after_max = [val for val in indice_above_cond[0] if val >= max_counts_ind[tr][sc]]
            stopp_ind = indice_above_cond_after_max[0]
            for i, each in enumerate(indice_above_cond_after_max):
                if i < len(indice_above_cond_after_max) - 1:
                    if np.isclose(each + 1, indice_above_cond_after_max[i + 1],
                                  atol=non_consectutive_time_bins_tolerance):
                        stopp_ind = each + 2
                    else:
                        # break for loop as soon as indices are not consecutive anymore
                        break
            # print(indice_above_cond_after_max)
            # print('stopp_ind:', stopp_ind)

            # print(spec_data.t[tr][indice_above_cond])
            bunch_begin_times[tr].append(spec_data.t[tr][start_ind])
            bunch_end_times[tr].append(spec_data.t[tr][stopp_ind])
            bunch_lenght_us[tr].append(bunch_end_times[tr][-1] - bunch_begin_times[tr][-1])
            # print(len(axes), tr, sc)
            pl_sc_tr = axes[sc][tr].plot(spec_data.t[tr], spec_data.t_proj[tr][sc],
                                         color='k', label='sc: %s, tr: %s' % (sc, tr), linewidth=1.)
            axes[sc][tr].autoscale(enable=True)
            axes[sc][tr].axvline(bunch_begin_times[tr][-1], color='g',
                                 label='start: %.2f µs' % bunch_begin_times[tr][-1],
                                 linewidth=1.5)
            axes[sc][tr].axvline(bunch_end_times[tr][-1], color='b',
                                 label='stopp: %.2f µs' % bunch_end_times[tr][-1],
                                 linewidth=1.5)
            axes[sc][tr].plot([spec_data.t[0][0]], [0], color='w',
                              label='length: %.2f µs' % bunch_lenght_us[tr][-1])
            axes[sc][tr].axvline(max_counts_times[tr][sc], color='r',
                                 label='max_cts: %.2f µs' % max_counts_times[tr][sc],
                                 linewidth=1.5)
            axes[sc][tr].axhline(cond_min, color='g',
                                 # label='%s percent of peak above background' % percentage_of_peak,
                                 linewidth=0.5, linestyle='--')
            axes[sc][tr].set_ylim(axes[sc][tr].get_ylim()[0], axes[sc][tr].get_ylim()[1] - 0.5)
            # in order not to have numbers overlap
            axes[sc][tr].set_xlabel('time of flight / µs')
            axes[sc][tr].set_ylabel('counts')
            if time_around_bunch[0] >= 0:
                # zoom in around bunch in plot
                axes[sc][tr].set_xlim(bunch_begin_times[tr][-1] - time_around_bunch[0], axes[sc][tr].get_xlim()[1])
            if time_around_bunch[1] >= 0:
                # zoom in around bunch in plot
                axes[sc][tr].set_xlim(axes[sc][tr].get_xlim()[0], bunch_end_times[tr][-1] + time_around_bunch[1])
            # now gaussian:
            if fit_gaussian:
                # cut time axis to relevant set:
                bin_width = round(spec_data.t[tr][1] - spec_data.t[tr][0], 2)
                additional_bins_rChi = additional_time_rChi // bin_width
                cut_ind_start = int(max(0, start_ind - additional_bins_rChi))
                cut_ind_stop = int(min(len(spec_data.t[tr]) - 1, stopp_ind + additional_bins_rChi))
                x_to_fit = spec_data.t[tr][cut_ind_start:cut_ind_stop]
                y_to_fit = sc_t_proj[cut_ind_start:cut_ind_stop]

                y_data_set = []  # "inverse from projection)
                for ind, each in enumerate(y_to_fit):
                    # create inverse to existing histogram
                    y_data_set += [x_to_fit[ind]] * int(each)
                mean = max_counts_times[tr][sc]
                sigma = (bunch_end_times[tr][-1] - bunch_begin_times[tr][-1]) / 2.35  # roughly...
                amp_start = max_counts_ind[tr][sc] * sigma * np.sqrt(2 * np.pi)
                x = spec_data.t[tr]
                y = sc_t_proj
                popt, pcov = curve_fit(Physics.gaussian_offset, x, y,
                                       p0=[mean, sigma, amp_start, sc_back_mean],
                                       bounds=([bunch_begin_times[tr][-1] - bin_width, 0, 0, 0],
                                               [bunch_end_times[tr][-1], np.inf, np.inf, np.inf])
                                       )
                mean_fit = popt[0]
                sigma_fit = popt[1]
                amp_fit = popt[2]
                off_fit = popt[3]
                y_gauss = np.array([Physics.gaussian_offset(t_i, *popt) for t_i in x])
                norm_fact = np.max(y_gauss)
                y_gauss_norm = y_gauss / norm_fact
                y_to_fit_norm = y_to_fit / norm_fact
                residuals = y_to_fit_norm - y_gauss_norm[cut_ind_start:cut_ind_stop]
                errs = np.sqrt(y_to_fit)
                errs = [1 if e == 0 else e for e in errs]  # replace errors with 0 by 1
                errs_norm = errs / norm_fact
                numbers_freedom = len(y_to_fit) - 4  # four from gaussian
                rchisq = sum([res ** 2 / e ** 2 for res, e in zip(residuals, errs_norm)]) / numbers_freedom
                err_fit_pars = [np.sqrt(pcov[j][j]) for j in range(pcov.shape[0])]
                label_gauss = 'gaussian:\n x0=%.2f(%.0f)us\n sig=%.2f(%.0f)\n rChi²=%.2f' % (
                    mean_fit, err_fit_pars[0] * 100, sigma_fit, err_fit_pars[1] * 100, rchisq)
                axes[sc][tr].plot(x_to_fit, Physics.gaussian_offset(x_to_fit, *popt), linewidth=2, color='grey',
                                  label=label_gauss)
                gaussian_results[tr].append([[mean_fit, sigma_fit, amp_fit, off_fit, rchisq], err_fit_pars])
            axes[sc][tr].legend(loc=(1.01, 0.05))
    fig.set_facecolor('w')
    fig.suptitle('%s percentage of peak: %.2f%% rebinning: %d ns '
                 % (spec_data.file, percentage_of_peak, spec_data.softBinWidth_ns[0]), ha='center', va='top')

    if show_plt:
        plt.show(block=True)

    if save_to_path:
        if isinstance(save_to_path, list):
            for each_store_loc in save_to_path:
                print('saving to: %s' % each_store_loc)
                fig.savefig(each_store_loc, dpi=150, quality=95, facecolor='w')
        else:
            print('saving to: %s' % save_to_path)
            fig.savefig(save_to_path, dpi=150, quality=95, facecolor='w')
    # print(max_counts)
    # print(backgrounds)
    # print(bunch_begin_times)
    # print(max_counts_times)
    # print(bunch_end_times)
    # print(bunch_lenght_us)
    plt.close(fig)
    ret_dict = {'max_counts': max_counts,
                'backgrounds': backgrounds,
                'bunch_begin_times': bunch_begin_times,
                'max_counts_times': max_counts_times,
                'bunch_end_times': bunch_end_times,
                'bunch_lenght_us': bunch_lenght_us,
                'gaussian_res': gaussian_results
                }
    return spec_data, ret_dict


def convert_volt_axis_to_freq(x_axis_energy, mass, col, laser_freq, iso_center_freq):
    """
    will convert an x axis given in total voltage (accvolt + offset + line * kepco)
    to frequency relative to the iso center freq
    :return: list, x_axis in freq
    """
    x_axis_freq = deepcopy(x_axis_energy)
    for tr_ind, x_each_track in enumerate(x_axis_energy):
        for i, e in enumerate(x_each_track):
            v = Physics.relVelocity(Physics.qe * e, mass * Physics.u)
            v = -v if col else v

            f = Physics.relDoppler(laser_freq, v) - iso_center_freq
            x_axis_freq[tr_ind][i] = f
    return x_axis_freq


def convert_fit_volt_axis_to_freq(fit):
    """
     will convert an x axis given in total voltage (accvolt + offset + line * kepco)
     to frequency relative to the iso center freq
     :return: list, x_axis in freq
    """
    x_axis_in_freq = convert_volt_axis_to_freq(fit.meas.x, fit.spec.iso.mass,
                                               fit.meas.col, fit.meas.laserFreq, fit.spec.iso.freq)
    return x_axis_in_freq


def print_dict_pretty(dict):
    """ module for pretty printing a dictionary """
    print(json.dumps(dict, sort_keys=True, indent=4))


def translate_raw_data(raw_data):
    """ translate raw data to string for interpretation """
    step_complete = add_header_to23_bit(1, 4, 0, 1)  # binary for step complete
    scan_started = add_header_to23_bit(2, 4, 0, 1)  # binary for scan started
    new_bunch = add_header_to23_bit(3, 4, 0, 1)  # binary for new bunch
    dac_int_key = 2 ** 29 + 2 ** 28 + 2 ** 23  # binary key for an dac element
    header_index = 2 ** 23  # binary for the headerelement
    step_comp_ct = 0
    scan_start_ct = 0
    new_bunch_ct = 0
    dac_ct = 0
    pmt_ct = 0
    for each in raw_data:
        if each == step_complete:
            step_comp_ct += 1
            print(each, '\t step complete %d ' % step_comp_ct)
        elif each == scan_started:
            scan_start_ct += 1
            print(each, '\t scan started  %d' % scan_start_ct, '\t current step: %d' % step_comp_ct)
        elif each == new_bunch:
            new_bunch_ct += 1
            print(each, '\t new bunch %d ' % new_bunch_ct, '\t current step: %d' % step_comp_ct)
        elif not each & header_index:
            pmt_ct += 1
            time = each & (2 ** 23 - 1)
            act_pmt = each >> 24
            act_pmts = [i for i in range(0, 7) if 2 ** i & act_pmt]
            print(each, '\t pmt_event number %d, act pmt: %s, timestamp: %d' % (pmt_ct, str(act_pmts), time),
                  '\t current step: %d' % step_comp_ct)
        elif each & dac_int_key:
            dac_ct += 1
            print(each, '\t dac bit: %d' % dac_ct, '\t current step: %d' % step_comp_ct)
        else:
            print(each, '\t could not be resolved ', format(each, '#032b')[2:], '\t current step: %d' % step_comp_ct)


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


def line_to_total_volt(x, lineMult, lineOffset, offset, accVolt, voltDivRatio, offset_by_dev_mean={}):
    """
    Converts an DAC line voltage array x to a total voltage array depending on the conversion coefficients
    """
    if isinstance(voltDivRatio['offset'], float):  # just one number
        scanvolt = (x * lineMult + lineOffset) * voltDivRatio.get('lineMult', voltDivRatio['offset']) \
                   + offset * voltDivRatio['offset']
    else: # offset measured by different devices. Offset is then calculated by different voltDivRatio values.
        vals = list(voltDivRatio['offset'].values())
        mean_offset_div_ratio = np.mean(vals)
        # treat each offset with its own divider ratio
        # x axis is multiplied by mean divider ratio value anyhow, similiar to kepco scans

        mean_offset = np.mean([val * offset_by_dev_mean.get(key, offset) for key, val in voltDivRatio['offset'].items()])
        scanvolt = (lineMult * x + lineOffset) * voltDivRatio.get('lineMult', mean_offset_div_ratio) + mean_offset

    return accVolt*voltDivRatio['accVolt'] - scanvolt


def get_file_number_from_file_str(file_str, mass_index, end_result_len, app=None):
    numbers = []
    number_str = ''
    for i, letter in enumerate(file_str):
        if letter.isdigit():  # is either 0-9
            number_str += letter
        else:  # not a digit
            if number_str.isdigit():  # convert the consecutive number
                numbers += [number_str]
            number_str = ''  # reset number str
    if isinstance(mass_index, list):
        # [0] etc. .
        numbers = [val for n, val in enumerate(numbers) if n not in mass_index]
    if end_result_len > 0:
        # user want to check if the correct amount of integers is found
        if len(numbers) != end_result_len:
            # does not match, require user input!
            print('warning: ')
            print('file', file_str, 'nums', numbers)
            if app is None:
                app = QtWidgets.QApplication(sys.argv)

            print('opening dial:')
            text, ok_pressed = QtWidgets.QInputDialog.getText(None, 'Warning',
                                                              '%s has more or less than %s numbers: %s \n'
                                                              ' please write the desired file number(s) here'
                                                              'still as a list of strings please!:' %
                                                              (file_str, end_result_len, numbers),
                                                              QtWidgets.QLineEdit.Normal,
                                                              str(numbers)
                                                              )
            if ok_pressed:
                try:
                    numbers = ast.literal_eval(text)
                except Exception as e:
                    print('could not convert %s, error is: %s' % (text, e))
            else:
                return [], app
            # make sure it has the right length in the end!
            if len(numbers) > end_result_len:
                numbers = numbers[:end_result_len]
                print('warning, still incorrect amount of numbers! Will use %s fo file: %s' % (numbers, file_str))
            elif len(numbers) < end_result_len:
                numbers = numbers * (end_result_len // len(numbers) + 1)
                numbers = numbers[:end_result_len]
                print('warning, still incorrect amount of numbers! Will use %s fo file: %s' % (numbers, file_str))
    return numbers, app


def get_file_numbers(file_list, mass_index=[0], end_result_len=1, app=None, user_overwrite={}):
    """
    get all file numbers (=conescutive integer numbers)
    in the filenames that are listed in file_list.
    :param file_list: list, like  ['62_Ni_trs_run071.xml', ...]
    :param mass_index: list, indice of all expected mass numbers, that will be removed from the output.
    if the mass number is wanted -> use mass_index=[-1]
    for 62_Ni_trs_run071.xml
     -> mass_index=[0] -> [[71]]
     -> mass_index=None -> [[62, 71]]
     :param end_result_len: int, desired amount of numbers to be found, as a cross check.
     use -1/0 if you don't care
     :param user_overwrite: dict, key is orig. filenum, value is str that will be put as file_num_str
     e.g.  {'60_Ni_trs_run113_sum114.xml': ['113+114']}
     helpful to avoid user input on runtime
    :return: list of all file numbers each still as string, that might be convertable to int,
     but can also be something like '123+124' by user choice.
    """
    file_nums = []
    for f in file_list:
        if f in user_overwrite.keys():
            file_nums += user_overwrite[f]
        else:
            file_num, app = get_file_number_from_file_str(f, mass_index, end_result_len, app)
            file_nums += file_num
    return file_nums


def get_scan_step_from_track_dict(track_dict):
    """
    return the current scan and step number from the numberOfcompleted steps
    :param track_dict: dict, as in Service/Scan/draftScanParameters.py:136
    :return: tuple, (scan_num, step_num)
    """
    nOfSteps = track_dict['nOfSteps']
    nOfCompletedSteps = track_dict['nOfCompletedSteps']
    invertScan = track_dict['invertScan']
    current_scan = nOfCompletedSteps // nOfSteps
    odd_scan = current_scan % 2
    completed_steps_in_cur_scan = nOfCompletedSteps % nOfSteps
    current_step = nOfSteps - completed_steps_in_cur_scan if invertScan and odd_scan else completed_steps_in_cur_scan
    return current_scan, current_step


if __name__ == '__main__':
    # isodi = {'isotope': 'bbb', 'type': 'csdummy'}
    # newname = nameFileXml(isodi, 'E:\Workspace\AddedTestFiles')
    # print(newname)
    # get_all_tracks_of_xml_in_one_dict('E:/TildaDebugging3/sums/Test_trsdummy_run005.xml')
    # import Service.Scan.draftScanParameters as dft
    #
    # sc_dict = dft.draftScanDict
    # isodict = sc_dict['isotopeData']
    # print(nameFileXml(isodict, 'E:\\temp2'))
    # ret = get_all_tracks_of_xml_in_one_dict('E:\\temp2\\sums\\notrit_csdummy_run1546.xml')
    # print(ret['track0']['triton'])
    sample_dict0 = {'track0':
        {'triton':
            {'preScan':
                {'dummyDev': {
                    'calls': {'required': 2, 'data': [0, 1], 'acquired': 2},
                    'random': {'required': 4, 'data': [0, 1, 2, 3], 'acquired': 4}}}}}}
    sample_dict1 = {'track0':
        {'triton':
            {'preScan':
                {'dummyDev': {
                    'calls': {'required': 2, 'data': [2], 'acquired': 1},
                    'random': {'required': 4, 'data': [4, 5, 6], 'acquired': 3}}}}}}
    sample_dict2 = {}
    sample_dict3 = {'track0':
        {'triton':
            {'preScan':
                {'dummyDev2': {
                    'calls': {'required': 2, 'data': [2], 'acquired': 1},
                    'random': {'required': 4, 'data': [4, 5, 6], 'acquired': 3}}}}}}
    sample_dict4 = {'track0':
        {'triton':
            {'preScan':
                {'dummyDev': {
                    'calls1': {'required': 2, 'data': [2], 'acquired': 1},
                    'random1': {'required': 4, 'data': [4, 5, 6], 'acquired': 3}}}}}}
    sample_dict5 = {'track0':
        {'triton':
            {'postScan':
                {'dummyDev': {
                    'calls1': {'required': 2, 'data': [2], 'acquired': 1},
                    'random1': {'required': 4, 'data': [4, 5, 6], 'acquired': 3}}}}}}
    np1 = np.array([1, 1])
    np2 = np.array([2, 2])
    dmm_sample_dict0 = {'dummy_somewhere': np1, 'dummy_else': np2}
    dmm_sample_dict1 = {'dummy_somewhere': np2, 'dummy_else': np1, 'more_dummy': np2}
    test_d0 = {'lala': [0, 1]}
    test_d1 = {'lala': [1, 2], 'blub': [5, 6]}

    deepupdate(sample_dict0, sample_dict1)
    deepupdate(sample_dict0, sample_dict2)
    deepupdate(sample_dict0, sample_dict3)
    deepupdate(sample_dict0, sample_dict4)
    deepupdate(sample_dict0, sample_dict5)
    realdict = {'track0': {'triton': {'duringScan': {'dummyDev': {'random': {'required': -1, 'data': [], 'acquired': 0},
                                                                  'calls': {'required': -1, 'data': [3],
                                                                            'acquired': 1}}}}}}
    deepupdate(dmm_sample_dict0, dmm_sample_dict1)
    print(dmm_sample_dict0)
    for dev, chs in sample_dict0['track0']['triton']['preScan'].items():
        for ch_name, ch_data in chs.items():
            # print(ch_data)
            ch_data['acquired'] = len(ch_data['data'])

    #print_dict_pretty(sample_dict0)