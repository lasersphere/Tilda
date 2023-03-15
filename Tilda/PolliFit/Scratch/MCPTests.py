"""
Created on 

@author: simkaufm

Module Description:
"""

import re
import numpy as np
from copy import deepcopy
import datetime

file = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\MCP_Files\\60Ni_no_protonTrigger_Run018.mcp'


def find_data_list_in_str(orig_str, obj_name_str, multiple_occurence=2, data_begin_str='<', data_end_str='>>'):
    l_ind = [m.start() for m in re.finditer(obj_name_str, orig_str)]
    if multiple_occurence >= 2:
        del l_ind[::multiple_occurence]  # every object is mentioned twice in mcp file
    ret = ''
    names = []
    for ind in l_ind:
        names.append(orig_str[orig_str.find(',', ind) + 1:orig_str.find(',', orig_str.find(',', ind) + 1)])
        ret += orig_str[
               orig_str.find(data_begin_str, ind) + len(data_begin_str):orig_str.find(data_end_str, ind)]
        ret += '\t'
    # ret = ret[:-1]
    ret = ret.split('\t')[:-1]
    ret2 = []
    for ind, vals_str in enumerate(ret):
        if '.' in vals_str:
            ret2.append(np.fromstring(ret[ind], dtype=float, sep=','))
        else:
            ret2.append(np.fromstring(ret[ind], dtype=int, sep=','))
    return ret2, names
    # ret = [float(s) if '.' in s or 'e' in s.lower() else int(s) for s in ret.split(',')]
    # return ret


def get_date(mcp_file_as_string):
    date = mcp_file_as_string[mcp_file_as_string.find('@<') + 2:
    mcp_file_as_string.find(',', mcp_file_as_string.find('@<'))]
    date_t = datetime.datetime.strptime(date, '%a %b %d %H:%M:%S %Y')
    new_fmt = date_t.strftime('%Y-%M-%d %H:%M:%S')
    return new_fmt


def get_nr_of_scans(mcp_file_as_string):
    ind = mcp_file_as_string.find('<<')  # will not work for multiple tracks like this
    ind2 = mcp_file_as_string.find('<', ind + 2)
    lis = mcp_file_as_string[ind:ind2].split(',')[1:-1]
    scans = int(lis[0])
    completed_scans = int(lis[1])
    steps = int(lis[2])
    return (scans, completed_scans, steps)

with open(file, 'r') as f:
    lines = str(f.read().replace('\n', '').replace('\"', ''))
    volts = find_data_list_in_str(lines, 'SiclReaderObj')
    acc_volt = np.mean(volts[0][volts[1].index('lan[A-34461A-06386]:inst0')])
    prema = np.mean(find_data_list_in_str(lines, 'PremaVoltageObj')[0])
    agilent = np.mean(volts[0][volts[1].index('lan[A-34461A-06287]:inst0')])
    d_prema_agilent = prema - agilent
    offset = np.mean([prema, agilent])
    scalers = find_data_list_in_str(lines, 'PM_SpectrumObj')
    line_volt = find_data_list_in_str(lines, 'LineVoltageSweepObj')
    print(line_volt)
    cts = scalers[0]
    activePmtList = scalers[1]
    nrOfSteps = cts[0].shape[0]
    nrOfScalers = len(cts)
    nrLoops = len(volts[0])

    nrScans = get_nr_of_scans(lines)
    print('nrOfScans', nrScans)
    # ind_tr = lines.find('>>>>')
    # nOfTracks = len([m.start() for m in re.finditer(lines, '>>>>>')])
    # print(nOfTracks)

    # print('second track: ', lines.find('>>>>'), lines[ind_tr - 10:ind_tr + 10])
    date = get_date(lines)
    print(date)

    print(acc_volt)
    print(offset, '+/-', d_prema_agilent)
    print(len(cts), [ct.shape[0] for ct in cts])
    # print(find_data_list_in_str(lines, 'PM_SpectrumObj'))
    # print(find_data_list_in_str(lines, 'SiclReaderObj'))
    # print(find_data_list_in_str(lines, 'PremaVoltageObj'))

f.close()
