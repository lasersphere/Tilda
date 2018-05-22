"""
Created on 

@author: simkaufm

Module Description:  Script for testing the "offset displaying" similiar to the displaying of the isotope shift,
 here used for the 2016 & 2017 Ni-data
 Additionally two files of the 2017 Ni run for 67Ni were added, which required a lot of handwork,
  since the voltage and time axis was changed in between the two files run243 and run248
"""

import Analyzer
import TildaTools as TiTs
import os
import time
import logging
import sys

import numpy as np

app_log = logging.getLogger()
app_log.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
# ch.setFormatter(log_formatter)
app_log.addHandler(ch)

workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'
workdir17 = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\' \
          'Measurement_and_Analysis_Simon\\Ni_workspace2017\\Ni_2017'

mcp_file_folder = os.path.join(workdir, 'Ni_April2016_mcp')
tipa_file_folder = os.path.join(workdir, 'TiPaData')
tipa_files = sorted([file for file in os.listdir(tipa_file_folder) if file.endswith('.xml')])
tipa_files = [os.path.join(tipa_file_folder, file) for file in tipa_files]

data_folder17 = os.path.join(workdir17, 'sums')

db = os.path.join(workdir, 'Ni_workspace.sqlite')
db17 = os.path.join(workdir17, 'Ni_2017.sqlite')

run = 'wide_gate_asym'
run17 = 'AsymVoigt'

isotopes = ['%s_Ni' % i for i in range(58, 70)]
isotopes.remove('69_Ni')
isotopes.remove('67_Ni')
isotopes.remove('60_Ni')
# isotopes.remove('70_Ni')
isotopes.remove('59_Ni')  # not measured in 2017
isotopes.remove('63_Ni')  # not measured in 2017
offset_dict = {}
isotopes = ['67_Ni']
for iso in isotopes:
    offset_dict[iso] = Analyzer.combineShiftOffsetPerBunchDisplay(iso, run, db)
#
# for iso, offset in offset_dict.items():
#     offs, errs = offset
#     for off in offs:
#         print(iso, off)
    # print('errors now')
    # for err in errs:
    #     print(iso, errs)
raise Exception

file243 = '67_Ni_trs_run243.xml'
file248 = '67_Ni_trs_run248.xml'

# from InteractiveFit import InteractiveFit
#
# fit243 = InteractiveFit(run243, db17, run17, block=False)
# fit243.fit(show=True)
#
# fit248 = InteractiveFit(run248, db17, run17, block=False, clear_plot=False, data_fmt='b.')
# fit248.fit(clear_plot=True, data_fmt='b.')

# will try to add both files now
from Measurement.MeasLoad import load

file243_full = os.path.join(data_folder17, file243)
file248_full = os.path.join(data_folder17, file248)

meas243 = load(file243_full, db17, x_as_voltage=True, softw_gates=(db17, run17), raw=True)
meas248 = load(file248_full, db17, x_as_voltage=True, softw_gates=(db17, run17), raw=True)

# found via looking at the x axis:
to_cut = [
    ((0, 41), (20, 61)),  # tr0 ((meas243_start, meas243_stop), (meas248_start, meas248_stop)),
    ((0, 41), (0, 41)),  # tr1
    ((0, 26), (10, 36))  # tr2
    ]

for i in range(0, 3):  # 3 tracks
    meas243.x[i] = meas243.x[i][to_cut[i][0][0]:to_cut[i][0][1]]

    cts_copy = [np.zeros_like(meas243.x[i]) for sc_i in range(0, 4)]
    err_copy = [np.zeros_like(meas243.x[i]) for sc_i in range(0, 4)]
    for sc in range(0, 4):
        # for each scaler cut the uneccessary steps
        cts_copy[sc] = meas243.cts[i][sc][to_cut[i][0][0]:to_cut[i][0][1]]
        err_copy[sc] = meas243.err[i][sc][to_cut[i][0][0]:to_cut[i][0][1]]
    meas243.cts[i] = cts_copy  # replace whole track
    meas243.err[i] = err_copy
    # cut out all cts that are outside these steps
    valid_zf_cts_243 = np.where(
        (to_cut[i][0][0] <= meas243.time_res_zf[i]['step']) & (meas243.time_res_zf[i]['step'] < to_cut[i][0][1]))
    meas243.time_res_zf[i] = meas243.time_res_zf[i][valid_zf_cts_243]
    meas243.time_res_zf[i]['step'] -= to_cut[i][0][0]  # shift all step numbers by the beginning

    meas248.x[i] = meas248.x[i][to_cut[i][1][0]:to_cut[i][1][1]]
    cts248_copy = [np.zeros_like(meas248.x[i]) for sc_i in range(0, 4)]
    err248_copy = [np.zeros_like(meas248.x[i]) for sc_i in range(0, 4)]
    for sc in range(0, 4):
        cts248_copy[sc] = meas248.cts[i][sc][to_cut[i][1][0]:to_cut[i][1][1]]
        err248_copy[sc] = meas248.err[i][sc][to_cut[i][1][0]:to_cut[i][1][1]]
    meas248.cts[i] = cts248_copy
    meas248.err[i] = err248_copy

    # cut out all cts that are outside these steps
    valid_zf_cts_248 = np.where(
        (to_cut[i][1][0] <= meas248.time_res_zf[i]['step']) & (meas248.time_res_zf[i]['step'] < to_cut[i][1][1]))
    meas248.time_res_zf[i] = meas248.time_res_zf[i][valid_zf_cts_248]
    meas248.time_res_zf[i]['step'] -= to_cut[i][1][0]  # shift all step numbers by the beginning

    # timing:
    # delay in 243: 4500 & 2000 Bins, delay in 248: 3000 & 4000 Bins
    meas243.time_res_zf[i]['time'] += 1500  # add the additional delay to the timestamps

TiTs.add_specdata(meas248, [(1, meas243)], save_dir=data_folder17, filename='67_Ni_sum_run243_and_run248')
# additionally edited the following by hand:
# nOfCompletedSteps: tr0: 1640, tr1: 1640, tr2: 11492
# laserfreq: 14198.7574
# isotopeStartTime 2017-09-11 17:10:00

