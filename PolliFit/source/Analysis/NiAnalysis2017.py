"""
Created on 08.09.2017

@author: simkaufm

Module Description:  Analysis of the Nickel Data from COLLAPS taken on 07.09.2017 - 13.09.2017
"""

import math
import os
import sqlite3
import ast
from datetime import datetime, timedelta

import numpy as np

import Analyzer
import Physics
import Tools
from KingFitter import KingFitter
from InteractiveFit import InteractiveFit
import BatchFit
import MPLPlotter
import TildaTools as TiTs

''' working directory: '''

workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\' \
          'Measurement_and_Analysis_Simon\\Ni_workspace2017\\Ni_2017'

datafolder = os.path.join(workdir, 'sums')

db = os.path.join(workdir, 'Ni_2017.sqlite')

runs = ['Voigt', 'AsymExpVoigt', 'AsymVoigtFree', '2016Experiment']

isotopes = ['%s_Ni' % i for i in range(58, 71)]
isotopes.remove('69_Ni')
odd_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2]
even_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2 == 0]
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']

dif_doppl = Physics.diffDoppler(850344066.10401, 40000, 60)
print('diff doppler factor 60Ni', dif_doppl)

''' literature IS  '''
# for the 3d9(2D)4s  	 3D 3  -> 3d9(2D)4p  	 3P° 2 @352.454nm transition
# A. Steudel measured some isotop extrapolated_shifts:
# units are: mK = 10 ** -3 cm ** -1
iso_sh = {'58-60': (16.94, 0.09), '60-62': (16.91, 0.12), '62-64': (17.01, 0.26),
          '60-61': (9.16, 0.10), '61-62': (7.55, 0.12), '58-62': (34.01, 0.15), '58-64': (51.12, 0.31)}
# convert this to frequency/MHz
iso_sh_freq = {}
for key, val in iso_sh.items():
    iso_sh_freq[key] = (round(Physics.freqFromWavenumber(val[0] * 10 ** -3), 2),
                        round(Physics.freqFromWavenumber(val[1] * 10 ** -3), 2))

# 64_Ni has not been measured directly to 60_Ni, so both possible
# paths are taken into account and the weighted average is taken.
is_64_ni = [iso_sh_freq['60-62'][0] + iso_sh_freq['62-64'][0], - iso_sh_freq['58-60'][0] + iso_sh_freq['58-64'][0]]
err_is_64_ni = [round(math.sqrt(iso_sh_freq['62-64'][1] ** 2 + iso_sh_freq['60-62'][1] ** 2), 2),
                round(math.sqrt(iso_sh_freq['58-60'][1] ** 2 + iso_sh_freq['58-64'][1] ** 2), 2)]
mean_is_64 = Analyzer.weightedAverage(is_64_ni, err_is_64_ni)
print('isotope shifts for 64_ni', is_64_ni, err_is_64_ni)
print('mean:', mean_is_64)
#
literature_shifts = {
    '58_Ni': (-1 * iso_sh_freq['58-60'][0], iso_sh_freq['58-60'][1]),
    '60_Ni': (0, 0),
    '61_Ni': (iso_sh_freq['60-61'][0], iso_sh_freq['60-61'][1]),
    '62_Ni': (iso_sh_freq['60-62'][0], iso_sh_freq['60-62'][1]),
    '64_Ni': (round(mean_is_64[0], 2), round(mean_is_64[1], 2))
}
print('literatur shifts from A. Steudel (1980) in MHz:')
[print(key, val[0], val[1]) for key, val in sorted(literature_shifts.items())]
print(literature_shifts)

# results from last year with statistical error only
last_year_shifts = {
    '58_Ni': (-509.9, 0.7),
    '60_Ni': (0, 0),
    '61_Ni': (283.6, 1.1),
    '62_Ni': (505.6, 0.3),
    '64_Ni': (1029.6, 0.4)
}

# MPLPlotter.plot_par_from_combined(db, runs, list(sorted(literature_shifts.keys())), 'shift',
#                                   literature_dict=literature_shifts, plot_runs_seperate=True,
#                                   literature_name='A. Steudel (1980)',
#                                   show_pl=True)

''' Files: '''
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(''' SELECT date, file, type FROM Files ORDER BY date ''')
ret = cur.fetchall()
con.close()

files_to_correct_laser_freq = []
file_list = []

for f_date_tuple in ret:
    run_num = f_date_tuple[1].split('.')[0][-3:]
    file_size = os.path.getsize(os.path.join(datafolder, f_date_tuple[1]))
    smaller_then_2kb = 'deleted, stopped before saving' if file_size <= 2000 else ''
    print('%s\t%s\t%s\t%s' % (run_num, f_date_tuple[0], f_date_tuple[1], smaller_then_2kb))
    file_list.append((run_num, f_date_tuple[0], f_date_tuple[1], f_date_tuple[2]))
    if 34 < int(run_num) < 85:
        files_to_correct_laser_freq.append(f_date_tuple[1])
#
# # correct laser frequency in some files:
# print(files_to_correct_laser_freq)
#
# for f in files_to_correct_laser_freq:
#     print('changing laserfreq in %s to 14198.76240 cm-1' % f)
#     f_path = os.path.join(datafolder, f)
#     root = TiTs.load_xml(f_path)
#     laser_freq = root.find('header').find('laserFreq')
#     laser_freq.text = str(14198.76240)
#     TiTs.save_xml(root, f_path)
# # (val, statErr, rChi)


''' Kepco Fits:  '''
# Batchfitting done in gui...
kepco_res = {}
kepco_runs = ['kepco_pxi', 'kepco_agilent']
kepco_pars = ['m', 'b']

# for run in kepco_runs:
#     kepco_res[run] = {}
#     for par in kepco_pars:
#         con = sqlite3.connect(db)
#         cur = con.cursor()
#         cur.execute(''' SELECT val, statErr, rChi FROM Combined WHERE run = ? AND parname = ? ''', (run, par))
#         ret = cur.fetchall()
#         con.close()
#         kepco_res[run][par] = ret[0]
#
# print(kepco_res)
# slope_vals = [kepco_res[r]['m'][0] for r in kepco_runs]
# slope_errs = [kepco_res[r]['m'][1] * kepco_res[r]['m'][2] for r in kepco_runs]
# kepco_slope, kepco_slope_err, kepco_slope_rChi = Analyzer.weightedAverage(slope_vals, slope_errs)
# print('kepco slope: %.6f(%.0f)' % (kepco_slope, kepco_slope_err * 10 ** 6))
#
# offset_vals = [kepco_res[r]['b'][0] for r in kepco_runs]
# offset_errs = [kepco_res[r]['b'][1] * kepco_res[r]['b'][2] for r in kepco_runs]
# kepco_offset, kepco_offset_err, kepco_offset_rChi = Analyzer.weightedAverage(offset_vals, offset_errs)
# print('kepco offset: %.6f(%.0f)' % (kepco_offset, kepco_offset_err * 10 ** 6))


''' BatchFitting '''

run_hot_cec = 'AsymVoigtHotCec'
run_numbermax = 81

fits = {}
files_w_err = {}

for iso in isotopes:
    iso_list = []
    for run_num, date_str, file_str, iso_str in file_list:
        if iso_str == iso and int(run_num) <= run_numbermax:
            iso_list.append(file_str)
    print('starting Batchfit on Files: %s' % str(iso_list))
    fits_iso, files_w_err_iso = BatchFit.batchFit(iso_list, db, run_hot_cec)
    fits[iso] = fits_iso
    files_w_err[iso] = files_w_err_iso


print('Batchfit finished, files with error: ')
TiTs.print_dict_pretty(files_w_err)









# Tools.createDB(db)