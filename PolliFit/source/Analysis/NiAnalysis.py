"""
Created on 

@author: simkaufm

Module Description:  Analysis of the Nickel Data from COLLAPS taken on 28.04.-03.05.2016
"""

import math
import os
import sqlite3

import numpy as np

import Analyzer
import BatchFit
import Physics
import Tools
from KingFitter import KingFitter

''' working directory: '''

workdir = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder = os.path.join(workdir, 'Ni_April2016_mcp')

db = os.path.join(workdir, 'Ni_workspace.sqlite')

runs = ['narrow_gate', 'wide_gate']
runs = [runs[0]]

isotopes = ['%s_Ni' % i for i in range(58, 71)]
isotopes.remove('69_Ni')
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']

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
print(mean_is_64)
#
literature_shifts = {
    '58_Ni': (-1 * iso_sh_freq['58-60'][0], iso_sh_freq['58-60'][1]),
    '60_Ni': (0, 0),
    '61_Ni': (iso_sh_freq['60-61'][0], iso_sh_freq['60-61'][1]),
    '62_Ni': (iso_sh_freq['60-62'][0], iso_sh_freq['60-62'][1]),
    '64_Ni': (mean_is_64[0], mean_is_64[1])
}
print('literatur shifts from A. Steudel (1980) in MHz:')
[print(key, val[0], val[1]) for key, val in sorted(literature_shifts.items())]


''' literature radii '''

# from Landolt-Börnstein - Group I Elementary Particles, Nuclei and Atoms, Fricke 2004
# http://materials.springer.com/lb/docs/sm_lbs_978-3-540-45555-4_30
# Root mean square nuclear charge radii <r^2>^{1/2}_{0µe}
lit_radii = {
    '58_Ni': (3.770, 0.004),
    '60_Ni': (3.806, 0.002),
    '61_Ni': (3.818, 0.003),
    '62_Ni': (3.836, 0.003),
    '64_Ni': (3.853, 0.003)
}

delta_lit_radii = {iso: [
    lit_vals[0] - lit_radii['60_Ni'][0],
    np.sqrt(lit_vals[1] ** 2 + lit_radii['60_Ni'][1] ** 2)]
                   for iso, lit_vals in sorted(lit_radii.items())}
delta_lit_radii.pop('60_Ni')
print('iso\t<r^2>^{1/2}_{0µe}\t\Delta<r^2>^{1/2}_{0µe}\t<r^2>^{1/2}_{0µe}(A-A_{60})\t\Delta <r^2>^{1/2}_{0µe}(A-A_{60})')
for iso, radi in sorted(lit_radii.items()):
    dif = delta_lit_radii.get(iso, (0, 0))
    print('%s\t%.3f\t%.3f\t%.5f\t%.5f' % (iso, radi[0], radi[1], dif[0], dif[1]))


''' crawling '''

# Tools.crawl(db, 'Ni_April2016_mcp')

# ''' laser wavelength: '''
# wavenum = 28393.0  # cm-1
# freq = Physics.freqFromWavenumber(wavenum)
# # freq -= 1256.32701
# print(freq, Physics.wavenumber(freq), 0.5 * Physics.wavenumber(freq))
#
# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute('''UPDATE Files SET laserFreq = ? ''', (freq, ))
# con.commit()
# con.close()
#
# ''' kepco scan results: '''
#
# line_mult = 0.050415562
# line_offset = 1.75 * 10 ** -10
# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute('''UPDATE Files SET lineMult = ?, lineOffset = ?''', (line_mult, line_offset))
# con.commit()
# con.close()
#
# ''' volt div ratio: '''
# volt_div_ratio = "{'accVolt': 1003.8, 'offset': 1003.7}"
# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute('''UPDATE Files SET voltDivRatio = ?''', (volt_div_ratio, ))
# con.commit()
# con.close()

''' diff doppler 60Ni 30kV'''
diffdopp60 = Physics.diffDoppler(850343019.777062, 30000, 60)  # 14.6842867127 MHz/V

''' transition wavelenght: '''
# observed_wavenum = 28364.39  # cm-1  observed wavenum from NIST, mass is unclear.
transition_freq = 850342663.9020721  # final value, observed from voltage calibration
# # transition_freq = Physics.freqFromWavenumber(observed_wavenum)
# print('transition frequency: %s ' % transition_freq)
#
# transition_freq += 1256.32701  # correction from fitting the 60_Ni references
#
# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute('''UPDATE Lines SET frequency = ?''', (transition_freq,))
# con.commit()
# con.close()

''' Batch fits '''
selected_isos = ['65_Ni']

# create static list because selection might be necessary as removing cw files.
ni58_files = Tools.fileList(db, isotopes[0])
ni58_bunch_files = [each for each in ni58_files if 'cw' not in each]


ni60_files = Tools.fileList(db, isotopes[2])
ni60_cont_files = [each for each in ni60_files if 'contin' in each]
ni60_bunch_files = [each for each in ni60_files if 'contin' not in each]
# print(ni60_bunch_files)

ni61_files = Tools.fileList(db, isotopes[3])

files_dict = {iso: Tools.fileList(db, iso) for iso in isotopes}
files_dict[isotopes[0]] = [each for each in files_dict[isotopes[0]] if 'cw' not in each]
files_dict[isotopes[2]] = [each for each in files_dict[isotopes[2]] if 'contin' not in each or '206'
                           in each or '208' in each or '209' in each]
# -> some files accidentially named continous
# print('fielList: %s ' % files_dict[isotopes[2]])
# BatchFit.batchFit(ni58_bunch_files, db, runs[0])
# Analyzer.combineRes(isotopes[2], 'center', runs[0], db)
# stables = ['67_Ni']
# pars = ['center', 'Al', 'Bl', 'Au', 'Bu', 'Int0']
# for iso in selected_isos:
#     files = files_dict[iso]
#     for run in runs:
#         # fits = BatchFit.batchFit(files, db, run)
#         for par in pars:
#             Analyzer.combineRes(iso, par, run, db)

''' isotope shift '''
# get all current configs:
# print('run \t iso \t val \t statErr \t rChi')
# for iso in selected_isos:
#     for run in runs:
#         con = sqlite3.connect(db)
#         cur = con.cursor()
#         cur.execute('''SELECT config, val, statErr, rChi FROM Combined WHERE iso = ? AND run = ? AND parname = ? ''',
#                     (iso, run, 'shift'))
#         data = cur.fetchall()
#         con.close()
#         if len(data):
#             config, val, statErr, rChi = data[0]
#             print('%s \t %s \t %s \t %s \t %s \n %s' % (run, iso, val, statErr, rChi, config))
#             print('\n')

# for iso in selected_isos:
#     if iso != '60_Ni':
#         for run in runs:
#             con = sqlite3.connect(db)
#             cur = con.cursor()
#             cur.execute(''' UPDATE Combined SET statErrForm = ? ''', ('applyChi(err, rChi)', ))
#             con.commit()
#             con.close()
#             Analyzer.combineShift(iso, run, db, show_plot=True)


''' Divider Ratio Determination '''
acc_div_start = 1000.05
offset_div_start = 1000
configs = {}
# get the relevant files which need to be fitted in the following:
# div_ratio_relevant_stable_files = {}
# div_ratio_relevant_stable_files['60_Ni'] = []
# for iso, cfg in sorted(configs.items()):
#     div_ratio_relevant_stable_files[iso] = []
#     for each in cfg:
#         [div_ratio_relevant_stable_files['60_Ni'].append(file) for file in each[0] if
#          file not in div_ratio_relevant_stable_files['60_Ni']]
#         [div_ratio_relevant_stable_files[iso].append(file) for file in each[1]]
#         [div_ratio_relevant_stable_files['60_Ni'].append(file) for file in each[2] if
#          file not in div_ratio_relevant_stable_files['60_Ni']]
# div_ratio_relevant_stable_files['60_Ni'] = sorted(div_ratio_relevant_stable_files['60_Ni'])

# div_ratio_relevant_stable_files.pop('58_Ni')  # due to deviation of 58_Ni, do not fit this one.

# print('number of resonances that will be fitted: %s' %
#       float(sum([len(val) for key, val in div_ratio_relevant_stable_files.items()])))


# Analyzer.combineRes('60_Ni', 'sigma', runs[0], db, print_extracted=True, show_plot=True)


def chi_square_finder(acc_dev_list, off_dev_list):
    offset_div_ratios = [[]]
    acc_ratios = []
    run_chi_finder = runs[0]
    fit_res = [[]]
    chisquares = [[]]
    acc_vol_ratio_index = 0
    for acc in acc_dev_list:
        current_acc_div = acc_div_start + acc / 100
        freq = -442.4 * acc / 100 - 9.6  # value found by playing with gui
        freq += transition_freq
        # freq = transition_freq
        print('setting transition Frequency to: %s ' % freq)
        acc_ratios.append(current_acc_div)

        for off in off_dev_list:

            freq_correction = 17.82 * off / 100 - 9.536  # determined for the region around acc_div = 1000.05
            new_freq = freq + freq_correction

            curent_off_div = offset_div_start + off / 100

            con = sqlite3.connect(db)
            cur = con.cursor()
            divratio = str({'accVolt': current_acc_div, 'offset': curent_off_div})
            cur.execute('''UPDATE Files SET voltDivRatio = ? ''', (divratio,))
            cur.execute('''UPDATE Lines SET frequency = ?''', (new_freq,))
            con.commit()
            con.close()

            # Batchfitting:
            fitres = [(iso, run_chi_finder, BatchFit.batchFit(files, db, run_chi_finder)[1])
                      for iso, files in sorted(div_ratio_relevant_stable_files.items())]

            # combineRes only when happy with voltdivratio, otherwise no use...
            # [[Analyzer.combineRes(iso, par, run, db) for iso in stables] for par in pars]
            try:
                shifts = {iso: Analyzer.combineShift(iso, run_chi_finder, db) for iso in stables if iso not in ['58_Ni', '60_Ni']}
            except Exception as e:
                shifts = {}
                print(e)

            # calc red. Chi ** 2:
            chisq = 0
            for iso, shift_tuple in shifts.items():
                iso_shift_err = np.sqrt(np.square(shift_tuple[3]) + np.square(literature_shifts[iso][1]))
                iso_chisq = np.square((shift_tuple[2] - literature_shifts[iso][0]) / iso_shift_err)
                print('iso: %s chi sq: %s shift tuple: %s ' % (iso, iso_chisq, shift_tuple))
                chisq += iso_chisq
            chisquares[acc_vol_ratio_index].append(float(chisq))
            fit_res[acc_vol_ratio_index].append(fitres)
            offset_div_ratios[acc_vol_ratio_index].append(curent_off_div)

        acc_vol_ratio_index += 1
        chisquares.append([])
        fit_res.append([])
        offset_div_ratios.append([])
    chisquares = chisquares[:-1]  # delete last empty list in order not to confuse.
    fit_res = fit_res[:-1]
    offset_div_ratios = offset_div_ratios[:-1]
    print('acceleration voltage divider ratios: \n %s ' % str(acc_ratios))
    print('offset voltage divider ratios: \n %s ' % str(offset_div_ratios))
    print('Chi^2 are: \n %s ' % str(chisquares))

    print(fit_res)

    print('the following files failed during BatchFit: \n')
    for acc_volt_ind, each in enumerate(fit_res):
        print('for acc volt div ratio: %s' % acc_ratios[acc_volt_ind])
        for offset_volt_ind, inner_each in enumerate(each):
            [print(fit_res_tpl) for fit_res_tpl in inner_each if len(fit_res_tpl[2])]
    print('acc\toff\tchisquare')
    for acc_ind, acc_rat in enumerate(acc_ratios):
        for off_ind, off_rat in enumerate(offset_div_ratios[acc_ind]):
            print(('%s\t%s\t%s' % (acc_rat, off_rat, chisquares[acc_ind][off_ind])).replace('.', ','))
    return acc_ratios, offset_div_ratios, chisquares


# acc_ratios, offset_div_ratios, chisquares = chi_square_finder([375], [370])
#
print('plotting now')
try:
    # isotopes.remove('69_Ni')
    # # isotopes.remove('67_Ni')
    # print(isotopes)
    # files = .extract_shifts(runs, isotopes=isotopes)
    # print('extracted shifts are are: % s' % files)
    #
    # print('iso\tshift [MHz]\tstatErr [Mhz]\trChi')
    # [print('%s\t%s\t%s\t%s' % (key, val[0], val[1], val[2])) for key, val in sorted(files[runs[0]].items())]
    #
    # print('\n\nfor Excel: \n\n')
    # print('iso\tshift [MHz]\tstatErr [Mhz]\trChi')
    # for key, val in sorted(files[runs[0]].items()):
    #     out_str = '%s\t%s\t%s\t%s' % (key, val[0], val[1], val[2])
    #     out_str = out_str.replace('.', ',')
    #     print(out_str)
    # print('\n\n\niso\tAu\td_Au\tAl\td_Al\tAu/Al\td_Au/Al')
    # a_fac_runs = [runs[0], 'narrow_gate_67_Ni']
    # al = extract_shifts(a_fac_runs, isotopes, 'Al')
    # au = extract_shifts(a_fac_runs, isotopes, 'Au')
    # ratios = []
    # d_ratios = []
    # for run in a_fac_runs:
    #     for iso, a_low in sorted(al[run].items()):
    #         if a_low[0]:
    #             a_up = au[run][iso]
    #             ratio = a_up[0] / a_low[0]
    #             delta_ratio = np.sqrt(
    #                 (a_up[1]/a_low[0]) ** 2 + (a_up[0] * a_low[1] / (a_low[0] ** 2)) ** 2
    #             )
    #             ratios.append(ratio)
    #             d_ratios.append(delta_ratio)
    #             print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % (
    #                 iso, a_up[0], a_up[1], a_low[0], a_low[1], ratio, delta_ratio))
    # average, errorprop, rChi = Analyzer.weightedAverage(ratios, d_ratios)
    # print('\nAverage Au/Al: %.5f +/- %.5f \t rChi: %.5f' % (average, errorprop, rChi))
    # MPLPlotter.plt.errorbar(range(59, 69, 2), ratios, d_ratios)
    # MPLPlotter.get_current_axes().set_xlabel('mass')
    # MPLPlotter.get_current_axes().set_ylabel('A_upper/A_lower')
    # literature_shifts = {iso: (0, 0) for iso in isotopes}
    # plot_par_from_combined(['narrow_gate'], files)
    # MPLPlotter.show(True)
    # MPLPlotter.plot_par_from_combined(
    #     db,
    #     -1, isotopes,
    #     'shift', plot_runs_seperate=False
    # )
    pass
except Exception as e:
    print('plotting did not work, error is: %s' % e)


# print('------------------- Done -----------------')
# winsound.Beep(2500, 500)

# print('\a')


''' Fit on certain Files '''
# searchterm = 'Run167'
# certain_file = [file for file in ni60_files if searchterm in file][0]
# fit = InteractiveFit.InteractiveFit(certain_file, db, runs[0], block=True, x_as_voltage=True)
# fit.fit()


''' results: '''
# acc_divs_result = acc_ratios
# off_divs_result = offset_div_ratios[0]
# chisquares_result = chisquares
#
# import PyQtGraphPlotter as PGplt
# from PyQt5 import QtWidgets
# import sys
#
# x_range = (float(np.min(acc_divs_result)), np.max(acc_divs_result))
# x_scale = np.mean(np.ediff1d(acc_divs_result))
# y_range = (float(np.min(off_divs_result)), np.max(off_divs_result))
# y_scale = np.mean(np.ediff1d(off_divs_result))
#
# chisquares_result = np.array(chisquares_result)
#
# app = QtWidgets.QApplication(sys.argv)
# main_win = QtWidgets.QMainWindow()
# widg, plt_item = PGplt.create_image_view('acc_volt_div_ratio', 'offset_div_ratio')
# widg.setImage(chisquares_result,
#               pos=[x_range[0] - abs(0.5 * x_scale),
#                    y_range[0] - abs(0.5 * y_scale)],
#               scale=[x_scale, y_scale])
# try:
#     main_win.setCentralWidget(widg)
# except Exception as e:
#     print(e)
# main_win.show()
#
# app.exec()

''' King Plot Analysis '''
# delta_lit_radii.pop('61_Ni')
king = KingFitter(db, showing=True, litvals=delta_lit_radii)
run = -1
# isotopes = sorted(delta_lit_radii.keys())
king.kingFit(alpha=-49, findBestAlpha=True, run=run)
king.calcChargeRadii(isotopes=isotopes, run=run)
