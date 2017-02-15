"""
Created on 

@author: simkaufm

Module Description:  Analysis of the Nickel Data from COLLAPS taken on 28.04.-03.05.2016
"""

import math
import os
import sqlite3
import ast

import numpy as np

import Analyzer
import Physics
import Tools
from KingFitter import KingFitter
from InteractiveFit import InteractiveFit
import BatchFit
import MPLPlotter

''' working directory: '''

workdir = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder = os.path.join(workdir, 'Ni_April2016_mcp')

db = os.path.join(workdir, 'Ni_workspace.sqlite')

runs = ['narrow_gate', 'wide_gate', 'narrow_gate_67_Ni']
runs = [runs[0]]

isotopes = ['%s_Ni' % i for i in range(58, 71)]
isotopes.remove('69_Ni')
odd_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2]
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
# print('literatur shifts from A. Steudel (1980) in MHz:')
# [print(key, val[0], val[1]) for key, val in sorted(literature_shifts.items())]
print(literature_shifts)

''' literature radii '''

# from Landolt-Börnstein - Group I Elementary Particles, Nuclei and Atoms, Fricke 2004
# http://materials.springer.com/lb/docs/sm_lbs_978-3-540-45555-4_30
# Root mean square nuclear charge radii <r^2>^{1/2}_{0µe}
# lit_radii = {
#     '58_Ni': (3.770, 0.004),
#     '60_Ni': (3.806, 0.002),
#     '61_Ni': (3.818, 0.003),
#     '62_Ni': (3.836, 0.003),
#     '64_Ni': (3.853, 0.003)
# }   # have ben calculated more accurately below

baret_radii_lit = {
    '58_Ni': (4.8386, 0.0009 + 0.0019),
    '60_Ni': (4.8865, 0.0008 + 0.002),
    '61_Ni': (4.9005, 0.001 + 0.0017),
    '62_Ni': (4.9242, 0.0009 + 0.002),
    '64_Ni': (4.9481, 0.0009 + 0.0019)
}

v2_lit = {
    '58_Ni': 1.283517,
    '60_Ni': 1.283944,
    '61_Ni': 1.283895,
    '62_Ni': 1.283845,
    '64_Ni': 1.284133
}

lit_radii_calc = {iso: (val[0]/v2_lit[iso], val[1])for iso, val in sorted(baret_radii_lit.items())}

# using the more precise values by the self calculated one:
lit_radii = lit_radii_calc


delta_lit_radii = {iso: [
    lit_vals[0] ** 2 - lit_radii['60_Ni'][0] ** 2,
    np.sqrt(lit_vals[1] ** 2 + lit_radii['60_Ni'][1] ** 2)]
                   for iso, lit_vals in sorted(lit_radii.items())}
delta_lit_radii.pop('60_Ni')
# print('iso\t<r^2>^{1/2}_{0µe}\t\Delta<r^2>^{1/2}_{0µe}\t<r^2>^{1/2}_{0µe}(A-A_{60})\t\Delta <r^2>^{1/2}_{0µe}(A-A_{60})')
# for iso, radi in sorted(lit_radii.items()):
#     dif = delta_lit_radii.get(iso, (0, 0))
#     print('%s\t%.3f\t%.3f\t%.5f\t%.5f' % (iso, radi[0], radi[1], dif[0], dif[1]))

''' Moments '''
''' Quadrupole Moments '''
# literature values from PHYSICAL REVIEW VOLUME 170, NUM HER 1 5 JUNE 1968
# Hyperfine-Structure Studies of Ni", and the Nuclear Ground-State
# Electric Quadrupole Moment*
# W. J. CHILDs AND L. S. 600DMAN
# Argonne Eationa/ Laboratory, Argonne, Illinois

q_literature_61_Ni = 0.162  # barn
d_q_literature_61_Ni = 0.015  # barn

# 3d9(2D)4s  	 3D 3
b_lower_lit = -102.979  # MHz
d_b_lower_lit = 0.016  # MHz

# e_Vzz value from this: e_Vzz = B / Q
e_Vzz_lower = b_lower_lit / q_literature_61_Ni
d_e_Vzz_lower = np.sqrt(
    (e_Vzz_lower / b_lower_lit * d_b_lower_lit) ** 2 +
    (e_Vzz_lower / q_literature_61_Ni * d_q_literature_61_Ni) ** 2)
print(e_Vzz_lower, d_e_Vzz_lower)

# for the upper state  3d9(2D)4p  	 3P° 2, no b factor was measured
# therefore i will need to get the e_Vzz_upper from my results on 61_Ni and the q_literature_61_Ni
b_upper_exp = -50.5033931843  # MHz
d_b_upper_exp = 1.421  # Mhz

e_Vzz_upper = b_upper_exp / q_literature_61_Ni
d_e_Vzz_upper = np.sqrt(
    (e_Vzz_upper / b_upper_exp * d_b_upper_exp) ** 2 +
    (e_Vzz_upper / q_literature_61_Ni * d_q_literature_61_Ni) ** 2)

print(e_Vzz_upper, d_e_Vzz_upper)

def quadrupol_moment(b, d_stat_b, d_syst_b, upper=True):
    if b:
        e_vzz = e_Vzz_lower
        d_e_vzz = d_e_Vzz_lower
        if upper:
            e_vzz = e_Vzz_upper
            d_e_vzz = d_e_Vzz_upper
        q = b / e_vzz
        d_stat_q = d_stat_b / e_vzz
        d_syst_q = np.sqrt(
            (b / e_vzz * d_syst_b) ** 2 +
            (b / (e_vzz ** 2) * d_e_vzz) ** 2
        )
        q_print = '%.3f(%.0f)[%.0f]' % (q, d_stat_q * 1000, d_syst_q * 1000)
        d_total = np.sqrt(d_stat_q ** 2 + d_syst_q ** 2)
        return q, d_stat_q, d_syst_q, d_total, q_print
    else:
        return 0, 0, 0, 0, '0.000(0)[0]'

''' magnetic moments '''
# µ = A µ_Ref / A_Ref * I / I_Ref
# µ_ref from:
# TABLE OF NUCLEAR MAGNETIC DIPOLE
# AND ELECTRIC QUADRUPOLE MOMENTS
# N.J. Stone
# Oxford Physics, Clarendon Laboratory, Parks Road, Oxford U.K. OX1 3PU and
# Department of Physics and Astronomy, University of Tennessee, Knoxville, USA, TN 37996-1200
# February 2014
# page 36

mu_ref = -0.75002  # nm -> nuclear magneton
d_mu_ref = 0.00004  # nm
i_ref = 3 / 2

# A_Ref_lower was taken from:
# PHYSICAL REVIEW VOLUME 170, NUM HER 1 5 JUNE 1968
# Hyperfine-Structure Studies of Ni", and the Nuclear Ground-State
# Electric Quadrupole Moment*
# W. J. CHILDs AND L. S. 600DMAN
# Argonne Eationa/ Laboratory, Argonne, Illinois
# (Received 31 January 1968)
# table III.

a_low_ref = -454.974
d_a_low_ref = 0.003


def magnetic_moment(a_lower, d_stat_a_lower, d_syst_a_lower, nucl_spin):
    mu = a_lower * mu_ref / a_low_ref * nucl_spin / i_ref
    d_stat_mu = d_stat_a_lower * mu_ref / a_low_ref * nucl_spin / i_ref
    d_syst_mu = np.sqrt(
        (mu_ref / a_low_ref * nucl_spin / i_ref * d_syst_a_lower) ** 2 +
        (a_lower / a_low_ref * nucl_spin / i_ref * d_mu_ref) ** 2 +
        (a_lower * mu_ref / (a_low_ref ** 2) * nucl_spin / i_ref * d_a_low_ref) ** 2
    )
    # print(mu, d_stat_mu, d_syst_mu)
    d_total = np.sqrt(d_stat_mu ** 2 + d_stat_mu ** 2)
    print_val = '%.4f(%.0f)[%.0f]' % (mu, d_stat_mu * 10000, d_syst_mu * 10000)
    return mu, d_stat_mu, d_syst_mu, d_total, print_val


# Schmidt values as in R. Neugart and G. Neyens, Nuclear Moments, Lecture Notes in Physics 700 (2006),
# 135–189 -> Page 139:
def mu_schmidt(I, l, proton, g_l=0, g_s=-3.826):
    l_plus = I == l + 0.5
    # g_l = 0
    # g_s = -3.826
    if proton:
        g_l = 1
        g_s = 5.587
    if l_plus:
        mu_I = ((I - 0.5) * g_l + 0.5 * g_s)
    else:
        mu_I = I / (I + 1) * ((I + 1.5) * g_l - 0.5 * g_s)
    return mu_I

levels = [('2p 3/2', 1, 1.5), ('1f 5/2', 3, 2.5), ('2p 1/2', 1, 0.5), ('1g 9/2', 4, 4.5)]
# levels_michael = [('3s 1/2', 0, 0.5), ('2d 3/2', 2, 1.5), ('2d 5/2', 2, 2.5), ('1g 7/2', 4, 3.5), ('1h 11/2', 5, 5.5)]
# levels = levels_michael

mu_list_schmidt = [mu_schmidt(each[2], each[1], False) for each in levels]
print('level\t\mu(\\nu) / \mu_N')
for i, each in enumerate(levels):
    print('%s\t%.2f' % (each[0], mu_list_schmidt[i]))


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
''' volt div ratio: '''
volt_div_ratio = "{'accVolt': 1000.05, 'offset': {'prema': 1000.022, 'agilent': 999.985}}"
# volt_div_ratio = "{'accVolt': 1000.05, 'offset': 1000.0}"
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''UPDATE Files SET voltDivRatio = ?''', (volt_div_ratio, ))
con.commit()
con.close()

''' diff doppler 60Ni 30kV'''
diffdopp60 = Physics.diffDoppler(850343019.777062, 30000, 60)  # 14.6842867127 MHz/V

''' transition wavelenght: '''
# observed_wavenum = 28364.39  # cm-1  observed wavenum from NIST, mass is unclear.
# transition_freq = 850342663.9020721  # final value, observed from voltage calibration
transition_freq = 850343019.777  # value from NIST, mass unclear

# # transition_freq = Physics.freqFromWavenumber(observed_wavenum)
# print('transition frequency: %s ' % transition_freq)
#
transition_freq += 1256.32701  # correction from fitting the 60_Ni references
#
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''UPDATE Lines SET frequency = ?''', (transition_freq,))
con.commit()
con.close()

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
configs = {}
# print('run \t iso \t val \t statErr \t rChi')
# runs = ['narrow_gate_67_Ni']

for iso in isotopes:
    for run in runs:
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''SELECT config, val, statErr, rChi FROM Combined WHERE iso = ? AND run = ? AND parname = ? ''',
                    (iso, run, 'shift'))
        data = cur.fetchall()
        con.close()
        if len(data):
            config, val, statErr, rChi = data[0]
            # print('%s \t %s \t %s \t %s \t %s \n %s' % (run, iso, val, statErr, rChi, config))
            configs[iso] = ast.literal_eval(config)

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
# get the relevant files which need to be fitted in the following:
div_ratio_relevant_stable_files = {}
div_ratio_relevant_stable_files['60_Ni'] = []


# for voltage divider ratio determination use only the ones with known references.
# configs = {iso: cfg_sel for iso, cfg_sel in list(configs.items()) if iso in
#            iso in list(literature_shifts.keys())}
# configs.pop('60_Ni')

for iso, cfg in sorted(configs.items()):
    div_ratio_relevant_stable_files[iso] = []
    for each in cfg:
        [div_ratio_relevant_stable_files['60_Ni'].append(ref_before) for ref_before in each[0] if
         ref_before not in div_ratio_relevant_stable_files['60_Ni']]
        [div_ratio_relevant_stable_files[iso].append(file) for file in each[1]]
        [div_ratio_relevant_stable_files['60_Ni'].append(ref_after) for ref_after in each[2] if
         ref_after not in div_ratio_relevant_stable_files['60_Ni']]
div_ratio_relevant_stable_files['60_Ni'] = sorted(div_ratio_relevant_stable_files['60_Ni'])

# div_ratio_relevant_stable_files.pop('58_Ni')  # due to deviation of 58_Ni, do not fit this one.

# div_ratio_relevant_stable_files = {iso: files_sel for iso, files_sel in
#                                    div_ratio_relevant_stable_files.items() if iso in list(literature_shifts.keys())}

# print('number of resonances that will be fitted: %s' %
#       float(sum([len(val) for key, val in div_ratio_relevant_stable_files.items()])))
# for iso_name, files in sorted(div_ratio_relevant_stable_files.items()):
#     print(iso_name, files)
#     if iso_name == '60_Ni':
#         BatchFit.batchFit(files, db, 'narrow_gate')  # refRun
#     else:
#         BatchFit.batchFit(files, db, runs[0])

# raise Exception

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

            # con = sqlite3.connect(db)
            # cur = con.cursor()
            # divratio = str({'accVolt': current_acc_div, 'offset': curent_off_div})
            # cur.execute('''UPDATE Files SET voltDivRatio = ? ''', (divratio,))
            # cur.execute('''UPDATE Lines SET frequency = ?''', (new_freq,))
            # con.commit()
            # con.close()

            # Batchfitting:
            fitres = [(iso, run_chi_finder, BatchFit.batchFit(files, db, run_chi_finder)[1])
                      for iso, files in sorted(div_ratio_relevant_stable_files.items())]

            # combineRes only when happy with voltdivratio, otherwise no use...
            # [[Analyzer.combineRes(iso, par, run, db) for iso in stables] for par in pars]
            try:
                shifts = {iso: Analyzer.combineShift(iso, run_chi_finder, db) for iso in
                          stables if iso not in ['60_Ni']}
            except Exception as e:
                shifts = {}
                print('error while combining shifts: %s' % e)

            # calc red. Chi ** 2:
            chisq = 0
            for iso, shift_tuple in shifts.items():
                iso_shift_err = max(np.sqrt(np.square(shift_tuple[3]) + np.square(literature_shifts[iso][1])), 1)
                iso_chisq = np.square((shift_tuple[2] - literature_shifts[iso][0]) / iso_shift_err)
                print('iso: %s chi sq: %s shift tuple: %s ' % (iso, iso_chisq, shift_tuple))
                chisq += iso_chisq
            chisquares[acc_vol_ratio_index].append(float(chisq))
            # fit_res[acc_vol_ratio_index].append(fitres)
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


# acc_ratios, offset_div_ratios, chisquares = chi_square_finder([0], [0])
#
# the error for the voltage determination has been found to be:
# 1.5 * 10 ** -4 for the accvolt ratio and the offset ratio
# for par in ['shift', 'Al', 'Au', 'Bl', 'Bu']:
#     con = sqlite3.connect(db)
#     cur = con.cursor()
#     syst_error = str('systE(accVolt_d=%s, offset_d=%s)' % ('1.5 * 10 ** -4', '1.5 * 10 ** -4'))
#     cur.execute('''UPDATE Combined SET systErrForm = ? WHERE parname = ?''', (syst_error, par))
#     con.commit()
#     con.close()
#
#
# print('plotting now')



# try:
#     shifts = {iso: Analyzer.combineShift(iso, 'narrow_gate', db) for iso in
#               isotopes if iso not in ['67_Ni', '60_Ni']}
# except Exception as e:
#     shifts = {}
#     print('error while combining shifts: %s' % e)

# try:
#     shifts = {iso: Analyzer.combineShift(iso, 'narrow_gate_67_Ni', db) for iso in
#               ['67_Ni']}
# except Exception as e:
#     shifts = {}
#     print('error while combining shifts of 67_Ni: %s' % e)


try:
    # print(literature_shifts)
    # MPLPlotter.plot_par_from_combined(db, runs, list(literature_shifts.keys()), 'shift',
    #                                   literature_dict=literature_shifts,
    #                                   literature_name='A. Steudel (1980)')
    # isotopes.remove('69_Ni')
    # isotopes.remove('67_Ni')
    # print(isotopes)
    # files = Tools.fileList(runs, isotopes=isotopes)
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
    a_fac_runs = [runs[0], 'narrow_gate_67_Ni']

    al = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Al', print_extracted=False)
    au = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Au', print_extracted=False)
    bl = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Bl', print_extracted=False)
    bu = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Bu', print_extracted=False)
    ratios = []
    d_ratios = []
    q_moments = []
    magn_moments = []
    print('iso\tI\tAu [MHz]\trChi Au\tAl [MHz]\trChi Al\tAu/Al\td_Au/Al'
          '\tBu [MHz]\trChi Bu\tBl [MHz]\trChi Bl\tBu/Bl\td_Bu/Bl\tQ_l [b]\tQ_u [b]\tQ_m [b]'
          '\tµ [nm]')
    for run in a_fac_runs:
        for iso, a_low in sorted(al[run].items()):
            mass = int(iso[:2])
            nucl_spin = 0
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute(''' SELECT I FROM Isotopes WHERE iso = ? ''', (iso,))
            data = cur.fetchall()
            con.close()
            if data:
                nucl_spin = data[0][0]
            if a_low[0]:
                a_up = au[run][iso]
                b_up = bu[run][iso]
                b_low = bl[run][iso]
                ratio = a_up[0] / a_low[0]
                delta_ratio = np.sqrt(
                    (a_up[1]/a_low[0]) ** 2 + (a_up[0] * a_low[1] / (a_low[0] ** 2)) ** 2
                )
                b_ratio = 0.0
                delta_b_ratio = 0.0

                b_up = [0.0 if each is None else each for each in b_up]
                b_low = [0.0 if each is None else each for each in b_low]

                if b_low[0]:
                    b_ratio = b_up[0] / b_low[0]
                    delta_b_ratio = np.sqrt(
                        (b_up[1] / b_low[0]) ** 2 + (b_up[0] * b_low[1] / (b_low[0] ** 2)) ** 2
                    )
                ratios.append(ratio)
                d_ratios.append(delta_ratio)
                q_from_upper = quadrupol_moment(b_up[0], b_up[1], b_up[2])
                q_from_lower = quadrupol_moment(b_low[0], b_low[1], b_low[2])
                if q_from_lower[0] and q_from_upper[0]:
                    q_mean = Analyzer.weightedAverage(
                        [q_from_upper[0], q_from_lower[0]], [q_from_upper[1], q_from_lower[1]])
                else:
                    q_mean = (0, 0, 0)
                q_mean_print = '%.3f(%.0f)' % (q_mean[0], q_mean[1] * 1000)
                mu = magnetic_moment(a_low[0], a_low[1], a_low[2], nucl_spin)
                mu_print = mu[4]
                print('%s\t%s'
                      '\t%.3f(%.0f)[%.0f]\t%.2f'
                      '\t%.3f(%.0f)[%.0f]\t%.2f\t%.3f\t%.3f'
                      '\t%.3f(%.0f)[%.0f]\t%.2f'
                      '\t%.3f(%.0f)[%.0f]\t%.2f\t%.3f\t%.3f'
                      '\t%s\t%s\t%s'
                      '\t%s' % (
                    iso, nucl_spin,
                    a_up[0], a_up[1] * 1000, a_up[2] * 1000, a_up[3],
                    a_low[0], a_low[1] * 1000, a_low[2] * 1000, a_low[3], ratio, delta_ratio,
                    b_up[0], b_up[1] * 1000, b_up[2] * 1000, b_up[3],
                    b_low[0], b_low[1] * 1000, b_low[2] * 1000, b_low[3], b_ratio, delta_b_ratio,
                    q_from_lower[4], q_from_upper[4], q_mean_print,
                    mu_print
                ))
                q_moments.append((mass, nucl_spin, q_mean[0], q_mean[1]))
                magn_moments.append((mass, nucl_spin, mu[0], mu[1]))

    # print('magnetic moments: %s ' % magn_moments)
    # # optimize schmidt values:
    # # g_l_0 = 0
    # # g_s_0 = -3.826
    # g_l_0 = 0.125
    # g_s_0 = -2.04
    # g_l_difs = np.arange(-0.01, 0.01, 0.005)
    # g_s_difs = np.arange(-0.01, 0.01, 0.005)
    # chi_squares = [[]]
    # best_chi = [99999999999999, 0, 0]
    # for i, g_l_dif in enumerate(g_l_difs):
    #     g_l = g_l_0 + g_l_dif
    #     for j, g_s_dif in enumerate(g_s_difs):
    #         g_s = g_s_0 + g_s_dif
    #         chi_square = 0
    #         for magn_i, each in enumerate(magn_moments):
    #             if each[1] != 2.5:  # l = 1
    #                 l = 1
    #             else:
    #                 l = 0
    #             dif = mu_schmidt(each[1], l, False, g_l=g_l, g_s=g_s) - each[2]
    #             d_dif = each[3]
    #             chi_square += np.square(dif / d_dif)
    #         if chi_square < best_chi[0]:
    #             best_chi = chi_square, g_l, g_s
    #
    # print('best chi square: %.3f %.3f %.3f' % best_chi)
    # mu_list_schmidt = [
    #     (each[0], each[2], mu_schmidt(each[2], each[1], False, g_l=best_chi[1], g_s=best_chi[2])) for each in levels]
    #
    # print('level\t\mu(\\nu) / \mu_N')
    # for i, each in enumerate(mu_list_schmidt):
    #     print('%s\t%.2f' % (each[0], each[2]))
    #
    #
    # # plot magnetic moments
    # magn_mom_fig = MPLPlotter.plt.figure(0, facecolor='white')
    # magn_mom_axes = MPLPlotter.plt.axes()
    # magn_mom_axes.margins(0.1, 0.1)
    # magn_mom_axes.set_xlabel('A')
    # magn_mom_axes.set_ylabel('µ [nm]')
    # magn_mom_axes.set_xticks([each[0] for each in magn_moments])
    # magn_mom_by_spin = []
    # colors = ['b', 'g', 'k']
    # markers = ['o', 's', 'D']
    # for i, spin in enumerate([0.5, 1.5, 2.5]):
    #     spin_list_x = [mu[0] for mu in magn_moments if mu[1] == spin]
    #     spin_list_y = [mu[2] for mu in magn_moments if mu[1] == spin]
    #     spin_list_y_err = [mu[3] for mu in magn_moments if mu[1] == spin]
    #     if spin_list_x:
    #         label = 'spin: %s' % spin
    #         spin_line, cap_line, barline = MPLPlotter.plt.errorbar(
    #             spin_list_x, spin_list_y, spin_list_y_err, axes=magn_mom_axes,
    #             linestyle='None', marker=markers[i], label=label, color=colors[i]
    #         )
    #     for each in mu_list_schmidt:
    #         if spin == each[1]:
    #             hor_line = MPLPlotter.plt.axhline(each[2], label='eff. schmidt: %s' % each[0], color=colors[i])
    # MPLPlotter.plt.legend(loc=2, title='magnetic moments')
    # MPLPlotter.show(True)
    #
    # # plot quadrupole moments
    # q_mom_fig = MPLPlotter.plt.figure(1, facecolor='white')
    # q_mom_axes = MPLPlotter.plt.axes()
    # q_mom_axes.margins(0.1, 0.1)
    # q_mom_axes.set_xlabel('A')
    # q_mom_axes.set_ylabel('Q [b]')
    # q_mom_axes.set_xticks([each[0] for each in q_moments])
    # q_mom_by_spin = []
    # colors = ['g', 'k']
    # markers = ['s', 'D']
    # for i, spin in enumerate([1.5, 2.5]):
    #     spin_list_x = [mu[0] for mu in q_moments if mu[1] == spin]
    #     spin_list_y = [mu[2] for mu in q_moments if mu[1] == spin]
    #     spin_list_y_err = [mu[3] for mu in q_moments if mu[1] == spin]
    #     if spin_list_x:
    #         label = 'spin: %s' % spin
    #         spin_line, cap_line, barline = MPLPlotter.plt.errorbar(
    #             spin_list_x, spin_list_y, spin_list_y_err, axes=q_mom_axes,
    #             linestyle='None', marker=markers[i], label=label, color=colors[i]
    #         )
    #     MPLPlotter.plt.legend(loc=0, title='quadrupole moments')
    # MPLPlotter.show(True)



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
    #     'Al', plot_runs_seperate=False
    # )
    # Tools.extract_from_combined(a_fac_runs, db, isotopes, par='Au', print_extracted=True)
    pass
except Exception as e:
    print('plotting did not work, error is: %s' % e)
#
#
# # print('------------------- Done -----------------')
# # winsound.Beep(2500, 500)
#
# # print('\a')
#
# raise Exception
''' Fit on certain Files '''
# searchterm = 'Run167'
# certain_file = [file for file in ni60_files if searchterm in file][0]
# fit = InteractiveFit(certain_file, db, runs[0], block=True, x_as_voltage=True)
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
# delta_lit_radii.pop('64_Ni')  # just to see whoch point is what
king = KingFitter(db, showing=True, litvals=delta_lit_radii)
run = -1
# # isotopes = sorted(delta_lit_radii.keys())
king.kingFit(alpha=362, findBestAlpha=False, run=run)
# king.calcChargeRadii(isotopes=isotopes, run=run)

#
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(''' SELECT iso, val, statErr, systErr, rChi From Combined WHERE parname = ? ORDER BY iso''', ('shift',))
data = cur.fetchall()
con.close()
if data:
    print(data)
    print('iso\tshift [MHz]\t(statErr)[systErr]\t Chi^2')
    for iso in data:
        print('%s\t%.1f(%.0f)[%.0f]\t%.3f' % (iso[0], iso[1], iso[2] * 10, iso[3] * 10, iso[4]))
