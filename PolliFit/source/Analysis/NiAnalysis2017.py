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
isotopes.remove('59_Ni')  # not measured in 2017
isotopes.remove('63_Ni')  # not measured in 2017
odd_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2]
even_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2 == 0]
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']


# 2016 database etc.
workdir2016 = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder2016 = os.path.join(workdir2016, 'Ni_April2016_mcp')

db2016 = os.path.join(workdir2016, 'Ni_workspace.sqlite')
runs2016 = ['wide_gate_asym', 'wide_gate_asym_67_Ni']

isotopes2016 = ['%s_Ni' % i for i in range(58, 71)]
isotopes2016.remove('69_Ni')
odd_isotopes2016 = [iso for iso in isotopes2016 if int(iso[:2]) % 2]


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

lit_radii_calc = {iso: (val[0] / v2_lit[iso], val[1]) for iso, val in sorted(baret_radii_lit.items())}

# using the more precise values by the self calculated one:
lit_radii = lit_radii_calc

delta_lit_radii = {iso: [
    lit_vals[0] ** 2 - lit_radii['60_Ni'][0] ** 2,
    np.sqrt(lit_vals[1] ** 2 + lit_radii['60_Ni'][1] ** 2)]
                   for iso, lit_vals in sorted(lit_radii.items())}
delta_lit_radii.pop('60_Ni')
print(
    'iso\t<r^2>^{1/2}_{0µe}\t\Delta<r^2>^{1/2}_{0µe}\t<r^2>^{1/2}_{0µe}(A-A_{60})\t\Delta <r^2>^{1/2}_{0µe}(A-A_{60})')
for iso, radi in sorted(lit_radii.items()):
    dif = delta_lit_radii.get(iso, (0, 0))
    print('%s\t%.3f\t%.3f\t%.5f\t%.5f' % (iso, radi[0], radi[1], dif[0], dif[1]))


''' literature moments: '''
''' Moments '''
a_low_61_Ni_lit = (-454.972, 0.003)
b_low_61_Ni_lit = (-102.951, 0.016)

''' Quadrupole Moments '''
# literature values from PHYSICAL REVIEW VOLUME 170, NUM HER 1 5 JUNE 1968
# Hyperfine-Structure Studies of Ni", and the Nuclear Ground-State
# Electric Quadrupole Moment*
# W. J. CHILDs AND L. S. 600DMAN
# Argonne Eationa/ Laboratory, Argonne, Illinois

mass_q_literature_61_Ni = 61  # u
spin_q_literature = 3 / 2
q_literature_61_Ni = 0.162  # barn
d_q_literature_61_Ni = 0.015  # barn

lit_q_moments = [(61.1, 3 / 2, 0.162, 0.015), (61.2, 5 / 2, -0.2, 0.03), (61.3, 5 / 2, -0.08, 0.07)]

# 3d9(2D)4s  	 3D 3
b_lower_lit = -102.979  # MHz
d_b_lower_lit = 0.016  # MHz

# e_Vzz value from this: e_Vzz = Q / B
e_Vzz_lower = q_literature_61_Ni / b_lower_lit
d_e_Vzz_lower = np.sqrt(
    (e_Vzz_lower / b_lower_lit * d_b_lower_lit) ** 2 +
    (e_Vzz_lower / q_literature_61_Ni * d_q_literature_61_Ni) ** 2)
print(e_Vzz_lower, d_e_Vzz_lower)
print('eVzz_lower = %.3f(%.0f) b/ kHz' % (e_Vzz_lower * 1000, d_e_Vzz_lower * 1e6))

# for the upper state  3d9(2D)4p  	 3P° 2, no b factor was measured
# therefore i will need to get the e_Vzz_upper from my results on 61_Ni and the q_literature_61_Ni
b_upper_exp = -50.597  # MHz
d_b_upper_exp = 1.43  # Mhz

e_Vzz_upper = q_literature_61_Ni / b_upper_exp
d_e_Vzz_upper = np.sqrt(
    (e_Vzz_upper / b_upper_exp * d_b_upper_exp) ** 2 +
    (e_Vzz_upper / q_literature_61_Ni * d_q_literature_61_Ni) ** 2)

print(e_Vzz_upper, d_e_Vzz_upper)
print('eVzz_upper = %.3f(%.0f) b/ kHz' % (e_Vzz_upper * 1000, d_e_Vzz_upper * 1e6))


def quadrupol_moment(b, d_stat_b, d_syst_b, upper=True):
    if b:
        e_vzz = e_Vzz_lower
        d_e_vzz = d_e_Vzz_lower
        if upper:
            e_vzz = e_Vzz_upper
            d_e_vzz = d_e_Vzz_upper
        q = b * e_vzz
        d_stat_q = abs(d_stat_b * e_vzz)
        d_syst_q = np.sqrt(
            (e_vzz * d_syst_b) ** 2 +
            (b * d_e_vzz) ** 2
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

mass_mu_ref = 61
mu_ref = -0.75002  # nm -> nuclear magneton
d_mu_ref = 0.00004  # nm
i_ref = 3 / 2

# (spin, µ, d_µ)
lit_magnetic_moments = {
    # '57_Ni': (3/2, -0.7975, 0.0014),  # not measured
    '61_Ni': (61, 3 / 2, -0.75002, 0.00004),
    '65_Ni': (65, 5 / 2, 0.69, 0.06),
    '67_Ni': (67, 1 / 2, 0.601, 0.005)
}

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
def mu_schmidt(I, l, proton, g_l=0., g_s=-3.826):
    l_plus = I == l + 0.5
    # g_l = 0
    # g_s = -3.826
    if proton:
        g_l = 1.
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


''' Files from db: '''
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
normal_run = 'AsymVoigt'
exp_2016_run = '2016Experiment'
current_run = normal_run
run_numbermax = 81 * 100
run_number_min = 81

fits = {}
files_w_err = {}

# # overwrite isotopes if wanted:
# isotopes = ['67_Ni']
#
# for iso in isotopes:
#     iso_list = []
#     for run_num, date_str, file_str, iso_str in file_list:
#         if iso_str == iso and run_number_min < int(run_num) <= run_numbermax:
#             iso_list.append(file_str)
#     if len(iso_list):
#         print('starting Batchfit on Files: %s' % str(iso_list))
#         fits_iso, files_w_err_iso = BatchFit.batchFit(iso_list, db, current_run)
#         fits[iso] = fits_iso
#         files_w_err[iso] = files_w_err_iso
#
#
# print('Batchfit finished, files with error: ')
# TiTs.print_dict_pretty(files_w_err)


''' combine shifts '''

# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute(''' UPDATE Combined SET statErrForm = ? ''', ('applyChi(err, rChi)',))
# syst_error = str('systE(accVolt_d=%s, offset_d=%s)' % ('1.5 * 10 ** -4', '1.5 * 10 ** -4'))
# cur.execute('''UPDATE Combined SET systErrForm = ? WHERE parname = ?''', (syst_error, 'shift'))
# con.commit()
# con.close()
#
# for iso in isotopes:
#     if iso != '60_Ni':
#         Analyzer.combineShiftByTime(iso, current_run, db, show_plot=False)
#         # Analyzer.combineShift(iso, current_run, db, show_plot=False)
#
# print shifts:

con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(''' SELECT iso, val, statErr, systErr, rChi From Combined WHERE parname = ? AND run = ? ORDER BY iso''',
            ('shift', current_run))
data = cur.fetchall()
con.close()
iso_shift_plot_data_x = []
iso_shift_plot_data_y = []
iso_shift_plot_data_err = []
if data:
    print(data)
    print('iso\tshift [MHz]\t(statErr)[systErr]\t Chi^2')
    for iso in data:
        iso_shift_plot_data_x.append(int(iso[0][:2]))
        iso_shift_plot_data_y.append(float(iso[1]))
        err = np.sqrt(iso[2] ** 2 + iso[3] ** 2)
        iso_shift_plot_data_err.append(err)
        # print('%s\t%.1f(%.0f)[%.0f]\t%.3f' % (iso[0], iso[1], iso[2] * 10, iso[3] * 10, iso[4]))
        a = '%s\t%.2f\t%.2f' % (iso[0], iso[1], iso[2])
        a = a.replace('.', ',')
        print(a)


''' compare shifts '''

# create literature shift dict from 2016 run:
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(''' SELECT iso, val, statErr, systErr, rChi From Combined WHERE parname = ? AND run = ? ORDER BY iso''',
            ('shift', '2016Experiment'))
shifts2016 = cur.fetchall()
con.close()

shifts2016_dict = {}
for iso, val, statErr, systErr, rChi in shifts2016:
    shifts2016_dict[iso] = (val, statErr)

print('shifts from 2016:')
TiTs.print_dict_pretty(shifts2016_dict)
#
# MPLPlotter.plot_par_from_combined(db, [normal_run, run_hot_cec],
#                                   isotopes, 'shift', show_pl=True,
#                                   plot_runs_seperate=True, literature_dict=shifts2016_dict,
#                                   literature_name='2016 Data (as reference) (Simon)',
#                                   comments=[' (Simon)', ' (Simon)', '', ''],
#                                   markers=['s', 's', 'D', 'D'],
#                                   colors=[(1, 0, 0), (1, 0.1, 0.7), (0.4, 0.4, 1), (1, 0.3, 0.3)],
#                                   start_offset=-0.1,
#                                   legend_loc=3
#                                   )

MPLPlotter.plot_par_from_combined(db, [normal_run, '2016ExpLiang', '2017ExpLiang'],
                                  isotopes, 'shift', show_pl=True,
                                  plot_runs_seperate=True, literature_dict=shifts2016_dict,
                                  literature_name='2016 Data (as reference) (Simon)',
                                  comments=[' 2017Exp (Simon)', '', ''],
                                  markers=['s', 'D', 'D'],
                                  colors=[(1, 0, 0), (0.6, 0.4, 1), (1, 0.5, 0.3)],
                                  legend_loc=3
                                  )


''' King plot and charge radii '''

king = KingFitter(db, showing=True, litvals=delta_lit_radii, plot_y_mhz=False, font_size=18)
# king.kingFit(alpha=0, findBestAlpha=False, run=current_run, find_slope_with_statistical_error=False)
# king.calcChargeRadii(isotopes=isotopes, run=current_run, plot_evens_seperate=True)

king.kingFit(alpha=365, findBestAlpha=True, run=current_run)
radii_alpha = king.calcChargeRadii(isotopes=isotopes, run=current_run, plot_evens_seperate=True)
print('radii with alpha', radii_alpha)

''' compare radii '''

# create literature radii dict from 2016 run:
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(''' SELECT iso, val, statErr, systErr, rChi From Combined WHERE parname = ? AND run = ? ORDER BY iso''',
            ('delta_r_square', exp_2016_run))
radii2016 = cur.fetchall()
con.close()

radii2016_dict = {}
for iso, val, statErr, systErr, rChi in radii2016:
    radii2016_dict[iso] = (val, statErr, systErr)

print('shifts from 2016:')
TiTs.print_dict_pretty(radii2016_dict)

MPLPlotter.plot_par_from_combined(db, [normal_run],
                                  isotopes, 'delta_r_square', show_pl=True,
                                  plot_runs_seperate=True, literature_dict=radii2016_dict,
                                  literature_name='2016 Data (as reference)',
                                  use_syst_err_only=True,
                                  comments=[' - 2017 Data'],
                                  start_offset=-0.05)


''' A and B factors and moments '''

pars = ['Al', 'Au', 'Bl', 'Bu']
a_fac_runs = [run_hot_cec]


# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute(''' UPDATE Combined SET statErrForm = ? ''', ('applyChi(err, rChi)',))
# syst_error = str('systE(accVolt_d=%s, offset_d=%s)' % ('1.5 * 10 ** -4', '1.5 * 10 ** -4'))
# for par in pars:
#     cur.execute('''UPDATE Combined SET systErrForm = ? WHERE parname = ?''', (syst_error, par))
# con.commit()
# con.close()
#
# for iso in isotopes:
#     if int(iso[:2]) % 2:
#         for par in pars:
#             Analyzer.combineRes(iso, par, current_run, db)
#
#
# # get results from db:
# print('\n\n\niso\tAu\td_Au\tAl\td_Al\tAu/Al\td_Au/Al')
#
# al = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Al', print_extracted=False)
# au = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Au', print_extracted=False)
# bl = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Bl', print_extracted=False)
# bu = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Bu', print_extracted=False)
# ratios = []
# b_ratios = []
# d_ratios = []
# d_b_ratios = []
# q_moments = []
# magn_moments = []
# print('iso\tI\tAu [MHz]\trChi Au\tAl [MHz]\trChi Al\tAu/Al\td_Au/Al'
#       '\tBu [MHz]\trChi Bu\tBl [MHz]\trChi Bl\tBu/Bl\td_Bu/Bl\tQ_l [b]\tQ_u [b]\tQ_m [b]'
#       '\tµ [nm]')
# for run in a_fac_runs:
#     for iso, a_low in sorted(al[run].items()):
#         mass = int(iso[:2])
#         nucl_spin = 0
#         con = sqlite3.connect(db)
#         cur = con.cursor()
#         cur.execute(''' SELECT I FROM Isotopes WHERE iso = ? ''', (iso,))
#         data = cur.fetchall()
#         con.close()
#         if data:
#             nucl_spin = data[0][0]
#         if a_low[0]:
#             a_up = (0, 0, 0, 0) if au[run][iso] == (None, None, None, None) else au[run][iso]  # when fixed use 0's
#             b_up = (0, 0, 0, 0) if bu[run][iso] == (None, None, None, None) else bu[run][iso]
#             b_low = (0, 0, 0, 0) if bl[run][iso] == (None, None, None, None) else bl[run][iso]
#             ratio = a_up[0] / a_low[0]
#             delta_ratio = np.sqrt(
#                 (a_up[1] / a_low[0]) ** 2 + (a_up[0] * a_low[1] / (a_low[0] ** 2)) ** 2
#             )
#             b_ratio = 0.0
#             delta_b_ratio = 0.0
#
#             b_up = [0.0 if each is None else each for each in b_up]
#             b_low = [0.0 if each is None else each for each in b_low]
#
#             if b_low[0]:
#                 b_ratio = b_up[0] / b_low[0]
#                 delta_b_ratio = np.sqrt(
#                     (b_up[1] / b_low[0]) ** 2 + (b_up[0] * b_low[1] / (b_low[0] ** 2)) ** 2
#                 )
#                 b_ratios.append(b_ratio)
#                 d_b_ratios.append(delta_b_ratio)
#             ratios.append(ratio)
#             d_ratios.append(delta_ratio)
#             q_from_upper = quadrupol_moment(b_up[0], b_up[1], b_up[2], upper=True)
#             q_from_lower = quadrupol_moment(b_low[0], b_low[1], b_low[2], upper=False)
#             if q_from_lower[0] and q_from_upper[0]:
#                 q_mean = Analyzer.weightedAverage(
#                     [q_from_upper[0], q_from_lower[0]], [q_from_upper[1], q_from_lower[1]])
#             else:
#                 q_mean = (0, 0, 0)
#             q_mean_print = '%.3f(%.0f)' % (q_mean[0], q_mean[1] * 1000)
#             mu = magnetic_moment(a_low[0], a_low[1], a_low[2], nucl_spin)
#             mu_print = mu[4]
#             print('%s\t%s'
#                   '\t%.3f(%.0f)[%.0f]\t%.2f'
#                   '\t%.3f(%.0f)[%.0f]\t%.2f\t%.3f\t%.3f'
#                   '\t%.3f(%.0f)[%.0f]\t%.2f'
#                   '\t%.3f(%.0f)[%.0f]\t%.2f\t%.3f\t%.3f'
#                   '\t%s\t%s\t%s'
#                   '\t%s' % (
#                       iso, nucl_spin,
#                       a_up[0], a_up[1] * 1000, a_up[2] * 1000, a_up[3],
#                       a_low[0], a_low[1] * 1000, a_low[2] * 1000, a_low[3], ratio, delta_ratio,
#                       b_up[0], b_up[1] * 1000, b_up[2] * 1000, b_up[3],
#                       b_low[0], b_low[1] * 1000, b_low[2] * 1000, b_low[3], b_ratio, delta_b_ratio,
#                       q_from_lower[4], q_from_upper[4], q_mean_print,
#                       mu_print
#                   ))
#             q_moments.append((mass, nucl_spin, q_from_lower[0], q_from_lower[3]))
#             magn_moments.append((mass, nucl_spin, mu[0], mu[3]))
#
# print('magnetic moments: %s ' % magn_moments)

al2016 = Tools.extract_from_combined(runs2016, db2016, odd_isotopes2016, par='Al', print_extracted=False)
au2016 = Tools.extract_from_combined(runs2016, db2016, odd_isotopes2016, par='Au', print_extracted=False)
bl2016 = Tools.extract_from_combined(runs2016, db2016, odd_isotopes2016, par='Bl', print_extracted=False)
bu2016 = Tools.extract_from_combined(runs2016, db2016, odd_isotopes2016, par='Bu', print_extracted=False)

hf_factors2016_list = [al2016, au2016, bl2016, bu2016]
hf_factors2016 = {par: hf_factors2016_list[i] for i, par in enumerate(pars)}

# odd_isotopes = ['61_Ni', '65_Ni']

for par in pars:
    try:
        ref_dict = hf_factors2016[par][runs2016[0]]
        TiTs.print_dict_pretty(ref_dict)
        MPLPlotter.plot_par_from_combined(db, [normal_run],
                                          odd_isotopes, par, show_pl=True,
                                          plot_runs_seperate=True, literature_dict=ref_dict,
                                          literature_name='2016 Data (as reference)',
                                          start_offset=-0.05)
        MPLPlotter.clear()
    except Exception as e:
        print('error: %s' % e)

