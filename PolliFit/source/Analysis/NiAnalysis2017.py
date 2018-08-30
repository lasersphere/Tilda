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
import gc

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

run_hot_cec = 'AsymVoigtHotCec'
run_hot_cec_exp = 'VoigtAsy'
normal_run = 'AsymVoigt'
final_2017_run = '2017_Experiment'  # this will be the joined analysis of hot CEC and normal run!

run2016_final_db = '2016Experiment'
exp_liang_run16 = '2016ExpLiang'
exp_liang_run17 = '2017ExpLiang'
steudel_1980_run = 'Steudel_1980'
bradley_2016_3_ev_run = '2016_Bradley_3eV'

current_run = normal_run
other_run2017 = normal_run if current_run is run_hot_cec else run_hot_cec

# no need to scroll down
# select if the files need to be fitted or not:
perform_bacthfit = False
# same for combining shifts:
combine_shifts = False
# ... and the offset:
combine_offset = False
# combine final run values -> joined of hot cec and normal
combine_final_run = False


# for plotting:
run_colors = {
    normal_run: (0.4, 0, 0),  # dark red
    run_hot_cec: (0.6, 0, 0),  # dark red
    run2016_final_db: (0, 0, 1),  # blue
    steudel_1980_run: (0, 0.6, 0.5),  # turqoise
    exp_liang_run16: (0.6, 0.4, 1),  # purple
    exp_liang_run17: (1, 0.5, 0.3),  # orange
    run_hot_cec_exp: (0.4, 0, 0),  # dark red
    bradley_2016_3_ev_run: (0, 0.3, 1),
    final_2017_run: (1, 0, 0)  # red
}

run_markes = {
    normal_run: 's',  # square
    run_hot_cec: 's',  # square
    run2016_final_db: 'o',  # circle
    steudel_1980_run: 'o',  # circle
    exp_liang_run16: 'D',  # diamond
    exp_liang_run17: 'D',  # diamond
    run_hot_cec_exp: 's',
    bradley_2016_3_ev_run: '^',  # triangle up
    final_2017_run: 's'
}

run_comments = {
    normal_run: ' 2017 Simon (runs 83-end)',
    run_hot_cec: ' 2017 hot CEC (runs 62-82) Simon',
    run_hot_cec_exp: '2017 hot CEC (runs 62-82) Simon',
    run2016_final_db: '2016 Simon',
    steudel_1980_run: '1980 Steudel',
    exp_liang_run16: '2016 Liang',
    exp_liang_run17: '2017 Liang',
    bradley_2016_3_ev_run: '2016 Bradley',
    final_2017_run: '2017 Simon'
}

isotopes = ['%s_Ni' % i for i in range(58, 71 if current_run is normal_run else 65)]
isotopes.remove('59_Ni')  # not measured in 2017
isotopes.remove('63_Ni')  # not measured in 2017
if current_run is normal_run:
    isotopes.remove('69_Ni')

odd_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2]
even_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2 == 0]
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']

# isotopes = ['64_Ni']


# 2016 database etc.
workdir2016 = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder2016 = os.path.join(workdir2016, 'Ni_April2016_mcp')

db2016 = os.path.join(workdir2016, 'Ni_workspace.sqlite')
runs2016 = ['wide_gate_asym', 'wide_gate_asym_67_Ni']
run2016_final = 'wide_gate_asym'

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

# write literature shifts to db:
# con = sqlite3.connect(db)
# cur = con.cursor()
# for iso, shift_tuple in literature_shifts.items():
#     cur.execute(
#         '''INSERT OR IGNORE INTO Combined (iso, parname, run, config,
#  val, statErr, systErr) VALUES (?, ?, ?, ?, ?, ?, ?)''',
#         (iso, 'shift', 'Steudel_1980', '[]', shift_tuple[0], shift_tuple[1], 0.0))
# con.commit()
# con.close()

''' write 2016 Results to 2017 db '''
shifts_2016 = Tools.extract_from_combined([run2016_final], db2016, isotopes2016, par='shift')[run2016_final]
delta_r_square_2016 = Tools.extract_from_combined([run2016_final], db2016,
                                                  isotopes2016, par='delta_r_square')[run2016_final]
print(shifts_2016)
print(delta_r_square_2016)

con = sqlite3.connect(db)
cur = con.cursor()
# delete old
cur.execute('''DELETE FROM Combined WHERE run = ? AND parname = ?''', (run2016_final_db, 'shift'))
con.commit()
for iso, ret_tuple in shifts_2016.items():
    shift, stat_err, syst_err, r_chi = ret_tuple
    print(iso, shift, stat_err, syst_err, r_chi)
    cur.execute(
        '''INSERT OR IGNORE INTO Combined (iso, parname, run, config,
 val, statErr, systErr, rChi) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (iso, 'shift', '2016Experiment', '[]', shift, stat_err, syst_err, r_chi))

# delete old
cur.execute('''DELETE FROM Combined WHERE run = ? AND parname = ?''', (run2016_final_db, 'delta_r_square'))
con.commit()
for iso, ret_tuple in delta_r_square_2016.items():
    d_r_sq, stat_err, syst_err, r_chi = ret_tuple
    print(iso, d_r_sq, stat_err, syst_err, r_chi)
    cur.execute(
        '''INSERT OR IGNORE INTO Combined (iso, parname, run, config,
 val, statErr, systErr, rChi) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (iso, 'delta_r_square', '2016Experiment', '[]', d_r_sq, stat_err, syst_err, r_chi))

con.commit()
con.close()

''' Bradley results of 2016 '''
# Bradley had an undergrad look t the 2016 results.
# lineshape: lorentzian profile (free int.)
# offset side peaks: -2.155eV (unclear-> 2016_Bradley_2eV) and -3.517eV (final -> 2016_Bradley_3eV)
# errors are directly taken from weigthed mean

bradley_2016_3_ev = {
    '58_Ni': (-507.19, 0.5),
    '59_Ni': (-214.83, 0.99),
    '60_Ni': (0, 0),
    '61_Ni': (284.86, 0.94),
    '62_Ni': (504.74, 0.6),
    '63_Ni': (783.46, 0.38),
    '64_Ni': (1026.95, 0.38),
    '65_Ni': (1323.16, 0.55),
    '66_Ni': (1533.67, 0.41),
    '68_Ni': (1995.03, 0.8),
    '70_Ni': (2369.06, 7.72)
}

both_bradley_runs = {
    bradley_2016_3_ev_run: bradley_2016_3_ev
}

# write to db
con = sqlite3.connect(db)
cur = con.cursor()
for br_run in [bradley_2016_3_ev_run]:
    for iso, ret_tuple in both_bradley_runs[br_run].items():
        shift, stat_err = ret_tuple
        print(iso, shift, stat_err)
        cur.execute(
            '''INSERT OR IGNORE INTO Combined (iso, parname, run, config,
     val, statErr, systErr, rChi) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (iso, 'shift', br_run, '[]', shift, stat_err, 0, 0))


con.commit()
con.close()





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
# crawl first :(
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
# #
# laserfreq_to_be = 14198.7624  # freq in logbook seems to be wrong and
# # this frequency increases the laserfrequency by about 300MHz to match the absolute center
# # frequencies later in the beam time.
# # (note comment in morning shift on 08.09.2017:
# #  09:24 The doide laser lost lock since yesterday evening, which is the reason why the matisse can not be locked.
# #  Rearrange the input for the doide laser and locked into the same peak as yesterday day time. The FPI is relocked
# #  into the doide laser, and matisse is locked into the FPI
# # )
# # probably the wavemeter was calibrated wrong all night. Was it even calibrated multiple times or just once?
# #
# for f in files_to_correct_laser_freq:
#     print('changing laserfreq in %s to %s cm-1' % (f, laserfreq_to_be))
#     f_path = os.path.join(datafolder, f)
#     root = TiTs.load_xml(f_path)
#     laser_freq = root.find('header').find('laserFreq')
#     laser_freq.text = str(laserfreq_to_be)
#     TiTs.save_xml(root, f_path)


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


run_numbermax = 9999 if current_run is normal_run else 81
run_number_min = 81 if current_run is normal_run else 0

fits = {}
files_w_err = {}

# # overwrite isotopes if wanted:
# isotopes = ['67_Ni']

# # ['58_Ni', '59_Ni', '60_Ni', '61_Ni', '62_Ni', '63_Ni', '64_Ni', '65_Ni', '66_Ni', '67_Ni', '68_Ni', '70_Ni']
for iso in isotopes:
    if perform_bacthfit:
        if iso in ['58_Ni', '59_Ni', '60_Ni', '61_Ni', '62_Ni',
                   '63_Ni', '64_Ni', '65_Ni', '66_Ni', '67_Ni', '68_Ni', '70_Ni']:
            print('---------------- started batchfitting on iso %s -----------' % iso)
            iso_list = []
            for run_num, date_str, file_str, iso_str in file_list:
                if iso_str == iso and run_number_min < int(run_num) <= run_numbermax:
                    iso_list.append(file_str)
            if len(iso_list):
                print('starting Batchfit on Files: %s' % str(iso_list))
                fits_iso, files_w_err_iso = BatchFit.batchFit(iso_list, db, current_run)
                # BatchFit.batchFit(iso_list, db, current_run)
                fits[iso] = fits_iso
                files_w_err[iso] = files_w_err_iso
                gc.collect()


if perform_bacthfit:
    print('Batchfit finished, files with error: ')
    TiTs.print_dict_pretty(files_w_err)


''' combine hot CEC and normal run '''
# the fit results from the hot CEC batch fits and the normal run will be renamed to the name of the final run and
#  then can be used for the final analysis
#  configs of hot cec and normal run need to be joined!

if combine_final_run:
    # step 1 copy fit results from hot cec and rename them to normal
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''DELETE FROM FitRes WHERE run = ?''', (final_2017_run,))
    con.commit()
    cur.execute(''' INSERT INTO FitRes
     (
     file, iso, run, rChi, pars
     )
     SELECT file, iso, ?, rChi, pars
     FROM FitRes WHERE run = ?
     ''', (final_2017_run, run_hot_cec))
    cur.execute(''' INSERT INTO FitRes
     (
     file, iso, run, rChi, pars
     )
     SELECT file, iso, ?, rChi, pars
     FROM FitRes WHERE run = ?
     ''', (final_2017_run, normal_run))
    con.commit()
    con.close()

    # step 2 copy configs of all combined analysis for the normal run
    #  (all isos in normal run are also included in hot CEC run)
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''DELETE FROM Combined WHERE run = ?''', (final_2017_run,))
    con.commit()
    cur.execute(''' INSERT INTO Combined
     (
     iso, parname, run, config, final, rChi, val, statErr, statErrForm, systErr, systErrForm
     )
     SELECT iso, parname, ?, config, final, 0.0, 0.0, statErr, statErrForm, systErr, systErrForm
     FROM Combined WHERE run = ?
     ''', (final_2017_run, normal_run))
    con.commit()
    con.close()

    # step 3 add the configs from hot cec run to normal run into final_2017_run:
    con = sqlite3.connect(db)
    cur = con.cursor()
    for hot_par_name in ['shift', 'Al', 'Au', 'Bl', 'Bu']:
        print('working on par: ', hot_par_name)
        for iso_hot_cec in ['58_Ni', '61_Ni', '62_Ni', '64_Ni']:
            print('working on iso: ', iso_hot_cec)

            cur.execute(''' SELECT config FROM Combined WHERE parname = ? AND iso = ? AND run = ?''',
                        (hot_par_name, iso_hot_cec, run_hot_cec))
            data = cur.fetchall()
            if len(data):
                hot_cfg = data[0][0]
                hot_cfg_useful = len(hot_cfg) > 2  # more than '[]'
                cur.execute(''' SELECT config FROM Combined WHERE parname = ? AND iso = ? AND run = ?''',
                            (hot_par_name, iso_hot_cec, normal_run))
                normal_cfg = cur.fetchall()[0][0]
                normal_cfg_useful = len(normal_cfg) > 2
                if normal_cfg_useful and hot_cfg_useful:
                    joined_cfg = hot_cfg[:-1] + ', ' + normal_cfg[1:]
                else:
                    if normal_cfg_useful:
                        joined_cfg = normal_cfg
                    else:
                        joined_cfg = hot_cfg
                cur.execute('''UPDATE Combined SET config = ? WHERE iso = ? AND parname = ? AND run = ?''',
                            (joined_cfg, iso_hot_cec, hot_par_name, final_2017_run))
                print(' %s combined config for par %s  is: %s' % (iso_hot_cec, hot_par_name, joined_cfg))
                con.commit()

    con.close()


''' combine shifts '''
# PT50 from PTB was measuring the accvoltage, assumed precision 5E-5,
# JRL-10 (Ser.# 143) was measuring the offset voltage and was assumed again with 1.5E-4
# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute(''' UPDATE Combined SET statErrForm = ? ''', ('applyChi(err, rChi)',))
# syst_error = str('systE(accVolt_d=%s, offset_d=%s)' % ('5 * 10 ** -5', '1.5 * 10 ** -4'))
# cur.execute('''UPDATE Combined SET systErrForm = ? WHERE parname = ?''', (syst_error, 'shift'))
# con.commit()
# con.close()
#
offset_dict = {}
for iso in isotopes:
    if iso != '60_Ni':
        if combine_shifts:
            Analyzer.combineShiftByTime(iso, current_run, db, show_plot=False)
    if iso not in ['59_Ni', '60_Ni', '63_Ni', '69_Ni']:
        if combine_offset:
            try:
                # this does not work for all isotopes
                offset_dict[iso] = Analyzer.combineShiftOffsetPerBunchDisplay(iso, current_run, db)
            except Exception as e:
                print('-----------------------------------------------------------------')
                print('error: %s  occurred while combining offsets for iso %s' % (e, iso))
                print('-----------------------------------------------------------------')

for iso in isotopes:
    if iso != '60_Ni':
        if combine_final_run:
            print('combining now the runs %s and %s to the final run %s' % (run_hot_cec, normal_run, final_2017_run))
            Analyzer.combineShiftByTime(iso, final_2017_run, db, show_plot=False)


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

# compare 2016 results
runs_to_compare = [exp_liang_run16]  #, bradley_2016_3_ev_run]
MPLPlotter.plot_par_from_combined(db, runs_to_compare,
                                  isotopes2016, 'shift', show_pl=True,
                                  plot_runs_seperate=True, literature_run=run2016_final_db,
                                  literature_name=run_comments[run2016_final_db],
                                  lit_color=run_colors[run2016_final_db],
                                  lit_marker=run_markes[run2016_final_db],
                                  comments=[run_comments[r] for r in runs_to_compare],
                                  markers=[run_markes[r] for r in runs_to_compare],
                                  colors=[run_colors[r] for r in runs_to_compare],
                                  legend_loc=3,
                                  start_offset=-0.1,
                                  use_syst_err_only=False,
                                  use_full_error=False
                                  )


# # compare My 2017 results
# runs_to_compare = [normal_run, run_hot_cec]
# MPLPlotter.plot_par_from_combined(db, runs_to_compare,
#                                   isotopes, 'shift', show_pl=True,
#                                   plot_runs_seperate=True, literature_run=final_2017_run,
#                                   literature_name=run_comments[final_2017_run],
#                                   lit_color=run_colors[final_2017_run],
#                                   lit_marker=run_markes[final_2017_run],
#                                   comments=[run_comments[r] for r in runs_to_compare],
#                                   markers=[run_markes[r] for r in runs_to_compare],
#                                   colors=[run_colors[r] for r in runs_to_compare],
#                                   legend_loc=3,
#                                   start_offset=-0.1,
#                                   use_syst_err_only=False,
#                                   use_full_error=False
#                                   )

# compare 2017 results
runs_to_compare = [exp_liang_run17]
MPLPlotter.plot_par_from_combined(db, runs_to_compare,
                                  isotopes, 'shift', show_pl=True,
                                  plot_runs_seperate=True, literature_run=final_2017_run,
                                  literature_name=run_comments[final_2017_run],
                                  lit_color=run_colors[final_2017_run],
                                  lit_marker=run_markes[final_2017_run],
                                  comments=[run_comments[r] for r in runs_to_compare],
                                  markers=[run_markes[r] for r in runs_to_compare],
                                  colors=[run_colors[r] for r in runs_to_compare],
                                  legend_loc=3,
                                  start_offset=-0.1,
                                  use_syst_err_only=False,
                                  use_full_error=False
                                  )

# compare most of the 2016 / 2017 data relative to 2017 data
runs_to_compare = [exp_liang_run17, run2016_final_db, exp_liang_run16, steudel_1980_run]
MPLPlotter.plot_par_from_combined(db, runs_to_compare,
                                  isotopes, 'shift', show_pl=True,
                                  plot_runs_seperate=True, literature_run=final_2017_run,
                                  literature_name=run_comments[final_2017_run],
                                  lit_color=run_colors[final_2017_run],
                                  lit_marker=run_markes[final_2017_run],
                                  comments=[run_comments[r] for r in runs_to_compare],
                                  markers=[run_markes[r] for r in runs_to_compare],
                                  colors=[run_colors[r] for r in runs_to_compare],
                                  legend_loc=3,
                                  start_offset=-0.1,
                                  use_syst_err_only=False,
                                  use_full_error=False
                                  )


# compare most of the 2016 / 2017 data relative to 2016 data
runs_to_compare = [exp_liang_run16, final_2017_run, exp_liang_run17, steudel_1980_run]
MPLPlotter.plot_par_from_combined(db, runs_to_compare,
                                  isotopes, 'shift', show_pl=True,
                                  plot_runs_seperate=True, literature_run=run2016_final_db,
                                  literature_name=run_comments[run2016_final_db],
                                  lit_color=run_colors[run2016_final_db],
                                  lit_marker=run_markes[run2016_final_db],
                                  comments=[run_comments[r] for r in runs_to_compare],
                                  markers=[run_markes[r] for r in runs_to_compare],
                                  colors=[run_colors[r] for r in runs_to_compare],
                                  legend_loc=3,
                                  start_offset=-0.1,
                                  use_syst_err_only=False,
                                  use_full_error=False
                                  )

# compare most of the 2016 / 2017 data relative to 2016 data
runs_to_compare = [final_2017_run]
MPLPlotter.plot_par_from_combined(db, runs_to_compare,
                                  isotopes, 'shift', show_pl=True,
                                  plot_runs_seperate=True, literature_run=run2016_final_db,
                                  literature_name=run_comments[run2016_final_db],
                                  lit_color=run_colors[run2016_final_db],
                                  lit_marker=run_markes[run2016_final_db],
                                  comments=[run_comments[r] for r in runs_to_compare],
                                  markers=[run_markes[r] for r in runs_to_compare],
                                  colors=[run_colors[r] for r in runs_to_compare],
                                  legend_loc=3,
                                  start_offset=-0.1,
                                  use_syst_err_only=False,
                                  use_full_error=False
                                  )


''' King plot and charge radii '''

king = KingFitter(db, showing=True, litvals=delta_lit_radii, plot_y_mhz=False, font_size=26)
king.kingFit(alpha=0, findBestAlpha=False, run=current_run, find_slope_with_statistical_error=False)
# king.calcChargeRadii(isotopes=isotopes, run=current_run, plot_evens_seperate=True)

king.kingFit(alpha=365, findBestAlpha=True, run=final_2017_run)
radii_alpha = king.calcChargeRadii(isotopes=isotopes, run=final_2017_run, plot_evens_seperate=True)
print('radii with alpha', radii_alpha)

''' compare radii '''

# create literature radii dict from 2016 run:
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(''' SELECT iso, val, statErr, systErr, rChi From Combined WHERE parname = ? AND run = ? ORDER BY iso''',
            ('delta_r_square', run2016_final_db))
radii2016 = cur.fetchall()
con.close()

radii2016_dict = {}
for iso, val, statErr, systErr, rChi in radii2016:
    radii2016_dict[iso] = (val, statErr, systErr)

print('shifts from 2016:')
TiTs.print_dict_pretty(radii2016_dict)

runs_to_compare = [final_2017_run]  # , run_hot_cec]
MPLPlotter.plot_par_from_combined(db, runs_to_compare,
                                  isotopes, 'delta_r_square', show_pl=True,
                                  plot_runs_seperate=True, literature_run=run2016_final_db,
                                  literature_name=run_comments[run2016_final_db],
                                  use_syst_err_only=True,
                                  comments=[run_comments[r] for r in runs_to_compare],
                                  markers=[run_markes[r] for r in runs_to_compare],
                                  colors=[run_colors[r] for r in runs_to_compare],
                                  start_offset=-0.05)


''' A and B factors and moments '''

pars = ['Al', 'Au', 'Bl', 'Bu']
a_fac_runs = [final_2017_run]


# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute(''' UPDATE Combined SET statErrForm = ? ''', ('applyChi(err, rChi)',))
# syst_error = str('systE(accVolt_d=%s, offset_d=%s)' % ('1.5 * 10 ** -4', '1.5 * 10 ** -4'))
# for par in pars:
#     cur.execute('''UPDATE Combined SET systErrForm = ? WHERE parname = ?''', (syst_error, par))
# con.commit()
# con.close()
#
for iso in isotopes:
    if int(iso[:2]) % 2:
        for par in pars:
            Analyzer.combineRes(iso, par, final_2017_run, db)
#
#
# get results from db:
print('\n\n\niso\tAu\td_Au\tAl\td_Al\tAu/Al\td_Au/Al')

al = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Al', print_extracted=False)
au = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Au', print_extracted=False)
bl = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Bl', print_extracted=False)
bu = Tools.extract_from_combined(a_fac_runs, db, odd_isotopes, par='Bu', print_extracted=False)
ratios = []
b_ratios = []
d_ratios = []
d_b_ratios = []
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
            a_up = (0, 0, 0, 0) if au[run][iso] == (None, None, None, None) else au[run][iso]  # when fixed use 0's
            b_up = (0, 0, 0, 0) if bu[run][iso] == (None, None, None, None) else bu[run][iso]
            b_low = (0, 0, 0, 0) if bl[run][iso] == (None, None, None, None) else bl[run][iso]
            ratio = a_up[0] / a_low[0]
            delta_ratio = np.sqrt(
                (a_up[1] / a_low[0]) ** 2 + (a_up[0] * a_low[1] / (a_low[0] ** 2)) ** 2
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
                b_ratios.append(b_ratio)
                d_b_ratios.append(delta_b_ratio)
            ratios.append(ratio)
            d_ratios.append(delta_ratio)
            q_from_upper = quadrupol_moment(b_up[0], b_up[1], b_up[2], upper=True)
            q_from_lower = quadrupol_moment(b_low[0], b_low[1], b_low[2], upper=False)
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
            q_moments.append((mass, nucl_spin, q_from_lower[0], q_from_lower[3]))
            magn_moments.append((mass, nucl_spin, mu[0], mu[3]))

print('magnetic moments: %s ' % magn_moments)

al2016 = Tools.extract_from_combined([run2016_final], db2016, odd_isotopes2016,
                                     par='Al', print_extracted=False)[run2016_final]
au2016 = Tools.extract_from_combined([run2016_final], db2016, odd_isotopes2016,
                                     par='Au', print_extracted=False)[run2016_final]
bl2016 = Tools.extract_from_combined([run2016_final], db2016, odd_isotopes2016,
                                     par='Bl', print_extracted=False)[run2016_final]
bu2016 = Tools.extract_from_combined([run2016_final], db2016, odd_isotopes2016,
                                     par='Bu', print_extracted=False)[run2016_final]

a_bs_2016 = [('Al', al2016), ('Au', au2016), ('Bl', bl2016), ('Bu', bu2016)]
# write 2016 results to 2017db
con = sqlite3.connect(db)
cur = con.cursor()
for par, val_dict in a_bs_2016:
    for iso, ret_tuple in val_dict.items():
        val, stat_err, syst_err, r_chi = ret_tuple
        cur.execute(
            '''INSERT OR IGNORE INTO Combined (iso, parname, run, config,
     val, statErr, systErr, rChi) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (iso, par, run2016_final_db, '[]', val, stat_err, syst_err, r_chi))
        cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ?
 WHERE parname = ? AND iso = ? AND run = ?''', (val, stat_err, syst_err, r_chi, par, iso, run2016_final_db))
con.commit()
con.close()

# odd_isotopes = ['61_Ni', '65_Ni']

print('---------------------------- plotting A&Bs now ----------------------------')
for par in pars:
    try:
        runs_to_compare = [final_2017_run]
        MPLPlotter.plot_par_from_combined(db, runs_to_compare,
                                          odd_isotopes, par, show_pl=True,
                                          plot_runs_seperate=True, literature_run=run2016_final_db,
                                          literature_name=run_comments[run2016_final_db],
                                          lit_color=run_colors[run2016_final_db],
                                          lit_marker=run_markes[run2016_final_db],
                                          markers=[run_markes[r] for r in runs_to_compare],
                                          colors=[run_colors[r] for r in runs_to_compare],
                                          comments=[run_comments[r] for r in runs_to_compare],
                                          start_offset=-0.05)

        runs_to_compare = [exp_liang_run16, final_2017_run, exp_liang_run17]
        MPLPlotter.plot_par_from_combined(db, runs_to_compare,
                                          odd_isotopes, par, show_pl=True,
                                          plot_runs_seperate=True, literature_run=run2016_final_db,
                                          literature_name=run_comments[run2016_final_db],
                                          lit_color=run_colors[run2016_final_db],
                                          lit_marker=run_markes[run2016_final_db],
                                          markers=[run_markes[r] for r in runs_to_compare],
                                          colors=[run_colors[r] for r in runs_to_compare],
                                          comments=[run_comments[r] for r in runs_to_compare],
                                          start_offset=-0.05)

        MPLPlotter.clear()
    except Exception as e:
        print('error: %s' % e)

