"""
Created on 

@author: simkaufm

Module Description:  Analysis of the Nickel Data from COLLAPS taken on 28.04.-03.05.2016
"""

import ast
import math
import os
import sqlite3
from datetime import datetime

import numpy as np

import Analyzer
import BatchFit
import MPLPlotter
import Physics
import TildaTools as TiTs
import Tools
from KingFitter import KingFitter

''' settings '''

pars_to_combine = None  # ['center', 'Al', 'Au', 'Bl', 'Bu']
perform_batch_fit = False
combine_shifts = False
shifts_file = 'shifts_2016.txt'
comb_centers = False
show_moments_etc = True
perf_king_plot = False
print_plot_shifts = False


''' working directory: '''

workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder = os.path.join(workdir, 'Ni_April2016_mcp')

db = os.path.join(workdir, 'Ni_workspace.sqlite')

runs = ['wide_gate_asym', 'wide_gate_asym_67_Ni']

isotopes = ['%s_Ni' % i for i in range(58, 71)]
isotopes.remove('69_Ni')
odd_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2]
even_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2 == 0]
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']

Tools.add_missing_columns(db)

overwrites_for_file_num_determination = {'60_Ni_trs_run113_sum114.xml': ['113+114'],
                                         '60_Ni_trs_run192_sum193.xml': ['192+193'],
                                         '60_Ni_trs_run198_sum199_200.xml': ['198+199+200'],
                                         '60_Ni_trs_run117_sum118.xml': ['117+118'],
                                         '60_Ni_trs_run122_sum123.xml': ['122+123'],
                                         '67Ni_no_protonTrigger_3Tracks_Run191.mcp': ['191'],
                                         '67Ni_sum_Run061_Run062.xml': ['061+062'],
                                         '67Ni_sum_Run063_Run064_Run065.xml': ['063+064+065'],
                                         '67_Ni_sum_run243_and_run248.xml': ['243+248'],
                                         '60_Ni_trs_run239_sum240.xml': ['239+240'],
                                         '70Ni_protonTrigger_Run248_sum_252_254_259_265.xml': ['248+252+254+259+265'],
                                         '60_Ni_trs_run266_sum267.xml': ['266+267'],
                                         '70_Ni_trs_run268_plus_run312.xml': ['268+312']}

''' Masses '''
# masses = {
#     '56_Ni': (55942127.872, 0.452),
#     '57_Ni': (56939791.525, 0.624),
#     '58_Ni': (57935342.4, 0.5),
#     '59_Ni': (58934346.2, 0.5),
#     '60_Ni': (59930785.9, 0.5),
#     '61_Ni': (60931055.6, 0.5),
#     '62_Ni': (61928345.4, 0.6),
#     '63_Ni': (62929669.6, 0.6),
#     '64_Ni': (63927966.8, 0.6),
#     '65_Ni': (64930085.2, 0.6),
#     '66_Ni': (65929139.3, 1.5),
#     '67_Ni': (66931569, 3),
#     '68_Ni': (67931869, 3),
#     '69_Ni': (68935610, 4),
#     '70_Ni': (69936431.3, 2.3),
#     '71_Ni': (70940518.964, 2.401),
#     '72_Ni': (71941785.926, 2.401)
# }
#
# con = sqlite3.connect(db)
# cur = con.cursor()
# for iso, mass_tupl in masses.items():
#     cur.execute('''UPDATE Isotopes SET mass = ?, mass_d = ? WHERE iso = ? ''',
#                 (mass_tupl[0] * 10 ** -6, mass_tupl[1] * 10 ** -6, iso))
# con.commit()
# con.close()

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

# # write literature shifts to db:
# con = sqlite3.connect(db)
# cur = con.cursor()
# for iso, shift_tuple in literature_shifts.items():
#     cur.execute(
#         '''INSERT OR IGNORE INTO Combined (iso, parname, run, config,
#  val, statErr, systErr) VALUES (?, ?, ?, ?, ?, ?, ?)''',
#         (iso, 'shift', 'Steudel_1980', '[]', shift_tuple[0], shift_tuple[1], 0.0))
# con.commit()
# con.close()


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
    '58_Ni': (4.8386, np.sqrt(0.0009 ** 2 + 0.0019 ** 2)),
    '60_Ni': (4.8865, np.sqrt(0.0008 ** 2 + 0.002 ** 2)),
    '61_Ni': (4.9005, np.sqrt(0.001 ** 2 + 0.0017 ** 2)),
    '62_Ni': (4.9242, np.sqrt(0.0009 ** 2 + 0.002 ** 2)),
    '64_Ni': (4.9481, np.sqrt(0.0009 ** 2 + 0.0019 ** 2))
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

delta_lit_radii_58 = {iso: [
    lit_vals[0] ** 2 - lit_radii['58_Ni'][0] ** 2,
    np.sqrt(lit_vals[1] ** 2 + lit_radii['58_Ni'][1] ** 2)]
                   for iso, lit_vals in sorted(lit_radii.items())}
delta_lit_radii_58.pop('58_Ni')


print(
    'iso\t<r^2>^{1/2}_{0µe}\t\Delta<r^2>^{1/2}_{0µe}\t<r^2>^{1/2}_{0µe}(A-A_{60})\t\Delta <r^2>^{1/2}_{0µe}(A-A_{60})')
for iso, radi in sorted(lit_radii.items()):
    dif = delta_lit_radii.get(iso, (0, 0))
    print('%s\t%.3f\t%.3f\t%.5f\t%.5f' % (iso, radi[0], radi[1], dif[0], dif[1]))

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
# 5/2- states are isomere with 5.34 ns life time. Values out of Nuclear Moments 2014
# 3d9(2D)4s  	 3D 3
b_lower_lit = -102.979  # MHz
d_b_lower_lit = 0.016  # MHz

# Q = B / eVzz -> eVzz = B/Q  see Neugart_LNP_700

e_Vzz_lower = b_lower_lit / q_literature_61_Ni
d_e_Vzz_lower = np.sqrt(
    (d_b_lower_lit / q_literature_61_Ni) ** 2 +
    (b_lower_lit * d_q_literature_61_Ni / (q_literature_61_Ni ** 2)) ** 2
)
print(e_Vzz_lower, d_e_Vzz_lower)
print('eVzz_lower = %.3f(%.0f) b/ kHz' % (e_Vzz_lower * 1000, d_e_Vzz_lower * 1e6))

# for the upper state  3d9(2D)4p  	 3P° 2, no b factor was measured
# therefore i will need to get the e_Vzz_upper from my results on 61_Ni and the q_literature_61_Ni
# do not do this!


def quadrupol_moment(b, d_stat_b, d_syst_b):
    """
    get the quadrupolemoment for the lower B with the given eVzz from literature with
     Q = B / eVzz
    """
    if b:
        e_vzz = e_Vzz_lower
        d_e_vzz = d_e_Vzz_lower
        q = b / e_vzz
        d_stat_q = abs(d_stat_b / e_vzz)
        d_syst_q = np.sqrt(
            (d_syst_b / e_vzz) ** 2 +
            (b * d_e_vzz / (e_vzz ** 2)) ** 2
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
mu_list_schmidt_eff = [mu_schmidt(each[2], each[1], False, g_s=-2.6782) for each in levels]
print('level \t µ_free \t g_free \t µ_eff \t g_eff')
try:
    for ind, each in enumerate(levels):
        lvl, l, nuc_spin = each
        print('%s\t%.2f\t%.2f\t%.2f\t%.2f' % (lvl, mu_list_schmidt[ind], mu_list_schmidt[ind] / nuc_spin,
                                             mu_list_schmidt_eff[ind], mu_list_schmidt_eff[ind] / nuc_spin))
except Exception as e:
    print(e)
''' crawling '''

# Tools.crawl(db, 'Ni_April2016_mcp')

# ''' laser wavelength: '''
wavenum = 28393.0  # cm-1
freq = Physics.freqFromWavenumber(wavenum)
# freq -= 1256.32701
print(freq, Physics.wavenumber(freq), 0.5 * Physics.wavenumber(freq))
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
volt_div_ratio = "{'accVolt': 1000.05, 'offset': {'prema': 1000.022, 'agilent': 999.985}}"  # from datasheet
# volt_div_ratio = "{'accVolt': 1000.05, 'offset': {'prema': 1000.442, 'agilent': 1000.405}}"  # found chisquare min
# volt_div_ratio = "{'accVolt': 998.85, 'offset': {'prema': 999.222, 'agilent': 999.185}}"  # found chisquare min
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''UPDATE Files SET voltDivRatio = ?''', (volt_div_ratio,))
con.commit()
con.close()

''' diff doppler 60Ni 30kV'''
diffdopp60 = Physics.diffDoppler(850343019.777062, 30000, 60)  # 14.6842867127 MHz/V

''' transition wavelenght: '''
# observed_wavenum = 28364.39  # cm-1  observed wavenum from NIST, mass is unclear.
# # transition_freq = 850342663.9020721  # final value, observed from voltage calibration
transition_freq = 850343019.777  # value from NIST, mass unclear
#
# # # transition_freq = Physics.freqFromWavenumber(observed_wavenum)
# # print('transition frequency: %s ' % transition_freq)
# #
transition_freq += 1256.32701 - 460  # correction from fitting the 60_Ni references

# # volt_div_ratio = "{'accVolt': 1000.05, 'offset': {'prema': 1000.442, 'agilent': 1000.405}}"  # found chisquare min
# transition_freq = 850343804.45241  # see one line above ;)

# volt_div_ratio = "{'accVolt': 998.85, 'offset': {'prema': 999.222, 'agilent': 999.185}}" # found chisquare min
# transition_freq = 850344366.10401  # see one line above ;)

con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''UPDATE Lines SET frequency = ?''', (transition_freq,))
con.commit()
con.close()

''' isotope shift and batch fitting'''


def isotope_shift_batch_fitting(run_isos, run_67_ni, isotopes_batch_fit, pars=None,
                                combine_shift=True, perform_fit=True, combine_centers=False,
                                store_shifts_to=''):
    """
    get all configs for the used runs and fit those. then combine the reults.
    Be careful with 67Ni here scaler numeration is different.
    Currently only works for runs = ['wide_gate_asym', 'wide_gate_asym_67_Ni']
    """
    # get all current configs:
    configs = {}
    files_with_error = []
    # print('run \t iso \t val \t statErr \t rChi')
    print('will fit the isotopes: %s' % isotopes_batch_fit)
    st_time = datetime.now()
    print('started at: %s' % st_time)

    # get all needed files, sorted by run:
    # note: the config for 67Ni should also be stored under run_iso name
    configs[run_isos] = {}
    for iso in isotopes_batch_fit:
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute(
            '''SELECT config, val, statErr, rChi FROM Combined WHERE iso = ? AND run = ? AND parname = ? ''',
            (iso, run_isos, 'shift'))
        data = cur.fetchall()
        con.close()
        if len(data):
            config, val, statErr, rChi = data[0]
            # print('%s \t %s \t %s \t %s \t %s \n %s' % (run, iso, val, statErr, rChi, config))
            configs[run_isos][iso] = ast.literal_eval(config)
        else:
            print('error: run: %s iso: %s no config found' % (run_isos, iso))

    # sort files in cfg by ref or iso
    all_iso_shift_files = {}
    all_iso_shift_files_by_iso = {}
    for run_name in [run_isos, run_67_ni]:
        all_iso_shift_files[run_name] = {'refs': [], 'isoFiles': []}
        all_iso_shift_files_by_iso[run_name] = {'refs': {}, 'isoFiles': {}}
        for iso, cfg in configs[run_isos].items():
            all_iso_shift_files_by_iso[run_name]['isoFiles'][iso] = []
            all_iso_shift_files_by_iso[run_name]['refs'][iso] = []
            if ('67' in iso and '67' in run_name) or ('67' not in iso and '67' not in run_name):
                for each_cfg in cfg:
                    ref_before = each_cfg[0]
                    iso_files = each_cfg[1]
                    ref_after = each_cfg[2]
                    all_iso_shift_files[run_name]['refs'] += ref_before + ref_after
                    all_iso_shift_files[run_name]['isoFiles'] += iso_files
                    all_iso_shift_files_by_iso[run_name]['isoFiles'][iso] += iso_files
                    all_iso_shift_files_by_iso[run_name]['refs'][iso] += ref_before + ref_after
        all_iso_shift_files[run_name]['refs'] = sorted(list(set(all_iso_shift_files[run_name]['refs'])))
        all_iso_shift_files[run_name]['isoFiles'] = sorted(list(set(all_iso_shift_files[run_name]['isoFiles'])))

    print('-------------- all_all_iso_shift_files -------------')
    print(all_iso_shift_files)

    for run in [run_isos, run_67_ni]:
        if perform_fit:
            print('run is: ', run, [run_isos, run_67_ni])
            ret = TiTs.select_from_db(db, 'Lines.reference, lines.refRun',
                                      'Runs JOIN Lines ON Runs.lineVar = Lines.lineVar', [['Runs.run'], [run]],
                                      caller_name='Ni_analysis')
            if ret is not None:
                ref, refRun = ret[0]
            else:
                raise Exception('refRun not found')
            print('---------- working on %s with refRun: %s -------------' % (run, refRun))
            ref_fits, ref_files_w_error = BatchFit.batchFit(all_iso_shift_files[run]['refs'], db, refRun)
            print('------------------- all ref files of run %s fitted -------------------' % run)
            iso_fits, iso_files_w_error = BatchFit.batchFit(all_iso_shift_files[run]['isoFiles'], db, run)
            print('------------------- all files of run %s fitted -------------------' % run)
            files_with_error.append(ref_files_w_error)
            files_with_error.append(iso_files_w_error)
    if perform_fit and '67_Ni' in isotopes_batch_fit:
        # rename the fit reults for 'wide_gate_asym_67_Ni' to 'wide_gate_asym'
        # in order to have all all in combined for run 'wide_gate_asym'
        # will only be necessary if a fit was performed otherwise results should be already renamed
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''DELETE FROM FitRes WHERE run = ? AND iso = ?''', (run_isos, '67_Ni'))  # delete old Fit results
        con.commit()
        cur.execute('''UPDATE FitRes SET run = ? WHERE iso = ?''', (run_isos, '67_Ni'))  # rename
        con.commit()
        con.close()

    syst_error = str('systE(accVolt_d=%s, offset_d=%s)' % ('1.5 * 10 ** -4', '1.5 * 10 ** -4'))
    # now combine results
    for iso in isotopes_batch_fit:
        if pars is not None:
            for par in pars:
                Analyzer.combineRes(iso, par, run_isos, db)  # create db entry then add error formulas
                if par in ['sigma', 'Al', 'Au', 'Bl', 'Bu', 'center']:
                    relevant_files = [isos[0] for refs_before, isos, refs_after in configs[run_isos][iso]]
                    con = sqlite3.connect(db)
                    cur = con.cursor()
                    cur.execute('''UPDATE Combined SET systErrForm = ? WHERE parname = ?''', (syst_error, par))
                    cur.execute(''' UPDATE Combined SET statErrForm = ? ''', ('applyChi(err, rChi)',))
                    cur.execute(''' UPDATE Combined SET config = ? WHERE iso = ? AND run = ? AND parname = ? ''',
                                (str(relevant_files), iso, run_isos, par))
                    con.commit()
                    con.close()
                    Analyzer.combineRes(iso, par, run_isos, db)  # then combine again
                    if iso == '67_Ni' and par == 'Au':
                        # 67_Ni only has one file were A&B can be left free,
                        #  therefore it needs to be treated seperately
                        ni67_Al = 1090.33170375
                        ni67_Al_stat_err = 2.352703835
                        ni67_Au = 424.756765644
                        ni67_Au_stat_err = 3.37324439988
                        accvolt_d = 1.5 * 10 ** -4
                        offset_d = 1.5 * 10 ** -4
                        # Au needs to be combined from all Al in 67Ni by multiplikation with the found ratio:
                        # 0.3893 +/- 0.0006
                        Analyzer.combineRes(iso, par, run_isos, db, combine_from_par='Al',
                                            combine_from_multipl=0.3893, combine_from_mult_err=0.0006)

        if iso != '60_Ni' and combine_shift:
            Analyzer.combineShiftByTime(
                iso, run_isos, db,
                show_plot=False, overwrite_file_num_det=overwrites_for_file_num_determination
            )  # create db entry then add error formulas
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute(''' UPDATE Combined SET statErrForm = ? ''', ('applyChi(err, rChi)',))
            syst_error = str('systE(accVolt_d=%s, offset_d=%s)' % ('1.5 * 10 ** -4', '1.5 * 10 ** -4'))
            cur.execute('''UPDATE Combined SET systErrForm = ? WHERE parname = ?''', (syst_error, 'shift'))
            con.commit()
            con.close()
            Analyzer.combineShiftByTime(
                iso, run_isos, db, show_plot=False,
                overwrite_file_num_det=overwrites_for_file_num_determination,
                store_to_file_in_combined_plots=store_shifts_to)  # then combine again
        if combine_centers:
            # now combine the center of all relevant iso files, to have the plot on the harddrive
            Analyzer.combineRes(iso, 'center', run_isos,
                                db, only_this_files=all_iso_shift_files_by_iso[run_isos]['isoFiles'][iso])
    done_time = datetime.now()
    elapsed = done_time - st_time
    print('finished bacthfitting at %s after %.1f min' % (done_time, elapsed.seconds / 60))
    return files_with_error


# Batchfitting and parameter combination Isotope Shift ...


files_w_err = isotope_shift_batch_fitting(
    'wide_gate_asym', 'wide_gate_asym_67_Ni', isotopes,
    pars=pars_to_combine, combine_shift=combine_shifts,
    perform_fit=perform_batch_fit, combine_centers=comb_centers, store_shifts_to=shifts_file)
print('--------------------------------------------------------------------')
print('files with error during batchfit: ', (['wide_gate_asym', 'wide_gate_asym_67_Ni'], files_w_err))
print('--------------------------------------------------------------------')

''' Divider Ratio Determination '''
acc_div_start = 1000.05
offset_prema_div_start = 1000.022
offset_agi_div_start = 999.985
# get the relevant files which need to be fitted in the following:
div_ratio_relevant_stable_files = {}
div_ratio_relevant_stable_files['60_Ni'] = []


def chi_square_finder(acc_dev_list, offset_dev_list, runs):
    offset_prema_div_ratios = [[]]
    offset_agi_div_ratios = [[]]
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

        for off_div in offset_dev_list:

            freq_correction = 17.82 * off_div / 100 - 9.536  # determined for the region around acc_div = 1000.05
            new_freq = freq + freq_correction

            curent_prema_off_div = offset_prema_div_start + off_div / 100

            current_agi_off_div = offset_agi_div_start + off_div / 100

            con = sqlite3.connect(db)
            cur = con.cursor()
            volt_div_ratio = "{'accVolt': %.3f, 'offset': {'prema': %.3f, 'agilent': %.3f}}" % (
                current_acc_div, curent_prema_off_div, current_agi_off_div)

            cur.execute('''UPDATE Files SET voltDivRatio = ? ''', (volt_div_ratio,))
            cur.execute('''UPDATE Lines SET frequency = ?''', (new_freq,))
            con.commit()
            con.close()

            # Batchfitting:
            fitres = isotope_shift_batch_fitting(runs, stables, combine_shift=False)

            # combineRes only when happy with voltdivratio, otherwise no use...
            # [[Analyzer.combineRes(iso, par, run, db) for iso in stables] for par in pars]
            try:
                shifts = {iso: Analyzer.combineShiftByTime(
                    iso, run_chi_finder, db, overwrite_file_num_det=overwrites_for_file_num_determination) for iso in
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
            offset_prema_div_ratios[acc_vol_ratio_index].append(curent_prema_off_div)
            offset_agi_div_ratios[acc_vol_ratio_index].append(current_agi_off_div)
            try:
                off_str = str(round(current_agi_off_div, 3)).replace('.', '_')
                acc_str = str(round(current_acc_div, 3)).replace('.', '_')
                save_name = os.path.join(workdir, 'divider_ratio', 'acc_%s_off_%s.png' % (acc_str, off_str))
                MPLPlotter.plot_par_from_combined(db, ['wide_gate_asym'], list(literature_shifts.keys()), 'shift',
                                                  literature_run='Steudel_1980', plot_runs_seperate=False,
                                                  literature_name='A. Steudel (1980)\n '
                                                                  'acc_ratio: %s\n off_ratio: %s' % (acc_str, off_str),
                                                  show_pl=False,
                                                  save_path=save_name)
                MPLPlotter.clear()
            except Exception as e:
                print('could not plot, moving on, error was %s' % e)

        acc_vol_ratio_index += 1
        chisquares.append([])
        fit_res.append([])
        offset_prema_div_ratios.append([])
        offset_agi_div_ratios.append([])
    chisquares = chisquares[:-1]  # delete last empty list in order not to confuse.
    fit_res = fit_res[:-1]
    offset_prema_div_ratios = offset_prema_div_ratios[:-1]
    offset_agi_div_ratios = offset_agi_div_ratios[:-1]
    print('acceleration voltage divider ratios: \n %s ' % str(acc_ratios))
    print('offset voltage divider ratios prema: \n %s ' % str(offset_prema_div_ratios))
    print('offset voltage divider ratios agilent: \n %s ' % str(offset_agi_div_ratios))
    print('Chi^2 are: \n %s ' % str(chisquares))

    print(fit_res)

    print('the following files failed during BatchFit: \n')
    for acc_volt_ind, each in enumerate(fit_res):
        print('for acc volt div ratio: %s' % acc_ratios[acc_volt_ind])
        for offset_volt_ind, inner_each in enumerate(each):
            [print(fit_res_tpl) for fit_res_tpl in inner_each if len(fit_res_tpl[2])]
    print('acc\toff_prema\toff_agi\tchisquare')
    for acc_ind, acc_rat in enumerate(acc_ratios):
        for off_ind, off_rat in enumerate(offset_prema_div_ratios[acc_ind]):
            print(('%.3f\t%.3f\t%.3f\t%.3f' % (acc_rat, off_rat, offset_agi_div_ratios[acc_ind][off_ind],
                                               chisquares[acc_ind][off_ind])).replace('.', ','))
    return acc_ratios, offset_prema_div_ratios, offset_agi_div_ratios, chisquares


def chisquare_finder_kepco(run_chi_finder, kepco_dif_list):
    chisquare_list = []
    starting_lineMult = 0.050415562
    best_chi_sq_kepco = (99999999, starting_lineMult)
    for kepco_dif in kepco_dif_list:
        kepco_dif += starting_lineMult
        # set db to next value
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute('''UPDATE Files SET lineMult = ? ''', (kepco_dif,))
        # cur.execute('''UPDATE Lines SET frequency = ?''', (new_freq,))
        con.commit()
        con.close()

        # Batchfitting:
        isotope_shift_batch_fitting([run_chi_finder], stables)

        try:
            shifts = {iso: Analyzer.combineShiftByTime(
                iso, run_chi_finder, db, overwrite_file_num_det=overwrites_for_file_num_determination) for iso in
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
        print('chisquare: %.3f, lineMult: %.5f' % (chisq, kepco_dif))
        chisquare_list.append((chisq, kepco_dif))
        if chisq <= best_chi_sq_kepco[0]:
            best_chi_sq_kepco = (chisq, kepco_dif)
    # setting db to best value and refitting
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''UPDATE Files SET lineMult = ? ''', (best_chi_sq_kepco[1],))
    # cur.execute('''UPDATE Lines SET frequency = ?''', (new_freq,))
    con.commit()
    con.close()
    # Batchfitting:
    isotope_shift_batch_fitting([run_chi_finder], stables)
    print('chisquares: ', chisquare_list)

    return best_chi_sq_kepco


# remember:
#  acc_div_start = 1000.05
# offset_prema_div_start = 1000.022
# offset_agi_div_start = 999.985

# acc_volt_deviation_list = [-120]  # each ele will eb divided by 100 and then added to the current ratio
# off_volt_deviation_list = [-80]  # each ele will eb divided by 100 and then added to the current ratio
# # acc_volt_deviation_list = list(range(-190, -125, 5))
# # off_volt_deviation_list = list(range(-30, 85, 5))
# # each ele will eb divided by 100 and then added to the current ratio
# # off_volt_deviation_list += list(range(-14, 10, 3))
#
# # stables.remove('61_Ni')
# # literature_shifts.pop('61_Ni')
# # print('stables: %s' % stables)
#
# fits_to_fit = len(acc_volt_deviation_list) * len(off_volt_deviation_list)
# start_time = datetime.now()
# # time_per_analysis_s = 180  # roughly
# time_per_analysis_s = 155  # roughly fixed Al Bl
# # time_per_analysis_s = 62  # roughly no 61Ni
# expected_elaps_m = time_per_analysis_s * fits_to_fit / 60
# expect_stop = start_time + timedelta(seconds=(time_per_analysis_s * fits_to_fit))
# print('Fitting of %s runs started at %s, should be finished in %.1f m at %s'
#       % (fits_to_fit, start_time, expected_elaps_m, expect_stop))
#
# in_ret = input('continue? [y/n]')
# if in_ret not in ['y', 'Y']:
#     raise Exception('decided to stop')
# start_time = datetime.now()  # overwrite because start time need time to decide
# acc_ratios, offset_prema_div_ratios, offset_agi_div_ratios, chisquares = chi_square_finder(
#     acc_volt_deviation_list, off_volt_deviation_list, ['wide_gate_asym'])
# print('----------------------------')
# print('result of chisquare min: ', acc_ratios, offset_prema_div_ratios, offset_agi_div_ratios, chisquares)
# print('----------------------------')
# done_time = datetime.now()
# elapsed = done_time - start_time
# time_per_fit = elapsed.seconds / fits_to_fit
# print('done at %s after %.1f min with roughly %.1f s/ratio_analysis' % (done_time, elapsed.seconds / 60, time_per_fit))
# # raise Exception('wohooo volt div determination done!')
#
# if len(acc_volt_deviation_list) == 1:
#     # make 2d plot
#     import MPLPlotter
#
#     deg_freedom = 3  # for leaving out 61_Ni
#     red_ch_sq = np.array(chisquares[0]) / deg_freedom
#     MPLPlotter.plot([offset_agi_div_ratios[0], red_ch_sq])
#     MPLPlotter.show(True)

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

try:
    if show_moments_etc:
        # kepco_dif = [each / 10000 for each in range(-20, -11)]
        # # best_chi_square = chisquare_finder_kepco('narrow_gate_asym', [0])
        # best_chi_square = ('???', 0.050415562)
        # print('best chi square kepco: ', best_chi_square)
        # print('literature shifts',  literature_shifts)
        MPLPlotter.plot_par_from_combined(db, ['wide_gate_asym'], list(literature_shifts.keys()), 'shift',
                                          literature_run='Steudel_1980', plot_runs_seperate=False,
                                          literature_name='A. Steudel (1980)')
        # print(isotopes)

        # print('iso\tshift [MHz]\tstatErr [Mhz]\trChi')
        # [print('%s\t%s\t%s\t%s' % (key, val[0], val[1], val[2])) for key, val in sorted(files[runs[0]].items())]

        # print('\n\nfor Excel: \n\n')
        # print('iso\tshift [MHz]\tstatErr [Mhz]\trChi')
        # for key, val in sorted(files[runs[0]].items()):
        #     out_str = '%s\t%s\t%s\t%s' % (key, val[0], val[1], val[2])
        #     out_str = out_str.replace('.', ',')
        #     print(out_str)
        print('\n\n\niso\tAu\td_Au\tAl\td_Al\tAu/Al\td_Au/Al')
        a_fac_runs = ['wide_gate_asym', 'wide_gate_asym_67_Ni']

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
              '\tBu [MHz]\trChi Bu\tBl [MHz]\trChi Bl\tBu/Bl\td_Bu/Bl\tQ_l [b] '
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
                    q_from_lower = quadrupol_moment(b_low[0], b_low[1], b_low[2])
                    mu = magnetic_moment(a_low[0], a_low[1], a_low[2], nucl_spin)
                    mu_print = mu[4]
                    print('%s\t%s'
                          '\t%.2f(%.0f)[%.0f]\t%.2f'
                          '\t%.2f(%.0f)[%.0f]\t%.2f\t%.3f\t%.3f'
                          '\t%.2f(%.0f)[%.0f]\t%.2f'
                          '\t%.2f(%.0f)[%.0f]\t%.2f\t%.3f\t%.3f'
                          '\t%s'
                          '\t%s' % (
                              iso, nucl_spin,
                              a_up[0], a_up[1] * 100, a_up[2] * 100, a_up[3],
                              a_low[0], a_low[1] * 100, a_low[2] * 100, a_low[3], ratio, delta_ratio,
                              b_up[0], b_up[1] * 100, b_up[2] * 100, b_up[3],
                              b_low[0], b_low[1] * 100, b_low[2] * 100, b_low[3], b_ratio, delta_b_ratio,
                              q_from_lower[4],
                              mu_print
                          ))
                    q_moments.append((mass, nucl_spin, q_from_lower[0], q_from_lower[3]))
                    magn_moments.append((mass, nucl_spin, mu[0], mu[3]))

        print('magnetic moments: %s ' % magn_moments)
        # optimize schmidt values:
        # g_l_0 = 0
        # g_s_0 = -3.826
        g_l_0 = 0.125
        g_s_0 = -2.04
        g_l_difs = np.arange(-0.01, 0.01, 0.005)
        g_s_difs = np.arange(-0.01, 0.01, 0.005)
        chi_squares = [[]]
        best_chi = [99999999999999, 0, 0]
        for i, g_l_dif in enumerate(g_l_difs):
            g_l = g_l_0 + g_l_dif
            for j, g_s_dif in enumerate(g_s_difs):
                g_s = g_s_0 + g_s_dif
                chi_square = 0
                for magn_i, each in enumerate(magn_moments):
                    if each[1] != 2.5:  # l = 1
                        l = 1
                    else:
                        l = 0
                    dif = mu_schmidt(each[1], l, False, g_l=g_l, g_s=g_s) - each[2]
                    d_dif = each[3]
                    chi_square += np.square(dif / d_dif)
                if chi_square < best_chi[0]:
                    best_chi = chi_square, g_l, g_s

        print('best chi square: %.3f %.3f %.3f' % best_chi)
        mu_list_schmidt = [
            (each[0], each[2], mu_schmidt(each[2], each[1], False, g_l=0, g_s=-2.6782)) for each in levels]

        print('level\t\mu(\\nu) / \mu_N')
        for i, each in enumerate(mu_list_schmidt):
            print('%s\t%.2f' % (each[0], each[2]))

        # plot magnetic moments
        magn_mom_fig = MPLPlotter.plt.figure(0, facecolor='white')
        magn_mom_axes = MPLPlotter.plt.axes()
        magn_mom_axes.margins(0.1, 0.1)
        magn_mom_axes.set_xlabel('A')
        magn_mom_axes.set_ylabel('µ [nm]')
        magn_mom_axes.set_xticks([each[0] for each in magn_moments])
        magn_mom_by_spin = []
        colors = ['b', 'g', 'k']
        markers = ['o', 's', 'D']
        for i, spin in enumerate([0.5, 1.5, 2.5]):
            spin_list_x = [mu[0] for mu in magn_moments if mu[1] == spin]
            spin_list_y = [mu[2] for mu in magn_moments if mu[1] == spin]
            spin_list_y_err = [mu[3] for mu in magn_moments if mu[1] == spin]
            if len(spin_list_x):
                label = 'spin: %s' % spin
                spin_line, cap_line, barline = MPLPlotter.plt.errorbar(
                    spin_list_x, spin_list_y, spin_list_y_err, axes=magn_mom_axes,
                    linestyle='None', marker=markers[i], label=label, color=colors[i]
                )
            lit_spin_list_x = [lit_mu[0] + 0.15 for iso, lit_mu in lit_magnetic_moments.items() if lit_mu[1] == spin]
            lit_spin_list_y = [lit_mu[2] for iso, lit_mu in lit_magnetic_moments.items() if lit_mu[1] == spin]
            lit_spin_list_y_err = [lit_mu[3] for iso, lit_mu in lit_magnetic_moments.items() if lit_mu[1] == spin]
            if len(lit_spin_list_x):
                label = 'spin: %s (lit.)' % spin
                lit_spin_line, lit_cap_line, lit_barline = MPLPlotter.plt.errorbar(
                    lit_spin_list_x, lit_spin_list_y, lit_spin_list_y_err, axes=magn_mom_axes,
                    linestyle='None', marker=markers[i], label=label, color=colors[i],
                    markerfacecolor='w', markeredgewidth=1.5, markeredgecolor=colors[i]
                )
            # #  display rel. to 0
            # for j, each in enumerate(lit_spin_list_x):
            #     same_mass = [mu for mu in magn_moments if mu[0] == each - 0.15]
            #     if len(same_mass):
            #         label = 'spin: %s (exp.)' % spin
            #         MPLPlotter.plt.errorbar(
            #             same_mass[0][0], 0, same_mass[0][3], axes=magn_mom_axes,
            #             linestyle='None', marker=markers[i], label=label, color=colors[i]
            #         )
            #         label = 'spin: %s (lit.)' % spin
            #         MPLPlotter.plt.errorbar(
            #             each, lit_spin_list_y[j] - same_mass[0][2], lit_spin_list_y_err[j], axes=magn_mom_axes,
            #             linestyle='None', marker=markers[i], label=label, color=colors[i],
            #             markerfacecolor='w', markeredgewidth=1.5, markeredgecolor=colors[i]
            #         )
            #         print(same_mass)

            for each in mu_list_schmidt:
                if spin == each[1]:
                    hor_line = MPLPlotter.plt.axhline(each[2], label='eff. schmidt: %s' % each[0], color=colors[i])
        MPLPlotter.plt.legend(loc=2, title='magnetic moments')
        MPLPlotter.show(True)

        # plot g factor g = µ/I
        g_fac_fig = MPLPlotter.plt.figure(1, facecolor='white')
        g_fac_axes = MPLPlotter.plt.axes()
        g_fac_axes.margins(0.1, 0.1)
        g_fac_axes.set_xlabel('A')
        g_fac_axes.set_ylabel('g [nm]')
        g_fac_axes.set_xticks([each[0] for each in magn_moments])
        g_fac_by_spin = []
        colors = ['b', 'g', 'k']
        markers = ['o', 's', 'D']
        for i, spin in enumerate([0.5, 1.5, 2.5]):
            spin_list_x = [mu[0] for mu in magn_moments if mu[1] == spin]
            spin_list_y = [mu[2] / spin for mu in magn_moments if mu[1] == spin]
            spin_list_y_err = [mu[3] / spin for mu in magn_moments if mu[1] == spin]
            if len(spin_list_x):
                label = 'spin: %s' % spin
                spin_line, cap_line, barline = MPLPlotter.plt.errorbar(
                    spin_list_x, spin_list_y, spin_list_y_err, axes=g_fac_axes,
                    linestyle='None', marker=markers[i], label=label, color=colors[i]
                )
            lit_spin_list_x = [lit_mu[0] + 0.15 for iso, lit_mu in lit_magnetic_moments.items() if lit_mu[1] == spin]
            lit_spin_list_y = [lit_mu[2] / spin for iso, lit_mu in lit_magnetic_moments.items() if lit_mu[1] == spin]
            lit_spin_list_y_err = [lit_mu[3] / spin for iso, lit_mu in lit_magnetic_moments.items() if lit_mu[1] == spin]
            if len(lit_spin_list_x):
                label = 'spin: %s (lit.)' % spin
                lit_spin_line, lit_cap_line, lit_barline = MPLPlotter.plt.errorbar(
                    lit_spin_list_x, lit_spin_list_y, lit_spin_list_y_err, axes=g_fac_axes,
                    linestyle='None', marker=markers[i], label=label, color=colors[i],
                    markerfacecolor='w', markeredgewidth=1.5, markeredgecolor=colors[i]
                )
            for each in mu_list_schmidt:
                if spin == each[1]:
                    hor_line = MPLPlotter.plt.axhline(each[2] / spin, label='eff. g: %s' % each[0], color=colors[i])
        MPLPlotter.plt.legend(loc=2, title='g-factor')
        MPLPlotter.show(True)

        # plot quadrupole moments
        q_mom_fig = MPLPlotter.plt.figure(2, facecolor='white')
        q_mom_axes = MPLPlotter.plt.axes()
        q_mom_axes.margins(0.1, 0.1)
        q_mom_axes.set_xlabel('A')
        q_mom_axes.set_ylabel('Q [b]')
        q_mom_axes.set_xticks([each[0] for each in q_moments])
        q_mom_by_spin = []
        colors = ['g', 'k']
        markers = ['s', 'D']
        for i, spin in enumerate([1.5, 2.5]):
            spin_list_x = [mu[0] for mu in q_moments if mu[1] == spin]
            spin_list_y = [mu[2] for mu in q_moments if mu[1] == spin]
            spin_list_y_err = [mu[3] for mu in q_moments if mu[1] == spin]
            if spin_list_x:
                label = 'spin: %s' % spin
                spin_line, cap_line, barline = MPLPlotter.plt.errorbar(
                    spin_list_x, spin_list_y, spin_list_y_err, axes=q_mom_axes,
                    linestyle='None', marker=markers[i], label=label, color=colors[i]
                )
            x = []
            y = []
            y_err = []
            for each in lit_q_moments:
                if spin == each[1]:
                    x.append(each[0])
                    y.append(each[2])
                    y_err.append(each[3])
            if len(x):
                label = 'spin: %s (lit.)' % spin
                spin_line_lit, cap_line_lit, barline_lit = MPLPlotter.plt.errorbar(
                    x, y, y_err, axes=q_mom_axes,
                    linestyle='None', marker=markers[i], label=label, color=colors[i], markerfacecolor='w',
                    markeredgewidth=1.5, markeredgecolor=colors[i]
                )
        MPLPlotter.plt.legend(loc=0, title='quadrupole moments')
        MPLPlotter.show(True)

        # # Plot A ratios
        # average, errorprop, rChi = Analyzer.weightedAverage(ratios, d_ratios)
        # print('\nAverage Au/Al: %.5f +/- %.5f \t rChi: %.5f' % (average, errorprop, rChi))
        # MPLPlotter.plt.errorbar(range(59, 69, 2), ratios, d_ratios,
        #                         linestyle='None', marker='o', label=label, color='r')
        # MPLPlotter.get_current_axes().set_xlabel('mass')
        # MPLPlotter.get_current_axes().set_ylabel('A_upper/A_lower')
        # # literature_shifts = {iso: (0, 0) for iso in isotopes}
        # # plot_par_from_combined(['narrow_gate'], files)
        # MPLPlotter.get_current_figure().set_facecolor('w')
        # MPLPlotter.show(True)
        #
        # #Plot b ratios
        # average, errorprop, rChi = Analyzer.weightedAverage(b_ratios, d_b_ratios)
        # print('\nAverage Bu/Bl: %.5f +/- %.5f \t rChi: %.5f' % (average, errorprop, rChi))
        # MPLPlotter.plt.errorbar([59, 61, 65], b_ratios, d_b_ratios,
        #                         linestyle='None', marker='o', label=label, color='r')
        # MPLPlotter.get_current_axes().set_xlabel('mass')
        # MPLPlotter.get_current_axes().set_ylabel('B_upper/B_lower')
        # # literature_shifts = {iso: (0, 0) for iso in isotopes}
        # # plot_par_from_combined(['narrow_gate'], files)
        # MPLPlotter.get_current_figure().set_facecolor('w')
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
# off_divs_result = offset_agi_div_ratios[0]
# chisquares_result = chisquares
#
# import PyQtGraphPlotter as PGplt
# from PyQt5 import QtWidgets
# import sys
# print('plotting image')
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
# input('anything to proceed')
# app.exec()
# input('anything to proceed')
''' Run comparison '''
# for now use wide_gate_asym
# runs_comp = ['wide_gate', 'wide_gate_asym']
# shift_ret = Tools.extract_from_combined(runs_comp, db, isotopes, 'shift')
# wide_g_sym = shift_ret['wide_gate']
# # sym['67_Ni'] = shift_ret['narrow_gate_67_Ni']['67_Ni']
# wide_g = shift_ret['wide_gate_asym']
# # asym['67_Ni'] = shift_ret['narrow_gate_asym_67_Ni']['67_Ni']
# x = []
# y_sym = []
# y_dif = []
# y_err_sym = []
# y_err_asym = []
# for iso, shift_tuple in sorted(wide_g_sym.items()):
#     x += int(iso[:2]),
#     y_sym += 0,
#     y_dif += shift_tuple[0] - wide_g[iso][0],
#     # y_err_sym += np.sqrt(shift_tuple[1] ** 2 + shift_tuple[2] ** 2),
#     # y_err_asym += np.sqrt(wide_g[iso][1] ** 2 + wide_g[iso][2] ** 2),
#     # only statistical errors:
#     y_err_sym += shift_tuple[1],
#     y_err_asym += wide_g[iso][1],
#
# # MPLPlotter.plt.errorbar([new_x-0.1 for new_x in x], y_sym, y_err_sym,
# #                         marker='o', color='b', linestyle='None', label='wide - wide')
# # MPLPlotter.plt.errorbar([new_x+0.1 for new_x in x], y_dif, y_err_asym,
# #                         marker='o', color='r', linestyle='None', label='wide - wide_asym')
# # MPLPlotter.get_current_figure().set_facecolor('w')
# # MPLPlotter.plt.margins(0.1)
# # MPLPlotter.plt.xlabel('A')
# # MPLPlotter.plt.ylabel('shift(wide) - shift(wide_asym) /MHz')
# # MPLPlotter.plt.legend()
# # MPLPlotter.plt.show(True)
#
# extract_dict = Tools.extract_from_fitRes(-1, db, isotopes)
# print('extracted dict:', extract_dict)
# sym_fitres = extract_dict['wide_gate']
# # sym_fitres = TiTs.merge_dicts(extract_dict['wide_gate'], extract_dict['wide_gate_67_Ni'])
# asym_fitres = extract_dict['wide_gate_asym']
# # asym_fitres = TiTs.merge_dicts(extract_dict['wide_gate_asym'], extract_dict['wide_gate_asym_67_Ni'])
#
# run_nums = []
# r_chi_sym = []
# r_chi_diff = []
# center_diff = []
# center_diff_s_err = []
# center_diff_a_err = []
# au_dif = []
# au_dif_s_err = []
# au_dif_a_err = []
# al_dif = []
# al_dif_s_err = []
# al_dif_a_err = []
# bu_diff = []
# bu_dif_s_err = []
# bu_dif_a_err = []
# bl_diff = []
# bl_dif_s_err = []
# bl_dif_a_err = []
# sigma_dif = []
# sigma_dif_s_err = []
# sigma_dif_a_err = []
# for file_name, fitres_tpl in sorted(sym_fitres.items()):
#     sym_iso, sym_run_num, sym_r_chi_sq, sym_pars_dict = fitres_tpl
#     asym_iso, asym_run_num, asym_r_chi_sq, asym_pars_dict = asym_fitres.get(file_name, [None] * 4)
#     if asym_iso is not None:
#         run_nums += sym_run_num,
#         r_chi_sym += sym_r_chi_sq,
#         r_chi_diff += asym_r_chi_sq,
#         center_diff += [sym_pars_dict['center'][0] - asym_pars_dict['center'][0]]
#         center_diff_s_err += sym_pars_dict['center'][1],
#         center_diff_a_err += asym_pars_dict['center'][1],
#         au_dif += [sym_pars_dict['Au'][0] - asym_pars_dict['Au'][0]]
#         au_dif_s_err += sym_pars_dict['Au'][1],
#         au_dif_a_err += asym_pars_dict['Au'][1],
#         al_dif += [sym_pars_dict['Al'][0] - asym_pars_dict['Al'][0]]
#         al_dif_s_err += sym_pars_dict['Al'][1],
#         al_dif_a_err += asym_pars_dict['Al'][1],
#         bu_diff += [sym_pars_dict['Bu'][0] - asym_pars_dict['Bu'][0]]
#         bu_dif_s_err += sym_pars_dict['Bu'][1],
#         bu_dif_a_err += asym_pars_dict['Bu'][1],
#         bl_diff += [sym_pars_dict['Bl'][0] - asym_pars_dict['Bl'][0]]
#         bl_dif_s_err += sym_pars_dict['Bl'][1],
#         bl_dif_a_err += asym_pars_dict['Bl'][1],
#         sigma_dif += [sym_pars_dict['sigma'][0] - asym_pars_dict['sigma'][0]]
#         sigma_dif_s_err += sym_pars_dict['sigma'][1],
#         sigma_dif_a_err += asym_pars_dict['sigma'][1],
#
# # plot rChi square difference:
# # MPLPlotter.clear()
# # MPLPlotter.plt.errorbar([new_x-0.1 for new_x in run_nums], r_chi_sym, marker='o', color='b', linestyle='None')
# # MPLPlotter.plt.errorbar([new_x+0.1 for new_x in run_nums], r_chi_diff, marker='o', color='r', linestyle='None')
# # MPLPlotter.get_current_figure().set_facecolor('w')
# # MPLPlotter.plt.margins(0.1)
# # MPLPlotter.plt.xlabel('run number')
# # MPLPlotter.plt.ylabel('rChiSq')
# # MPLPlotter.plt.show(True)
#
# # plot all parameters as difference between par(Voigt) - par(asymVoigt)
# # for par, dif_list, sym_err, asym_err in [('center', center_diff, center_diff_s_err, center_diff_a_err),
# #                                          ('Au', au_dif, au_dif_s_err, au_dif_a_err),
# #                                          ('Al', al_dif, al_dif_s_err, al_dif_a_err),
# #                                          ('Bu', bu_diff, bu_dif_s_err, bu_dif_a_err),
# #                                          ('Bl', bl_diff, bl_dif_s_err, bl_dif_a_err),
# #                                          ('sigma', sigma_dif, sigma_dif_s_err, sigma_dif_a_err)]:
# #     MPLPlotter.plt.errorbar(
# #         [new_x-0.1 for new_x in run_nums], [0] * len(dif_list), sym_err, marker='o', color='b', linestyle='None')
# #     MPLPlotter.plt.errorbar(
# #         [new_x+0.1 for new_x in run_nums], dif_list, asym_err, marker='o', color='r', linestyle='None')
# #     MPLPlotter.get_current_figure().set_facecolor('w')
# #     MPLPlotter.plt.margins(0.1)
# #     MPLPlotter.plt.xlabel('run number')
# #     MPLPlotter.plt.ylabel('%s(Voigt) - %s(asymVoigt)' % (par, par))
# #     MPLPlotter.plt.show(True)

''' King Plot Analysis '''
run = 'wide_gate_asym'
if perf_king_plot:
    # raise (Exception('stopping before king fit'))
    # delta_lit_radii.pop('62_Ni')  # just to see which point is what
    king = KingFitter(db, showing=True, litvals=delta_lit_radii, plot_y_mhz=False, font_size=18, ref_run=run)
    # run = 'narrow_gate_asym'
    # isotopes = sorted(delta_lit_radii.keys())
    # print(isotopes)
    # king.kingFit(alpha=0, findBestAlpha=False, run=run, find_slope_with_statistical_error=True)
    king.kingFit(alpha=0, findBestAlpha=False, run=run, find_slope_with_statistical_error=False)
    king.calcChargeRadii(isotopes=isotopes, run=run, plot_evens_seperate=True)

    # king.kingFit(alpha=378, findBestAlpha=True, run=run, find_slope_with_statistical_error=True)
    king.kingFit(alpha=361, findBestAlpha=True, run=run)
    radii_alpha = king.calcChargeRadii(isotopes=isotopes, run=run, plot_evens_seperate=True)
    print('radii with alpha', radii_alpha)
    # king.calcChargeRadii(isotopes=isotopes, run=run)
#
if print_plot_shifts:
    # # print / plot isotope shift:
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(''' SELECT iso, val, statErr, systErr, rChi From Combined WHERE parname = ? AND run = ? ORDER BY iso''',
                ('shift', run))
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
            radii_iso, radii_iso_err = (radii_alpha.get(iso[0], [0])[0], radii_alpha.get(iso[0], [0, 0])[1])
            print('%s\t%.1f(%.0f)[%.0f]\t%.3f' % (iso[0], iso[1], iso[2] * 10, iso[3] * 10, iso[4]))
            # for latex:
            # print('$ %s $ & %.1f(%.0f)[%.0f] & %.3f(%.0f) \\\\'
            #       % (iso[0][:2], iso[1], iso[2] * 10, iso[3] * 10, radii_iso, radii_iso_err * 1000))
    #
    MPLPlotter.plt.figure(facecolor='w')
    fontsize_ticks = 18
    MPLPlotter.plt.errorbar(iso_shift_plot_data_x, iso_shift_plot_data_y, iso_shift_plot_data_err,
                            marker='o', linestyle='None')
    MPLPlotter.plt.xticks(fontsize=fontsize_ticks)
    MPLPlotter.plt.yticks(fontsize=fontsize_ticks)
    MPLPlotter.plt.margins(0.1)
    MPLPlotter.plt.xlabel('A', fontsize=fontsize_ticks)
    MPLPlotter.plt.ylabel(r'$\delta \nu$ (MHz)', fontsize=fontsize_ticks)
    MPLPlotter.show(True)
