"""
Created on 18.09.2018

@author: simkaufm

Module Description: short script to extract values from db etc.
"""

from fractions import Fraction
import Tools
import TildaTools as TiTs
import numpy as np
import os
import sqlite3

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
save_shift_as_pdf = False
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
# isotopes = ['58_Ni']

odd_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2]
even_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2 == 0]
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']

isotopes16 = ['58_Ni', '59_Ni', '60_Ni', '61_Ni', '62_Ni', '63_Ni',
              '64_Ni', '65_Ni', '66_Ni', '67_Ni', '68_Ni', '70_Ni']

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
}   # have ben calculated more accurately below

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
# lit_radii = lit_radii_calc

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

''' results '''

con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(''' SELECT iso, I, mass, mass_d From main.Isotopes ORDER BY iso''')
spins = cur.fetchall()
con.close()
print(spins)
spins_dict = {}
masses_dict = {}
for iso, spin, mass, mass_d in spins:
    if iso in isotopes:
        spins_dict[iso] = spin
        masses_dict[iso] = (mass, mass_d)

shifts = Tools.extract_from_combined([final_2017_run], db, isotopes, par='shift')
d_r2_dict_17 = Tools.extract_from_combined([final_2017_run], db, isotopes, par='delta_r_square')
print(d_r2_dict_17)
# {run: {iso: [val, statErr, systErr, rChi], iso...}}
# Header:
print('------------------ tab:isoShiftChargeRadii2017 ------------------')
print('\hline \hline')
print(' A  &'
      ' $ I $ &'
      ' $ \delta \\nu_{IS}^{60,A} $ / MHz &'
      ' $ \delta \langle r_c^2\\rangle^{60,A} $ / fm$^2$ &'
      ' $ r_c $ / fm \cite{fricke04} &'
      ' $ \delta \langle r_c^2\\rangle^{60,A} $ / fm$^2$ \cite{fricke04} \\\\'
      '\hline')
for iso in isotopes:
    massnum = int(iso[:2])
    spin = str(Fraction(spins_dict[iso]))
    shift, shift_stat_err, shift_syst_err = shifts[final_2017_run].get(iso, [0., 0., 0., 0.])[:-1]
    shift_str = '%.1f(%.0f)[%.0f]' % (shift, shift_stat_err * 10, shift_syst_err * 10)
    d_r2val, d_r2val_stat_err, d_r2val_syst_err = d_r2_dict_17[final_2017_run].get(iso, [0., 0., 0., 0.])[:-1]
    d_r2val_str = '%.3f(%.0f)' % (d_r2val, d_r2val_syst_err * 1000)
    fricke_abs, fricke_abs_err = lit_radii.get(iso, [False, False])
    fricke_abs_radius_str = '%.3f(%.0f)' % (fricke_abs, fricke_abs_err * 1000) if fricke_abs else ''
    fricke_rel, fricke_rel_err = delta_lit_radii.get(iso, [False, False])
    fricke_rel_radius_str = '%.3f(%.0f)' % (fricke_rel, fricke_rel_err * 1000) if fricke_rel else ''
    print('  %s  &'
          '  %s  &'
          '  %s  &'
          '  %s  &'
          '  %s  &'
          '  %s  \\\\' % (massnum, spin, shift_str, d_r2val_str, fricke_abs_radius_str, fricke_rel_radius_str))
print('\hline')


au16 = Tools.extract_from_combined([final_2017_run], db, isotopes, par='Au')[final_2017_run]
al16 = Tools.extract_from_combined([final_2017_run], db, isotopes, par='Al')[final_2017_run]
bu16 = Tools.extract_from_combined([final_2017_run], db, isotopes, par='Bu')[final_2017_run]
bl16 = Tools.extract_from_combined([final_2017_run], db, isotopes, par='Bl')[final_2017_run]
print('------------------ tab:2017AandB ------------------')
print('\hline \hline')
print('A & I & $A_u$ / MHz & $A_l$  / MHz & $B_u$  / MHz & $B_l$ / MHz\\\\')
print('\hline')
for odd_i in odd_isotopes:
    massnum = int(odd_i[:2])
    spin = str(Fraction(spins_dict[odd_i]))
    au_iso, au_iso_stat_err, au_iso_syst_err, au_iso_rchisq = au16.get(
        odd_i, (0.0, 0.0, 0.0, 0.0))  # val, statErr, systErr, rChi
    al_iso, al_iso_stat_err, al_iso_syst_err, al_iso_rchisq = al16.get(
        odd_i, (0.0, 0.0, 0.0, 0.0))  # val, statErr, systErr, rChi
    bu_iso, bu_iso_stat_err, bu_iso_syst_err, bu_iso_rchisq = bu16.get(
        odd_i, (0.0, 0.0, 0.0, 0.0))  # val, statErr, systErr, rChi
    bl_iso, bl_iso_stat_err, bl_iso_syst_err, bl_iso_rchisq = bl16.get(
        odd_i, (0.0, 0.0, 0.0, 0.0))  # val, statErr, systErr, rChi
    a_up_str = '%.1f(%.0f)[%.0f]' % (au_iso, au_iso_stat_err * 10, max(1, au_iso_syst_err * 10))
    a_l_str = '%.1f(%.0f)[%.0f]' % (al_iso, al_iso_stat_err * 10, max(1, al_iso_syst_err * 10))
    b_up_str = ''
    b_l_str = ''
    if bu_iso > 0:
        b_l_str = '%.1f(%.0f)[%.0f]' % (bl_iso, bl_iso_stat_err * 10, max(1, bl_iso_syst_err * 10))
        b_up_str = '%.1f(%.0f)[%.0f]' % (bu_iso, bu_iso_stat_err * 10, max(1, bu_iso_syst_err * 10))

    print('%s & %s & %s & %s & %s & %s\\\\' % (
        massnum, spin, a_up_str, a_l_str, b_up_str, b_l_str
    ))
print('\hline')

shift_68_ni, shift_stat_err_68_ni, shift_syst_err_68_ni = shifts[final_2017_run]['68_Ni'][:-1]
shift_68_ni_err = np.sqrt(shift_stat_err_68_ni ** 2 + shift_syst_err_68_ni ** 2)
d_r2_68_ni, d_r2_stat_err_68_ni, d_r2_syst_err_68_ni = d_r2_dict_17[final_2017_run]['68_Ni'][:-1]
d_r2_68_ni_err = np.sqrt(d_r2_stat_err_68_ni ** 2 + d_r2_syst_err_68_ni ** 2)

mass_68, mass_68_err = masses_dict['68_Ni']
ref_mass, ref_massErr = masses_dict['60_Ni']

isotopeMasses = [mass_68]
massErr = [mass_68_err]

#from Kingfitter.py:
red_mass_68 = [i*ref_mass/(i-ref_mass) for i in isotopeMasses][0]
        # from error prop:
red_mass_68_err = [
    ((iso_m_d * (ref_mass ** 2)) / (iso_m - ref_mass) ** 2) ** 2 +
    ((ref_massErr * (iso_m ** 2)) / (iso_m - ref_mass) ** 2) ** 2
    for iso_m, iso_m_d in zip(isotopeMasses, massErr)
][0]
print(red_mass_68, red_mass_68_err)
mod_isotope_shift = red_mass_68 * shift_68_ni
mod_isotope_shift_err = np.sqrt((shift_68_ni * red_mass_68_err) ** 2 +
                                (red_mass_68 * shift_68_ni_err) ** 2)
print('%.3f(%.0f) uGHz' % (mod_isotope_shift * 10 ** -3, mod_isotope_shift_err))
print(mod_isotope_shift, mod_isotope_shift_err)

mod_d_r2 = red_mass_68 * d_r2_68_ni  # u fm2
mod_d_r2_err = np.sqrt((d_r2_68_ni * red_mass_68_err) ** 2 +
                       (red_mass_68 * d_r2_68_ni_err) ** 2)

print('%.3f(%.0f) ufm2' % (mod_d_r2, mod_d_r2_err))
print(mod_d_r2, mod_d_r2_err)

# make sure this was updated before with NiAnalysis2017.py
d_r2_dict_16 = Tools.extract_from_combined([run2016_final_db], db, isotopes16, par='delta_r_square')
TiTs.print_dict_pretty(d_r2_dict_16)

print('------------------ fig:dr2EvolutionFinal ------------------')
print('# copy to origin')
print('# iso\tmass\tdr2\tstatErr\tsystErr\tfullErr')
for iso in isotopes16:
    massnum = int(iso[:2])
    d_r2val_17, d_r2val_stat_err_17, d_r2val_syst_err_17, d_r2val_rChi_17 = d_r2_dict_17[final_2017_run].get(
        iso, [0., 0., 0., 0.])
    d_r2val_16, d_r2val_stat_err_16, d_r2val_syst_err_16, d_r2val_rChi_16 = d_r2_dict_16[run2016_final_db].get(
        iso, [0., 0., 0., 0.])
    dr2_final = d_r2val_16 if iso in ['59_Ni', '63_Ni'] else d_r2val_17
    dr2_final_stat_err = d_r2val_stat_err_16 if iso in ['59_Ni', '63_Ni'] else d_r2val_stat_err_17
    dr2_final_syst_err = d_r2val_syst_err_16 if iso in ['59_Ni', '63_Ni'] else d_r2val_syst_err_17
    dr2_final_full_err = np.sqrt(dr2_final_stat_err ** 2 + dr2_final_syst_err ** 2)
    print('%s\t%d\t%.5f\t%.5f\t%.5f\t%.5f' % (iso, massnum, dr2_final, dr2_final_stat_err,
                                              dr2_final_syst_err, dr2_final_full_err))
