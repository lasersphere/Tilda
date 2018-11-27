"""
Created on 18.09.2018

@author: simkaufm

Module Description: short script to extract values from db etc.
"""

from fractions import Fraction
import Tools
import numpy as np
import os
import sqlite3

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

''' results '''

con = sqlite3.connect(db)
cur = con.cursor()
cur.execute(''' SELECT iso, I From main.Isotopes ORDER BY iso''')
spins = cur.fetchall()
con.close()
print(spins)
spins_dict = {}
for iso, spin in spins:
    if iso in isotopes:
        spins_dict[iso] = spin

shifts = Tools.extract_from_combined([runs[0]], db, isotopes, par='shift')
d_r2_dict = Tools.extract_from_combined([runs[0]], db, isotopes, par='delta_r_square')
print(d_r2_dict)
# {run: {iso: [val, statErr, systErr, rChi], iso...}}
# Header:
print('\hline \hline'
      ' A  &'
      ' $ I $ &'
      ' $ \delta \\nu_{IS}^{60,A} $ / MHz &'
      ' $ \delta \langle r_c^2\\rangle^{60,A} $ / fm$^2$ &'
      ' $ r_c $ / fm \cite{fricke04} &'
      ' $ \delta \langle r_c^2\\rangle^{60,A} $ / fm$^2$ \cite{fricke04} \\\\'
      '\hline')
for iso in isotopes:
    massnum = int(iso[:2])
    spin = str(Fraction(spins_dict[iso]))
    shift, shift_stat_err, shift_syst_err = shifts[runs[0]].get(iso, [0., 0., 0., 0.])[:-1]
    shift_str = '%.1f(%.0f)[%.0f]' % (shift, shift_stat_err * 10, shift_syst_err * 10)
    d_r2val, d_r2val_stat_err, d_r2val_syst_err = d_r2_dict[runs[0]].get(iso, [0., 0., 0., 0.])[:-1]
    d_r2val_str = '%.3f(%.0f)' % (d_r2val, d_r2val_syst_err * 1000)
    fricke_abs, fricke_abs_err = lit_radii_calc.get(iso, [False, False])
    fricke_abs_radius_str = '%.4f(%.0f)' % (fricke_abs, fricke_abs_err * 10000) if fricke_abs else ''
    fricke_rel, fricke_rel_err = delta_lit_radii.get(iso, [False, False])
    fricke_rel_radius_str = '%.4f(%.0f)' % (fricke_rel, fricke_rel_err * 10000) if fricke_rel else ''
    print('  %s  &'
          '  %s  &'
          '  %s  &'
          '  %s  &'
          '  %s  &'
          '  %s  \\\\' % (massnum, spin, shift_str, d_r2val_str, fricke_abs_radius_str, fricke_rel_radius_str))
print('\hline')

moments = ['iso & I & $A_u$ / MHz & $A_l$  / MHz & $A_u$/$A_l$	& $B_u$  / MHz & $B_l$ / MHz & $B_u$/$B_l$ \\\\',
           '59 & 3/2 &	-176.07(155)[4] & 	-452.70(112)[10] &	0.389(4) &'
           '	-31.53(549)[1] & -56.65(682)[1] & 	0.557(118)\\\\',
           '61 & 3/2 &	-177.15(35)[4] & 	-454.92(26)[10] &	0.389(1) &'
           '	-51.17(136)[1] & -103.85(171)[2] & 	0.493(015)\\\\',
           '63 & 1/2 &	351.39(85)[8] & 	904.19(61)[20] &	0.389(1) &'
           '	0.00(0)[0] & 0.00(0)[0] & 	0.000(000)\\\\',
           '65 & 5/2 &	107.86(24)[2] & 	276.68(17)[6] &	0.390(1) &'
           '	-28.71(177)[1] & -60.35(216)[1] & 	0.476(034)\\\\',
           '67 & 1/2 &	424.00(18)[10] & 	1089.21(35)[25] &	0.389(0) &'
           '	0.00(0)[0] & 0.00(0)[0] & 	0.000(000)']
for each in moments:
    print(each)
