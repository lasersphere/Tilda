"""
Created on 

@author: simkaufm

Module Description:  copied values from Cu paper (M.L. Bissel et al., PRC 93, 064318 (2016))
and Zn from Radii draft from Xiaofei.
"""

import numpy as np

# electron-scattering radii, do not use!
cu_65_abs_radii_sc77 = (3.892, 0.036)  # Landolt-Börnstein Vol. 20
cu_65_abs_radii_ke77 = (3.986, 0.019)  # Landolt-Börnstein Vol. 20
cu_65_abs_radii_sbk78 = (3.954, 0.013)  # Landolt-Börnstein Vol. 20


#muonic model dependetn radii:
cu_65_abs_radii_muonic = (3.902, 0.)
cu_65_abs_radii = cu_65_abs_radii_muonic  # Landolt-Börnstein Vol. 20

# copied from: Cu-radii: M.L. Bissel et al., PRC 93, 064318 (2016)
cu_shifts_delt_rchisq = [
    [58, 1, -1975, 10, -0.833, 0.013, 0.091],
    [59, 3 / 2, -1717.4, 70, -0.635, 0.009, 0.071],
    [60, 2, -1415.0, 60, -0.511, 0.008, 0.057],
    [61, 3 / 2, -1147.9, 50, -0.359, 0.006, 0.040],
    [62, 1, -825.0, 38, -0.293, 0.005, 0.033],
    [63, 3 / 2, -576.1, 11, -0.148, 0.001, 0.017],
    [64, 1, -249.4, 22, -0.116, 0.003, 0.013],
    [65, 3 / 2, 0, 0, 0, 0., 0.],
    [66, 1, 304.8, 33, 0.033, 0.004, 0.012],
    [67, 3 / 2, 561.4, 35, 0.115, 0.005, 0.018],
    [68, 1, 858.8, 37, 0.133, 0.005, 0.031],
    [69, 3 / 2, 1079.0, 20, 0.238, 0.003, 0.034],
    [70, 6, 1347.3, 23, 0.271, 0.003, 0.044],
    [71, 3 / 2, 1526.5, 91, 0.407, 0.011, 0.044],
    [72, 2, 1787.1, 38, 0.429, 0.005, 0.055],
    [73, 3 / 2, 1984, 12, 0.523, 0.015, 0.058],
    [74, 2, 2260, 14, 0.505, 0.018, 0.072],
    [75, 5 / 2, 2484, 16, 0.546, 0.021, 0.080]
]

cu_shifts_delt_rchisq_isomeres = [
    [68, 6, 812.5, 26, 0.192, 0.003, 0.031],
    [70, 3, 1334.6, 84, 0.287, 0.011, 0.044],
    [70, 1, 1307.1, 83, 0.323, 0.011, 0.044],
]

cu_z = 29

print('----------- cu radii for origin ------------')
print('A\tN\td_rCh_sq\terr_d_rCh_sq\trCh\trChErr')
for mass, spin, shift, shift_err_10, delta_r_ch, delta_r_ch_stat_err, delta_r_ch_syst_err in cu_shifts_delt_rchisq:
    nutr_num = mass - cu_z
    rCh = np.sqrt(cu_65_abs_radii[0] ** 2 + delta_r_ch)
    delta_rch_full_err = np.sqrt(delta_r_ch_stat_err ** 2 + delta_r_ch_syst_err ** 2)
    rChErr = np.sqrt(
        (0.5 * (cu_65_abs_radii[0] ** 2 + delta_r_ch) ** -0.5 * 2 * cu_65_abs_radii[0] * cu_65_abs_radii[
            1]) ** 2 +
        (0.5 * (cu_65_abs_radii[0] ** 2 + delta_r_ch) ** -0.5 * delta_rch_full_err) ** 2
    )
    print('%d\t%d\t%.5f\t%.5f\t%.5f\t%.5f' % (mass, nutr_num, delta_r_ch, delta_rch_full_err, rCh, rChErr))

''' similar for Zn: '''

zn_shifts_delt_rchisq = [
    [62, 0, -239.5, 11, 09.9, -0.493, 0.003, 0.052],
    [63, 3 / 2, -191.2, 32, 08.7, -0.389, 0.009, 0.043],
    [64, 0, -141.2, 12, 06.6, -0.279, 0.004, 0.034],
    [65, 5 / 2, -121.8, 23, 05.1, -0.257, 0.007, 0.025],
    [66, 0, -63.6, 15, 03.8, -0.121, 0.004, 0.016],
    [67, 5 / 2, -41.4, 21, 01.6, -0.089, 0.006, 0.008],
    [68, 0, 0, 0, 0, 0, 0, 0.],
    [69, 1 / 2, 19.5, 20, 01.5, 0.026, 0.006, 0.009],
    [70, 0, 69.5, 9, 02.9, 0.142, 0.003, 0.015],
    [71, 1 / 2, 108.8, 24, 04.4, 0.227, 0.007, 0.023],
    [72, 0, 140.6, 10, 05.7, 0.292, 0.003, 0.030],
    [73, 1 / 2, 158.9, 12, 07.1, 0.318, 0.003, 0.037],
    [74, 0, 187.9, 13, 08.3, 0.375, 0.004, 0.044],
    [75, 7 / 2, 187.7, 10, 09.6, 0.349, 0.003, 0.051],
    [76, 0, 221.3, 14, 010.8, 0.421, 0.004, 0.057],
    [77, 7 / 2, 236.0, 16, 012.0, 0.440, 0.005, 0.064],
    [78, 0, 255.7, 11, 013.1, 0.474, 0.003, 0.070],
    [79, 9 / 2, 259.3, 10, 014.2, 0.461, 0.003, 0.077],
    [80, 0, 268.4, 12, 016.1, 0.465, 0.004, 0.084]
]

zn_shifts_delt_rchisq_isomeres = [
    [69, 9 / 2, 35.7, 11, 01.5, 0.073, 0.003, 0.008],
    [71, 9 / 2, 96.3, 11, 04.3, 0.191, 0.003, 0.023],
    [73, 5 / 2, 160.4, 19, 07.1, 0.322, 0.006, 0.037],
    [75, 1 / 2, 195.8, 21, 09.6, 0.373, 0.006, 0.051],
    [77, 1 / 2, 241.2, 38, 012.0, 0.455, 0.011, 0.064],
    [79, 1 / 2, 320.6, 29, 014.2, 0.639, 0.008, 0.075]
]

zn_z = 30

zn_68_abs_radii = (3.964, 0)  # Landolt-Börnstein Vol. 20, error is not given.

print('----------- zn radii for origin ------------')
print('A\tN\td_rCh_sq\terr_d_rCh_sq\trCh\trChErr')
for mass, spin, shift, shift_err_10, shift_syst_err,\
    delta_r_ch, delta_r_ch_stat_err, delta_r_ch_syst_err in zn_shifts_delt_rchisq:
    # print(mass, spin, shift, shift_err_10, delta_r_ch, delta_r_ch_stat_err, delta_r_ch_syst_err)
    nutr_num = mass - zn_z
    rCh = np.sqrt(zn_68_abs_radii[0] ** 2 + delta_r_ch)
    delta_rch_full_err = np.sqrt(delta_r_ch_stat_err ** 2 + delta_r_ch_syst_err ** 2)
    rChErr = np.sqrt(
        (0.5 * (zn_68_abs_radii[0] ** 2 + delta_r_ch) ** -0.5 * 2 * zn_68_abs_radii[0] * zn_68_abs_radii[
            1]) ** 2 +
        (0.5 * (zn_68_abs_radii[0] ** 2 + delta_r_ch) ** -0.5 * delta_rch_full_err) ** 2
    )
    print('%d\t%d\t%.5f\t%.5f\t%.5f\t%.5f' % (mass, nutr_num, delta_r_ch, delta_rch_full_err, rCh, rChErr))

