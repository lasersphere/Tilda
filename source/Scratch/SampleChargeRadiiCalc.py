"""
Created on 09.01.2018

@author: simkaufm

Module Description: in order to check teh calculation of the charge radii this is done here once again
 without the link to the existing PolliFit scripts
"""
import numpy as np


# isotopic shift of 68_ni from the database from the 2016 Run
is_68_ni, is_68_ni_stat_err, is_68_ni_syst_err = (2001.2781095785253, 1.5111597204327243, 9.890225266545826)

# masses from AME:
mass_68_ni, mass_68_ni_err = (67.93186899999999, 0.000003)
mass_60_ni, mass_60_ni_err = (59.9307859, 0.0000005)  # reference

reduced_mass_68 = mass_68_ni * mass_60_ni / (mass_68_ni - mass_60_ni)
# from error prop:
reduced_mass_68_err = np.sqrt(
    (mass_68_ni_err * mass_60_ni / (mass_68_ni - mass_60_ni) ** 2) ** 2 +
    (mass_68_ni * mass_60_ni_err / (mass_68_ni - mass_60_ni) ** 2) ** 2
)

# King Fit parameters from performing the King Fit
field_shift, field_shift_err = (-740.362558937, 93.0050930225)  # MHz/fm^2
mass_shift, mass_shift_err = (1246035.06494, 33993.9703469)  # u MHz

# charge radii
# M * dv = F * M * drCh2 + K
#
# M = m60 * m68 / (m68 - m60)  # reduced mass 68
# dv, isotope shift 68-60
# F, field shift constant from King Fit
# drCh2, change in mean square charge radii
# K, mass shift constant from King Fit
#
# solved for drCh2:
# drCh2 = (M * dv - K) / (M * F) = dv / F - K / (M * F)

ch_r2 = is_68_ni/field_shift - mass_shift / (reduced_mass_68 * field_shift)
print('ch_r2\t %.4f' % ch_r2)  # -> 0.6044798541811236

# error from gaussian error prop:
ch_r2_err = np.sqrt(
    (is_68_ni_stat_err / field_shift) ** 2 +
    (field_shift_err * (mass_shift / (reduced_mass_68 * field_shift ** 2) - is_68_ni / (field_shift ** 2))) ** 2 +
    (mass_shift_err / (reduced_mass_68 * field_shift)) ** 2 +
    (reduced_mass_68_err * (mass_shift / (reduced_mass_68 ** 2 * field_shift))) ** 2
)
print('ch_r2_err\t %.4f' % ch_r2_err)  # -> 0.11795333504


''' now  x-axis shifted by alpha to reduce error in mass_shift: '''
field_shift, field_shift_err = (-740.362558937, 93.0050930225)  # MHz/fm^2  # stayed the same.
mass_shift_alpha, mass_shift_alpha_err = (977283.456043, 3370.53834392)  # u MHz  # one order more precise
alpha = 363


# charge radii same as before but x-axis shifted by alpha:
# from:
# M * dv = F * (M * drCh2 - alpha) + K
# to
# drCh2 = (M * dv - K + F * alpha) / (M * F)
# or equally:
# drCh2 = dv / F - K / (M * F) + alpha / M

ch_r2_alpha = is_68_ni/field_shift - mass_shift_alpha / (reduced_mass_68 * field_shift) + alpha / reduced_mass_68
print('ch_r2_alpha\t %.4f' % ch_r2_alpha)  # -> 0.6044798541735079

# error from gaussian error prop:
ch_r2_alpha_err = np.sqrt(
    (is_68_ni_stat_err / field_shift) ** 2 +
    (field_shift_err * (mass_shift_alpha / (reduced_mass_68 * field_shift ** 2) - is_68_ni / (field_shift ** 2))) ** 2 +
    (mass_shift_alpha_err / (reduced_mass_68 * field_shift)) ** 2 +
    (reduced_mass_68_err * (mass_shift_alpha / (reduced_mass_68 ** 2 * field_shift) - alpha / reduced_mass_68 ** 2)) ** 2
)
print('ch_r2_alpha_err\t %.4f' % ch_r2_alpha_err)  # -> 0.0164749530254


''' Now with "Deyan's Method" '''
# -> fit with stat errors only first, fix the gained field shift value to that
# and then fit again with total errors to get the mass shift factor
# This means:
#  king.kingFit(alpha=0, findBestAlpha=False, run=run, find_slope_with_statistical_error=True)
# intercept:  1263152.29956 ( 3477.0416787 ) u MHz 	 percent: 0.28
# slope:  -787.363369176 ( 45.4285575469 ) MHz/fm^2 	 percent: -5.77
mass_shift_deyan, mass_shift_deyan_err = (1263152.29956, 3477.0416787)  # u MHz 	 percent: 0.28
field_shift_deyan, field_shift_deyan_err = (-787.363369176, 45.4285575469)  # MHz/fm^2 	 percent: -5.77

# charge radii
ch_r2_deyan = is_68_ni/field_shift_deyan - mass_shift_deyan / (reduced_mass_68 * field_shift_deyan)
print('ch_r2_deyan\t %.4f' % ch_r2_deyan)  # -> 0.6111212333747518

# error from gaussian error prop:
ch_r2_deyan_err = np.sqrt(
    (is_68_ni_stat_err / field_shift_deyan) ** 2 +
    (field_shift_deyan_err * (mass_shift_deyan / (reduced_mass_68 * field_shift_deyan ** 2) - is_68_ni / (field_shift_deyan ** 2))) ** 2 +
    (mass_shift_deyan_err / (reduced_mass_68 * field_shift_deyan)) ** 2 +
    (reduced_mass_68_err * (mass_shift_deyan / (reduced_mass_68 ** 2 * field_shift_deyan))) ** 2
)
print('ch_r2_deyan_err\t %.4f' % ch_r2_deyan_err)  # -> 0.0363629757839


''' finally print combined: '''
print('------- combined --------')
print('method\tdrchi^2\terr drchi^2\t %%')
print('normal\t%.4f\t%.4f\t%.2f %%' % (ch_r2, ch_r2_err, (ch_r2_err / ch_r2 * 100)))
print('alpha\t%.4f\t%.4f\t%.2f %%' % (ch_r2_alpha, ch_r2_alpha_err, (ch_r2_alpha_err / ch_r2_alpha * 100)))
print('Deyan\t%.4f\t%.4f\t%.2f %%' % (ch_r2_deyan, ch_r2_deyan_err, (ch_r2_deyan_err / ch_r2_deyan * 100)))

# method	drchi^2	err drchi^2	 %
# normal	0.6045  0.1180      19.51 %
# alpha	    0.6045  0.0165      2.73 %
# Deyan	    0.6111  0.0364      5.95 %
