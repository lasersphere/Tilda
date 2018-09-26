"""
Created on 

@author: simkaufm

Module Description: This is a script for plotting the central wavelength's on the different Ni isotopes.

"""

import Tools
import math
import Analyzer
import Physics
import Service.DatabaseOperations.DatabaseOperations as TiDb

db_path = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_isotopes.sqlite'
# TiDb.createTildaDB(db_path)

# for the 3d9(2D)4s  	 3D 3  -> 3d9(2D)4p  	 3P° 2 @352.454nm transition
# A. Steudel measured some isotop extrapolated_shifts:
# units are: mK = 10 ** -3 cm ** -1
iso_sh = {'58-60': (16.94, 0.09), '60-62': (16.91, 0.12), '62-64': (17.01, 0.26),
          '60-61': (9.16, 0.10), '61-62': (7.55, 0.12), '58-62': (34.01, 0.15), '58-64': (51.12, 0.31)}
iso_sh_wave = {}

spins = {
    '57': 1.5,
    '59': 1.5,
    '61': 1.5,
    '63': 0.5,
    '65': 2.5,
    '67': 0.5,
    '69': 4.5,
    '71': 4.5,
}

a_factors = {
    '57': (-483.8, 0, -513.79, 0),
    '59': (127.4, 0, 135.29, 0),
    '61': (-455, 0, -483.2, 0),
    '63': (273.72, 0, 290.685, 0),
    '65': (251.15, 0, 266.72, 0),
    '67': (1093.79, 0, 1161.58, 0),
    '69': (-267.88, 99.44, -284.48, 0),
    '71': (-269.54, 42.15, -268.25, 0),
}

# convert this to frequency/MHz
for key, val in iso_sh.items():
    iso_sh_wave[key] = (round(Physics.freqFromWavenumber(val[0] * 10 ** -3), 2),
                        round(Physics.freqFromWavenumber(val[1] * 10 ** -3), 2))

# get the mean isotope shift between two masses in the given range 58-64
lis_58_64 = [iso_sh_wave['58-60'][0], iso_sh_wave['60-62'][0], iso_sh_wave['62-64'][0]]
lis_errs_58_64 = [iso_sh_wave['58-60'][1], iso_sh_wave['60-62'][1], iso_sh_wave['62-64'][1]]
mean_shift = Analyzer.average(lis_58_64, lis_errs_58_64)
print('mean isotope shift between two even isotopes from (58-60/60-62/62-64): ', mean_shift, 'MHz')
tisa_nu0 = 0850343019.777062

diff_doppl_lis = [Physics.diffDoppler(tisa_nu0, 30000, m) for m in range(58, 73)]
for i, diff in enumerate(diff_doppl_lis):
    print('for mass: %d the diff doppler shift is: %0.2f' % (i + 58, diff))

''' 58_Ni as reference: '''
ref = 58
extrapolated_shifts_58 = [(iso, round(((iso - ref) * (mean_shift[0] * 0.5)), 1), 0)
                          for iso in range(56, 74)]
known_shifts_58 = [
    (58, 0, 0), (60, iso_sh_wave['58-60'][0], iso_sh_wave['58-60'][1]),
    (61, iso_sh_wave['58-60'][0] + iso_sh_wave['60-61'][0],
     round(math.sqrt(iso_sh_wave['58-60'][1] ** 2 + iso_sh_wave['60-61'][1] ** 2)), 2),
    (62, iso_sh_wave['58-62'][0], iso_sh_wave['58-62'][1]),
    (64, iso_sh_wave['58-64'][0], iso_sh_wave['58-64'][1])
]

for i, iso in enumerate(extrapolated_shifts_58):
    replace = [known_iso for known_iso in known_shifts_58 if known_iso[0] == iso[0]]
    if replace:
        extrapolated_shifts_58[i] = replace[0]

shifts_58 = sorted(extrapolated_shifts_58, key=lambda x: x[0])

# print('referenced on 58_Ni')
# print('isotope', '\t', 'shift [MHz]', '\t', 'error [MHz]')
# for iso in shifts_58:
#     print(iso[0], '\t', iso[1], '\t', iso[2])

''' 60_Ni as reference: '''
ref = 60
extrapolated_shifts_60 = [(iso, round(((iso - ref) * (mean_shift[0] * 0.5)), 1), 0)
                          for iso in range(56, 74)]

known_shifts_60 = [
    (60, 0, 0), (58, -1 * iso_sh_wave['58-60'][0], iso_sh_wave['58-60'][1]),
    (61, iso_sh_wave['60-61'][0], iso_sh_wave['60-61'][1]),
    (62, iso_sh_wave['60-62'][0], iso_sh_wave['60-62'][1]),
    (64, iso_sh_wave['62-64'][0] + iso_sh_wave['60-62'][0],
     round(math.sqrt(iso_sh_wave['62-64'][1] ** 2 + iso_sh_wave['60-62'][1] ** 2)), 2)
]
for i, iso in enumerate(extrapolated_shifts_60):
    replace = [known_iso for known_iso in known_shifts_60 if known_iso[0] == iso[0]]
    if replace:
        extrapolated_shifts_60[i] = replace[0]
shifts_60 = sorted(extrapolated_shifts_60, key=lambda x: x[0])
# print('referenced on 60_Ni')
# print('isotope', '\t', 'shift [MHz]', '\t', 'error [MHz]')
# for iso in shifts_60:
#     print(iso[0], '\t', iso[1], '\t', iso[2])

# ''' insert isos into db '''
# for iso in shifts_58:
#     Tools._insertIso(db_path, '%s_Ni_ref58' % iso[0], iso[0], 0, spins.get(str(iso[0]), 0), iso[1],
#                      a_factors.get(str(iso[0]), (0, 0, 0, 0))[0], a_factors.get(str(iso[0]), (0, 0, 0, 0))[1],
#                      a_factors.get(str(iso[0]), (0, 0, 0, 0))[2], a_factors.get(str(iso[0]), (0, 0, 0, 0))[3],
#                      0, 0, 5000, 0)
#
# for iso in shifts_60:
#     Tools._insertIso(db_path, '%s_Ni_ref60' % iso[0], iso[0], 0, spins.get(str(iso[0]), 0), iso[1],
#                      a_factors.get(str(iso[0]), (0, 0, 0, 0))[0], a_factors.get(str(iso[0]), (0, 0, 0, 0))[1],
#                      a_factors.get(str(iso[0]), (0, 0, 0, 0))[2], a_factors.get(str(iso[0]), (0, 0, 0, 0))[3],
#                      0, 0, 5000, 0)


dye_wavenum = 32695.46
tisa_wavenum = 28364.39

print('dye line wavenumber/cm-1: ', dye_wavenum, 'dye line freq/MHz: ', Physics.freqFromWavenumber(dye_wavenum))
print('tisa line wavenumber/cm-1: ', tisa_wavenum, 'tisa line freq/MHz: ', Physics.freqFromWavenumber(tisa_wavenum))

isos = ['%s_Ni_ref60' % i for i in range(56, 73)]
# isos = [isos[0]]
# Tools.centerPlot(db_path, isos, width=2e6, linevar='tisa_60')

Tools.isoPlot(db_path, '67_Ni_ref60', linevar='tisa_60', as_freq=False, laserfreq=851200725.9994, col=True)


''' tisa laser lab wavelengths'''
sh_30 = 28393
fun_30 = sh_30 / 2
sh_40 = 28397.5
fun_40 = sh_40 / 2
wavenums = [sh_30, fun_30, sh_40, fun_40]

freqs = []
wavelens = []
for wavenum in wavenums:
    freq = Physics.freqFromWavenumber(wavenum)
    wavelen = Physics.wavelenFromFreq(freq)
    freqs.append(freq)
    wavelens.append(wavelen)
print('tisa:')
print(wavenums)
print(freqs)
print(wavelens)

''' dye laser lab wavelengths'''
sh_30 = 32728.5
fun_30 = sh_30 / 2
sh_40 = 32733.5
fun_40 = sh_40 / 2
wavenums = [sh_30, fun_30, sh_40, fun_40]

freqs = []
wavelens = []
for wavenum in wavenums:
    freq = Physics.freqFromWavenumber(wavenum)
    wavelen = Physics.wavelenFromFreq(freq)
    freqs.append(freq)
    wavelens.append(wavelen)
print('dye:')
print(wavenums)
print(freqs)
print(wavelens)

