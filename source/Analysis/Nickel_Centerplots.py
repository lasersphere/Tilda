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

# convert this to frequency/MHz
for key, val in iso_sh.items():
    iso_sh_wave[key] = (round(Physics.freqFromWavenumber(val[0] * 10 ** -3), 2),
                        round(Physics.freqFromWavenumber(val[1] * 10 ** -3), 2))

# get the mean isotope shift between two masses in the given range 58-64
lis_58_64 = [iso_sh_wave['58-60'][0], iso_sh_wave['60-62'][0], iso_sh_wave['62-64'][0]]
lis_errs_58_64 = [iso_sh_wave['58-60'][1], iso_sh_wave['60-62'][1], iso_sh_wave['62-64'][1]]
mean_shift = Analyzer.average(lis_58_64, lis_errs_58_64)
print('mean isotope shift between two even isotopes from (58-60/60-62/62-64): ', mean_shift, 'MHz')

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
#     Tools._insertIso(db_path, '%s_Ni_ref58' % iso[0], iso[0], 0, 0, iso[1], 0, 0, 0, 0, 0, 0, 5000, 0)
#
# for iso in shifts_60:
#     Tools._insertIso(db_path, '%s_Ni_ref60' % iso[0], iso[0], 0, 0, iso[1], 0, 0, 0, 0, 0, 0, 5000, 0)


dye_wavenum = 32695.46
tisa_wavenum = 28364.39

print('dye line wavenumber/cm-1: ', dye_wavenum, 'dye line freq/MHz: ', Physics.freqFromWavenumber(dye_wavenum))
print('tisa line wavenumber/cm-1: ', tisa_wavenum, 'tisa line freq/MHz: ', Physics.freqFromWavenumber(tisa_wavenum))

isos = ['%s_Ni_ref58' % i for i in range(56, 73)]
# isos = [isos[0]]
Tools.centerPlot(db_path, isos, width=2e6, linevar='dye_58')


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