"""
Created on 

@author: simkaufm

Module Description:  Analysis of the Nickel Data from COLLAPS taken on 28.04.-03.05.2016
"""

import math
import os
import sqlite3
import winsound

import numpy as np

import Analyzer
import BatchFit
import MPLPlotter
import Physics
import Tools

''' working directory: '''

workdir = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder = os.path.join(workdir, 'Ni_April2016_mcp')

db = os.path.join(workdir, 'Ni_workspace.sqlite')

runs = ['narrow_gate', 'wide_gate']
runs = [runs[0]]

isotopes = ['%s_Ni' % i for i in range(58, 71)]
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

literature_shifts = {
    '58_Ni': (-1 * iso_sh_freq['58-60'][0], iso_sh_freq['58-60'][1]),
    '60_Ni': (0, 0),
    '61_Ni': (iso_sh_freq['60-61'][0], iso_sh_freq['60-61'][1]),
    '62_Ni': (iso_sh_freq['60-62'][0], iso_sh_freq['60-62'][1]),
    '64_Ni': (mean_is_64[0], mean_is_64[1])
}
# print('literatur shifts from A. Steudel (1980) in MHz:')
# [print(key, val[0], val[1]) for key, val in sorted(literature_shifts.items())]


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
# ''' volt div ratio: '''
# volt_div_ratio = "{'accVolt': 1000.05, 'offset': 1000}"
# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute('''UPDATE Files SET voltDivRatio = ?''', (volt_div_ratio, ))
# con.commit()
# con.close()

''' diff doppler 60Ni 30kV'''
diffdopp60 = Physics.diffDoppler(850343019.777062, 30000, 60)  # 14.6842867127 MHz/V

''' transition wavelenght: '''
observed_wavenum = 28364.39  # cm-1  observed wavenum from NIST, mass is unclear.
transition_freq = Physics.freqFromWavenumber(observed_wavenum)
print('transition frequency: %s ' % transition_freq)

transition_freq += 1256.32701  # correction from fitting the 60_Ni references
# transition_freq += 900

con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''UPDATE Lines SET frequency = ?''', (transition_freq,))
con.commit()
con.close()

''' Batch fits '''

# # create static list because selection might be necessary as removing cw files.
# ni58_files = Tools.fileList(db, isotopes[0])
# ni58_bunch_files = [each for each in ni58_files if 'cw' not in each]
#
# ni59_files = Tools.fileList(db, isotopes[1])
#
# ni60_files = Tools.fileList(db, isotopes[2])
# ni60_cont_files = [each for each in ni60_files if 'contin' in each]
# ni60_bunch_files = [each for each in ni60_files if 'contin' not in each]
# # print(ni60_bunch_files)
#
# ni61_files = Tools.fileList(db, isotopes[3])

files_dict = {iso: Tools.fileList(db, iso) for iso in isotopes}
files_dict[isotopes[0]] = [each for each in files_dict[isotopes[0]] if 'cw' not in each]
files_dict[isotopes[2]] = [each for each in files_dict[isotopes[2]] if 'contin' not in each or '206'
                           in each or '208' in each or '209' in each]
# -> some files accidentially named continous
# print('fielList: %s ' % files_dict[isotopes[2]])
# BatchFit.batchFit(ni58_bunch_files, db, runs[0])
# Analyzer.combineRes(isotopes[2], 'center', runs[0], db)
# stables = ['61_Ni']
pars = ['center', 'Al', 'Bl', 'Au', 'Bu', 'Int0']
# for iso in stables:
#     files = files_dict[iso]
#     for run in runs:
#         # fits = BatchFit.batchFit(files, db, run)
#         for par in pars:
#             Analyzer.combineRes(iso, par, run, db)

''' isotope shift '''
# get all current configs:
# print('run \t iso \t val \t statErr \t rChi')
# for iso in stables:
#     for run in runs:
#         con = sqlite3.connect(db)
#         cur = con.cursor()
#         cur.execute('''SELECT config, val, statErr, rChi FROM Combined WHERE iso = ? AND run = ? AND parname = ? ''',
#                     (iso, run, 'shift'))
#         data = cur.fetchall()
#         con.close()
#         if len(data):
#             config, val, statErr, rChi = data[0]
#             print('%s \t %s \t %s \t %s \t %s \n %s' % (run, iso, val, statErr, rChi, config))
#             print('\n')

# automatic configs gained with the gui:
# auto_cfg_58 = [([], ['58Ni_no_protonTrigger_Run006.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                ([], ['58Ni_no_protonTrigger_Run007.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                ([], ['58Ni_no_protonTrigger_Run008.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                ([], ['58Ni_no_protonTrigger_Run009.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                (['60Ni_no_protonTrigger_Run027.mcp'], ['58Ni_no_protonTrigger_Run028.mcp'],
#                 ['60Ni_no_protonTrigger_Run035.mcp']),
#                (['60Ni_no_protonTrigger_Run027.mcp'], ['58Ni_no_protonTrigger_Run029.mcp'],
#                 ['60Ni_no_protonTrigger_Run035.mcp']),
#                (['60Ni_no_protonTrigger_Run027.mcp'], ['58Ni_no_protonTrigger_Run030.mcp'],
#                 ['60Ni_no_protonTrigger_Run035.mcp']),
#                (['60Ni_no_protonTrigger_Run72.mcp'], ['58Ni_no_protonTrigger_Run073.mcp'],
#                 ['60Ni_no_protonTrigger_Run076.mcp']),
#                (['60Ni_no_protonTrigger_Run72.mcp'], ['58Ni_no_protonTrigger_Run074.mcp'],
#                 ['60Ni_no_protonTrigger_Run076.mcp']),
#                (['60Ni_no_protonTrigger_Run72.mcp'], ['58Ni_no_protonTrigger_Run075.mcp'],
#                 ['60Ni_no_protonTrigger_Run076.mcp']),
#                (['60Ni_no_protonTrigger_Run147.mcp'], ['58Ni_no_protonTrigger_Run148.mcp'],
#                 ['60Ni_no_protonTrigger_Run151.mcp']),
#                (['60Ni_no_protonTrigger_Run147.mcp'], ['58Ni_no_protonTrigger_Run149.mcp'],
#                 ['60Ni_no_protonTrigger_Run151.mcp']),
#                (['60Ni_no_protonTrigger_Run147.mcp'], ['58Ni_no_protonTrigger_Run150.mcp'],
#                 ['60Ni_no_protonTrigger_Run151.mcp']),
#                ([], ['58Ni_no_protonTrigger_Run210.mcp'], ['60Ni_no_protonTrigger_Run213.mcp']),
#                ([], ['58Ni_no_protonTrigger_Run211.mcp'], ['60Ni_no_protonTrigger_Run213.mcp']),
#                ([], ['58Ni_no_protonTrigger_Run212.mcp'], ['60Ni_no_protonTrigger_Run213.mcp'])]
#
# auto_cfg_61 = [([], ['61Ni_no_protonTrigger_Run010.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                ([], ['61Ni_no_protonTrigger_Run011.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                ([], ['61Ni_no_protonTrigger_Run012.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                ([], ['61Ni_no_protonTrigger_Run013.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                ([], ['61Ni_no_protonTrigger_Run014.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                (['60Ni_no_protonTrigger_Run119.mcp'], ['61Ni_no_protonTrigger_Run120.mcp'],
#                 ['60Ni_no_protonTrigger_Run125.mcp']),
#                (['60Ni_no_protonTrigger_Run119.mcp'], ['61Ni_no_protonTrigger_Run121.mcp'],
#                 ['60Ni_no_protonTrigger_Run125.mcp']),
#                (['60Ni_no_protonTrigger_Run119.mcp'], ['61Ni_no_protonTrigger_Run123.mcp'],
#                 ['60Ni_no_protonTrigger_Run125.mcp']),
#                (['60Ni_no_protonTrigger_Run119.mcp'], ['61Ni_no_protonTrigger_Run124.mcp'],
#                 ['60Ni_no_protonTrigger_Run125.mcp']),
#                (['60Ni_no_protonTrigger_Run158.mcp'], ['61Ni_no_protonTrigger_Run159.mcp'],
#                 ['60Ni_no_protonTrigger_Run161.mcp']),
#                (['60Ni_no_protonTrigger_Run158.mcp'], ['61Ni_no_protonTrigger_Run160.mcp'],
#                 ['60Ni_no_protonTrigger_Run161.mcp'])]
#
# auto_cfg_62 = [([], ['62Ni_no_protonTrigger_Run015.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                ([], ['62Ni_no_protonTrigger_Run016.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                ([], ['62Ni_no_protonTrigger_Run017.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
#                (['60Ni_no_protonTrigger_Run143.mcp'], ['62Ni_no_protonTrigger_Run144.mcp'],
#                 ['60Ni_no_protonTrigger_Run146.mcp']),
#                (['60Ni_no_protonTrigger_Run143.mcp'], ['62Ni_no_protonTrigger_Run145.mcp'],
#                 ['60Ni_no_protonTrigger_Run146.mcp']),
#                (['60Ni_no_protonTrigger_Run163.mcp'], ['62Ni_no_protonTrigger_Run164.mcp'],
#                 ['60Ni_no_protonTrigger_Run166.mcp']),
#                (['60Ni_no_protonTrigger_Run163.mcp'], ['62Ni_no_protonTrigger_Run165.mcp'],
#                 ['60Ni_no_protonTrigger_Run166.mcp'])]
#
# auto_cfg_64 = [
#     (['60Ni_no_protonTrigger_Run021.mcp'], ['64Ni_no_protonTrigger_Run022.mcp'], ['60Ni_no_protonTrigger_Run025.mcp']),
#     (['60Ni_no_protonTrigger_Run021.mcp'], ['64Ni_no_protonTrigger_Run023.mcp'], ['60Ni_no_protonTrigger_Run025.mcp']),
#     (['60Ni_no_protonTrigger_Run021.mcp'], ['64Ni_no_protonTrigger_Run024.mcp'], ['60Ni_no_protonTrigger_Run025.mcp']),
#     (['60Ni_no_protonTrigger_Run152.mcp'], ['64Ni_no_protonTrigger_Run153.mcp'], ['60Ni_no_protonTrigger_Run155.mcp']),
#     (['60Ni_no_protonTrigger_Run152.mcp'], ['64Ni_no_protonTrigger_Run154.mcp'], ['60Ni_no_protonTrigger_Run155.mcp']),
#     (['60Ni_no_protonTrigger_Run173.mcp'], ['64Ni_no_protonTrigger_Run174.mcp'], ['60Ni_no_protonTrigger_Run178.mcp']),
#     (['60Ni_no_protonTrigger_Run173.mcp'], ['64Ni_no_protonTrigger_Run175.mcp'], ['60Ni_no_protonTrigger_Run178.mcp']),
#     (['60Ni_no_protonTrigger_Run173.mcp'], ['64Ni_no_protonTrigger_Run176.mcp'], ['60Ni_no_protonTrigger_Run178.mcp']),
#     (['60Ni_no_protonTrigger_Run173.mcp'], ['64Ni_no_protonTrigger_Run177.mcp'], ['60Ni_no_protonTrigger_Run178.mcp'])]

# manually configured configs
manual_cfg_58 = [(['60Ni_no_protonTrigger_Run026.mcp', '60Ni_no_protonTrigger_Run027.mcp'],
                  ['58Ni_no_protonTrigger_Run028.mcp'],
                  ['60Ni_no_protonTrigger_Run035.mcp', '60Ni_no_protonTrigger_Run036.mcp',
                   '60Ni_no_protonTrigger_Run037.mcp']),
                 (['60Ni_no_protonTrigger_Run026.mcp', '60Ni_no_protonTrigger_Run027.mcp'],
                  ['58Ni_no_protonTrigger_Run029.mcp'],
                  ['60Ni_no_protonTrigger_Run035.mcp', '60Ni_no_protonTrigger_Run036.mcp',
                   '60Ni_no_protonTrigger_Run037.mcp']),
                 (['60Ni_no_protonTrigger_Run026.mcp', '60Ni_no_protonTrigger_Run027.mcp'],
                  ['58Ni_no_protonTrigger_Run030.mcp'],
                  ['60Ni_no_protonTrigger_Run035.mcp', '60Ni_no_protonTrigger_Run036.mcp',
                   '60Ni_no_protonTrigger_Run037.mcp']),
                 (['60Ni_no_protonTrigger_Run72.mcp'],
                  ['58Ni_no_protonTrigger_Run073.mcp'],
                  ['60Ni_no_protonTrigger_Run076.mcp', '60Ni_no_protonTrigger_Run077.mcp',
                   '60Ni_no_protonTrigger_Run078.mcp']),
                 (['60Ni_no_protonTrigger_Run72.mcp'],
                  ['58Ni_no_protonTrigger_Run074.mcp'],
                  ['60Ni_no_protonTrigger_Run076.mcp', '60Ni_no_protonTrigger_Run077.mcp',
                   '60Ni_no_protonTrigger_Run078.mcp']),
                 (['60Ni_no_protonTrigger_Run72.mcp'],
                  ['58Ni_no_protonTrigger_Run075.mcp'],
                  ['60Ni_no_protonTrigger_Run076.mcp', '60Ni_no_protonTrigger_Run077.mcp',
                   '60Ni_no_protonTrigger_Run078.mcp']),
                 (['60Ni_no_protonTrigger_Run146.mcp', '60Ni_no_protonTrigger_Run147.mcp'],
                  ['58Ni_no_protonTrigger_Run148.mcp'],
                  ['60Ni_no_protonTrigger_Run151.mcp', '60Ni_no_protonTrigger_Run152.mcp']),
                 (['60Ni_no_protonTrigger_Run146.mcp', '60Ni_no_protonTrigger_Run147.mcp'],
                  ['58Ni_no_protonTrigger_Run149.mcp'],
                  ['60Ni_no_protonTrigger_Run151.mcp', '60Ni_no_protonTrigger_Run152.mcp']),
                 (['60Ni_no_protonTrigger_Run146.mcp', '60Ni_no_protonTrigger_Run147.mcp'],
                  ['58Ni_no_protonTrigger_Run150.mcp'],
                  ['60Ni_no_protonTrigger_Run151.mcp', '60Ni_no_protonTrigger_Run152.mcp']),
                 (['60Ni_no_protonTrigger_continuous_Run206.mcp', '60Ni_no_protonTrigger_continuous_Run208.mcp',
                   '60Ni_no_protonTrigger_continuous_Run209.mcp'],
                  ['58Ni_no_protonTrigger_Run210.mcp'],
                  ['60Ni_no_protonTrigger_Run213.mcp', '60Ni_no_protonTrigger_Run214.mcp']),
                 (['60Ni_no_protonTrigger_continuous_Run206.mcp', '60Ni_no_protonTrigger_continuous_Run208.mcp',
                   '60Ni_no_protonTrigger_continuous_Run209.mcp'],
                  ['58Ni_no_protonTrigger_Run211.mcp'],
                  ['60Ni_no_protonTrigger_Run213.mcp', '60Ni_no_protonTrigger_Run214.mcp']),
                 (['60Ni_no_protonTrigger_continuous_Run206.mcp', '60Ni_no_protonTrigger_continuous_Run208.mcp',
                   '60Ni_no_protonTrigger_continuous_Run209.mcp'],
                  ['58Ni_no_protonTrigger_Run212.mcp'],
                  ['60Ni_no_protonTrigger_Run213.mcp', '60Ni_no_protonTrigger_Run214.mcp'])]

manual_cfg_62 = [([], ['62Ni_no_protonTrigger_Run015.mcp'],
                  ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
                   '60Ni_no_protonTrigger_Run021.mcp']),
                 ([], ['62Ni_no_protonTrigger_Run016.mcp'],
                  ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
                   '60Ni_no_protonTrigger_Run021.mcp']),
                 ([], ['62Ni_no_protonTrigger_Run017.mcp'],
                  ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
                   '60Ni_no_protonTrigger_Run021.mcp']),
                 (['60Ni_no_protonTrigger_Run142.mcp', '60Ni_no_protonTrigger_Run143.mcp'],
                  ['62Ni_no_protonTrigger_Run144.mcp'],
                  ['60Ni_no_protonTrigger_Run146.mcp', '60Ni_no_protonTrigger_Run147.mcp']),
                 (['60Ni_no_protonTrigger_Run142.mcp', '60Ni_no_protonTrigger_Run143.mcp'],
                  ['62Ni_no_protonTrigger_Run145.mcp'],
                  ['60Ni_no_protonTrigger_Run146.mcp', '60Ni_no_protonTrigger_Run147.mcp']),
                 (['60Ni_no_protonTrigger_Run162.mcp', '60Ni_no_protonTrigger_Run163.mcp'],
                  ['62Ni_no_protonTrigger_Run164.mcp'],
                  ['60Ni_no_protonTrigger_Run166.mcp', '60Ni_no_protonTrigger_Run167.mcp']),
                 (['60Ni_no_protonTrigger_Run162.mcp', '60Ni_no_protonTrigger_Run163.mcp'],
                  ['62Ni_no_protonTrigger_Run165.mcp'],
                  ['60Ni_no_protonTrigger_Run166.mcp', '60Ni_no_protonTrigger_Run167.mcp'])]

manual_cfg_64 = [
    (['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp', '60Ni_no_protonTrigger_Run021.mcp'],
     ['64Ni_no_protonTrigger_Run022.mcp'],
     ['60Ni_no_protonTrigger_Run025.mcp', '60Ni_no_protonTrigger_Run026.mcp', '60Ni_no_protonTrigger_Run027.mcp']),
    (['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp', '60Ni_no_protonTrigger_Run021.mcp'],
     ['64Ni_no_protonTrigger_Run023.mcp'],
     ['60Ni_no_protonTrigger_Run025.mcp', '60Ni_no_protonTrigger_Run026.mcp', '60Ni_no_protonTrigger_Run027.mcp']),
    (['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp', '60Ni_no_protonTrigger_Run021.mcp'],
     ['64Ni_no_protonTrigger_Run024.mcp'],
     ['60Ni_no_protonTrigger_Run025.mcp', '60Ni_no_protonTrigger_Run026.mcp', '60Ni_no_protonTrigger_Run027.mcp']),
    (['60Ni_no_protonTrigger_Run151.mcp', '60Ni_no_protonTrigger_Run152.mcp'],
     ['64Ni_no_protonTrigger_Run153.mcp'],
     ['60Ni_no_protonTrigger_Run155.mcp', '60Ni_no_protonTrigger_Run156.mcp']),
    (['60Ni_no_protonTrigger_Run151.mcp', '60Ni_no_protonTrigger_Run152.mcp'],
     ['64Ni_no_protonTrigger_Run154.mcp'],
     ['60Ni_no_protonTrigger_Run155.mcp', '60Ni_no_protonTrigger_Run156.mcp']),
    (['60Ni_no_protonTrigger_Run172.mcp', '60Ni_no_protonTrigger_Run173.mcp'],
     ['64Ni_no_protonTrigger_Run174.mcp'],
     ['60Ni_no_protonTrigger_Run178.mcp']),
    (['60Ni_no_protonTrigger_Run172.mcp', '60Ni_no_protonTrigger_Run173.mcp'],
     ['64Ni_no_protonTrigger_Run175.mcp'],
     ['60Ni_no_protonTrigger_Run178.mcp']),
    (['60Ni_no_protonTrigger_Run172.mcp', '60Ni_no_protonTrigger_Run173.mcp'],
     ['64Ni_no_protonTrigger_Run176.mcp'],
     ['60Ni_no_protonTrigger_Run178.mcp']),
    (['60Ni_no_protonTrigger_Run172.mcp', '60Ni_no_protonTrigger_Run173.mcp'],
     ['64Ni_no_protonTrigger_Run177.mcp'],
     ['60Ni_no_protonTrigger_Run178.mcp'])]

manual_cfg_61 = [
    # files 010 - 013 & 120 are ignored for now due to bad buncher settings.
    # ([], ['61Ni_no_protonTrigger_Run010.mcp'],
    #  ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
    #   '60Ni_no_protonTrigger_Run021.mcp']),
    # ([], ['61Ni_no_protonTrigger_Run011.mcp'],
    #  ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
    #   '60Ni_no_protonTrigger_Run021.mcp']),
    # ([], ['61Ni_no_protonTrigger_Run012.mcp'],
    #  ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
    #   '60Ni_no_protonTrigger_Run021.mcp']),
    # ([], ['61Ni_no_protonTrigger_Run013.mcp'],
    #  ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
    #   '60Ni_no_protonTrigger_Run021.mcp']),
    # ([], ['61Ni_no_protonTrigger_Run014.mcp'],
    #  ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
    #   '60Ni_no_protonTrigger_Run021.mcp']),
    # (['60Ni_no_protonTrigger_Run117.mcp', '60Ni_no_protonTrigger_Run118.mcp', '60Ni_no_protonTrigger_Run119.mcp'],
    #  ['61Ni_no_protonTrigger_Run120.mcp'],
    #  ['60Ni_no_protonTrigger_Run125.mcp', '60Ni_no_protonTrigger_Run126.mcp']),
    (['60Ni_no_protonTrigger_Run117.mcp', '60Ni_no_protonTrigger_Run118.mcp', '60Ni_no_protonTrigger_Run119.mcp'],
     ['61Ni_no_protonTrigger_Run121.mcp'],
     ['60Ni_no_protonTrigger_Run125.mcp', '60Ni_no_protonTrigger_Run126.mcp']),
    (['60Ni_no_protonTrigger_Run117.mcp', '60Ni_no_protonTrigger_Run118.mcp', '60Ni_no_protonTrigger_Run119.mcp'],
     ['61Ni_no_protonTrigger_Run123.mcp'],
     ['60Ni_no_protonTrigger_Run125.mcp', '60Ni_no_protonTrigger_Run126.mcp']),
    (['60Ni_no_protonTrigger_Run117.mcp', '60Ni_no_protonTrigger_Run118.mcp', '60Ni_no_protonTrigger_Run119.mcp'],
     ['61Ni_no_protonTrigger_Run124.mcp'],
     ['60Ni_no_protonTrigger_Run125.mcp', '60Ni_no_protonTrigger_Run126.mcp']),
    (['60Ni_no_protonTrigger_Run157.mcp', '60Ni_no_protonTrigger_Run158.mcp'],
     ['61Ni_no_protonTrigger_Run159.mcp'],
     ['60Ni_no_protonTrigger_Run161.mcp']),
    (['60Ni_no_protonTrigger_Run157.mcp', '60Ni_no_protonTrigger_Run158.mcp'],
     ['61Ni_no_protonTrigger_Run160.mcp'],
     ['60Ni_no_protonTrigger_Run161.mcp'])]

configs = {'58_Ni': manual_cfg_58, '62_Ni': manual_cfg_62, '64_Ni': manual_cfg_64, '61_Ni': manual_cfg_61}


# stables = ['61_Ni']


for iso in stables:
    if iso != '60_Ni':
        sel_config = configs[iso]
        for run in runs:
            con = sqlite3.connect(db)
            cur = con.cursor()
            # cur.execute('''UPDATE Combined SET config = ? WHERE iso = ? AND run = ? AND parname = ? ''',
            #             (str(sel_config), iso, run, 'shift'))
            cur.execute(''' UPDATE Combined SET statErrForm = ? ''', ('applyChi(err, rChi)', ))
            con.commit()
            con.close()
    #         Analyzer.combineShift(iso, run, db, show_plot=True)


def extract_shifts(runs_list):
    is_stables_exp = {}
    for selected_run in runs_list:
        is_stables_exp[selected_run] = {}
        is_stables_exp[selected_run]['60_Ni'] = (0, 0, 0)
        for iso in stables:
            connection = sqlite3.connect(db)
            cursor = connection.cursor()
            cursor.execute('''SELECT val, statErr, rChi FROM Combined WHERE iso = ? AND run = ? AND parname = ? ''',
                        (iso, selected_run, 'shift'))
            data = cursor.fetchall()
            connection.close()
            if len(data):
                is_stables_exp[selected_run][iso] = data[0]
    for selected_run in runs_list:
        print('Isotope Shifts for run %s: iso | literature | exp | dif ' % selected_run)
        [print(key1, literature_shifts[key1], value, round(literature_shifts[key1][0] - value[0], 3))
         for key1, value in sorted(is_stables_exp[run].items())]
    return is_stables_exp


def plot_iso_shift_stables(runs_for_shift, val_statErr_rChi_shift_dict):
    for each in runs_for_shift:
        try:
            vals = [(int(key_pl[:2]), val_pl[0], literature_shifts[key_pl][0]) for key_pl, val_pl in
                    sorted(val_statErr_rChi_shift_dict[each].items())]
            errs = [(int(key_pl2[:2]), val_pl2[1], literature_shifts[key_pl2][1]) for key_pl2, val_pl2 in
                    sorted(val_statErr_rChi_shift_dict[each].items())]
            x = [valo[0] for valo in vals]
            # exp_y = [val[1] for val in vals]
            # exp_y_err = [val[1] for val in errs]
            # lit_y = [val[2] for val in vals]
            # lit_y_err = [val[2] for val in errs]
            exp_y = [0 for valo in vals]
            exp_y_err = [valo[1] for valo in errs]
            lit_y = [valo[1] - valo[2] for valo in vals]
            lit_y_err = [valo[2] for valo in errs]
            MPLPlotter.plt.errorbar(x, exp_y, exp_y_err, label='experimental values', linestyle='None', marker="o")
            MPLPlotter.plt.errorbar(x, lit_y, lit_y_err, label='literature values', linestyle='None', marker="o")
            MPLPlotter.plt.legend()
            MPLPlotter.plt.margins(0.25)
            # MPLPlotter.show(True)
        except Exception as err:
            print('error while plotting: %s' % err)


''' Divider Ratio Determination '''
acc_div_start = 1000.05
offset_div_start = 1000

# get the relevant files which need to be fitted in the following:
div_ratio_relevant_stable_files = {}
div_ratio_relevant_stable_files['60_Ni'] = []
for iso, cfg in sorted(configs.items()):
    div_ratio_relevant_stable_files[iso] = []
    for each in cfg:
        [div_ratio_relevant_stable_files['60_Ni'].append(file) for file in each[0] if
         file not in div_ratio_relevant_stable_files['60_Ni']]
        [div_ratio_relevant_stable_files[iso].append(file) for file in each[1]]
        [div_ratio_relevant_stable_files['60_Ni'].append(file) for file in each[2] if
         file not in div_ratio_relevant_stable_files['60_Ni']]
div_ratio_relevant_stable_files['60_Ni'] = sorted(div_ratio_relevant_stable_files['60_Ni'])

# div_ratio_relevant_stable_files.pop('58_Ni')  # due to deviation of 58_Ni, do not fit this one.

print('number of resonances that will be fitted: %s' %
      float(sum([len(val) for key, val in div_ratio_relevant_stable_files.items()])))


# Analyzer.combineRes('60_Ni', 'sigma', runs[0], db, print_extracted=True, show_plot=True)


def chi_square_finder(acc_dev_list, off_dev_list):
    offset_div_ratios = [[]]
    acc_ratios = []
    run = runs[0]
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

            con = sqlite3.connect(db)
            cur = con.cursor()
            divratio = str({'accVolt': current_acc_div, 'offset': curent_off_div})
            cur.execute('''UPDATE Files SET voltDivRatio = ? ''', (divratio,))
            cur.execute('''UPDATE Lines SET frequency = ?''', (new_freq,))
            con.commit()
            con.close()

            # Batchfitting:
            fitres = [(iso, run, BatchFit.batchFit(files, db, run)[1])
                      for iso, files in sorted(div_ratio_relevant_stable_files.items())]

            # combineRes only when happy with voltdivratio, otherwise no use...
            # [[Analyzer.combineRes(iso, par, run, db) for iso in stables] for par in pars]
            try:
                shifts = {iso: Analyzer.combineShift(iso, run, db) for iso in stables if iso not in ['58_Ni', '60_Ni']}
            except Exception as e:
                shifts = {}
                print(e)

            # calc red. Chi ** 2:
            chisq = 0
            for iso, shift_tuple in shifts.items():
                iso_shift_err = np.sqrt(np.square(shift_tuple[3]) + np.square(literature_shifts[iso][1]))
                iso_chisq = np.square((shift_tuple[2] - literature_shifts[iso][0]) / iso_shift_err)
                print('iso: %s chi sq: %s shift tuple: %s ' % (iso, iso_chisq, shift_tuple))
                chisq += iso_chisq
            chisquares[acc_vol_ratio_index].append(float(chisq))
            fit_res[acc_vol_ratio_index].append(fitres)
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


acc_ratios, offset_div_ratios, chisquares = chi_square_finder(range(100, 250, 5), range(100, 250, 5))
#
# print('plotting now')
# try:
#     files = extract_shifts(runs)
#     print('files are: % s' % files)
#     plot_iso_shift_stables(['narrow_gate'], files)
#     MPLPlotter.show(True)
# except Exception as e:
#     print('plotting did not work, error is: %s' % e)


print('------------------- Done -----------------')
winsound.Beep(2500, 500)

# print('\a')


''' Fit on certain Files '''
# searchterm = 'Run167'
# certain_file = [file for file in ni60_files if searchterm in file][0]
# fit = InteractiveFit.InteractiveFit(certain_file, db, runs[0], block=True, x_as_voltage=True)
# fit.fit()


''' results: '''
acc_divs_result = [998.05, 998.25, 998.4499999999999, 998.65, 998.8499999999999, 999.05, 999.25, 999.4499999999999,
                   999.65, 999.8499999999999, 1000.05, 1000.25, 1000.4499999999999, 1000.65, 1000.8499999999999,
                   1001.05, 1001.25, 1001.4499999999999, 1001.65, 1001.8499999999999, 1002.05]
off_divs_result = [998.0, 998.2, 998.4, 998.6, 998.8, 999.0, 999.2, 999.4, 999.6, 999.8, 1000.0, 1000.2, 1000.4, 1000.6,
                   1000.8, 1001.0, 1001.2, 1001.4, 1001.6, 1001.8, 1002.0]
chisquares_result = [[1.7805125361029144, 3.212610317509303, 7.498024978976431, 14.984382673936475, 25.275387463102426,
                      39.04452171192091, 55.62602677943423, 75.47017993996532, 98.33111261177751, 124.18503499174972,
                      153.0342410220741, 184.86935425227387, 219.68384336983877, 258.2899338539038, 299.1566989581964,
                      345.24570825867886, 392.30911702180447, 442.2597846670234, 495.22323085234524, 551.0925370592823,
                      610.0058571534938],
                     [2.246113312252546, 1.5608686435073218, 3.1376629138213024, 7.367971507767217, 14.687018223386657,
                      24.989602396064427, 38.41397498785054, 55.37775968987857, 74.57526303759234, 97.32938885981422,
                      123.07363874940808, 151.8279254075697, 183.5746358406928, 218.306325873263, 256.01680078334584,
                      297.6231408701635, 341.38917427686175, 390.62701111836935, 439.86251132531845, 493.40004609956793,
                      549.1998509994778],
                     [4.772111837997043, 1.8576471833885564, 1.3104502084489198, 3.0347340281032658, 7.262927100380592,
                      14.43720626882433, 24.79636095447255, 37.814122063569364, 54.379055020959385, 73.68827081695704,
                      96.31941106237748, 121.95399278138905, 150.60671925220396, 182.26188181490303, 216.9082222980515,
                      254.53808817063185, 296.06577902299205, 339.7585099784147, 388.9218122735674, 438.66320914802657,
                      491.55392106148366],
                     [9.979671954225424, 4.433789760797801, 1.5281490019701798, 1.0712731334677619, 2.9215019287455557,
                      7.184278756760318, 14.243446185605075, 24.420649785960386, 37.49255690995948, 53.658300647935796,
                      73.42962984920196, 95.31558542099735, 120.85797286272341, 149.37543692166633, 180.93296758938678,
                      215.49038665259584, 253.03782198136759, 293.56630146550174, 338.1031159984398, 384.77334357199396,
                      436.8638630402372],
                     [18.025720455260863, 9.720419185139137, 4.172554614542382, 1.262479861128303, 0.8673076197225715,
                      2.82096854353975, 7.1380219106289315, 14.113187307570715, 24.11471224576134, 37.28864646237167,
                      52.98303644786452, 72.31394607717598, 94.32806892897476, 119.73761991937849, 148.13901248521228,
                      179.59088679142596, 214.0546419545081, 251.51610203649992, 291.9652508538069, 336.42404719800305,
                      383.0220850892987],
                     [29.141616130703852, 17.899669019097193, 9.531697022024476, 3.960234639627085, 1.0552429561036383,
                      0.7065878434907398, 2.748812881758989, 7.131337759177531, 14.053589796396802, 23.8913256141504,
                      36.86727604650602, 52.653670953383454, 71.51952530073807, 94.07293635829937, 118.63141504726323,
                      146.95619180356107, 178.2413238617604, 212.6044901710618, 249.97553257662685, 290.3415032492089,
                      333.6929351559211],
                     [43.37633656896273, 29.1380812652451, 17.82665644854114, 9.40022203032428, 3.7919214041534586,
                      0.8972361839912995, 0.5868396259891292, 2.7103375651975887, 7.167552806578008, 14.070419169471958,
                      23.76107729336787, 36.537069151935455, 52.468562097803826, 70.78838388449411, 92.85445669918559,
                      117.55071053381506, 145.7304463328526, 176.89128318190728, 211.1434824334874, 248.41851182433678,
                      288.69811950136926],
                     [60.854479831448586, 43.57271598696368, 29.244798955026127, 17.79655820460716, 9.314875158438614,
                      3.6702982744725228, 0.7742734321649545, 0.5010517492422581, 2.702428454327435, 7.244270435903258,
                      14.162112321674456, 23.732314531196522, 36.310823016039315, 52.03008792223833, 70.47617466771581,
                      92.00132552646045, 117.31184179216993, 144.52653897409033, 175.62950572429008, 209.6781420129861,
                      246.84841387771002],
                     [81.36533537840393, 61.052248009656125, 43.70469755153951, 29.31173818285904, 17.854482243310468,
                      9.302193725608483, 3.605084617841741, 0.6859495536532983, 0.432568784057798, 2.697688570516501,
                      7.32453082528711, 14.280924780146174, 23.766078418934036, 36.14875681850439, 51.64996989630782,
                      70.2824719859293, 91.173125548633, 115.94592635210552, 143.278864925429, 174.22041249337914,
                      208.10110159842952],
                     [104.99272379589726, 81.64561772821261, 61.27012081159337, 43.85961039748569, 29.404723256471094,
                      17.888646468526222, 9.2849901994527, 3.55085527444679, 0.6193882705073971, 0.39092176022550396,
                      2.726961101260926, 7.456066605447764, 14.487859334023923, 23.942501333227632, 36.155335530318496,
                      51.441297958856424, 69.85042452032961, 90.90648490922162, 115.0457805671644, 143.06296214705205,
                      172.92419545385837],
                     [131.72668081962954, 105.34735892076884, 81.94129614210254, 61.50511104316875, 44.03434096622407,
                      29.519472950205653, 17.9464638945853, 9.292467294317532, 3.520483711104848, 0.572726811942922,
                      0.3616519361720485, 2.7622440432610333, 7.5971154170030895, 14.7293059429047, 24.19996440003286,
                      36.28562159285546, 51.356383611049566, 69.54533105001029, 90.82927683083753, 114.24947076192703,
                      141.70363324797316],
                     [161.55859922802264, 132.14922999206846, 105.71317264540838, 82.24948233236599, 61.754935109873415,
                      44.22530435824139, 29.652310932032115, 18.02398672919141, 9.320158223871731, 3.5092344612705446,
                      0.5413325527584895, 0.3401214844056956, 2.794680711013145, 7.736908630912261, 14.988042659572432,
                      24.517623238502757, 36.5300676485171, 51.398372349338366, 69.37043946840103, 90.42217066695613,
                      114.03889486975828],
                     [194.48137432150338, 162.04387679314115, 132.58009815742045, 106.08886132807379, 82.56877018272301,
                      62.01717783111795, 44.4299879429425, 29.80047845328494, 18.11780811870139, 9.364491874183086,
                      3.51354517121505, 0.5217310159293416, 0.32306379192934187, 2.8193613002995055, 7.866457742474141,
                      15.247742323946385, 24.874436806998055, 36.86790101669938, 51.57054483663716, 69.32219880793524,
                      90.15680443314682],
                     [230.4885720094755, 195.0256996382302, 162.5359204485029, 133.01852093101047, 106.47325025755993,
                      82.89749595843614, 62.29004114712828, 44.64660986576715, 29.961807954724136, 18.225381752849,
                      9.422783753124966, 3.5304966161293727, 0.5113677126496607, 0.3083601834428845, 2.8355089236077116,
                      7.970123558309868, 15.494891945322413, 25.248543201560185, 37.27607905879249, 51.864641667719404,
                      69.39839941151281],
                     [269.5751936629454, 231.08923133777222, 195.57570353739328, 163.03376863037025, 133.46393373840732,
                      106.86448949737745, 83.23474551814581, 62.57219247397818, 44.873439601818916, 30.133706981890146,
                      18.34453977005046, 9.492768612700146, 3.5579151811502, 0.5083207615250321, 0.2947234132936537,
                      2.8422804103293347, 8.041843209506403, 15.719089642496277, 25.619710624164036, 37.73083066579814,
                      52.26086187459943],
                     [311.73627018767763, 270.22983903296586, 231.69435151894248, 196.13013438047017, 163.5374046821709,
                      133.91517924423374, 107.26267476462485, 83.57917872461292, 62.862242987889296, 45.10932542953984,
                      30.315449202900844, 18.47374936261836, 9.572474255300678, 3.5940367042055934, 0.5111553951567884,
                      0.28133670144284767, 2.8398086640232236, 8.090341482635809, 15.898100237273631, 25.97168737710299,
                      38.207402810242996],
                     [356.6961983325394, 312.44202831165904, 270.8873547000207, 232.3029043360204, 196.68896231303563,
                      164.04552018047082, 134.3713085630627, 107.66672796035984, 83.92987576161339, 63.15931600301978,
                      45.352591869847416, 30.505213707777507, 18.61135273127392, 9.660494210612596, 3.6375660417724522,
                      0.5188313947057777, 0.26781950803717847, 2.828559594767909, 8.11690883262855, 16.01198329475409,
                      26.291417990984353],
                     [404.97661339469835, 357.45168518532864, 313.1513908000213, 271.54874149745837, 232.91557259700016,
                      197.25182788733295, 164.55782140913482, 134.83242557258976, 108.07565807501915, 84.28623694219593,
                      63.46274739808915, 45.60270977456497, 30.70241662841586, 18.756358795566094, 9.75571666994571,
                      3.687320863043692, 0.5305470143856217, 0.25399300466252506, 2.8093549458184435, 8.123454093495956,
                      16.089326794061495],
                     [456.4619293541281, 405.6822829677992, 358.4812925486415, 313.8622743857221, 272.21269596795315,
                      233.53107021053566, 197.8183927108409, 165.07392281362442, 135.29800609605377, 108.48948616741579,
                      84.64776500939914, 63.77175449089276, 45.858643801910354, 30.90561370732172, 18.9077134147824,
                      9.85714177513029, 3.742425242355959, 0.5456871435980812, 0.23980658196180718, 2.783110032049627,
                      8.11226308174651],
                     [510.7484550577699, 457.3129908883984, 406.4885136387266, 359.24197696464375, 314.57657274606083,
                      272.8792141739907, 234.1495738633743, 198.3879440823261, 165.59354041265325, 135.76693478128672,
                      108.90757034689659, 85.01372699749842, 64.08544721395005, 46.11988501181415, 31.114714482858055,
                      19.064875458517875, 9.964023996018003, 3.802200438457848, 0.5637995244225125, 0.22532098703970566,
                      2.750783014628007],
                     [568.2047794727658, 511.9458105676367, 457.95603149641977, 407.2963882416916, 360.0049722863955,
                      315.29307117113115, 273.5480099278461, 234.77054207936612, 198.9602728361786, 166.11670069443628,
                      136.23952166594069, 109.32903548097968, 85.38399119477135, 64.40369103228761, 46.38579997112259,
                      31.328241385086567, 19.226825718076576, 10.075644369554837, 3.8660924148171505,
                      0.5845114242995966, 0.2105937389755355]]

acc_divs_result = acc_ratios
off_divs_result = offset_div_ratios[0]
chisquares_result = chisquares

import PyQtGraphPlotter as PGplt
from PyQt5 import QtWidgets
import sys

x_range = (float(np.min(acc_divs_result)), np.max(acc_divs_result))
x_scale = np.mean(np.ediff1d(acc_divs_result))
y_range = (float(np.min(off_divs_result)), np.max(off_divs_result))
y_scale = np.mean(np.ediff1d(off_divs_result))

chisquares_result = np.array(chisquares_result)

app = QtWidgets.QApplication(sys.argv)
main_win = QtWidgets.QMainWindow()
widg, plt_item = PGplt.create_image_view('acc_volt_div_ratio', 'offset_div_ratio')
widg.setImage(chisquares_result,
              pos=[x_range[0] - abs(0.5 * x_scale),
                   y_range[0] - abs(0.5 * y_scale)],
              scale=[x_scale, y_scale])
try:
    main_win.setCentralWidget(widg)
except Exception as e:
    print(e)
main_win.show()

app.exec()
