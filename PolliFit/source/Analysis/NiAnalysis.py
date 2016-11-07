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
runs = [runs[0]]


# for iso in stables:
#     sel_config = configs[iso]
#     for run in runs:
#         con = sqlite3.connect(db)
#         cur = con.cursor()
#         cur.execute('''UPDATE Combined SET config = ? WHERE iso = ? AND run = ? AND parname = ? ''',
#                     (str(sel_config), iso, run, 'shift'))
#         con.commit()
#         con.close()
#         Analyzer.combineShift(iso, run, db, show_plot=True)


def extract_shifts(runs):
    is_stables_exp = {}
    for run in runs:
        is_stables_exp[run] = {}
        is_stables_exp[run]['60_Ni'] = (0, 0, 0)
        for iso in stables:
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute('''SELECT val, statErr, rChi FROM Combined WHERE iso = ? AND run = ? AND parname = ? ''',
                        (iso, run, 'shift'))
            data = cur.fetchall()
            con.close()
            if len(data):
                is_stables_exp[run][iso] = data[0]
    for run in runs:
        print('Isotope Shifts for run %s: iso | literature | exp | dif ')
        [print(key, literature_shifts[key], val, round(literature_shifts[key][0] - val[0], 3))
         for key, val in sorted(is_stables_exp[run].items())]
    return is_stables_exp


def plot_iso_shift_stables(runs, val_statErr_rChi_shift_dict):
    for run in runs:
        vals = [(int(key[:2]), val[0], literature_shifts[key][0]) for key, val in
                sorted(val_statErr_rChi_shift_dict[run].items())]
        errs = [(int(key[:2]), val[1], literature_shifts[key][1]) for key, val in
                sorted(val_statErr_rChi_shift_dict[run].items())]
        x = [val[0] for val in vals]
        # exp_y = [val[1] for val in vals]
        # exp_y_err = [val[1] for val in errs]
        # lit_y = [val[2] for val in vals]
        # lit_y_err = [val[2] for val in errs]
        exp_y = [0 for val in vals]
        exp_y_err = [val[1] for val in errs]
        lit_y = [val[1] - val[2] for val in vals]
        lit_y_err = [val[2] for val in errs]
        MPLPlotter.plt.errorbar(x, exp_y, exp_y_err, label='experimental values', linestyle='None', marker="o")
        MPLPlotter.plt.errorbar(x, lit_y, lit_y_err, label='literature values', linestyle='None', marker="o")
        MPLPlotter.plt.legend()
        MPLPlotter.plt.margins(0.25)
        MPLPlotter.show(True)


''' Divider Ratio Determination '''
acc_div_start = 1000.05
offset_div_start = 1000

# get the relevant files whihc need to be fitted in the following:
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
print('number of resonances that will be fitted: %s' %
      float(sum([len(val) for key, val in div_ratio_relevant_stable_files.items()])))

# Analyzer.combineRes('60_Ni', 'sigma', runs[0], db, print_extracted=True, show_plot=True)

offset_div_ratios = [[]]
acc_ratios = []
run = runs[0]
fit_res = [[]]
chisquares = [[]]
acc_vol_ratio_index = 0
for acc in range(-300, 160, 20):
    current_acc_div = acc_div_start + acc / 100
    freq = -442.4 * acc / 100 - 9.6  # value found by playing with gui
    freq += transition_freq
    # freq = transition_freq
    print('setting transition Frequency to: %s ' % freq)
    acc_ratios.append(current_acc_div)

    for off in range(-200, 320, 20):

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
            shifts = {iso: Analyzer.combineShift(iso, run, db) for iso in stables if iso is not '60_Ni'}
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
print('plotting now')
# try:
#     files = extract_shifts(runs)
#     print('files are: % s' % files)
#     plot_iso_shift_stables(runs, files)
# except Exception as e:
#     print('plotting did not work, error is: %s' % e)
print('acc\toff\tchisquare')
for acc_ind, acc_rat in enumerate(acc_ratios):
    for off_ind, off_rat in enumerate(offset_div_ratios[acc_ind]):
        print(('%s\t%s\t%s' % (acc_rat, off_rat, chisquares[acc_ind][off_ind])).replace('.', ','))

print('------------------- Done -----------------')
winsound.Beep(2500, 500)

# print('\a')


''' Fit on certain Files '''
# searchterm = 'Run167'
# certain_file = [file for file in ni60_files if searchterm in file][0]
# fit = InteractiveFit.InteractiveFit(certain_file, db, runs[0], block=True, x_as_voltage=True)
# fit.fit()


''' results: '''
# acc_divs_result = [999.05, 999.15, 999.25, 999.3499999999999, 999.4499999999999, 999.55, 999.65, 999.75,
#                    999.8499999999999,
#                    999.9499999999999, 1000.05, 1000.15, 1000.25, 1000.3499999999999, 1000.4499999999999, 1000.55,
#                    1000.65,
#                    1000.75, 1000.8499999999999, 1000.9499999999999, 1001.05]
# off_divs_result = [1000.0, 1000.1, 1000.2, 1000.3, 1000.4, 1000.5, 1000.6, 1000.7, 1000.8, 1000.9, 1001.0, 1001.1,
#                    1001.2,
#                    1001.3, 1001.4, 1001.5, 1001.6, 1001.7, 1001.8, 1001.9, 1002.0]
# chisquares_result = [[45.01229218452062, 50.69843336760621, 57.57091254503574, 65.95232538868225, 76.0286674750511,
#                       87.53181718302639, 100.74691611764959, 114.42829118023371, 130.05260317653625, 147.02513088815945,
#                       165.33799304917838, 185.05297511989085, 206.00955604765454, 228.28458491624468,
#                       251.87493801827125,
#                       276.7769487105806, 302.9772733650898, 330.49906541713153, 359.33045680033314, 389.46050452588867,
#                       420.90991262576375],
#                      [40.70252722340542, 44.78891494822458, 50.380662521575644, 57.47901702193742, 65.47512850491682,
#                       75.49099858802444, 86.94612598903147, 99.81312414681997, 114.3914775202434, 129.3777958430921,
#                       146.33288087322313, 164.6304789789784, 184.3323919524939, 205.277316931893, 227.53985949803592,
#                       251.11732496925129, 276.00747532516414, 302.19411035047716, 329.7021117860459, 358.5190372694503,
#                       388.63371157787094],
#                      [38.09206614326835, 40.608119293175, 44.59690284296598, 50.08983758782575, 57.09533786300952,
#                       65.2702875337721, 74.95931385318192, 86.36059893572731, 99.1859731644205, 113.73177307042357,
#                       128.69406787131425, 145.62937808125398, 163.90814898257062, 183.5232846934079, 204.5327313749487,
#                       226.78342270764165, 250.3493392192738, 275.2274429839773, 301.41482272401845, 328.8966690777025,
#                       357.6993220579358],
#                      [37.08867921701339, 38.117358009367145, 40.5489872744721, 44.44015400553168, 49.83031198956954,
#                       56.736961485632904, 65.15705908616953, 74.43823441390309, 85.77899232741846, 98.55724733008354,
#                       112.74420246075937, 128.64787369034175, 144.9162588046655, 163.1778746142208, 182.77907021284426,
#                       203.77686683719872, 226.01678513884266, 249.57072478925605, 274.4367530149972, 300.6131495746095,
#                       328.082518228193],
#                      [37.544713069752184, 37.206481839308275, 38.17610089385888, 40.52654471654518, 44.32080269493294,
#                       49.60553842625534, 56.40788212024812, 64.73184221320564, 74.20840137282563, 85.20407087830237,
#                       97.92901699878949, 112.07522582640256, 127.94959629246354, 144.19333819871665, 162.4366257262252,
#                       182.02347205381335, 202.93570041320913, 225.2378705261305, 248.78107599763388, 273.636223672751,
#                       299.8008063214031],
#                      [39.29726627010662, 37.713696288037724, 37.35058407755004, 38.268452117237544, 40.542228308335886,
#                       44.24145076825673, 49.419847454440465, 56.11282426186213, 64.33374490055421, 74.07488655248413,
#                       84.63979237711168, 97.30442862645369, 111.40416526990163, 126.90964161892434, 144.1377621833409,
#                       161.68518550725338, 181.2564751107336, 202.15594215792387, 224.44735562213268, 247.98070300271056,
#                       272.8249313534212],
#                      [42.1984214409951, 39.43854871648868, 37.849708636355544, 37.463434053145, 38.332847785353295,
#                       40.5301421215089, 44.13151919838761, 49.19775621101241, 55.77251681814542, 63.878570060798616,
#                       73.51520251679169, 84.28656963232626, 96.57711478583904, 110.61893565836078, 126.07947009275813,
#                       143.2650328334084, 160.7823616855226, 180.33100574001338, 201.21167474048409, 223.41719781737888,
#                       246.99901471686888],
#                      [46.27503594450667, 42.375576762708626, 39.63029909984655, 38.04733943927857, 37.65175918000588,
#                       38.48792286676931, 40.623315008899084, 44.136664246403065, 49.09772421915428, 55.55773396363064,
#                       63.54825035032329, 73.07764941078621, 84.13385611144128, 95.97090226660136, 109.95285630824522,
#                       125.36790357758748, 142.18223325505087, 160.72257965050161, 179.54272525024786, 200.4092117871999,
#                       222.6028188913641],
#                      [51.48045708492768, 46.43634648370378, 42.55532142289299, 39.827365645458435, 38.25525314966466,
#                       37.85716633785571, 38.66998165227774, 40.75362065291508, 44.18547934003058, 49.04376893860447,
#                       55.38823602376921, 63.258869580573034, 72.67396669766967, 83.62787067455001, 95.69957126059775,
#                       109.29396097365512, 124.65679396653475, 141.43269716110336, 159.94409647051572,
#                       178.74044592935547,
#                       199.595180939685],
#                      [57.84114308363239, 51.62762069609453, 46.59848721003208, 42.73614367258196, 40.02807574854775,
#                       38.47103418348684, 38.07654901624738, 38.87555164826893, 40.918165059659124, 44.2775744847119,
#                       49.03709925835706, 55.266751906369024, 63.01463240327765, 72.30835440851038, 83.15171685750843,
#                       95.52787115834286, 108.6466669324098, 123.94987187097124, 140.6837675258115, 158.81292073253877,
#                       178.6734809966617],
#                      [65.39551271327853, 57.979885523922796, 51.774932703491345, 46.76006401343994, 42.91666756320386,
#                       40.22966521303391, 38.691126682156906, 38.30676943722937, 39.09974777012387, 41.11265856881137,
#                       44.41120447690589, 49.078671315396306, 55.1953867496316, 62.819121874902756, 71.98629603717785,
#                       82.71066806083252, 94.98205635367225, 108.35334271207404, 123.25041942015872, 139.93310506346396,
#                       158.0248169951207],
#                      [74.18300588301015, 65.53431028837025, 58.11980405955974, 51.922035294140414, 46.92030828244353,
#                       43.0955365774301, 40.43037613970301, 38.913427117966926, 38.54432064329892, 39.33892333281119,
#                       41.3327763841767, 44.58267478462577, 49.167809458929675, 55.175341528481596, 62.67500453189094,
#                       71.7119994332561, 82.30976653866911, 94.46721285258833, 108.16284145628589, 122.56265745551482,
#                       139.18619432565254],
#                      [84.23755209479184, 74.32871359864507, 65.67439372798466, 58.25977741227725, 52.06759035504961,
#                       47.078222981192106, 43.271228195656064, 40.628370280794094, 39.13482678759556, 38.785463283514474,
#                       39.589170348760966, 41.573581760536456, 44.78729473656004, 49.302527719877084, 55.20767928346528,
#                       62.584717135010294, 71.4895154668232, 81.95456995491995, 93.98913549518274, 107.57796088172931,
#                       122.24841834825321],
#                      [95.58580423296578, 84.39590581584558, 74.47678714767973, 65.81597651821131, 58.40011854646201,
#                       52.21227734307449, 47.233203239603256, 43.44316293949161, 40.822325885857886, 39.353556170584106,
#                       39.02753051236919, 39.846367955562584, 41.82995650656702, 45.01978142371188, 49.478536577869235,
#                       55.29143580608694, 62.54943621607397, 71.32139484356212, 81.64944883464521, 93.55263262600918,
#                       107.02469817209109],
#                      [108.24739713131194, 95.76090188142152, 84.55688328494423, 74.62656278547041, 65.95857814460635,
#                       58.53981982717646, 52.354557333588495, 47.38487427372461, 43.61049765337234, 41.010872399151815,
#                       39.56755386415483, 39.267317028996054, 40.10657621620653, 42.09810547609991, 45.274710000851776,
#                       49.6908596747304, 55.42471380882687, 62.57032665810783, 71.21044935576788, 81.39860752348679,
#                       93.16343592190432],
#                      [122.2397086369685, 108.44335747577676, 95.93930305146068, 84.72080491696518, 74.77818092990988,
#                       66.101782385375, 58.679301222634514, 52.49552138893444, 47.5331204301546, 43.772574610737365,
#                       41.19334952021371, 39.77518896802299, 39.502250924142395, 40.36632994137729, 42.3728548695135,
#                       45.54628566860792, 49.932999944763345, 55.60264769081772, 62.64612959919809, 71.15770615046326,
#                       81.20495772826713],
#                      [137.57083527401437, 122.45749508485977, 108.64183527769525, 96.12040542225033, 84.88594505786432,
#                       74.93088230704406, 66.24564964020715, 58.81794219111548, 52.6339881125714, 47.67762210214909,
#                       43.92913334823578, 41.36879398042708, 39.97545047753183, 39.730731144173554, 40.62263391277908,
#                       42.65021263213999, 45.82990371259062, 50.199459000327316, 55.81970461243599, 62.775123647575285,
#                       71.16452163140704],
#                      [154.25013627340465, 137.81237881482815, 122.67805834621329, 108.84368166324647, 96.30401166387783,
#                       85.05352702582168, 75.08562939937947, 66.38989801545918, 58.95596366889503, 52.77044966414341,
#                       47.81823338085237, 44.08028523464607, 41.53706449538599, 40.16720658261256, 39.95089877154166,
#                       40.872473243278606, 42.926099518002104, 46.120084896021105, 50.48361874790676, 56.069054037935786,
#                       62.95185093626671],
#                      [172.28103505157972, 154.51490104226366, 138.0562041365761, 122.90143897590445, 109.04760403448905,
#                       96.48973236467259, 85.2230475055506, 75.24066114191626, 66.53427319935419, 59.09301735866695,
#                       52.90472540282428, 47.954858896424255, 44.22577618321952, 41.69786210711671, 40.35039109387793,
#                       40.161538404416646, 41.114218906308714, 43.19748071338121, 46.41270670657353, 50.78048192364703,
#                       56.34421720617451],
#                      [191.6671480920405, 172.5703219379408, 154.7826476854429, 138.3029759965934, 123.12785262112402,
#                       109.25391717413494, 96.6779053119614, 85.39454381661938, 75.39738507624179, 66.67897863827034,
#                       59.22936158355949, 53.03672665188634, 48.08768438851183, 44.36538523644146, 41.85102315622719,
#                       40.52373568712301, 40.361932823617245, 41.345900616083235, 43.46100013185151, 46.702993913624404,
#                       51.08436571206633],
#                      [212.41202854398034, 191.98101129698924, 172.8620786872209, 155.0528841551094, 138.5516125232318,
#                       123.35607917881491, 109.46227069630658, 96.86761178734632, 85.56739718575864, 75.55492559623629,
#                       66.82390633513141, 59.364773410824185, 53.16691803064188, 48.21695758056237, 44.499782745489114,
#                       41.996845592977145, 40.68807737184187, 40.55165498756001, 41.5663417847371, 43.71481731676147,
#                       46.98778710120829]]

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
