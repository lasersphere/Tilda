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

con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''UPDATE Lines SET frequency = ?''', (transition_freq, ))
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
auto_cfg_58 = [([], ['58Ni_no_protonTrigger_Run006.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               ([], ['58Ni_no_protonTrigger_Run007.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               ([], ['58Ni_no_protonTrigger_Run008.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               ([], ['58Ni_no_protonTrigger_Run009.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               (['60Ni_no_protonTrigger_Run027.mcp'], ['58Ni_no_protonTrigger_Run028.mcp'],
                ['60Ni_no_protonTrigger_Run035.mcp']),
               (['60Ni_no_protonTrigger_Run027.mcp'], ['58Ni_no_protonTrigger_Run029.mcp'],
                ['60Ni_no_protonTrigger_Run035.mcp']),
               (['60Ni_no_protonTrigger_Run027.mcp'], ['58Ni_no_protonTrigger_Run030.mcp'],
                ['60Ni_no_protonTrigger_Run035.mcp']),
               (['60Ni_no_protonTrigger_Run72.mcp'], ['58Ni_no_protonTrigger_Run073.mcp'],
                ['60Ni_no_protonTrigger_Run076.mcp']),
               (['60Ni_no_protonTrigger_Run72.mcp'], ['58Ni_no_protonTrigger_Run074.mcp'],
                ['60Ni_no_protonTrigger_Run076.mcp']),
               (['60Ni_no_protonTrigger_Run72.mcp'], ['58Ni_no_protonTrigger_Run075.mcp'],
                ['60Ni_no_protonTrigger_Run076.mcp']),
               (['60Ni_no_protonTrigger_Run147.mcp'], ['58Ni_no_protonTrigger_Run148.mcp'],
                ['60Ni_no_protonTrigger_Run151.mcp']),
               (['60Ni_no_protonTrigger_Run147.mcp'], ['58Ni_no_protonTrigger_Run149.mcp'],
                ['60Ni_no_protonTrigger_Run151.mcp']),
               (['60Ni_no_protonTrigger_Run147.mcp'], ['58Ni_no_protonTrigger_Run150.mcp'],
                ['60Ni_no_protonTrigger_Run151.mcp']),
               ([], ['58Ni_no_protonTrigger_Run210.mcp'], ['60Ni_no_protonTrigger_Run213.mcp']),
               ([], ['58Ni_no_protonTrigger_Run211.mcp'], ['60Ni_no_protonTrigger_Run213.mcp']),
               ([], ['58Ni_no_protonTrigger_Run212.mcp'], ['60Ni_no_protonTrigger_Run213.mcp'])]

auto_cfg_61 = [([], ['61Ni_no_protonTrigger_Run010.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               ([], ['61Ni_no_protonTrigger_Run011.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               ([], ['61Ni_no_protonTrigger_Run012.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               ([], ['61Ni_no_protonTrigger_Run013.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               ([], ['61Ni_no_protonTrigger_Run014.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               (['60Ni_no_protonTrigger_Run119.mcp'], ['61Ni_no_protonTrigger_Run120.mcp'],
                ['60Ni_no_protonTrigger_Run125.mcp']),
               (['60Ni_no_protonTrigger_Run119.mcp'], ['61Ni_no_protonTrigger_Run121.mcp'],
                ['60Ni_no_protonTrigger_Run125.mcp']),
               (['60Ni_no_protonTrigger_Run119.mcp'], ['61Ni_no_protonTrigger_Run123.mcp'],
                ['60Ni_no_protonTrigger_Run125.mcp']),
               (['60Ni_no_protonTrigger_Run119.mcp'], ['61Ni_no_protonTrigger_Run124.mcp'],
                ['60Ni_no_protonTrigger_Run125.mcp']),
               (['60Ni_no_protonTrigger_Run158.mcp'], ['61Ni_no_protonTrigger_Run159.mcp'],
                ['60Ni_no_protonTrigger_Run161.mcp']),
               (['60Ni_no_protonTrigger_Run158.mcp'], ['61Ni_no_protonTrigger_Run160.mcp'],
                ['60Ni_no_protonTrigger_Run161.mcp'])]

auto_cfg_62 = [([], ['62Ni_no_protonTrigger_Run015.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               ([], ['62Ni_no_protonTrigger_Run016.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               ([], ['62Ni_no_protonTrigger_Run017.mcp'], ['60Ni_no_protonTrigger_Run018.mcp']),
               (['60Ni_no_protonTrigger_Run143.mcp'], ['62Ni_no_protonTrigger_Run144.mcp'],
                ['60Ni_no_protonTrigger_Run146.mcp']),
               (['60Ni_no_protonTrigger_Run143.mcp'], ['62Ni_no_protonTrigger_Run145.mcp'],
                ['60Ni_no_protonTrigger_Run146.mcp']),
               (['60Ni_no_protonTrigger_Run163.mcp'], ['62Ni_no_protonTrigger_Run164.mcp'],
                ['60Ni_no_protonTrigger_Run166.mcp']),
               (['60Ni_no_protonTrigger_Run163.mcp'], ['62Ni_no_protonTrigger_Run165.mcp'],
                ['60Ni_no_protonTrigger_Run166.mcp'])]

auto_cfg_64 = [
    (['60Ni_no_protonTrigger_Run021.mcp'], ['64Ni_no_protonTrigger_Run022.mcp'], ['60Ni_no_protonTrigger_Run025.mcp']),
    (['60Ni_no_protonTrigger_Run021.mcp'], ['64Ni_no_protonTrigger_Run023.mcp'], ['60Ni_no_protonTrigger_Run025.mcp']),
    (['60Ni_no_protonTrigger_Run021.mcp'], ['64Ni_no_protonTrigger_Run024.mcp'], ['60Ni_no_protonTrigger_Run025.mcp']),
    (['60Ni_no_protonTrigger_Run152.mcp'], ['64Ni_no_protonTrigger_Run153.mcp'], ['60Ni_no_protonTrigger_Run155.mcp']),
    (['60Ni_no_protonTrigger_Run152.mcp'], ['64Ni_no_protonTrigger_Run154.mcp'], ['60Ni_no_protonTrigger_Run155.mcp']),
    (['60Ni_no_protonTrigger_Run173.mcp'], ['64Ni_no_protonTrigger_Run174.mcp'], ['60Ni_no_protonTrigger_Run178.mcp']),
    (['60Ni_no_protonTrigger_Run173.mcp'], ['64Ni_no_protonTrigger_Run175.mcp'], ['60Ni_no_protonTrigger_Run178.mcp']),
    (['60Ni_no_protonTrigger_Run173.mcp'], ['64Ni_no_protonTrigger_Run176.mcp'], ['60Ni_no_protonTrigger_Run178.mcp']),
    (['60Ni_no_protonTrigger_Run173.mcp'], ['64Ni_no_protonTrigger_Run177.mcp'], ['60Ni_no_protonTrigger_Run178.mcp'])]

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
        [div_ratio_relevant_stable_files['60_Ni'].append(file) for file in each[0] if file not in div_ratio_relevant_stable_files['60_Ni']]
        [div_ratio_relevant_stable_files[iso].append(file) for file in each[1]]
        [div_ratio_relevant_stable_files['60_Ni'].append(file) for file in each[2] if file not in div_ratio_relevant_stable_files['60_Ni']]
div_ratio_relevant_stable_files['60_Ni'] = sorted(div_ratio_relevant_stable_files['60_Ni'])
print('number of resonances that will be fitted: %s' %
      float(sum([len(val) for key, val in div_ratio_relevant_stable_files.items()])))

# Analyzer.combineRes('60_Ni', 'sigma', runs[0], db, print_extracted=True, show_plot=True)

offset_div_ratios = []
acc_ratios = []
run = runs[0]
fit_res = [[]]
chisquares = [[]]
acc_vol_ratio_index = 0
for acc in [0]:
    current_acc_div = acc_div_start + acc / 100
    freq = -442.4 * acc / 100 - 9.6  # value found by playing with gui
    freq += transition_freq
    # freq = transition_freq
    print('setting transition Frequency to: %s ' % freq)

    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute('''UPDATE Lines SET frequency = ?''', (freq,))
    con.commit()
    con.close()

    for off in [0]:
        curent_off_div = offset_div_start + off / 100

        con = sqlite3.connect(db)
        cur = con.cursor()
        divratio = str({'accVolt': current_acc_div, 'offset': curent_off_div})
        cur.execute('''UPDATE Files SET voltDivRatio = ? ''', (divratio, ))

        # cur.execute('''UPDATE Lines SET frequency='''+str(freq))

        con.commit()
        con.close()
        offset_div_ratios.append(curent_off_div)
        acc_ratios.append(current_acc_div)

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

    acc_vol_ratio_index += 1
    chisquares.append([])
    fit_res.append([])
chisquares = chisquares[:-1]  # delete last empty list in order not to confuse.
fit_res = fit_res[:-1]
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
try:
    files = extract_shifts(runs)
    print('files are: % s' % files)
    plot_iso_shift_stables(runs, files)
except Exception as e:
    print('plotting did not work, error is: %s' % e)

print('------------------- Done -----------------')
winsound.Beep(2500, 500)

# print('\a')


''' Fit on certain Files '''
# searchterm = 'Run167'
# certain_file = [file for file in ni60_files if searchterm in file][0]
# fit = InteractiveFit.InteractiveFit(certain_file, db, runs[0], block=True, x_as_voltage=True)
# fit.fit()
