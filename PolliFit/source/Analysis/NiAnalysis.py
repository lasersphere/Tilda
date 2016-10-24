"""
Created on 

@author: simkaufm

Module Description:  Analysis of the Nickel Data from COLLAPS taken on 28.04.-03.05.2016
"""

import os

import Physics
import Tools

''' working directory: '''

workdir = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder = os.path.join(workdir, 'Ni_April2016_mcp')

db = os.path.join(workdir, 'Ni_workspace.sqlite')

runs = ['narrow_gate', 'wide_gate']
isotopes = ['%s_Ni' % i for i in range(58, 71)]
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']

''' crawling '''

# Tools.crawl(db, 'Ni_April2016_mcp')

# ''' laser wavelength: '''
# wavenum = 28393.0  # cm-1
# freq = Physics.freqFromWavenumber(wavenum)
# freq -= 1256.32701
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
# print(transition_freq)


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
# pars = ['center', 'Al', 'Bl', 'Au', 'Bu', 'Int0']
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
                  ['60Ni_no_protonTrigger_Run018.mcp',
                   '60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
                   '60Ni_no_protonTrigger_Run021.mcp']),
                 ([], ['62Ni_no_protonTrigger_Run016.mcp'],
                  ['60Ni_no_protonTrigger_Run018.mcp',
                   '60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
                   '60Ni_no_protonTrigger_Run021.mcp']),
                 ([], ['62Ni_no_protonTrigger_Run017.mcp'],
                  ['60Ni_no_protonTrigger_Run018.mcp',
                   '60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
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

manual_cfg_61 = [([], ['61Ni_no_protonTrigger_Run010.mcp'],
                  ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
                   '60Ni_no_protonTrigger_Run021.mcp']),
               ([], ['61Ni_no_protonTrigger_Run011.mcp'],
                ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
                 '60Ni_no_protonTrigger_Run021.mcp']),
               ([], ['61Ni_no_protonTrigger_Run012.mcp'],
                ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
                 '60Ni_no_protonTrigger_Run021.mcp']),
               ([], ['61Ni_no_protonTrigger_Run013.mcp'],
                ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
                 '60Ni_no_protonTrigger_Run021.mcp']),
               ([], ['61Ni_no_protonTrigger_Run014.mcp'],
                ['60Ni_no_protonTrigger_Run019.mcp', '60Ni_no_protonTrigger_Run020.mcp',
                 '60Ni_no_protonTrigger_Run021.mcp']),
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

# configs = {'58_Ni': manual_cfg_58, '62_Ni': manual_cfg_62, '64_Ni': manual_cfg_64}
# stables = ['64_Ni']
# runs = [runs[0]]
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

''' Fit on certain Files '''
# searchterm = 'Run167'
# certain_file = [file for file in ni60_files if searchterm in file][0]
# fit = InteractiveFit.InteractiveFit(certain_file, db, runs[0], block=True, x_as_voltage=True)
# fit.fit()
