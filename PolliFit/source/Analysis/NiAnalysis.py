"""
Created on 

@author: simkaufm

Module Description:  Analysis of the Nickel Data from COLLAPS taken on 28.04.-03.05.2016
"""

import os
import sqlite3

import InteractiveFit
import Physics
import Tools

''' working directory: '''

workdir = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder = os.path.join(workdir, 'Ni_April2016_mcp')

db = os.path.join(workdir, 'Ni_workspace.sqlite')

runs = ['narrow_gate', 'wide_gate']
isotopes = ['%s_Ni' % i for i in range(58, 71)]
''' crawling '''

# Tools.crawl(db, 'Ni_April2016_mcp')

# ''' laser wavelength: '''
wavenum = 28393.0  # cm-1
freq = Physics.freqFromWavenumber(wavenum)
# freq -= 1250
print(freq, Physics.wavenumber(freq), 0.5 * Physics.wavenumber(freq))

con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''UPDATE Files SET laserFreq = ? ''', (freq, ))
con.commit()
con.close()
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
ni60_files = Tools.fileList(db, isotopes[2])
ni60_cont_files = [each for each in ni60_files if 'contin' in each]
ni60_bunch_files = [each for each in ni60_files if 'contin' not in each]
# print(ni60_bunch_files)

# BatchFit.batchFit(ni60_bunch_files, db, runs[0])
# Analyzer.combineRes(isotopes[2], 'center', runs[0], db)


''' Fit on certain Files '''
searchterm = 'Run167'
certain_file = [file for file in ni60_files if searchterm in file][0]
fit = InteractiveFit.InteractiveFit(certain_file, db, runs[0], block=True, x_as_voltage=True)
# fit.fit()