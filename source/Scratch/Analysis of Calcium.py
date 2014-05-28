'''
Created on 31.03.2014

@author: gorges
'''
import os
import sqlite3
import math

import MPLPlotter as plot

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from datetime import datetime
import BatchFit
import Analyzer


path = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/'
db = os.path.join(path, 'CaD1.sqlite')


'''Fitting the files with Voigt-Fits!'''
files = ["Ca_000.tld","Ca_001.tld","Ca_002.tld","Ca_004.tld","Ca_005.tld","Ca_006.tld","Ca_007.tld","Ca_010.tld","Ca_011.tld","Ca_012.tld","Ca_013.tld"]
BatchFit.batchFit(files, db , 'Run0')
#BatchFit.batchFit(files, db , 'Run1')

'''Mean of sigma and gamma for 40_Ca and Run0'''
# con = sqlite3.connect(os.path.join(path, 'CaD1.sqlite'))
# cur = con.cursor()
# cur.execute('''SELECT pars FROM FitRes WHERE Iso = ? AND Run = "Run0"''', ('40_Ca',))
# data = cur.fetchall()
# center40 = [eval(i[0])['center'][0] for i in data]
# meanC40 = math.fsum(center40)/len(center40)
# print('center:',center40, meanC40)
# sigma = [eval(i[0])['sigma'][0] for i in data]
# print('sigma:',sigma, math.fsum(sigma)/len(sigma))
# gamma = [eval(i[0])['gamma'][0] for i in data]
# print('gamma:', gamma, math.fsum(gamma)/len(gamma))

'''Calculate the isotope shift to 40_Ca'''
#print(Analyzer.combineShift('42_Ca', 'Run1', db))
print(Analyzer.combineShift('44_Ca', 'Run0', db))
#print(Analyzer.combineShift('48_Ca', 'Run1', db))