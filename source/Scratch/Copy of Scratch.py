'''
Created on 31.03.2014

@author: hammen
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


# path = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/'
# files = ["Ca_000.tld","Ca_001.tld","Ca_002.tld","Ca_004.tld","Ca_005.tld","Ca_006.tld","Ca_007.tld","Ca_010.tld","Ca_011.tld","Ca_012.tld","Ca_013.tld"]#,"Ca_015.tld","Ca_020.tld","Ca_021.tld"]
# BatchFit.batchFit(files, (1,-1), path, 'AnaDB.sqlite', 'calciumD1.sqlite', 'Run1')
# 
# 
# con = sqlite3.connect(os.path.join(path, 'AnaDB.sqlite'))
# cur = con.cursor()
# 
# cur.execute('''SELECT pars FROM Results WHERE Iso = ? AND Run = "Run1"''', ('40_Ca',))
# print('40_Ca')
# data = cur.fetchall()
# center40 = [eval(i[0])['center'][0] for i in data]
# meanC40 = math.fsum(center40)/len(center40)
# print('center:',center40, meanC40)
# # sigma = [eval(i[0])['sigma'][0] for i in data]
# # print('sigma:',sigma, math.fsum(sigma)/len(sigma))
# # gamma = [eval(i[0])['gamma'][0] for i in data]
# #print('gamma:', gamma, math.fsum(gamma)/len(gamma))
# cur.execute('''SELECT pars FROM Results WHERE Iso = ? AND Run = "Run1"''', ('42_Ca',))
# print('42_Ca')
# data = cur.fetchall()
# center42 = [eval(i[0])['center'][0] for i in data]
# meanC42 = math.fsum(center42)/len(center42)
# print('center:',center42, meanC42)
# # sigma = [eval(i[0])['sigma'][0] for i in data]
# # print('sigma:',sigma)
# # gamma = [eval(i[0])['gamma'][0] for i in data]
# # print('gamma:', gamma)
# cur.execute('''SELECT pars FROM Results WHERE Iso = ? AND Run = "Run1"''', ('44_Ca',))
# print('44_Ca')
# data = cur.fetchall()
# center44 = [eval(i[0])['center'][0] for i in data]
# meanC44 = math.fsum(center44)/len(center44)
# print('center:',center44, meanC44)
# # sigma = [eval(i[0])['sigma'][0] for i in data]
# # print('sigma:',sigma)
# # gamma = [eval(i[0])['gamma'][0] for i in data]
# # print('gamma:', gamma)
# cur.execute('''SELECT pars FROM Results WHERE Iso = ? AND Run = "Run1"''', ('48_Ca',))
# print('48_Ca')
# data = cur.fetchall()
# center48 = [eval(i[0])['center'][0] for i in data]
# meanC48 = math.fsum(center48)/len(center48)
# print('center:',center48, meanC48)
# # sigma = [eval(i[0])['sigma'][0] for i in data]
# # print('sigma:',sigma)
# # gamma = [eval(i[0])['gamma'][0] for i in data]
# # print('gamma:', gamma)
# 
# con2 = sqlite3.connect(os.path.join(path, 'calciumD1.sqlite'))
# cur2 = con2.cursor()
# cur2.execute('''SELECT center FROM Isotopes''')
# theo = cur2.fetchall()
# print(theo)
# print('Isotopieverschiebung\nAbweichungen zur Theorie: C40-42:', meanC40-meanC42 + theo[1][0], 'MHz\n')
# print('Abweichungen zur Theorie: C40-44:', meanC40-meanC44 + theo[2][0], 'MHz\n')
# print('Abweichungen zur Theorie: C40-48:', meanC40-meanC48 + theo[3][0], 'MHz\n'
# a = [5]
# a += [7]
# a = a+ 3
# print(a)
print(Analyzer.combineShift('44_Ca', 'Run0', 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/SuperAnaDB.sqlite'))