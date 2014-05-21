'''
Created on 16.05.2014

@author: hammen
'''

import numpy
import Analyzer


path = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/'
files = ["Ca_000.tld","Ca_001.tld","Ca_002.tld","Ca_003.tld","Ca_004.tld","Ca_005.tld","Ca_006.tld","Ca_007.tld","Ca_010.tld","Ca_011.tld","Ca_012.tld","Ca_013.tld","Ca_015.tld","Ca_020.tld","Ca_021.tld"]
a = Analyzer.extract('40_Ca', 'Run0', (0, -1), 'center', path + 'AnaDB.sqlite')

print(a)

b = Analyzer.weightedAverage(*a)
print(b)