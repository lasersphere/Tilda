'''
Created on 16.05.2014

@author: hammen
'''
import Tools
import BatchFit
import Analyzer

path = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/'
files = ["Ca_000.tld","Ca_001.tld","Ca_002.tld","Ca_003.tld","Ca_004.tld","Ca_005.tld","Ca_006.tld","Ca_007.tld","Ca_010.tld","Ca_011.tld","Ca_012.tld","Ca_013.tld","Ca_015.tld","Ca_020.tld","Ca_021.tld"]

Tools.createDB('../../test/Project/AnaDB.sqlite')

#BatchFit.batchFit(files, ([0], -1), path + 'SuperAnaDB.sqlite')

#Analyzer.combineRes('40_Ca', 'sigma', 'Run0', ([0], -1), path + 'SuperAnaDB.sqlite')