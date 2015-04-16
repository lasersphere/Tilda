'''
Created on 16.05.2014

@author: hammen
'''
import Tools
import BatchFit
import Analyzer
import MPLPlotter

path = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Simon/BuncherMessungen2014/Ca43/CaD1.sqlite'
#path2 = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/CaD2.sqlite'
#files = ["Ca_000.tld","Ca_001.tld","Ca_002.tld","Ca_003.tld","Ca_004.tld","Ca_005.tld","Ca_006.tld","Ca_007.tld","Ca_010.tld","Ca_011.tld","Ca_012.tld","Ca_013.tld","Ca_015.tld","Ca_020.tld","Ca_021.tld"]

#Tools.createDB(path)

#BatchFit.batchFit(['Ca_004.tld'], path, 'Run1')

#Analyzer.combineRes('40_Ca', 'center', 'Run1', path)

Tools.isoPlot(path, '43_Ca')
#MPLPlotter.plot('V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Simon/BuncherMessungen2014/Ca43/')

#Tools.centerPlot(path, ['40_Ca', '42_Ca', '44_Ca', '48_Ca'])

#Tools.centerPlot(path2, ['40_Ca', '42_Ca', '44_Ca', '48_Ca'])
