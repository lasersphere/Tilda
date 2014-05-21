'''
Created on 31.03.2014

@author: hammen
'''

from Measurement.SimpleImporter import SimpleImporter
import MPLPlotter as plot

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from datetime import datetime
import BatchFit
path = 'V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/'
files = ["Ca_000.tld","Ca_001.tld","Ca_002.tld","Ca_003.tld","Ca_004.tld","Ca_005.tld","Ca_006.tld","Ca_007.tld","Ca_010.tld","Ca_011.tld","Ca_012.tld","Ca_013.tld","Ca_015.tld","Ca_020.tld","Ca_021.tld"]
BatchFit.batchFit(files, (1,-1), path, 'AnaDB.sqlite', 'calciumD1.sqlite', 'Run1')
#Test