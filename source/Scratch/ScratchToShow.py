'''
Created on 31.03.2014

@author: gorges
'''
import os
import sqlite3
import math

import MPLPlotter as plot
import numpy as np

from DBIsotope import DBIsotope
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from datetime import datetime
import BatchFit
import Analyzer
import Tools
import Gui
import Physics

path = 'V:/User/Christian/databases'
db = os.path.join(path, 'Ni_isotopes.sqlite')

print(Physics.freqFromWavenumber(22094.549))
print(Physics.freqFromWavelength(452.60032))
Tools.centerPlot(db, ['56_Ni', '65_Ni','70_Ni'])


Tools.isoPlot(db, '63_Ni') # I = 0.5
Tools.isoPlot(db, '57_Ni') # I = 1.5
Tools.isoPlot(db, '65_Ni') # I = 2.5
Tools.isoPlot(db, '69_Ni') # I = 4.5

path = 'V:/Projekte/COLLAPS/ROC/ROC_October'
db = os.path.join(path, 'CaD2.sqlite')
Tools.isoPlot(db, '43_Ca') # I = 0.5
Tools.isoPlot(db, '51_Ca') # I = 0.5



# Tools.centerPlot(db, ['61_Zn','62_Zn','63_Zn','64_Zn','65_Zn','66_Zn','67_Zn','68_Zn','69_Zn', '70_Zn','71_Zn','72_Zn','73_Zn','74_Zn','75_Zn','76_Zn','77_Zn','78_Zn','79_Zn','80_Zn', '81_Zn'])
