'''
Created on 11.07.2016

@author: gorges
'''

import os, sqlite3, math
from datetime import datetime
import numpy as np

import time
import MPLPlotter as plot
import DBIsotope
import SPFitter
import Spectra.FullSpec as FullSpec
import BatchFit
import Analyzer
import Tools
import Physics
from KingFitter import KingFitter

import InteractiveFit as IF

db = 'V:/Projekte/COLLAPS/Bi/Bi.sqlite'

# Tools.isoPlot(db, '208_Bi')

# Tools.crawl(db)
# print(Physics.freqFromWavenumber(10352.30))
# con = sqlite3.connect(db)
# cur = con.cursor()
# cur.execute('''UPDATE Files SET laserFreq=977511301.4076173,
#  voltDivRatio="{'accVolt':1000., 'offset':1.}", lineMult=1., lineOffset=0''')
# con.commit()
# con.close()
BatchFit.batchFit(['208Bi_no_protonTrigger_einScaler.mcp'], db)
