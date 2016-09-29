'''
Created on 28.09.2016

@author: gorges
'''
import os, sqlite3, math
from datetime import datetime
import numpy as np

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

db = 'V:/Projekte/COLLAPS/Sn/Measurement_and_Analysis_Christian/TILDA/Sn_TILDA.sqlite'

'''preparing a list of isotopes'''
# isoL = ['109_Sn','112_Sn']
# for i in range(114,121, 1):
#     isoL.append(str(str(i)+'_Sn'))
# isoL.append('122_Sn')
# isoL = ['109_Sn']
# for i in range(112,135, 1):
#     isoL.append(str(str(i)+'_Sn'))
freq = 662305065
# isoL = ['112_Sn','114_Sn','115_Sn','116_Sn','118_Sn','119_Sn','120_Sn','122_Sn','125_Sn','126_Sn','128_Sn','131_Sn','132_Sn','133_Sn','134_Sn']

'''Crawling'''
Tools.crawl(db)
con = sqlite3.connect(db)
cur = con.cursor()
cur.execute('''UPDATE Files SET laserFreq=662929863.205568,
  voltDivRatio="{'accVolt':9997.1, 'offset':1000.85}",
  lineMult=0.050425, lineOffset=0.00015''')
con.commit()
con.close()

'''Fitting the spectra with Voigt-Fits!'''
# BatchFit.batchFit(Tools.fileList(db,'124_Sn'), db,'Run0')
