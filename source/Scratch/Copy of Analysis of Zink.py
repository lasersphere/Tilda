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
import Physics

path = 'V:/Projekte/TRIGA/Measurements and Analysis_Christian/Zink'
db = os.path.join(path, 'Zink.sqlite')
Tools.centerPlot(db, ['61_Zn', '70_Zn','81_Zn'])

# Tools.centerPlot(db, ['61_Zn','62_Zn','63_Zn','64_Zn','65_Zn','66_Zn','67_Zn','68_Zn','69_Zn', '70_Zn','71_Zn','72_Zn','73_Zn','74_Zn','75_Zn','76_Zn','77_Zn','78_Zn','79_Zn','80_Zn', '81_Zn'])
