'''
Created on 31.03.2014

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

db = 'V:/User/Christian/databases/Fe.sqlite'

'''performing a King fit analysis'''
litvals = {'57_Fe':[0.124,.028],
            '58_Fe':[0.283,.028],
           '54_Fe':[-0.313,.026]}

king = KingFitter(db, litvals,showing=True)
king.kingFit(alpha=0,findBestAlpha=False)
king.calcChargeRadii()