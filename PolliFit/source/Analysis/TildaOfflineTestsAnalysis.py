"""
Created on 

@author: simkaufm

Module Description: On 25th & 26nd Nov. 2015 Tilda was run several times for testing Tilda itself,
but also for testing if the asymetry in the peaks could be caused by the scanning direction.

Therefore
    track0 is: -2 to -1.698804
    track1 is: -2 to -1.698804 (invert direction = TRUE)
    track3 is: -1.698804 to -2

Nevertheless, ion current was really low and also deviating of around 10%,
whis for sure will influence the results.
"""

import Tools
import numpy as np
import os

from InteractiveFit import InteractiveFit
import BatchFit
import Analyzer
import MPLPlotter as plot
from matplotlib.dates import DateFormatter
import datetime


"""
databases:
"""

db = 'R:\Projekte\TRIGA\Measurements and Analysis_Simon' + \
     '\Tilda Offline Tests 15_11\BothDaysCombined\BothDaysCombined.sqlite'

'''crawl'''
# Tools.crawl(db)

# fit = InteractiveFit('cs_sum_40Ca_006.xml', db, 'Run1')
# fit.fit()
run = 'Run0'
files = Tools.fileList(db, '40Ca')
dirty_files = ['cs_sum_40Ca_000.xml', 'cs_sum_40Ca_001.xml', '26th_cs_sum_40Ca_016.xml']
files = [file for file in files if file not in dirty_files]
#BatchFit.batchFit(files, db, run)

Analyzer.combineRes('40Ca', 'center', run, db, show_plot=False)
Analyzer.combineRes('40Ca', 'sigma', run, db, show_plot=False)
Analyzer.combineRes('40Ca', 'Int0', run, db, show_plot=False)
# plot.plotAverage('40Ca', 'center', run, db)