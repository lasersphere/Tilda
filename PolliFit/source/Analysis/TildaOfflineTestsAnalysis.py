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

from InteractiveFit import InteractiveFit
import BatchFit
import Analyzer
import matplotlib.pyplot as plot
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
avg, stat_err, sys_err = Analyzer.combineRes('40Ca', 'center', run, db)

vals, err, date = Analyzer.extract('40Ca', 'center', run, db, [])

date = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in date]
plot.subplots_adjust(bottom=0.2)
plot.xticks(rotation=25)
ax = plot.gca()
xfmt = DateFormatter('%Y-%m-%d %H:%M:%S')
ax.xaxis.set_major_formatter(xfmt)
try:
    plot.errorbar(date, vals, yerr=err, fmt='k.')
    avg_l = np.asarray([[avg, avg], [0, len(vals)]])
    stat_err_l_p = np.asarray([avg+stat_err, avg+stat_err])
    stat_err_l_m = np.asarray([avg-stat_err, avg-stat_err])
    x = (sorted(date)[0], sorted(date)[-1])
    y = (avg, avg)
    #
    print(np.asarray([0, len(vals)]), stat_err_l_p, stat_err_l_m)
    plot.plot(x, y, 'r')
    plot.fill_between(x, stat_err_l_p, stat_err_l_m, alpha=0.5)
    plot.show()

except Exception as e:
    print(e)