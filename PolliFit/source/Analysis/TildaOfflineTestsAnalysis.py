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
import matplotlib.patches as mpatches
import pickle
from copy import deepcopy
from matplotlib.dates import DateFormatter
import datetime
import Measurement.XMLImporter as Xml

"""
databases:
"""

db = 'R:\Projekte\TRIGA\Measurements and Analysis_Simon' + \
     '\Tilda Offline Tests 15_11\BothDaysCombined\BothDaysCombined.sqlite'

""" folder """
work_dir = os.path.split(db)[0]
combine_plots_dir = os.path.join(work_dir, 'combined_plots')
if not os.path.exists(combine_plots_dir):
    os.makedirs(combine_plots_dir)

'''crawl'''
# Tools.crawl(db)

# fit = InteractiveFit('cs_sum_40Ca_006.xml', db, 'Run1')
# fit.fit()
files = Tools.fileList(db, '40Ca')
dirty_files = ['cs_sum_40Ca_000.xml', 'cs_sum_40Ca_001.xml', '26th_cs_sum_40Ca_016.xml']
files = [file for file in files if file not in dirty_files]
three_track_files = []
# for f in files:
#     try:
#         meas = Xml.XMLImporter(os.path.join(os.path.split(db)[0], 'sums', f))
#         if meas.nrTracks == 3:
#             three_track_files.append(f)
#     except Exception as e:
#         print(e)

''' batch fitting '''

# for i in range(0, 2):
#     for t in range(0, 3):
#         run = 's' + str(i) + 't' + str(t)
#         BatchFit.batchFit(three_track_files, db, run)


''' Average '''
pick_file = os.path.join(combine_plots_dir, 'plot_data.dat')

# pl = {}
# for i in range(0, 2):
#     for t in range(0, 3):
#         run = 's' + str(i) + 't' + str(t)
#         pl[run] = {'center':{}, 'sigma':{}, 'Int0':{}}
#         avgc, statErrc, systErrc, plotdatac = Analyzer.combineRes('40Ca', 'center', run, db, show_plot=False)
#         avgs, statErrs, systErrs, plotdatas = Analyzer.combineRes('40Ca', 'sigma', run, db, show_plot=False)
#         avgi, statErri, systErri, plotdatai = Analyzer.combineRes('40Ca', 'Int0', run, db, show_plot=False)
#         pl[run]['center'] = plotdatac
#         pl[run]['sigma'] = plotdatas
#         pl[run]['Int0'] = plotdatai
# pickle.dump(pl, open(pick_file, "wb"))

""" plotting of combined tracks. """
pl = pickle.load(open(pick_file, "rb"))

for par in ['center', 'sigma', 'Int0']:
    for s in range(0, 2):

        plot.clear()
        s0t0_l = list(pl['s' + str(s) + 't0'][par])[:-1]
        s0t0_l.append(('k.', 'k'))
        s0t1_l = list(pl['s' + str(s) + 't1'][par])[:-1]
        s0t1_l.append(('bo', 'b'))
        s0t2_l = list(pl['s' + str(s) + 't2'][par])[:-1]
        s0t2_l.append(('r+', 'r'))
        plot.plotAverage(*s0t0_l)
        plot.plotAverage(*s0t1_l)
        plot.plotAverage(*s0t2_l)
        ax = plot.get_current_axes()
        fig = plot.get_current_figure()
        fig.set_size_inches(10, 8, forward=True)

        black_patch = mpatches.Patch(color='black', label='-2V to -1.7V')
        blue_patch = mpatches.Patch(color='blue', label='inverting')
        red_patch = mpatches.Patch(color='red', label='-1.7V to -2V')
        plot.get_current_axes().legend(handles=[black_patch, blue_patch, red_patch],
                                       bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                       ncol=3, mode="expand", borderaxespad=0.)
        plot.get_current_axes().text(0.95, 0.01, 's' + str(s) + '_t0_to_t3_' + par,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax.transAxes,
        color='green', fontsize=20)
        plot.save(os.path.join(combine_plots_dir,
                               's' + str(s) + '_t0_to_t3_' + par + '.png'))

        # plot.show(True)
