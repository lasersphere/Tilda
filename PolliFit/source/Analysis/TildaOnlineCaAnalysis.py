"""
Created on 

@author: simkaufm

Module Description: On 23.02 & 26.02.2016 several spectra where taken using the online ion source
in the common beamline with Ca-granulate in it. So no capilar flow system here.
The Buncher was used in Cw as in bunching mode.

On 23.02. the settling time after the voltage has ben set by the DAC,
has been varied in order to see effects on the lineshape.
Also timeresolved measurements were performed on the cw- beam in order to see some fluctuations.

On 24.02. aim was to get bunches.
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
from copy import deepcopy, copy
from matplotlib.dates import DateFormatter
import datetime
from Measurement.XMLImporter import XMLImporter

"""
databases:
"""

db = 'R:\Projekte\TRIGA\Measurements and Analysis_Simon\Bunchermessungen2016' + \
     '\CalciumOnline_160223_analysis\CalciumOnline_160223.sqlite'

""" folder """
work_dir = os.path.split(db)[0]
combine_plots_dir = os.path.join(work_dir, 'combined_plots')
if not os.path.exists(combine_plots_dir):
    os.makedirs(combine_plots_dir)

'''crawl'''
# Tools.crawl(db)

# trs_test_file = os.path.join(work_dir, 'sums\\40_Ca_trs_002.xml')
# # print(trs_test_file)
# f = XMLImporter(trs_test_file)
# print(f.x)
# f.preProc(db)
# print(f.cts)
# print(f.dwell)

# ''' cs fits: '''
# # fit = InteractiveFit('cs_40_Ca_broad_000.xml', db, 'sc0', block=False)
# # fit.fit()
# files = Tools.fileList(db, '40_Ca_broad')
# dirty_files = ['cs_40_Ca_broad_001.xml', 'cs_40_Ca_broad_002.xml',
#                '40_Ca_broad_cs_000.xml', '40_Ca_broad_cs_002.xml', '40_Ca_broad_cs_003.xml',
#                '40_Ca_broad_cs_004.xml', '40_Ca_broad_cs_005.xml']
# files = [file for file in files if file not in dirty_files]
# cs_file = []
# for f in files:
#     try:
#         meas = XMLImporter(os.path.join(os.path.split(db)[0], 'sums', f))
#         if meas.seq_type == 'cs':
#             cs_file.append(f)
#     except Exception as e:
#         print(e)
# print(cs_file)
# ''' batch fitting '''
# runs = ['sc0', 'sc1', 'sc0+sc1']
#
# for run in runs:
#     BatchFit.batchFit(cs_file, db, run)

# ''' trs fits: '''
# fit = InteractiveFit('trs_40_Ca_001.xml', db, 'sc1', block=False)
# fit.fit()
files = Tools.fileList(db, '40_Ca')
dirty_files = ['trs_40_Ca_000.xml', 'trs_40_Ca_009.xml', 'trs_40_Ca_010.xml',
               '40_Ca_trs_000.xml', '40_Ca_trs_001.xml', '40_Ca_trs_002.xml',
               '40_Ca_trs_006.xml']
files = [file for file in files if file not in dirty_files]
files = [file for file in files if '_proj' not in file]
trs_file = []
for f in files:
    try:
        meas = XMLImporter(os.path.join(os.path.split(db)[0], 'sums', f))
        if meas.seq_type == 'trs':
            trs_file.append(f)
    except Exception as e:
        print(e)
print(trs_file)
''' batch fitting '''
runs = ['sc0', 'sc1', 'sc0+sc1']

for run in runs:
    BatchFit.batchFit(trs_file, db, run)


# ''' Average '''
# pick_file = os.path.join(combine_plots_dir, 'plot_data.dat')
#
# pl = {}
# for run in runs:
#     pl[run] = {'center':{}, 'sigma':{}, 'Int0':{}}
#     avgc, statErrc, systErrc, plotdatac = Analyzer.combineRes('40_Ca_broad', 'center', run, db, show_plot=True)
#     avgs, statErrs, systErrs, plotdatas = Analyzer.combineRes('40_Ca_broad', 'sigma', run, db, show_plot=False)
#     avgi, statErri, systErri, plotdatai = Analyzer.combineRes('40_Ca_broad', 'Int0', run, db, show_plot=False)
#     pl[run]['center'] = plotdatac
#     pl[run]['sigma'] = plotdatas
#     pl[run]['Int0'] = plotdatai
# pickle.dump(pl, open(pick_file, "wb"))  # saving the plot data in order nto to connect to the db everytime

""" plotting of combined tracks. """
# pl = pickle.load(open(pick_file, "rb"))

# for par in ['center', 'sigma', 'Int0']:
#     for s in range(0, 2):
#         plot.clear()
#         s0t0_l = list(pl['s' + str(s) + 't0'][par])[:-1]
#         s0t0_l.append(('k.', 'k'))
#         s0t1_l = list(pl['s' + str(s) + 't1'][par])[:-1]
#         s0t1_l.append(('bo', 'b'))
#         s0t2_l = list(pl['s' + str(s) + 't2'][par])[:-1]
#         s0t2_l.append(('r+', 'r'))
#         plot.plotAverage(*s0t0_l)
#         plot.plotAverage(*s0t1_l)
#         plot.plotAverage(*s0t2_l)
#         ax = plot.get_current_axes()
#         fig = plot.get_current_figure()
#         fig.set_size_inches(10, 8, forward=True)
#
#         black_patch = mpatches.Patch(color='black', label='-2V to -1.7V')
#         blue_patch = mpatches.Patch(color='blue', label='inverting')
#         red_patch = mpatches.Patch(color='red', label='-1.7V to -2V')
#         plot.get_current_axes().legend(handles=[black_patch, blue_patch, red_patch],
#                                        bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#                                        ncol=3, mode="expand", borderaxespad=0.)
#         plot.get_current_axes().text(0.95, 0.01, 's' + str(s) + '_t0_to_t3_' + par,
#                                      verticalalignment='bottom', horizontalalignment='right',
#                                      transform=ax.transAxes,
#                                      color='green', fontsize=20)
#         plot.save(os.path.join(combine_plots_dir,
#                                's' + str(s) + '_t0_to_t3_' + par + '.png'))
#
#         # plot.show(True)


''' NOW IT IS REFERENCED ON ONE TRACK AND THE OTHERS ARE PLOTTED AS A DIFFERENCE TO IT. '''
# ref = 's1t0'
# dif_to = ['s1t1', 's1t2']
# plot_styles = [['k.', 'k'], ['r+', 'r']]
# for i_p, par in enumerate(['center', 'sigma', 'Int0']):
#     to_plot = []
#     plot.clear()
#     pl_name = 'ref_' + ref + '_par_' + par
#     for i_r, r in enumerate(dif_to):
#         plt_list = list(pl[ref][par])
#         # difference diff = a - b
#         plt_list[1] = np.asarray(pl[ref][par][1]) - np.asarray(pl[r][par][1])
#         # uncertainty of difference, by gaussian errorprop d_diff = sqrt( (d_a) ** 2 + (d_b) ** 2 )
#         plt_list[2] = np.sqrt(np.square(np.asarray(pl[ref][par][2])) + np.square(np.asarray(pl[r][par][2])))
#         avg, err, r_chi_sq = Analyzer.weightedAverage(plt_list[1], plt_list[2])
#         plt_list[3] = avg
#         plt_list[4] = Analyzer.applyChi(err, r_chi_sq)  # don't want to connect to db here
#         plt_list[5] = 0  # don't want to connect to db here
#         plt_list[6] = plot_styles[i_r]
#         to_plot.append(deepcopy(plt_list))
#
#     for p in to_plot:
#         plot.plotAverage(*p)
#     ax = plot.get_current_axes()
#     fig = plot.get_current_figure()
#     fig.set_size_inches(10, 8, forward=True)
#     black_patch = mpatches.Patch(color='black', label=ref + ' - ' + dif_to[0])
#     red_patch = mpatches.Patch(color='red', label=ref + ' - ' + dif_to[1])
#     ax.legend(handles=[black_patch, red_patch],
#               bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
#               ncol=2, mode="expand", borderaxespad=0.)
#     ax.text(0.95, 0.01, pl_name,
#             verticalalignment='bottom', horizontalalignment='right',
#             transform=ax.transAxes,
#             color='green', fontsize=20)
#     plot.save(os.path.join(combine_plots_dir,
#                            pl_name + '.png'))
    # plot.show(True)
