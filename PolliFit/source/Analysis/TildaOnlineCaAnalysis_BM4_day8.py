"""
Created on 

@author: simkaufm

Module Description:  on the 03.03.2016, the Kepco settlement time meas. was repeated and bunching was performed


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
#
db = 'R:\Projekte\TRIGA\Measurements and Analysis_Simon\Bunchermessungen2016' + \
     '\CalciumOnline_160322_analysis\CalciumOnline_160322_analysis_8.sqlite'
# db = 'E:\\temp_analysis\CalciumOnline_160322_analysis\CalciumOnline_160322_analysis_8.sqlite'

""" folder """
work_dir = os.path.split(db)[0]
combine_plots_dir = os.path.join(work_dir, 'combined_plots_8')
if not os.path.exists(combine_plots_dir):
    os.makedirs(combine_plots_dir)

pick_file = os.path.join(combine_plots_dir, 'plot_data.dat')
isos = ['40_Ca']
pars = ['center', 'sigma', 'Int0']
runs = ['sc0', 'sc1', 'sc0+sc1']

'''crawl'''
# Tools.crawl(db)
# raise Exception
# trs_test_file = os.path.join(work_dir, 'sums\\40_Ca_trs_002.xml')
# # print(trs_test_file)
# f = XMLImporter(trs_test_file)
# print(f.x)
# f.preProc(db)
# print(f.cts)
# print(f.dwell)

''' fitting all files for the kepco settle time related stuff: '''

#
# files = ['40_Ca_cs_0%s.xml' % i for i in range(11, 22)]
cs_single = ['40_Ca_cs_011.xml', '40_Ca_cs_012.xml', '40_Ca_cs_013.xml', '40_Ca_cs_014.xml',
             '40_Ca_cs_015.xml', '40_Ca_cs_016.xml', '40_Ca_cs_017.xml', '40_Ca_cs_018.xml',
             '40_Ca_cs_019.xml', '40_Ca_cs_020.xml', '40_Ca_cs_021.xml']

# trs_cw_short = ['40_Ca_trs_0%s.xml' % i if i >= 10 else '40_Ca_trs_00%s.xml' % i for i in range(8, 19)]
trs_cw_short = ['40_Ca_trs_008.xml', '40_Ca_trs_009.xml', '40_Ca_trs_010.xml', '40_Ca_trs_011.xml',
                '40_Ca_trs_012.xml', '40_Ca_trs_013.xml', '40_Ca_trs_014.xml', '40_Ca_trs_015.xml',
                '40_Ca_trs_016.xml', '40_Ca_trs_017.xml', '40_Ca_trs_018.xml']

# trs_bunched_short = ['40_Ca_trs_0%s.xml' % i if i >= 10 else '40_Ca_trs_00%s.xml' % i for i in range(21, 32)]
trs_bunched_short = ['40_Ca_trs_021.xml', '40_Ca_trs_022.xml', '40_Ca_trs_023.xml', '40_Ca_trs_024.xml',
                     '40_Ca_trs_025.xml', '40_Ca_trs_026.xml', '40_Ca_trs_027.xml', '40_Ca_trs_028.xml',
                     '40_Ca_trs_029.xml', '40_Ca_trs_030.xml', '40_Ca_trs_031.xml']

# trs_bunched_short_2 = ['40_Ca_trs_0%s.xml' % i if i >= 10 else '40_Ca_trs_00%s.xml' % i for i in range(38, 48)]
trs_bunched_short_2 = ['40_Ca_trs_038.xml', '40_Ca_trs_039.xml', '40_Ca_trs_040.xml', '40_Ca_trs_041.xml',
                       '40_Ca_trs_042.xml', '40_Ca_trs_043.xml', '40_Ca_trs_044.xml', '40_Ca_trs_045.xml',
                       '40_Ca_trs_046.xml', '40_Ca_trs_047.xml']

# trs_bunch_trig = ['40_Ca_trs_0%s.xml' % i if i >= 10 else '40_Ca_trs_00%s.xml' % i for i in range(48, 60)]
trs_bunch_trig = ['40_Ca_trs_048.xml', '40_Ca_trs_049.xml', '40_Ca_trs_050.xml', '40_Ca_trs_051.xml',
                  '40_Ca_trs_052.xml', '40_Ca_trs_053.xml', '40_Ca_trs_054.xml', '40_Ca_trs_055.xml',
                  '40_Ca_trs_056.xml', '40_Ca_trs_057.xml', '40_Ca_trs_058.xml', '40_Ca_trs_059.xml']
files = trs_bunch_trig
print(files)

# files = ['cs_40_Ca_broad_003.xml',
#          'trs_40_Ca_001.xml', 'trs_40_Ca_002.xml', 'trs_40_Ca_003.xml'
#          ]
# files = files[-2:]  # -300 center 4000 IntScale
# fit = InteractiveFit(files[-1], db, 'sc1', block=True)
# fit.fit()
# #
# raise Exception
''' batch fitting '''

for run in runs:
    BatchFit.batchFit(files, db, run)

''' Average '''

pl = dict.fromkeys(isos, dict.fromkeys(runs, dict.fromkeys(pars, {})))
for iso in isos:
    pl[iso] = dict.fromkeys(runs, {})
    for run in runs:
        # pl[iso][run] = {'center': {}, 'sigma': {}, 'Int0': {}}
        pl[iso][run] = {}
        avgc, statErrc, systErrc, plotdatac = Analyzer.combineRes(iso, 'center', run, db, show_plot=False)
        avgs, statErrs, systErrs, plotdatas = Analyzer.combineRes(iso, 'sigma', run, db, show_plot=False)
        avgi, statErri, systErri, plotdatai = Analyzer.combineRes(iso, 'Int0', run, db, show_plot=False)
        pl[iso][run]['center'] = deepcopy(plotdatac)
        pl[iso][run]['sigma'] = deepcopy(plotdatas)
        pl[iso][run]['Int0'] = deepcopy(plotdatai)

pickle.dump(pl, open(pick_file, "wb"))  # saving the plot data in order nto to connect to the db everytime

# print(isos[0], runs[1], pl[isos[0]][runs[0]])
# print(isos[1], runs[1], pl[isos[1]][runs[0]])

""" plotting of combined tracks. """
pl = pickle.load(open(pick_file, "rb"))
for par in pars:  # , 'sigma', 'Int0']:
    for iso in isos:
        plot.clear()
        print('plotting ', iso)
        s0t0_l = list(pl[iso][runs[0]][par])[:-1]
        s0t0_l.append(('k.', 'k'))
        s0t1_l = list(pl[iso][runs[1]][par])[:-1]
        s0t1_l.append(('bo', 'b'))
        s0t2_l = list(pl[iso][runs[2]][par])[:-1]
        s0t2_l.append(('r+', 'r'))
        plot.plotAverage(*s0t0_l)
        plot.plotAverage(*s0t1_l)
        plot.plotAverage(*s0t2_l)
        ax = plot.get_current_axes()
        fig = plot.get_current_figure()
        fig.set_size_inches(10, 8, forward=True)
        # plot.show(True)


        black_patch = mpatches.Patch(color='black', label='sc0')
        blue_patch = mpatches.Patch(color='blue', label='sc1')
        red_patch = mpatches.Patch(color='red', label='sc0+sc1')
        plot.get_current_axes().legend(handles=[black_patch, blue_patch, red_patch],
                                       bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
                                       ncol=3, mode="expand", borderaxespad=0.)
        plot.get_current_axes().text(0.95, 0.01, iso + '_' + par,
                                     verticalalignment='bottom', horizontalalignment='right',
                                     transform=ax.transAxes,
                                     color='green', fontsize=20)
        plot.save(os.path.join(combine_plots_dir,
                               iso + '_' + par + '.png'))
        #
        # plot.show(True)
#

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
