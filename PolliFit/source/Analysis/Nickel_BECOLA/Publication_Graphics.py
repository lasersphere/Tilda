"""
Created on 2021-02-12

@author: fsommer

Module Description:  Plotting of all Graphics for the Thesis
"""

import os
import ast
from glob import glob
from datetime import datetime, timedelta
import logging
import re
import operator  # used for sorting legends

from math import ceil, floor, log10
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.font_manager as font_manager
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable
from itertools import *
from operator import itemgetter

from scipy.optimize import curve_fit
from scipy import odr
from scipy import interpolate
import scipy.constants as sc

from Measurement.XMLImporter import XMLImporter
import TildaTools as TiTs

class PlotThesisGraphics:
    def __init__(self):
        """ Folder Locations """
        # working directory:
        user_home_folder = os.path.expanduser("~")  # get user folder to access ownCloud
        owncould_path = 'ownCloud\\User\\Felix\\IKP411_Dokumente\\BeiträgePosterVorträge\\PhDThesis\\Grafiken\\'
        self.fig_dir = os.path.join(user_home_folder, owncould_path)
        self.ffe = '.png'  # file format ending

        """ Colors """
        # Define global color names TODO: Fill and propagate
        self.black = (0, 0, 0)
        self.grey = (181/255, 181/255, 181/255) # 40% grey
        self.dark_blue = (36/255, 53/255, 114/255)  # TUD1d
        self.blue = (0/255, 131/255, 204/255)  # TUD2b
        self.green = (153/255, 192/255, 0/255)  # TUD4b
        self.dark_green = (0/255, 113/255, 94/255)  # TUD3d
        self.yellow = (253/255, 202/255, 0/255)  # TUD6b
        self.orange = (245/255, 163/255, 0/255)  # TUD7b
        self.dark_orange = (190/255, 111/255, 0/255)  # TUD7d
        self.red = (230/255, 0/255, 26/255)  # TUD9b
        self.dark_red = (156/255, 28/255, 38/255)  # TUD9d
        self.purple = (114/255, 16/255, 133/255)  # TUD11b
        self.dark_purple = (76 / 255, 34 / 255, 106 / 255)  # TUD11d

        self.colorlist = [self.black, self.blue, self.green, self.orange, self.red, self.purple,
                          self.dark_blue, self.dark_green, self.yellow, self.dark_purple]
        self.markerlist = ['s', '<', 'D', 'v', 'p', '>', '*', '^']

        self.pmt_colors = {'scaler_0': self.blue, 'scaler_1': self.dark_orange, 'scaler_2': self.purple, 'scaler_c012': self.red}
        self.isotope_colors = {'60Ni': self.blue, '58Ni': self.black, '56Ni': self.green, '55Ni': self.orange, '54Ni': self.orange, '62Ni': self.purple, '64Ni': self.purple}

        # Define Color map
        c_dict = {  # TUD 2b-10b
            'red': ((0., 0 / 255, 0 / 255), (1 / 8, 0 / 255, 0 / 255), (2 / 8, 153 / 255, 153 / 255),
                    (3 / 8, 201 / 255, 201 / 255), (4 / 8, 253 / 255, 253 / 255), (5 / 8, 245 / 255, 245 / 255),
                    (6 / 8, 236 / 255, 236 / 255), (7 / 8, 230 / 255, 230 / 255), (1, 166 / 255, 166 / 255)),
            'green': ((0., 131 / 255, 131 / 255), (1 / 8, 157 / 255, 157 / 255), (2 / 8, 192 / 255, 192 / 255),
                      (3 / 8, 212 / 255, 212 / 255), (4 / 8, 202 / 255, 202 / 255), (5 / 8, 163 / 255, 163 / 255),
                      (6 / 8, 101 / 255, 101 / 255), (7 / 8, 0 / 255, 0 / 255), (1, 0 / 255, 0 / 255)),
            'blue': ((0., 204 / 255, 204 / 255), (1 / 8, 129 / 255, 129 / 255), (2 / 8, 0 / 255, 0 / 255),
                     (3 / 8, 0 / 255, 0 / 255), (4 / 8, 0 / 255, 0 / 255), (5 / 8, 0 / 255, 0 / 255),
                     (6 / 8, 0 / 255, 0 / 255), (7 / 8, 26 / 255, 26 / 255), (1, 132 / 255, 132 / 255))
        }
        self.custom_cmap = mpl.colors.LinearSegmentedColormap('my_colormap', c_dict, 1024)

        # Define a color gradient with negative values blue, 0 black and positive values green
        self.colorgradient = {-5: (36/255, 53/255, 114/255),  # TUD1d
                              -4: (0/255, 78/255, 115/255),  # TUD2d
                              -3: (0/255, 104/255, 157/255),  # TUD2c
                              -2: (0/255, 131/255, 204/255),  # TUD2b
                              -1: (0/255, 156/255, 218/255),  # TUD2a
                              0: (0, 0, 0),
                              1: (175/255, 204/255, 80/255),  # TUD4a
                              2: (153/255, 192/255, 0/255),  # TUD4b
                              3: (127/255, 171/255, 22/255),  # TUD4c
                              4: (106/255, 139/255, 55/255),  # TUD4d
                              5: (0/255, 113/255, 94/255)  # TUD3d
                              }

        """ Global Style Settings """
        # https://matplotlib.org/2.0.2/users/customizing.html
        font_plot = {'family': 'sans-serif',
                     'sans-serif': 'Verdana',
                     'weight': 'normal',
                     'stretch': 'ultra-condensed',
                     'size': 8}  # wie in Arbeit

        mpl.rc('font', **font_plot)
        # mpl.rcParams['mathtext.fontset'] = 'custom'
        # mpl.rcParams['mathtext.rm'] = 'STIXGeneral'
        mpl.rc('lines', linewidth=1)

        self.text_style = {'family': 'sans-serif',
                           'size': 8.,
                           }

        self.point_style = {'linestyle': '',
                           'marker': 'o',
                           'markersize': 2,
                           'color': self.black}

        self.data_style = self.ch_dict(self.point_style, {'capsize': 2})

        self.fit_style = {'linestyle': '-',
                          'linewidth': 1,
                          'marker': '',
                          'color': self.red}

        self.bar_style = {'align': 'center',
                          'hatch': '/',
                          'edgecolor': self.blue,
                          'fill': False}

        """ Global Size Settings """
        # Thesis document proportions
        self.dpi = 300  # resolution of Graphics

        scl = 25.4  # Scale Factor. Adapt to fit purpose  17 25.4
        w_a4_in = 210/scl  # A4 format pagewidth in inch
        w_rat = 0.45  # Textwidth used within A4 ~70%
        self.w_in = w_a4_in * w_rat

        h_a4_in = 297/scl  # A4 format pageheight in inch
        h_rat = 0.95  # Textheight used within A4  ~38 lines text
        self.h_in = h_a4_in*h_rat


    def lineshape_compare(self):
        """
        Three plots comparing Voigt, SatVoigt and ExpVoigt
        :return:
        """
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\Lineshapes\\')

        # Create plots for all three lineshapes
        widths = [1, 1, 1]
        heights = [1, 0.4]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        f, axes = plt.subplots(nrows=2, ncols=3, sharex=True , sharey='row', gridspec_kw=gs_kw)

        # define output size of figure
        width, height = 1, 0.3
        # f.set_dpi(300.)
        f.set_size_inches((self.w_in * width, self.h_in * height))

        # First Col: Voigt
        rChiVoigt = 2.18
        x_0, cts_0, res_0, cts_err_0 = np.loadtxt(
            glob(os.path.join(folder, 'Voigt\\BECOLA_6501_data__*'))[0],
            delimiter=', ', skiprows=1, unpack=True)
        x_fit_0, fit_0 = np.loadtxt(
            glob(os.path.join(folder, 'Voigt\\BECOLA_6501_fit_fullShape__*'))[0],
            delimiter=', ', skiprows=1, unpack=True)
        axes[0, 0].errorbar(x_0, cts_0, cts_err_0, **self.data_style)  # data
        axes[0, 0].annotate(r'$\chi^2_\mathrm{{red}}={:.2f}$'.format(rChiVoigt), (-950, 7000))
        axes[0, 0].plot(x_fit_0, fit_0, **self.fit_style)  # fit
        axes[1, 0].errorbar(x_0, res_0, cts_err_0, **self.data_style)  # residuals
        axes[0, 0].set_title('Voigt')

        # Second Col: SatVoigt
        rChiSatVoigt = 0.92
        x_1, cts_1, res_1, cts_err_1 = np.loadtxt(glob(os.path.join(folder, 'SatVoigt\\BECOLA_6501_data__*'))[0],
                                          delimiter=', ', skiprows=1, unpack=True)
        x_fit_1, fit_1 = np.loadtxt(glob(os.path.join(folder, 'SatVoigt\\BECOLA_6501_fit_fullShape__*'))[0],
                                delimiter=', ', skiprows=1, unpack=True)
        x_mp_1, mp_1 = np.loadtxt(glob(os.path.join(folder, 'SatVoigt\\BECOLA_6501_fit_mainPeak__*'))[0],
                                delimiter=', ', skiprows=1, unpack=True)
        x_sp_1, sp_1 = np.loadtxt(glob(os.path.join(folder, 'SatVoigt\\BECOLA_6501_fit_sidePeak0__*'))[0],
                                delimiter=', ', skiprows=1, unpack=True)
        axes[0, 1].errorbar(x_1, cts_1, cts_err_1, label='data', **self.data_style)  # data
        axes[0, 1].annotate(r'$\chi^2_\mathrm{{red}}={:.2f}$'.format(rChiSatVoigt), (-950, 7000))
        da, la = axes[0, 1].get_legend_handles_labels()
        fi = axes[0, 1].plot(x_fit_1, fit_1, label='fit', **self.fit_style)  # fit
        mp = axes[0, 1].plot(x_mp_1, mp_1, label='main peak', **self.ch_dict(self.fit_style, {'linestyle': '--'}))  # mainPeak
        sp = axes[0, 1].plot(x_sp_1, sp_1, label='satellite peak', **self.ch_dict(self.fit_style, {'linestyle': ':'}))  # sidePeak0
        axes[0, 1].legend(handles=[da[0], fi[0], mp[0], sp[0]], bbox_to_anchor=(0.5, 1.3), loc='center', ncol=4)
        axes[1, 1].errorbar(x_1, res_1, cts_err_1, **self.data_style)  # residuals
        axes[0, 1].set_title('SatVoigt')

        # Third Col: ExpVoigt
        rChiExpVoigt = 1.25
        x_2, cts_2, res_2, cts_err_2 = np.loadtxt(
            glob(os.path.join(folder, 'ExpVoigt\\BECOLA_6501_data__*'))[0],
            delimiter=', ', skiprows=1, unpack=True)
        x_fit_2, fit_2 = np.loadtxt(
            glob(os.path.join(folder, 'ExpVoigt\\BECOLA_6501_fit_fullShape__*'))[0],
            delimiter=', ', skiprows=1, unpack=True)
        axes[0, 2].errorbar(x_2, cts_2, cts_err_2, **self.data_style)  # data
        axes[0, 2].annotate(r'$\chi^2_\mathrm{{red}}={:.2f}$'.format(rChiExpVoigt), (-950, 7000))
        axes[0, 2].plot(x_fit_2, fit_2, **self.fit_style)  # fit
        axes[1, 2].errorbar(x_2, res_2, cts_err_2, **self.data_style)  # residuals
        axes[0, 2].set_title('ExpVoigt')

        # set axes:
        axes[0, 0].set_ylabel('cts/ arb.u.')
        axes[0, 0].set_yticks([1000, 3000, 5000, 7000])
        axes[1, 0].set_ylabel('res/ arb.u.')
        axes[1, 0].set_yticks([-200, 0, 200])
        for row in range(3):
            axes[1, row].set_xlabel('relative frequency/ MHz')
            axes[1, row].set_xticks([-900, -700, -500, -300, -100])

        f.tight_layout()
        plt.savefig(folder+'Lineshapes' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def tof_determination(self):
        """
        Simple plot comparing different methods to extract the midTof parameter
        :return:
        """
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\midTof\\')

        f, ax = plt.subplots()

        # define output size of figure
        width, height = 1, 0.4
        f.set_size_inches((self.w_in * width, self.h_in * height))

        # get the data TODO: add uncertainties!
        iso, Int0, TProj = np.loadtxt(os.path.join(folder, 'data.txt'), skiprows=1, unpack=True)

        ax.plot(iso, Int0, label='SNR optimization', **self.ch_dict(self.point_style, {'markersize': 7, 'color': 'green'}))
        ax.plot(iso, TProj, label='t-projection fit', **self.ch_dict(self.point_style, {'markersize': 7, 'color': 'blue'}))
        # do a linear regression
        m, b = self.lin_regression(iso, Int0, (1, 0))
        ax.plot(iso, self._line(iso, m, b), label='mid-tof={:.1f}\u00B7A+{:.1f}bins'.format(m, b), **self.fit_style)
        # annotate the results into the graph
        ax.legend(loc='upper left')

        # set axes:
        ax.set_ylabel('mid-tof/ bins')

        ax.set_xlabel('isotope mass number A')
        ax.set_xlim(53, 65)
        ax.set_xticks(iso)

        f.tight_layout()
        plt.savefig(folder + 'TOF' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def timeres_plot(self):
        """
        Plot time resolved data from one run as an example
        :return:
        """
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\TRS_data\\')
        file = 'BECOLA_6501.xml'

        # Import Measurement from file using Importer
        midtof = 5.3615
        gatewidth = 0.07648*2*2  # sigma*2*numberOfSigmas

        xml = XMLImporter(os.path.join(folder, file), x_as_volt=True,
                        softw_gates=[[-100, 100, midtof-gatewidth/2, midtof+gatewidth/2],
                                     [-100, 100, midtof-gatewidth/2, midtof+gatewidth/2],
                                     [-100, 100, midtof-gatewidth/2, midtof+gatewidth/2]])
        xml.preProc(os.path.join(folder, 'Ni_Becola.sqlite'))
        trs = xml.time_res[0]  # array of dimensions tracks, pmts, steps, bins
        t_proj = xml.t_proj[0]
        t_proj_err = np.sqrt(t_proj)
        v_proj = xml.cts[0]
        v_proj_err = xml.err[0]
        x_axis = -np.array(xml.x[0]) + xml.accVolt
        step_width = xml.stepSize[0] * xml.lineMult
        # Get sizes of arrays
        scal_data_size, x_data_size, t_data_size = np.shape(trs)
        # create mesh for time-resolved plot
        X, Y = np.meshgrid(np.arange(x_data_size), np.arange(t_data_size))
        # cts data is stored in data_array. Either per scaler [ScalerNo, Y, X]
        Z = trs[0][X, Y]
        # Z = data_arr.sum(axis=0)[X, Y]  # or summed for all scalers

        # Create plots for trs and projections
        f = plt.figure()
        widths = [0.1, 1, 0.2]
        heights = [0.4, 1]
        spec = mpl.gridspec.GridSpec(nrows=2, ncols=3,
                                     width_ratios=widths, height_ratios=heights,
                                     wspace=0.07, hspace=0.07)

        ax_col = f.add_subplot(spec[1, 0])
        ax_trs = f.add_subplot(spec[1, 1])
        ax_tpr = f.add_subplot(spec[1, 2], sharey=ax_trs)
        ax_vpr = f.add_subplot(spec[0, 1], sharex=ax_trs)

        # define output size of figure
        width, height = 1, 0.4
        # f.set_dpi(300.)
        f.set_size_inches((self.w_in * width, self.h_in * height))

        # create timeresolved plot
        im =ax_trs.pcolormesh(x_axis, np.arange(t_data_size), Z, cmap=self.custom_cmap)
        # im = ax_trs.imshow(Z, cmap=custom_cmap, interpolation='none', aspect='auto')  # cmap=pyl.cm.RdBu
        # work on x axis
        ax_trs.xaxis.set_ticks_position('top')
        ax_trs.axes.tick_params(axis='x', direction='out',
                                bottom=True, top=False, labelbottom=True, labeltop=False)
        ax_trs.set_xlabel('DAC scan voltage/ V')
        ax_trs.set_xlim((x_axis[0], x_axis[-1]))
        # work on y axis
        ax_trs.set_ylim((100*(midtof-2*gatewidth), 100*(midtof+2*gatewidth)))
        ax_trs.axes.tick_params(axis='y', direction='out',
                                left=False, right=False, labelleft=False, labelright=False)
        f.colorbar(im, cax=ax_col)  # create plot legend
        ax_col.axes.tick_params(axis='y', direction='out',
                                left=True, right=False, labelleft=True, labelright=False)
        ax_col.yaxis.set_label_position('left')
        ax_col.set_ylabel('cts/arb.u.')

        # create Voltage projection
        # ax_vpr.errorbar(x_axis, v_proj[0], v_proj_err[0], **self.data_style)  # x_step_projection_sc0
        ax_vpr.bar(x_axis, v_proj[0], width=step_width, color=self.grey, edgecolor=self.grey)
        ax_vpr.axes.tick_params(axis='x', direction='in',
                                top=True, bottom=True,
                                labeltop=False, labelbottom=False)
        ax_vpr.axes.tick_params(axis='y', direction='out',
                                left=True, right=False, labelleft=True)
        ax_vpr.set_yticks([1000, 2000, 3000])
        ax_vpr.set_yticklabels(['1k','2k','3k'])
        ax_vpr.set_ylabel('cts/arb.u.')

        # create time projection
        # ax_tpr.errorbar(t_proj[0], np.arange(t_data_size), xerr=t_proj_err[0], **self.data_style)  # y_time_projection_sc0
        ax_tpr.barh(np.arange(t_data_size), t_proj[0], height=1, color=self.grey, edgecolor=self.grey)
        ax_tpr.axes.tick_params(axis='x', direction='out',
                                top=False, bottom=True, labeltop=False, labelbottom=True)
        plt.setp(ax_tpr.get_xticklabels(), rotation=-90)
        ax_tpr.axes.tick_params(axis='y', direction='in',
                                left=True, right=True, labelleft=False, labelright=True)
        ax_tpr.set_xticks([1000, 2000, 3000])
        ax_tpr.set_xticklabels(['1k', '2k', '3k'])
        ax_tpr.set_xlabel('cts/arb.u.')
        ax_tpr.yaxis.set_label_position('right')
        ax_tpr.set_ylabel('time/bins')

        # add horizontal lines for timegates
        ax_trs.axhline(100 * (midtof - gatewidth/2), **self.fit_style)
        ax_tpr.axhline(100 * (midtof - gatewidth/2), **self.fit_style)
        ax_trs.axhline(100 * (midtof + gatewidth/2), **self.fit_style)
        ax_tpr.axhline(100 * (midtof + gatewidth/2), **self.fit_style)

        # add the TILDA logo top right corner
        logo = mpl.image.imread(os.path.join(folder, 'Tilda256.png'))
        ax_log = f.add_subplot(spec[0, 2])
        ax_log.axis('off')
        ax_log.imshow(logo)

        # f.tight_layout()
        plt.savefig(folder + 'TRS' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def isotope_shifts(self):
        """
        Plot overview of Nickel 56 isotope shifts for different files and PMTs.
        Based on plot_shifts() from Ni_StandartizedAnalysis
        :return:
        """
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\IsoShifts\\')

        # Define which Isotope(s) to plot
        isolist = ['56Ni_cal']
        scaler_list = ['scaler_0', 'scaler_1', 'scaler_2', 'scaler_c012']
        parameter = 'shift_iso-{}'.format('60')

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'Ni_StandartizedAnalysis_*'))[0]
        results = self.import_results(load_results_from)

        # plot settings
        digits = 1  # number of significant digits
        unit = 'MHz'
        plotAvg = True

        # make one separate plot for each isotope
        for iso in isolist:
            fig, ax1 = plt.subplots()

            # define output size of figure
            width, height = 1, 0.4
            fig.set_size_inches((self.w_in * width, self.h_in * height))

            x_type = 'file_times'  # alternative: 'file_numbers', 'file_times'
            x_ax = results[iso][x_type]
            file_numbers = results[iso]['file_numbers']  # for use as secondary axis

            for sc in scaler_list:
                scaler = sc

                # get the values
                centers = np.array(results[iso][scaler][parameter]['vals'])
                zero_arr = np.zeros(
                    len(centers))  # prepare zero array with legth of centers in case no errors are given
                # get the errors
                centers_d_stat = np.array(results[iso][scaler][parameter].get('d_stat', zero_arr))
                centers_d_syst = np.array(results[iso][scaler][parameter].get('d_syst', zero_arr))

                # calculate weighted average:
                if not np.any(centers_d_stat == 0) and not np.sum(1 / centers_d_stat ** 2) == 0:
                    wavg, wavg_d, wstd, std, std_avg = self.calc_weighted_avg(centers, centers_d_stat)
                    d = std_avg  # take the standard deviation of the mean
                    wavg_d = '{:.0f}'.format(10 ** digits * d)  # times 10 for representation in brackets
                else:  # some values don't have error, just calculate mean instead of weighted avg
                    wavg = centers.mean()
                    wavg_d = '-'

                # determine color by scaler
                col = self.pmt_colors[scaler]
                labelstr = 'PMT{}'.format(scaler.split('_')[-1])
                if scaler == 'scaler_c012':
                    # for the combined scaler use color determined by isotope
                    col = self.isotope_colors[iso[:4]]
                    labelstr = 'wAvg PMTs'

                # plot label with number:
                # plt_label = '{0} {1:.{4:d}f}({2}){3}' \
                #     .format(labelstr, wavg, wavg_d, unit, digits)
                # plot label without number:
                plt_label = labelstr

                # Do the plotting
                if scaler == 'scaler_c012':
                    # plot values as points
                    ax1.plot(x_ax, np.array(centers), '--', color=col)
                    # plot error band for statistical errors
                    ax1.fill_between(x_ax,
                                     np.array(centers) - centers_d_stat,
                                     np.array(centers) + centers_d_stat,
                                     alpha=0.8, facecolor='none',
                                     hatch='/', edgecolor=col,
                                     label=plt_label)
                else:
                    # define a very small offset to the time axis for scalers 0, 2 so its better visible
                    x_off = (int(scaler[-1])-1) * timedelta(0, 600)  # (days, seconds)
                    x_ax_off = [t+x_off for t in x_ax]
                    # plot values as dots with statistical errorbars
                    ax1.errorbar(x_ax_off, np.array(centers), yerr=np.array(centers_d_stat), label=plt_label,
                                 **self.ch_dict(self.data_style, {'color': col, 'markeredgecolor': col,
                                                                  'markersize': 4, 'capsize': 4}))

                # For the combined scaler, also plot the weighted avg over all scalers
                if scaler == 'scaler_c012' and plotAvg:
                    avg_parameter = 'avg_{}'.format(parameter)
                    # also plot average isotope shift
                    avg_shift = results[iso][scaler][avg_parameter]['vals'][0]
                    avg_shift_d = results[iso][scaler][avg_parameter]['d_stat'][0]
                    avg_shift_d_syst = results[iso][scaler][avg_parameter]['d_syst'][0]
                    # plot weighted average as red line
                    labelstr = 'wAvg files'
                    # plot label with number:
                    # plt_label = '{0}: {1:.{5:d}f}({2:.0f})[{3:.0f}]{4}'\
                    #     .format(labelstr, avg_shift, 10 ** digits * avg_shift_d, 10 ** digits * avg_shift_d_syst, unit, digits)
                    # plot label without number:
                    plt_label = labelstr

                    ax1.axhline(y=avg_shift, color=self.red)
                    # plot error of weighted average as red shaded box around that line
                    ax1.axhspan(ymin=avg_shift - avg_shift_d, ymax=avg_shift + avg_shift_d,
                                color=self.red,
                                alpha=0.4,
                                label=plt_label)

                    # plot systematic error as lighter red shaded box around that line
                    ax1.axhspan(ymin=avg_shift - avg_shift_d_syst - avg_shift_d,
                                ymax=avg_shift + avg_shift_d_syst + avg_shift_d,
                                color=self.red,
                                alpha=0.2)


            # work on the axes
            ax1.margins(0.05)
            if x_type == 'file_times':
                # create primary axis with the dates
                hours_fmt = mpl.dates.DateFormatter('%Hh')
                ax1.xaxis.set_major_formatter(hours_fmt)
                # create a days axis
                ax_day = ax1.twiny()
                ax_day.xaxis.set_ticks_position("bottom")
                ax_day.xaxis.set_label_position("bottom")
                ax_day.spines["bottom"].set_position(("axes", -0.07))  # Offset the days axis below the hours
                alldates = [datetime(2018, 4, d, 0, 0, 0) for d in range(13, 24, 1)]
                ax_day.set_xticks(alldates)  # same tick locations
                ax_day.set_xbound(ax1.get_xbound())
                days_fmt = mpl.dates.DateFormatter('%b.%d')
                ax_day.xaxis.set_major_formatter(days_fmt)
                ax_day.set_xlabel('date and time')
                ax_day.xaxis.set_label_coords(0.5, -0.1)  # set label position a little closer to axis
                # create a secondary axis with the run numbers
                ax_num = ax1.twiny()
                ax_num.set_xlabel('data set number')
                ax_num.xaxis.set_label_coords(0.5, 1.1)  # set label position a little closer to axis
                ax_num.set_xticks(x_ax)  # same tick locations
                ax_num.set_xbound(ax1.get_xbound())  # same axis range
                ax_num.set_xticklabels(file_numbers, rotation=90)
            else:
                plt.xlabel('run number')
            ax1.set_ylabel('isotope shift A-60 / {}'.format(unit))
            ax1.get_yaxis().get_major_formatter().set_useOffset(False)

            # create labels and title
            cal = ''
            if 'cal' in iso:
                cal = 'calibrated '
            # title = ax1.set_title('Isotope shifts in {} for all data sets of {}{}'.format(unit, cal, iso[:4]))
            # title.set_y(1.2)
            # fig.subplots_adjust(top=0.85)
            ax1.legend(bbox_to_anchor=(0.5, 1.2), loc="center", ncol=5, columnspacing=1.2, handletextpad=0.4)  # title='Scaler', columnspacing=0.5,

            plt.savefig(folder + 'iso_shifts_{}'.format(iso,)+self.ffe, dpi=self.dpi, bbox_inches='tight')
            plt.close()
            plt.clf()

    def calibration(self):
        """
        Plot calibration data.
        :return:
        """
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\Calibration\\')

        fig, ax1 = plt.subplots()
        # define output size of figure
        width, height = 1, 0.2
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        # Define which Isotope(s) to plot
        isolist = ['56Ni_cal']
        scaler = 'scaler_c012'
        parameter = 'shift_iso-{}'.format('60')

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'Ni_StandartizedAnalysis_*'))[0]
        results = self.import_results(load_results_from)

        x_type = 'file_times'  # alternative: 'file_numbers', 'file_times'

        hatches = ['/', '.']

        for num, iso in enumerate(['58Ni_cal', '60Ni_cal']):
            col = self.isotope_colors[iso[:4]]
            x_ax = results[iso][x_type]

            # get data
            cal = np.array(results[iso][scaler]['acc_volts']['vals'])
            cal_d = np.array(results[iso][scaler]['acc_volts']['d_stat'])
            cal_d_syst = np.array(results[iso][scaler]['acc_volts']['d_syst'])

            ax1.plot(x_ax, cal, ':', color=col)
            # # plot error band for statistical errors
            # ax1.fill_between(x_ax,
            #                  cal - cal_d,
            #                  cal + cal_d,
            #                  alpha=0.5, edgecolor=col, facecolor=col)
            # plot error band for systematic errors on top of statistical errors
            ax1.fill_between(x_ax,
                             cal - cal_d_syst - cal_d,
                             cal + cal_d_syst + cal_d,
                             label='{} reference'.format(iso[:4]),
                             alpha=0.8, facecolor='none',
                             hatch=hatches[num], edgecolor=col,
                             )

        # and finally plot the interpolated voltages
        for iso in isolist:
            col = self.isotope_colors[iso[:4]]
            x_ax = results[iso][x_type]

            # get data
            cal = np.array(results[iso][scaler]['acc_volts']['vals'])
            cal_d = np.array(results[iso][scaler]['acc_volts']['d_stat'])
            cal_d_syst = np.array(results[iso][scaler]['acc_volts']['d_syst'])

            ax1.errorbar(x_ax, cal, yerr=cal_d+cal_d_syst, label='{} interpolated'.format(iso[:4]),
                         **self.ch_dict(self.data_style, {'color': col, 'markeredgecolor': col, 'markersize': 4}))

        # format y-axis#
        plt.ylabel('beam particle energy / eV')
        ax1.get_yaxis().get_major_formatter().set_useOffset(False)

        # make x-axis dates
        plt.xlabel('date')
        days_fmt = mpl.dates.DateFormatter('%b.%d')
        ax1.xaxis.set_major_formatter(days_fmt)
        ax1.legend(bbox_to_anchor=(0.5, 1.2), loc="center", ncol=3, columnspacing=1.2, handletextpad=0.4)  # title='Scaler', columnspacing=0.5,
        plt.xticks(rotation=0)  # rotate date labels 45 deg for better readability
        plt.margins(0.05)

        plt.savefig(folder + 'calibration_{}'.format(iso)+self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def gatewidth(self):
        """
        Draw the Gate-width analysis results for one file.
        :return:
        """
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\Gates\\')

        fig, ax1 = plt.subplots()
        # define output size of figure
        width, height = 1, 0.4
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        # Define which Isotope(s) to plot
        iso = '58Ni'
        file = 'BECOLA_6501.xml'   # '60Ni_cal_6502.xml'
        scaler = 'scaler_012'
        include_devs = [-2.0, -1.0, 1.0, 2.0]  # midtof = 0 will be plotted separately

        # get data from results file
        load_results_from = glob(os.path.join(folder, '*all2018_FINAL*'))[0]
        results = self.import_results(load_results_from)
        # get the x-axis  --> its actually the powers of two of the gatewidth (2**x)
        xax = results[iso][scaler]['full_data']['xax']
        xax = TiTs.numpy_array_from_string(xax, -1, datatytpe=np.float)  # convert str to np.array
        # get the recommended sigma for this file
        f_indx = results[iso]['file_names'].index(file)
        rec_sigma = results[iso][scaler]['bestSNR_sigma']['vals'][f_indx]
        # get the SNR results
        SNR = results[iso][scaler]['full_data'][file]['SNR']['tof_0.0']['vals']
        SNR_d = results[iso][scaler]['full_data'][file]['SNR']['tof_0.0']['d_stat']
        # SNR = TiTs.numpy_array_from_string(SNR, -1, datatytpe=np.float)  # convert str to np.array
        # SNR_d = TiTs.numpy_array_from_string(SNR_d, -1, datatytpe=np.float)  # convert str to np.array
        # get the center fit results
        tof_dict = {}
        for key, item in results[iso][scaler]['full_data'][file]['centroid'].items():
            newkey = float(key.split('_')[-1])
            newvals = item['vals']
            new_d = item['d_stat']
            # newvals = TiTs.numpy_array_from_string(item['vals'], -1, datatytpe=np.float)  # convert str to np.array
            # new_d = TiTs.numpy_array_from_string(item['d_stat'], -1, datatytpe=np.float)  # convert str to np.array
            tof_dict[newkey] = {'vals': newvals, 'd_stat': new_d}

        ''' plotting '''
        # plot the midtof data
        for key, item in sorted(tof_dict.items()):
            if key in include_devs:  # midtof = 0 will be plotted separately
                y = np.array(item['vals'])
                y_d = np.array(item['d_stat'])
                col = self.colorgradient[int(key)]
                ax1.plot(xax, y, zorder=2, **self.ch_dict(self.fit_style, {'linewidth': 2, 'color': col}))
                ax1.fill_between(xax, y - y_d, y + y_d, label='{0:{1}.0f} bin'.format(key, '+' if key else ' '),
                                 alpha=0.4, facecolor=col, edgecolor='none', zorder=1
                                 )
        ax1.errorbar(xax, tof_dict[0.0]['vals'], tof_dict[0.0]['d_stat'], label='0 bin', zorder=3,
                     **self.ch_dict(self.data_style,
                                    {'markersize': 4, 'linestyle': '-', 'linewidth': 2, 'capsize': 5, 'capthick': 2}))

        # plot the gatewidth used in file
        # ax.axvline(x=np.log(200 * self.tof_width_sigma * self.tof_sigma[iso]) / np.log(2), color='red')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.set_ylabel('fit centroid rel. to analysis value / MHz', color='k')
        ax1.set_xlabel('gate size [bins]')

        # plot SNR error band
        axSNR = ax1.twinx()
        axSNR.set_ylabel('signal-to-noise ratio', color='red')  # we already handled the x-label with ax1
        axSNR.plot(xax, SNR, label='SNR', **self.ch_dict(self.fit_style, {'linewidth': 2}))
        axSNR.tick_params(axis='y', labelcolor='red')

        axSNR.axvline(x=np.log(rec_sigma*2*2)/np.log(2), color=self.red, linestyle='--')

        import matplotlib.ticker as ticker
        # convert the x-axis to real gate widths (2**x):
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x_b2, _: '{:.0f}'.format(2 ** x_b2)))
        plt.margins(0.05)
        ax1.legend(title='Offset of midTof parameter relativ to analysis value', bbox_to_anchor=(0.5, 1.2),
                   loc="center", ncol=6, columnspacing=0.7, handletextpad=0.1)

        plt.savefig(folder + 'gatewidth' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def SNR_analysis(self):
        """
        SNR ANalysis
        :return:
        """
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\Gates\\')

        # get data from results file
        load_results_from = glob(os.path.join(folder, '*AllIso*'))[0]  #AllIsoInclOffline #FINAL
        results = self.import_results(load_results_from)

        # Define isotopes
        iso_list = ['54Ni', '55Ni', '56Ni', '58Ni', '60Ni', '62Ni', '64Ni',]  #['56Ni', '58Ni', '60Ni']  #
        file_name_ex = ['Sum54Nic_9999.xml', 'Sum55Nic_9999.xml', 'Sum56Nic_9999.xml',
                        'Sum58Nic_9999.xml', 'Sum60Nic_9999.xml', 'Sum62Nic_9999.xml', 'Sum64Nic_9999.xml']  # file_name of example file for each isotope
        global_guess = {'54Ni': 5.19, '55Ni': 5.23, '56Ni': 5.28, '58Ni': 5.36, '60Ni': 5.47, '62Ni': 5.59, '64Ni': 5.68}
        scaler = 'scaler_012'

        midSNR_per_iso = []
        midSNR_d_per_iso = []
        midTOF_per_iso = []
        midTOF_d_per_iso = []

        remove_isos = []
        for num, iso in enumerate(iso_list):
            if iso in results.keys():
                base_dir = results[iso][scaler]
                file = 'Sum{}c_9999.xml'.format(iso)  #file_name_ex[num]
                file_indx = results[iso]['file_names'].index(file)
                # get the x-axis  --> its actually the powers of two of the gatewidth (2**x)
                xax = base_dir['full_data']['xax']
                xax = TiTs.numpy_array_from_string(xax, -1, datatytpe=np.float)  # convert str to np.array
                gatewidth_variation_arr = np.array([2**x_b2 for x_b2 in xax])
                # get the gate results
                midtof_var = []
                SNR_arr = []
                SNR_d_arr = []
                Centr_arr = []
                Centr_d_arr = []
                for m_key, vals in base_dir['full_data'][file]['SNR'].items():
                    m_v = ast.literal_eval(m_key[4:])
                    midtof_var.append(m_v)
                midtof_variation_arr = np.array(sorted(midtof_var))
                for m in midtof_variation_arr:
                    m_key = 'tof_{:.1f}'.format(m)
                    # get the results from the SNR analysis
                    SNR = base_dir['full_data'][file]['SNR'][m_key]['vals']
                    SNR_d = base_dir['full_data'][file]['SNR'][m_key]['d_stat']
                    SNR_arr.append(np.array(SNR))  # convert to np.array
                    SNR_d_arr.append(np.array(SNR_d))  # convert to np.array
                    # get the results from the SNR analysis
                    Centr = base_dir['full_data'][file]['centroid'][m_key]['vals']
                    Centr_d = base_dir['full_data'][file]['centroid'][m_key]['d_stat']
                    Centr_arr.append(np.array(Centr))  # convert str to np.array
                    Centr_d_arr.append(np.array(Centr_d))  # convert str to np.array
                # make proper 2-D arrays from the data lists again
                SNR_arr = np.array(SNR_arr)
                SNR_d_arr = np.array(SNR_d_arr)
                Centr_arr = np.array(Centr_arr)
                Centr_d_arr = np.array(Centr_d_arr)

                # Get the analysis values for this file
                # rec_mid_SNR = (base_dir['bestSNR_mid']['vals'][file_indx], base_dir['bestSNR_mid']['d_fit'][file_indx])
                # rec_sig_SNR = (base_dir['bestSNR_sigma']['vals'][file_indx], base_dir['bestSNR_sigma']['d_fit'][file_indx])
                # rec_mid_TOF = (base_dir['fitTOF_mid']['vals'][file_indx], base_dir['fitTOF_mid']['d_fit'][file_indx])
                # rec_sig_TOF = (base_dir['fitTOF_sigma']['vals'][file_indx], base_dir['fitTOF_sigma']['d_fit'][file_indx])
                # Get the recommended values for this isotope
                rec_mid_SNR = (base_dir['recommended_mid_SNR']['vals'][0],
                               base_dir['recommended_mid_SNR']['d_fit'][0])
                rec_sig_SNR = (base_dir['recommended_sigma_SNR']['vals'][0],
                               base_dir['recommended_sigma_SNR']['d_fit'][0])
                rec_mid_TOF = (base_dir['recommended_mid_TOF']['vals'][0],
                               base_dir['recommended_mid_TOF']['d_fit'][0])
                rec_sig_TOF = (base_dir['recommended_sigma_TOF']['vals'][0],
                               base_dir['recommended_sigma_TOF']['d_fit'][0])
                midSNR_per_iso.append(rec_mid_SNR[0])
                midSNR_d_per_iso.append(rec_mid_SNR[1])
                midTOF_per_iso.append(rec_mid_TOF[0])
                midTOF_d_per_iso.append(rec_mid_TOF[1])
                #
                fit_pars = base_dir['SNR_fit']['vals'][file_indx]
                fit_errs = base_dir['SNR_fit']['d_fit'][file_indx]

                # def SNR_model(x, y, SNRmax, mid, tof_sigma):
                #     """
                #     Assuming the ion bunch is gauss-shaped in the time domain, the signal-to-noise ratio should
                #     be defined through the integral over the gaussian and the (constant) background
                #     :param x: the x-position on the plotwindow/meshgrid
                #     :param y: the y-position on the plotwindow/meshgrid
                #     :param SNRmax: The maximum SNR. Scales the height but doesn't influence the positon
                #     :param mid: the real mid-tof correction to the assumed value
                #     :param tof_sigma: the real tof-sigma. This determines most of the SNR analysis
                #     :return: SNR, depending on gate choice
                #     """
                #     from scipy import special
                #     # give x and y a real meaning:
                #     width = gatewidth_variation_arr[x]  # the x-parameter is the gatewidth defined by log array
                #     tof_off = midtof_variation_arr[y]  # the y parameter is the tof-variation. We assume midtof=0
                #     # Integrate the gaussian from midtof+y-x/2 to midtof+y+x/2
                #     # The integral is defined through imaginary errorfunction erf
                #     intensity = np.sqrt(np.pi / 2) * tof_sigma * (
                #             special.erf((tof_off + width / 2 - mid) / (np.sqrt(2) * tof_sigma))
                #             - special.erf((tof_off - width / 2 - mid) / (np.sqrt(2) * tof_sigma)))
                #     intensity = intensity / (2.10177 * tof_sigma)  # scaling for int=1 at 2.8 sigma
                #     background = width / 2.8 * tof_sigma  # scales linear by width. scaling for bg=1 at 2.8 sigma
                #     # In reality, the background will not be linear but have a beam induced straylight component
                #     # if iso in self.isotopes_stable and '58' not in iso:
                #     #     # beam bg of stable isotopes will be dominated by 58Ni
                #     #     mid_bb = self.tof_mid['58Ni']-self.tof_mid[iso]+tof_off
                #     # else:
                #     #     # beam background will be dominated by the resonance isotope
                #     #     mid_bb = mid
                #     # bg_stray = beamb * np.sqrt(np.pi/2)*tof_sigma * (  # scaling of 2.7 has been extracted from time-res data
                #     #         special.erf((tof_off + width / 2 - mid_bb) / (np.sqrt(2) * tof_sigma))
                #     #         - special.erf((tof_off - width / 2 - mid_bb) / (np.sqrt(2) * tof_sigma)))
                #     # bg_stray = bg_stray / 2.10177*tof_sigma  # scaling for bg_stray = beamb at 2.8 sigma
                #     # background = (background + bg_stray)/(1+beamb)
                #
                #     # return the signal to noise calculated from this
                #     return SNRmax * intensity / np.sqrt(background)

                def SNR_model(x, y, SNRmax, mid, tof_sigma, beamb):
                    """
                    Assuming the ion bunch is gauss-shaped in the time domain, the signal-to-noise ratio should
                    be defined through the integral over the gaussian and the (constant) background
                    :param x: the x-position on the plotwindow/meshgrid
                    :param y: the y-position on the plotwindow/meshgrid
                    :param SNRmax: The maximum SNR. Scales the height but doesn't influence the positon
                    :param mid: the real mid-tof correction to the assumed value
                    :param tof_sigma: the real tof-sigma. This determines most of the SNR analysis
                    :return: SNR, depending on gate choice
                    """
                    from scipy import special
                    # give x and y a real meaning:
                    width = gatewidth_variation_arr[x]  # the x-parameter is the gatewidth defined by log array
                    tof_off = midtof_variation_arr[y]  # the y parameter is the tof-variation. We assume midtof=0
                    # Integrate the gaussian from midtof+y-x/2 to midtof+y+x/2
                    # The integral is defined through imaginary errorfunction erf
                    intensity = np.sqrt(np.pi / 2) * tof_sigma * (
                            special.erf((tof_off - mid + width / 2) / (np.sqrt(2) * tof_sigma))
                            - special.erf((tof_off - mid - width / 2) / (np.sqrt(2) * tof_sigma)))
                    intensity = intensity / (2.10177 * tof_sigma)  # scaling for int=1 at 2.8 sigma
                    background = width / (2.8 * tof_sigma)  # scales linear by width. scaling for bg=1 at 2.8 sigma
                    # In reality, the background will not be linear but have a beam induced straylight component
                    if iso in ['58Ni', '60Ni', '62Ni', '64Ni']:
                        # beam bg of stable isotopes will be dominated by 58Ni
                        mid_58 = 100 * global_guess['58Ni'] - (100 * global_guess[iso] + mid)
                        sigma_58 = 100 * 0.078
                        bg_stray = beamb * np.sqrt(np.pi / 2) * sigma_58 * (
                                special.erf((tof_off - mid_58 + width / 2) / (np.sqrt(2) * sigma_58))
                                - special.erf((tof_off - mid_58 - width / 2) / (np.sqrt(2) * sigma_58))) \
                                   / (2.10177 * sigma_58)
                        # 60Ni contribution is next. Scaled by natural abundance ratio
                        mid_60 = 100 * global_guess['60Ni'] - (100 * global_guess[iso] + mid)
                        sigma_60 = 100 * 0.074
                        bg_stray += 26 / 68 * beamb * np.sqrt(np.pi / 2) * sigma_60 * (
                                special.erf((tof_off - mid_60 + width / 2) / (np.sqrt(2) * sigma_60))
                                - special.erf((tof_off - mid_60 - width / 2) / (np.sqrt(2) * sigma_60))) \
                                    / (2.10177 * sigma_60)
                    else:
                        # beam background for radioactive isotopes is small to negligible compared to laser bg
                        bg_stray = 0
                        beamb = 0
                        # if we wanna try anyways then the beam bg is of course only the radioactive isotope
                        # mid_bb = mid
                        # sigma_bb = tof_sigma
                        # bg_stray = beamb * np.sqrt(np.pi/2)*sigma_bb * (  # scaling of 2.7 has been extracted from time-res data
                        #         special.erf((tof_off - mid_bb + width / 2) / (np.sqrt(2) * sigma_bb))
                        #         - special.erf((tof_off - mid_bb - width / 2) / (np.sqrt(2) * sigma_bb)))
                        # bg_stray = bg_stray / (2.10177*sigma_bb)  # scaling for bg_stray = beamb at 2.8 sigma
                    background = (background + bg_stray) / (1 + beamb)

                    # norm = np.sqrt(2.8*tof_sigma)/(2.10177*tof_sigma)
                    norm = 1
                    # return the signal to noise calculated from this
                    return SNRmax * norm * intensity / np.sqrt(background)

                ''' do all the plotting for this isotope'''
                f, ax = plt.subplots()
                # define output size of figure
                width, height = 1, 0.6
                f.set_size_inches((self.w_in * width, self.h_in * height))

                x = np.arange(gatewidth_variation_arr.shape[0])  # gatewidth_variation_arr
                y = np.arange(midtof_variation_arr.shape[0])  # midtof_variation_arr
                X, Y = np.meshgrid(x, y)
                # print the SNR values
                im = ax.imshow(SNR_arr, cmap=self.custom_cmap, interpolation='nearest')  #'nearest'
                pars = fit_pars
                # pars = [np.amax(SNR_arr), rec_mid_SNR[0]-global_guess[iso], rec_sig_SNR[0], 5]
                ax.contour(X, Y, SNR_model(X, Y, *pars), 15, colors='w', antialiased=True)
                # TODO: log scale the y errors
                scale = 34/(6-2.6)  # 10
                # xerr = [[-scale*(np.log(rec_sig_SNR[0]*2.8-rec_sig_SNR[1])/np.log(2)-2)
                #          +scale*(np.log(rec_sig_SNR[0]*2.8)/np.log(2)-2)],
                #         [scale*(np.log(rec_sig_SNR[0]*2.8+rec_sig_SNR[1])/np.log(2)-2)
                #          -scale*(np.log(rec_sig_SNR[0]*2.8)/np.log(2)-2)]]
                xerr = [[-scale * (np.log(fit_pars[2] * 2.8 - fit_errs[2]) / np.log(2) - 2)
                         + scale * (np.log(fit_pars[2] * 2.8) / np.log(2) - 2)],
                        [scale * (np.log(fit_pars[2] * 2.8 + fit_errs[2]) / np.log(2) - 2)
                         - scale * (np.log(fit_pars[2] * 2.8) / np.log(2) - 2)]]
                # ax.errorbar(x=scale*(np.log(fit_pars[2]*2.8)/np.log(2)-2),
                #             y=fit_pars[1]-midtof_variation_arr[0],  #-global_guess[iso]
                #             xerr=xerr, yerr=fit_errs[1], **self.ch_dict(self.data_style,
                #                                                                      {'color': 'w', 'markersize': 5}))
                ax.axhline(fit_pars[1]-midtof_variation_arr[0], color='w', lw=2, ls='--')

                ax.set_xlabel('gate size / bins')
                ax.set_ylabel('gate center / bins')
                x_tick_pos = [4, 14, 24, 34]
                ax.set_xticks(x_tick_pos)
                ax.set_xticklabels(['{:.0f}'.format(gatewidth_variation_arr[m]) for m in x_tick_pos])
                # ax.set_xticks(np.arange(41))
                # ax.set_xticklabels(['{:.1f}'.format(m) for m in np.logspace(2, 6, 41, base=2)])
                plt.setp(ax.get_xticklabels(), rotation=0, ha="center", va='top', rotation_mode="default")  # in case rotate
                y_tick_pos = [1, 3, 5, 7, 9]
                ax.set_yticks(y_tick_pos)
                ax.set_yticklabels(['{:.0f}'.format(midtof_variation_arr[m]+100*global_guess[iso]) for m in y_tick_pos])

                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                cbar = f.colorbar(im, cax=cax)
                cbar.set_label('signal-to-noise ratio')

                plt.savefig(folder + '{}_SNR'.format(iso)+self.ffe, dpi=self.dpi, bbox_inches='tight')
                plt.close()
                plt.clf()
            else:
                # isotope not in SNR analysis
                remove_isos.append(iso)

        # remove all isotopes that were not in analysis
        for isotope in remove_isos:
            iso_list.remove(isotope)
        # Now plot the mid gate parameter
        f, ax = plt.subplots()
        # define output size of figure
        width, height = 1, 0.2
        f.set_size_inches((self.w_in * width, self.h_in * height))

        iso_ax = np.array([ast.literal_eval(i[:2]) for i in iso_list])
        ax.errorbar(iso_ax, midSNR_per_iso, yerr=midSNR_d_per_iso, label='SNR analysis',
                    **self.ch_dict(self.data_style, {'color': self.purple, 'markersize': 4, 'capsize': 4}))
        ax.errorbar(iso_ax, midTOF_per_iso, yerr=midTOF_d_per_iso, label='TOF fitting',
                    **self.ch_dict(self.data_style, {'color': self.blue, 'markersize': 4, 'capsize': 4}))
        # do a linear regression
        m, b = self.lin_regression(np.sqrt(iso_ax), midSNR_per_iso, (1, 0))
        ax.plot(iso_ax, self._line(np.sqrt(iso_ax), m, b), label='fit', **self.fit_style)  #label='mid-tof={:.1f}\u00B7A+{:.1f}bins'.format(m, b)

        ax.set_xlim((iso_ax[0]-0.5, iso_ax[-1]+0.5))
        ax.set_xticks(iso_ax)
        ax.set_xticklabels([r'$\mathregular{^{54}Ni}$', r'$\mathregular{^{55}Ni}$', r'$\mathregular{^{56}Ni}$',
                            r'$\mathregular{^{58}Ni}$', r'$\mathregular{^{60}Ni}$', r'$\mathregular{^{62}Ni}$',
                            r'$\mathregular{^{64}Ni}$'])
        ax.set_xlabel('Isotope')

        ax.set_ylabel('gate center / bins')

        ax.legend(bbox_to_anchor=(0.5, 1.15), loc="center", ncol=3, columnspacing=1.2, handletextpad=0.4)  # title='Scaler', columnspacing=0.5, bbox_to_anchor=(0.5, 1.1), loc="lower right"

        plt.savefig(folder + 'gate_center' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def all_spectra(self):
        """
        Plot the spectra for all isotopes on top of each other together with a line for the centroid.
        :return:
        """
        # Specify in which folder input and output should be found
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\AllSpectra\\')

        # load all data and define isotopes

        iso_list = ['54Ni', '55Ni', '56Ni', '58Ni', '60Ni']  #['55Ni', '56Ni', '58Ni', '60Ni']
        isotope_shifts = [-1917.4, -1433.0, -1003.6, -501.7, 0]

        # Create plots for all three lineshapes
        widths = [1]  # only one column, relative width = 1
        heights = [1] * iso_list.__len__()  # as many rows as isotopes. All same relative height.
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        f, axes = plt.subplots(nrows=iso_list.__len__(), ncols=1,
                               sharex=True, sharey=False,  # Share x-axes between all plots. Leave Y-axis free
                               gridspec_kw=gs_kw)

        # define output size of figure
        width, height = 0.9, 0.4
        f.set_size_inches((self.w_in * width, self.h_in * height))

        for num, iso in enumerate(sorted(iso_list)):
            # sorted and enumerated list to put each isotope at the right location
            x, cts, res, cts_err = np.loadtxt(
                glob(os.path.join(folder, '*{}*data*'.format(iso)))[0],  # name of isotope must be in filename
                delimiter=', ', skiprows=1, unpack=True)
            x_fit, fit = np.loadtxt(
                glob(os.path.join(folder, '*{}*fullShape*'.format(iso)))[0],  # name of isotope must be in filename
                delimiter=', ', skiprows=1, unpack=True)
            isoshift = isotope_shifts[num]

            # transform x-axis to GHz
            x_unit = 'GHz'
            x = x/1000
            x_fit = x_fit/1000
            isoshift = isoshift/1000

            # now plot
            cen_art = axes[num].axes.axvline(x=isoshift, label='c.o.g.', linestyle='--', color=self.blue, lw=2)
            dat_art = axes[num].errorbar(x, cts, cts_err, label='data', **self.data_style)  # data
            fit_art = axes[num].plot(x_fit, fit, label='fit', **self.fit_style)  # fit
            # create legend
            # axes[num].legend((dat_art, fit_art, cen_art), ('data', 'fit', 'centroid'),
            #                  bbox_to_anchor=(0.5, 1.3), loc='center', ncol=3)
            # Place isotope name in top right corner
            axes[num].text(0.8, 0.9, r'$^{{{0}}}$Ni'.format(iso[:2]),
                           horizontalalignment='left', verticalalignment='top',
                           transform=axes[num].transAxes,
                           **self.ch_dict(self.text_style, {'size': 14})
                           )

        custom_ticks = False
        if custom_ticks:
            # set y-axes:
            f.text(-0.02, 0.5, 'cts / arb. units', ha='center', va='center',
                   rotation='vertical')  # common label for all y-axes
            for ax in range(len(iso_list)):
                yti = axes[ax].get_yticks()
                # determine a few nice ticks
                n_ticks = 3  # number of ticks we aim for on the new axis (rounding stuff below can change it a little)
                significant_num = 1000  # cut at 1000 cts? If you want full numbers set to 1 (or 10, 100, whatever)
                sig_symb = 'k'  # symbol to be attached to numbers. e.g. 1000 -> 1k. Set empty string if not used.
                axrange = yti.max() - yti.min()  # get the range currently spanned by the ticks
                newspacing = round(axrange//(n_ticks), -int(floor(log10(abs(axrange//(n_ticks))))))  # A lot of rounding to get some reasonable spacing with nice numbers
                newspacing = ceil(newspacing/significant_num)*significant_num  # adapt spacing to the above set significant number if it is below.
                if yti.min() > 0:  # log10 doesnt't work for 0 of course
                    newmin = round(yti.min(), -int(floor(log10(abs(yti.min()))))+1)  # A starting value for the lowest tick
                    newmin = round(newmin, -int(log10(significant_num)))  # again adapt to significant number
                else:  # if yti.min=0, then the axis should start at 0
                    newmin = 0
                newticks = np.arange(newmin, yti.max(), newspacing)
                # set the new labels
                axes[ax].set_yticks(newticks)
                axes[ax].set_yticklabels(['{:.0f}{}'.format(i//significant_num, sig_symb) for i in newticks])
                axes[ax].axes.tick_params(axis='y', direction='in',
                                          left=True, right=True,  # ticks left and right
                                          labelleft=True, labelright=False)  # no ticklabels anywhere. Because they look cluttered
        else:
            # set y-axes:
            f.text(-0.08, 0.5, 'cts / arb. units', ha='center', va='center',
                   rotation='vertical')  # common label for all y-axes
            for ax in range(len(iso_list)):
                axes[ax].axes.tick_params(axis='y', direction='in',
                                          left=True, right=True,  # ticks left and right
                                          labelleft=False, labelright=False)  # no ticklabels anywhere. Because they look cluttered
        # set x-axis
        axes[-1].set_xlabel(r'Frequency relative to $\nu_0^{60}$ / '+'{}'.format(x_unit))
        axes[-1].set_xlim(-2.25, 1.05)
        axes[-1].set_xticks([-2, -1, 0, 1])  # for custom ticks
        for ax in range(len(iso_list)):
            axes[ax].axes.tick_params(axis='x', top=False, bottom=True)

        # set label
        da, la = axes[0].get_legend_handles_labels()
        handles = (da[2], da[1], da[0])
        labels = (la[2], la[1], la[0])
        axes[0].legend(handles, labels, bbox_to_anchor=(0.5, 1.3), loc='center', ncol=3)

        # make tight spacing between subplots
        f.tight_layout(pad=0)

        plt.savefig(folder + 'pub_all_spectra' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

        # Do another Plot for Nickel 55 only
        # Create plots for all three lineshapes
        widths = [1]
        heights = [1, 0.4]
        gs_kw = dict(width_ratios=widths, height_ratios=heights)
        f, axes = plt.subplots(nrows=2, ncols=1, sharex=True, sharey='row', gridspec_kw=gs_kw)
        # define output size of figure
        width, height = 0.6, 0.3
        f.set_size_inches((self.w_in * width, self.h_in * height))

        # sorted and enumerated list to put each isotope at the right location
        x, cts, res, cts_err = np.loadtxt(
            glob(os.path.join(folder, '*55*data*'))[0],  # name of isotope must be in filename
            delimiter=', ', skiprows=1, unpack=True)
        x_fit, fit = np.loadtxt(
            glob(os.path.join(folder, '*55*fullShape*'))[0],  # name of isotope must be in filename
            delimiter=', ', skiprows=1, unpack=True)

        axes[0].errorbar(x, cts, cts_err, **self.data_style)  # data
        axes[0].plot(x_fit, fit, **self.fit_style)  # fit
        axes[1].errorbar(x, res, cts_err, **self.data_style)  # residuals
        axes[0].set_title('Nickel 55 Summed Data Spectrum')

        # set axes:
        axes[0].set_ylabel('cts/ arb.u.')
        axes[0].set_yticks([11000, 11500, 12000, 12500, 13000])
        axes[0].set_ylim((10900, 13200))
        axes[0].set_yticklabels(['11.0k', '11.5k', '12.0k', '12.5k', '13.0k'])
        axes[1].set_ylabel('res/ arb.u.')
        axes[1].set_yticks([-500, 0, 500])
        axes[1].set_xlabel(r'Frequency relative to $\nu_0^{60}$ / '+'{}'.format(x_unit))
        axes[1].set_xlim((-2500, 1200))
        axes[1].set_xticks([-2000, -1000, 0, 1000])

        # make tight spacing between subplots
        f.tight_layout(pad=0)

        plt.savefig(folder + 'pub_Ni55_spectrum' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def level2plus_be2(self):
        """

        :return:
        """
        # Specify in which folder input and output should be found
        folder = os.path.join(self.fig_dir, 'Nickel\\General\\LevelE_BE2\\')

        fig, ax1 = plt.subplots()
        plot_secondary_ax = True
        if plot_secondary_ax:
            ax2 = ax1.twinx()  # create second axis on top of first
            ax2.set_yscale("log")
        # define output size of figure
        width, height = 1, 0.3
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        ''' import the data '''
        # Pritychenko et al. 2016: 10.1016/j.adt.2015.10.001
        # 78Ni: Taniuchi et al. 2019: 10.1038/s41586-019-1155-x
        data = pd.read_csv(glob(os.path.join(folder, 'Data_Clean*'))[0], delimiter='\t',
                           index_col=0, skiprows=[1])

        ''' define isotopes to include '''
        iso_dict = {'20Ca': (40, 48+1),
                    '22Ti': (50, 50+1),
                    '24Cr': (52, 52+1),
                    '26Fe': (54, 54+1),
                    '28Ni': (56, 78+1)
                    }

        ''' fill datasets '''
        e2plus_data = []
        e2plus_errs = []
        masses_e2p = []
        be2_data = []
        be2_errs_min = []
        be2_errs_plu = []
        masses_be2 = []
        for i, rng in sorted(iso_dict.items()):
            iso = i[2:]
            for mass in np.arange(rng[0], rng[1], 2):
                iso_A = '{}{}'.format(str(mass), iso)
                # get E2+ data
                dt1 = data.loc[iso_A, 'E2+']
                e2, e2_d, *_ = re.split('[()]', dt1)
                if '.' in e2:
                    sign_digit = len(e2.split('.')[1])  # get the significant digits of the value
                else:
                    sign_digit = 0
                e2plus_data.append(float(e2))
                e2plus_errs.append(float(e2_d)/10**sign_digit)
                masses_e2p.append(mass)
                # get B(E2) data
                dt2 = data.loc[iso_A, 'B(E2)u']
                if dt2 is not np.nan:
                    be2, be2_d, *_ = re.split('[()]', dt2)
                    if '.' in be2:
                        sign_digit = len(be2.split('.')[1])  # get the significant digits of the value
                    else:
                        sign_digit = 0
                    be2_data.append(float(be2))
                    # for B(E2) uncertainties are often given as (+a-b). That needs special care.
                    if '/' in be2_d:
                        p_err, m_err = be2_d.split('/')
                        be2_errs_min.append(float(p_err)/10**sign_digit)
                        be2_errs_plu.append(abs(float(m_err))/10**sign_digit)
                    else:
                        be2_errs_min.append(float(be2_d)/10**sign_digit)
                        be2_errs_plu.append(float(be2_d)/10**sign_digit)
                    masses_be2.append(mass)
        ''' Do the plotting '''
        ax1.bar(masses_e2p, e2plus_data, **self.bar_style)
        if plot_secondary_ax:
            ax2.errorbar(masses_be2, be2_data, [be2_errs_min, be2_errs_plu],
                         **self.ch_dict(self.data_style,
                                        {'color': self.red, 'markeredgecolor': self.red,
                                     'markersize': 3, 'linestyle': '-', 'capsize': 3}))

        ''' Style the axes '''
        # work on x axis
        ax1.axes.tick_params(axis='x', direction='out',
                             bottom=True, top=False, labelbottom=True, labeltop=False)
        # ax1.set_xlabel('Isotope')
        ax1.set_xlim((39, 79))
        ax1.set_xticks([40, 48, 56, 68, 78])
        ax1.set_xticklabels(['$^\mathregular{40}$Ca',
                             '$^\mathregular{48}$Ca',
                             '$^\mathregular{56}$Ni',
                             '$^\mathregular{68}$Ni',
                             '$^\mathregular{78}$Ni'], size=14)
        # Annotate the Regions
        ax1.annotate('Z=28', xy=(66, 200), size=14)
        ax1.annotate(s='', xy=(55, 100), xytext=(79, 100),
                     arrowprops=dict(arrowstyle='<->', lw=1.5))
        ax1.annotate('N=28', xy=(50, 300), size=14)
        ax1.annotate(s='', xy=(47, 200), xytext=(57, 200),
                     arrowprops=dict(arrowstyle='<->', lw=1.5))
        ax1.annotate('Z=20', xy=(42, 200), size=14)
        ax1.annotate(s='', xy=(39, 100), xytext=(49, 100),
                     arrowprops=dict(arrowstyle='<->', lw=1.5))
        # work on y axis
        ax1.set_ylabel('E(2+) / keV', color=self.blue)
        ax1.tick_params(axis='y', labelcolor=self.blue, color=self.blue, which='both')
        ax1.set_ylim((0, 4500))
        ax1.set_yticks([0, 1000, 2000, 3000, 4000])
        if plot_secondary_ax:
            ax2.set_ylabel('B(E2) / (e$^2$b$^2$)', color=self.red)
            ax2.set_ylim((0.005, 0.12))
            ax2.tick_params(axis='y', labelcolor=self.red, color=self.red, which='both')

        pltname = 'E2'
        if plot_secondary_ax:
            pltname += '_BE2'
        plt.savefig(folder + pltname + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def a_ratio_comparison(self):
        """

        :return:
        """
        # Specify in which folder input and output should be found
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\ARatio\\')

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # create second axis on top of first
        # define output size of figure
        width, height = 1, 0.3
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        ''' import the data '''
        data = pd.read_csv(glob(os.path.join(folder, 'data*'))[0], delimiter='\t', index_col=0)

        ''' fill datasets '''
        spins = []
        center = []
        center_errs = []
        A_rat = []
        A_rat_errs = []

        for index, row in data.iterrows():
            spins.append(index)
            ''' center values '''
            c = row['Center']
            ce, ce_d, *_ = re.split('[()]', c)
            if '.' in ce:
                sign_digit = len(ce.split('.')[1])  # get the significant digits of the value
            else:
                sign_digit = 0
            center.append(float(ce))
            center_errs.append(float(ce_d)/10**sign_digit)
            ''' A ratios '''
            ar = row['A_ratio']
            ara, ara_d, *_ = re.split('[()]', ar)
            if '.' in ara:
                sign_digit = len(ara.split('.')[1])  # get the significant digits of the value
            else:
                sign_digit = 0
            A_rat.append(float(ara))
            A_rat_errs.append(float(ara_d) / 10 ** sign_digit)

        spins = np.array(spins)
        center = np.array(center)
        center_errs = np.array(center_errs)
        A_rat = np.array(A_rat)
        A_rat_errs = np.array(A_rat_errs)

        ''' plot the data '''
        ax1.errorbar(spins-0.05, A_rat, yerr=A_rat_errs,   # shift x-axis by a tiny bit to the left
                     **self.ch_dict(self.data_style,
                                    {'color': self.blue, 'markeredgecolor': self.blue,
                                     'markersize': 5, 'linestyle': '', 'capsize': 5}))
        ax2.errorbar(spins+0.05, center, yerr=center_errs,  # shift x-axis by a tiny bit to the right
                     **self.ch_dict(self.data_style,
                                    {'color': self.red, 'markeredgecolor': self.red,
                                     'markersize': 3, 'linestyle': '', 'capsize': 3}))
        ''' plot the literature value '''
        lit_a_rat = 0.389
        lit_a_rat_d = 0.001
        #ax1.axhline(lit_a_rat, color=self.blue)
        ax1.axhspan(ymin=lit_a_rat-lit_a_rat_d, ymax=lit_a_rat+lit_a_rat_d, color=self.blue, alpha=0.5, linewidth=0)
        # ax1.errorbar(7/2, lit_a_rat, lit_a_rat_d, **self.ch_dict(self.data_style,
        #                             {'color': self.orange, 'markeredgecolor': self.orange,
        #                              'markersize': 6, 'linestyle': '-', 'capsize': 6}))

        ''' Style the axes '''
        # work on x axis
        ax1.axes.tick_params(axis='x', direction='in',
                             bottom=True, top=True, labelbottom=True, labeltop=False)
        ax1.set_xlabel('nuclear spin')
        #ax1.set_xlim()
        ax1.set_xticks(spins)
        ax1.set_xticklabels(["{:.0f}/2".format(2*s) for s in spins])
        # work on y axis
        ax1.set_ylabel('A-ratio', color=self.blue)
        ax1.set_yticks([0.30, 0.32, 0.34, 0.36, 0.38, 0.40, 0.42, 0.44, 0.46,  0.48,  0.50])
        ax1.set_ylim((0.30, 0.51))
        ax1.tick_params(axis='y', labelcolor=self.blue, color=self.blue, which='both')

        ax2.set_ylabel('centroid / MHz', color=self.red)
        ax2.set_ylim((-2100, -500))
        ax2.tick_params(axis='y', labelcolor=self.red, color=self.red, which='both')

        plt.savefig(folder + 'Aratio_compare' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def pumping_simulation(self):
        """
        :return:
        """
        # Specify in which folder input and output should be found
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\Pumping\\')

        fig, ax1 = plt.subplots()
        # define output size of figure
        width, height = 1, 0.3
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        ''' import the data '''
        data = np.loadtxt(glob(os.path.join(folder, 'data*'))[0], delimiter='\t', unpack=True)

        ''' fill datasets '''
        peak_14 = data[0]  # 11/2-->13/2
        obs_14 = (1, 0)
        peak_13 = data[2]  # 11/2-->11/2
        obs_13 = (1.93, 0.64)
        peak_12 = data[1]  # 9/2-->11/2
        obs_12 = (0.91, 0.24)
        peak_10 = data[4]  # 9/2-->9/2
        obs_10 = (1.07, 0.41)
        peak_9 = data[3]  # 7/2-->9/2
        obs_9 = (0.96, 0.25)
        number_ex = np.arange(len(peak_14))

        ''' plot '''
        # ax1.plot(number_ex, peak_14, color=self.purple, )  #**self.ch_dict(self.point_style, {'color': self.purple}))
        # ax1.axhspan(obs_14[0]-obs_14[1], obs_14[0]+obs_14[1], alpha=0.2, color=self.purple)
        ax1.plot(number_ex, peak_13, color=self.purple, lw=2, label=r'$\mathrm{11/2\to 11/2}$')  #**self.ch_dict(self.point_style, {'color': self.black}))
        ax1.axhspan(obs_13[0]-obs_13[1], obs_13[0]+obs_13[1], alpha=0.1, color=self.purple, hatch='.', fill=True, lw=.9)
        ax1.axhline(obs_13[0], color=self.purple, ls='--')
        ax1.plot(number_ex, peak_12, color=self.orange, lw=2, label=r'$\mathrm{9/2\to 11/2}$')  #**self.ch_dict(self.point_style, {'color': self.orange}))
        ax1.axhspan(obs_12[0]-obs_12[1], obs_12[0]+obs_12[1], alpha=0.1, color=self.orange, hatch='*', fill=True, lw=.9)
        ax1.axhline(obs_12[0], color=self.orange, ls='--')
        ax1.plot(number_ex, peak_10, color=self.blue, lw=2, label=r'$\mathrm{9/2\to 9/2}$')  #**self.ch_dict(self.point_style, {'color': self.blue}))
        ax1.axhspan(obs_10[0]-obs_10[1], obs_10[0]+obs_10[1], alpha=0.1, color=self.blue, hatch='|', fill=True, lw=.9)
        ax1.axhline(obs_10[0], color=self.blue, ls='--')
        ax1.plot(number_ex, peak_9, color=self.dark_green, lw=2, label=r'$\mathrm{7/2\to 9/2}$')  #**self.ch_dict(self.point_style, {'color': self.green}))
        ax1.axhspan(obs_9[0]-obs_9[1], obs_9[0]+obs_9[1], alpha=0.1, color=self.dark_green, hatch='/', fill=True, lw=.9)
        ax1.axhline(obs_9[0], color=self.dark_green, ls='--')

        ''' axes '''
        ax1.set_xlabel('number of excitations')
        ax1.set_ylabel('intensity relative to theory')

        ax1.legend(bbox_to_anchor=(0.5, 1.1), loc="center", ncol=4, columnspacing=1.2, handletextpad=0.4) #, prop=self.unic)  # title='Scaler', columnspacing=0.5,

        plt.savefig(folder + 'pumping' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def ce_population(self):
        """

        :return:
        """
        # Specify in which folder input and output should be found
        folder = os.path.join(self.fig_dir, 'ChargeExchange\\Population\\')

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # create second axis on top of first
        # define output size of figure
        width, height = 1, 0.3
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        ''' import the data '''
        data = np.loadtxt(glob(os.path.join(folder, 'Ni_on_Na*'))[0], delimiter=',', skiprows=1)
        bare_cs = np.loadtxt(glob(os.path.join(folder, 'barecs*'))[0], delimiter=',')

        en = data[:, 0]
        cs = data[:, 1]
        ip = data[:, 2]
        fp = data[:, 3]

        ax2.plot(bare_cs[:, 0], bare_cs[:, 1], **self.fit_style)

        markerline_ini, stemlines_ini, baseline_ini = ax1.stem(en, ip, label='initial')
        plt.setp(baseline_ini, 'color', self.black)
        plt.setp(markerline_ini, 'color', self.blue)
        plt.setp(stemlines_ini, 'color', self.blue)
        markerline_fin, stemlines_fin, baseline_fin = ax1.stem(en, fp, '--', label='final')
        plt.setp(baseline_fin, 'color', self.black)
        plt.setp(markerline_fin, 'color', self.green)
        plt.setp(stemlines_fin, 'color', self.green)

        # add arrows to indicate the spectroscopy levels
        ax1.arrow(0.025, 16, 0, -1, color=self.orange,
                  width=0.03, length_includes_head=True, head_width=0.08, head_length=0.4, overhang=0)
        ax1.arrow(3.54, 5, 0, -1, color=self.purple,
                  width=0.03, length_includes_head=True, head_width=0.08, head_length=0.4, overhang=0)

        ax1.set_xlim((-0.1, 5))
        ax1.set_xlabel('level energy / eV')
        ax1.axes.tick_params(axis='x', direction='in',
                             bottom=True, top=False, labelbottom=True, labeltop=False)

        ax1.set_ylabel('level population / %')
        ax2.set_ylabel('bare cross section / $10^{-15}$cm$^2$', color=self.red)
        ax2.tick_params(axis='y', labelcolor=self.red, color=self.red, which='both')
        ax2.set_ylim((0, 10))

        ax1.legend()

        plt.savefig(folder + 'population' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def ce_simulation(self):
        """

        :return:
        """
        # Specify in which folder input and output should be found
        folder = os.path.join(self.fig_dir, 'ChargeExchange\\Simulation\\')

        fig, ax1 = plt.subplots()
        ax2 = ax1.twinx()  # create second axis on top of first
        # define output size of figure
        width, height = 1, 0.4
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        ''' import the data '''
        react = 'pH'  # 'NiNa
        data_prob = np.loadtxt(glob(os.path.join(folder, '{}_prob*'.format(react)))[0], delimiter='\t')
        data_cs = np.loadtxt(glob(os.path.join(folder, '{}_cs*'.format(react)))[0], delimiter='\t')

        log_x1 = data_prob[:, 0]
        d1 = data_prob[:, 1]
        log_x2 = data_cs[:, 0]
        d2 = data_cs[:, 1]

        ax1.plot(log_x1, d1*100, **self.ch_dict(self.fit_style, {'color': self.blue}))
        ax2.plot(log_x2, d2*10**15, **self.ch_dict(self.fit_style, {'color': self.red}))
        b1_abs = 0
        if react == 'pH':
            b1_abs = 2.90221*10**-8
            b1 = np.log(b1_abs)
            b0 = np.log(10**-10)
            rectp = mpl.patches.Rectangle((b0, 0), width=b1-b0, height=50,
                                          color=self.blue, linestyle='--', fill=False, lw=2)
            # ax1.axhspan(0, 50, xmin=b2, xmax=np.log(b1))  # 0, np.log(b1)
            ax1.add_patch(rectp)

            # now plot the integration over the approximation
            ax2.plot(log_x1,
                     np.where(log_x1 < np.log(b1_abs),  np.pi/2*np.exp(log_x1)**2*10**15, np.pi/2*b1_abs**2*10**15),
                     c=self.red, ls=':', lw=2)

        # add arrows to indicate the truncation used
        ax2.arrow(np.log(10**-7), 2.2, 0, -0.15, color=self.red,
                  width=0.03, length_includes_head=True, head_width=0.08, head_length=0.04, overhang=0)

        # x-axis label
        xbox1 = mpl.offsetbox.TextArea('impact parameter b / cm', textprops=dict(color=self.blue, size=11))
        xbox2 = mpl.offsetbox.TextArea(r'integration limit $\mathregular{b_{max}}$ / cm', textprops=dict(color=self.red, size=11))
        xbox = mpl.offsetbox.HPacker(children=[xbox1, xbox2], align="center", pad=0, sep=20)

        anchored_xbox = mpl.offsetbox.AnchoredOffsetbox(loc=3, child=xbox, pad=0., frameon=False,
                                                        bbox_to_anchor=(0.2, -0.12),
                                                        bbox_transform=ax1.transAxes, borderpad=0.)
        ax1.add_artist(anchored_xbox)

        # ax1.set_xlabel('impact parameter b / cm', color=self.blue)
        # ax1.xaxis.set_label_coords(0.3, -0.1)
        ax1.set_xticks(np.log([10**-9, 10**-8, b1_abs, 10**-7, 10**-6]))
        ax1.set_xticks(np.log([n*10**-9 for n in range(9)]
                              + [n*10**-8 for n in range(9)]
                              + [n*10**-7 for n in range(9)]), minor=True)
        ax1.set_xticklabels([r'$10^{-9}$', '$10^{-8}$', 'b$_1$', '$10^{-7}$', '$10^{-6}$'])
        ax1.set_xlim((np.log(10**-9), np.log(10**-6)))

        ax1.set_ylabel('electron transfer probability / %', color=self.blue)
        ax1.tick_params(axis='y', labelcolor=self.blue, color=self.blue, which='both', direction='out')
        ax1.set_ylim((0, 100))

        ax2.set_ylabel('bare cross section / $10^{-15}$cm$^2$', color=self.red)
        ax2.tick_params(axis='y', labelcolor=self.red, color=self.red, which='both', direction='out')

        plt.savefig(folder + 'simulation' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def ce_efficiency(self):
        """

        :return:
        """
        # Specify in which folder input and output should be found
        folder = os.path.join(self.fig_dir, 'ChargeExchange\\Neutralization\\')

        fig, ax1 = plt.subplots()
        # define output size of figure
        width, height = 0.4, 0.3
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        ''' import the data '''
        data = np.loadtxt(glob(os.path.join(folder, 'data*'))[0], delimiter='\t')

        temp = data[:, 0]
        n_eff = data[:, 5]

        def neutr_eff(T, a):
            # as done in Klose.2012 with data from Haynes.2013
            T = T+273  # in Kelvin
            A = 8.489
            B = -7813
            C = -0.8253
            return (1-np.exp(-10**(5.006+A+B/T+C*np.log(T))*a))*100

        def neutr_eff_odr(beta, T):
            # as done in Klose.2012 with data from Haynes.2013
            T = T+273  # in Kelvin
            A = 8.489
            B = -7813
            C = -0.8253
            return (1-np.exp(-10**(5.006+A+B/T+C*np.log(T))*beta))*100

        popt, pcov = curve_fit(neutr_eff, temp, n_eff)  # uncertainties?
        perr = np.sqrt(np.diag(pcov))

        # fit_dat = odr.RealData(temp, n_eff, sx=np.full(temp.shape, 10), sy=np.array([0.1,0.1,10,10,10,10, 10]))
        # fit_mod = odr.Model(neutr_eff_odr)
        # fit_odr = odr.ODR(fit_dat, fit_mod, [6])
        # fit_odr.set_job(fit_type=0)
        # output = fit_odr.run()
        # print(output.beta, output.sd_beta)
        # popt = output.beta
        # perr = output.sd_beta

        pltrange = 700

        ax1.plot(np.arange(pltrange), neutr_eff(np.arange(pltrange), popt[0]), color=self.blue)
        ax1.fill_between(np.arange(pltrange),
                         neutr_eff(np.arange(pltrange), popt[0]-3*perr[0]),
                         neutr_eff(np.arange(pltrange), popt[0]+3*perr[0]),
                         alpha=0.2, color=self.blue)
        ax1.plot(temp, n_eff, **self.ch_dict(self.point_style, {'color': self.red, 'markersize': 6}))

        ax1.set_xlabel('temperature / °C')
        ax1.set_xlim(300, pltrange)
        plt.setp(ax1.get_xticklabels(), rotation=90)

        ax1.set_ylabel('charge exchange efficiency / %')
        ax1.set_ylim((0, 100))

        plt.savefig(folder + 'neutralization' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def voltage_deviations(self):
        """

        :return:
        """
        # Specify in which folder input and output should be found
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\Rebinning\\')

        # # Create plots for 3D and colorbar
        widths = [0.1, 1]
        heights = [0.33, 1, 0.42]

        # Create plots for trs and projections
        f = plt.figure()
        spec = mpl.gridspec.GridSpec(nrows=3, ncols=2,
                                     width_ratios=widths, height_ratios=heights,
                                     wspace=-0.23, hspace=0.07)
        ax = f.add_subplot(spec[0:3, 1], projection='3d')
        ax2 = f.add_subplot(spec[1, 0])

        # define output size of figure
        width, height = 0.5, 0.4
        f.set_size_inches((self.w_in * width, self.h_in * height))

        ''' import the data '''
        volt_dev_arr = np.loadtxt(glob(os.path.join(folder, 'volt_data*'))[0])
        # get x and y shapes and create meshgrid for data
        x = np.arange(volt_dev_arr.shape[1])
        y = np.arange(volt_dev_arr.shape[0])
        X, Y = np.meshgrid(x, y)
        Z = volt_dev_arr

        # higher resolution meshgrid
        new_x = np.arange(volt_dev_arr.shape[1])
        new_y = np.arange(volt_dev_arr.shape[0])
        newX, newY = np.mgrid[0:59:118j, 0:30:60j]  # np.meshgrid(new_x, new_y)
        tck = interpolate.bisplrep(X, Y, Z, s=1)
        newZ = interpolate.bisplev(newX[:, 0], newY[0, :], tck)

        # define fit function (a simple plane)
        def plane(x, y, mx, my, coff):
            my = 0
            return x * mx + y * my + coff

        def _plane(M, *args):
            """
            2D function generating a 3D plane
            :param M: xdata parameter, 2-dimensional array of x and y values
            :param args: slope and offset passed to plane function
            :return: array of z-values
            """
            x, y = M  # x: steps, y: scans
            arr = np.zeros(x.shape)
            arr += plane(x, y, *args)
            return arr

        # Define start parameters [mx, my, offset]
        p0 = [0, 0, Z[0, 0]]
        # make 1-D data (necessary to use curve_fit)
        xdata = np.vstack((X.ravel(), Y.ravel()))
        # fit
        popt, pcov = curve_fit(_plane, xdata, Z.ravel(), p0)

        # store results
        # offset_adj = popt[1] * self.nrOfScans / 2  # average over time dependence
        # self.volt_correct = ((popt[2] + offset_adj) * 1000, popt[0] * 1000)  # (avg offset step 0, slope per step)

        # calculate average and maximum deviation after correction
        fit_plane = plane(X, Y, 0, 0, 0)
        standard_v_dev = np.sqrt(np.square(1000 * (fit_plane - Z)).mean())
        print('standard deviation before correction: ' + str(standard_v_dev))
        # calculate average and maximum deviation after correction
        fit_plane = plane(X, Y, *popt)
        standard_v_dev = np.sqrt(np.square(1000 * (fit_plane - Z)).mean())
        max_v_dev = 1000 * (fit_plane - Z).max()

        # Plot the surface.
        ax.plot_surface(newX, newY, np.where(plane(newX, newY, *popt) < newZ, 1000 * plane(newX, newY, *popt), np.nan), rstride=1,
                        cstride=1, linewidth=0, antialiased=True, alpha=0.7)
        surf = ax.plot_surface(X, Y, 1000 * Z, cmap=self.custom_cmap, rstride=1, cstride=1,
                               linewidth=0, antialiased=True, vmin=0.045, vmax=0.315)
        ax.plot_surface(newX, newY, np.where(plane(newX, newY, *popt) > newZ, 1000 * plane(newX, newY, *popt), np.nan),
                        rstride=1, cstride=1, linewidth=0, antialiased=True, alpha=0.7)
        ax.view_init(elev=5., azim=-35.)
        ax.autoscale_view(tight=True)

        # Annotate the Regions
        ax.text(40, 30, -0.07, 'time', 'y')
        ax.text(-23, 0, -0.055, 'frequency', 'x')
        # add arrows to indicate the spectroscopy levels
        ax.arrow(-0.01, -0.092, 0.09, 0.011, color=self.orange,
                  width=0.002, length_includes_head=True, head_width=0.006, head_length=0.007, overhang=0)
        ax.arrow(-0.01, -0.092, -0.084, 0.0225, color=self.orange,
                 width=0.002, length_includes_head=True, head_width=0.006, head_length=0.007, overhang=0)



        ax.set_ylabel('scan number', labelpad=-1)
        #ax.set_zlabel('Voltage deviation / V')

        # ax.set_axis_off()
        # ax.set_xlabel('DAC set value / V', labelpad=0)
        ax.text(-20, 0, -0.02, 'DAC set value / V', 'x')
        plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='center', verticalalignment='center')
        dac_ticks = [5, 15, 25, 35, 45, 55]
        ax.set_xticks(dac_ticks)
        ax.set_xticklabels(['{:.0f}'.format(-45 + 1 * x) for x in dac_ticks])
        ax.tick_params(axis='x', pad=0, direction='inout', length=10)
        ax.tick_params(axis='y', pad=-5)
        ax.tick_params(axis='z', labelleft=False, labelright=False)
        dev_ticks = ax.get_zticks()

        # Add a color bar which maps values to colors.
        col = plt.colorbar(surf, cax=ax2)
        ax2.axes.tick_params(axis='y', direction='inout', left=True, right=True, labelleft=True, labelright=False)
        ax2.yaxis.set_label_position('left')
        col.set_ticks(dev_ticks)
        ax2.set_ylabel('DAC deviation / V')

        f.savefig(folder + 'volt_dev_3D' + self.ffe, dpi=self.dpi, bbox_inches='tight', pad_inches=0.2)
        plt.close()
        plt.clf()

        plt.imshow(volt_dev_arr, cmap=self.custom_cmap, interpolation='nearest')
        if popt[2] == 1:
            # error while fitting
            plt.title('DBEC_' + '6502' + ': Deviation from dac set voltage.\n'
                                             '!error while fitting!')
        else:
            plt.title('DBEC_' + '6502' + ': Deviation from dac set voltage.\n'
                                             'offset: {0:.4f}V\n'
                                             'step_slope: {1:.4f}V\n'
                                             'scan_slope: {2:.4f}V\n'
                                             'standard deviation after correction: {3:.4f}V\n'
                                             'maximum deviation after correction: {4:.4f}V.'
                      .format(popt[2] * 1000, popt[0] * 1000, popt[1] * 1000, standard_v_dev, max_v_dev))
        plt.xlabel('step number')
        plt.ylabel('scan number')

        plt.colorbar()
        plt.savefig(folder + 'volt_dev_2D' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def time_res_atoms(self):
        """
        Plot time resolved data from one run as an example
        :return:
        """
        folder = os.path.join(self.fig_dir, 'ChargeExchange\\Resonances\\')
        file = 'Ca_Ion2020_trs_run203.xml'

        # Import Measurement from file using Importer
        midtof = 20.5
        gatewidth = 5
        t_axis_offset = 0  # 18 offset to scale to compare with different time plots
        t_plotrange = (18, 28)
        x_plotrange = (-85, -5)

        xml = XMLImporter(os.path.join(folder, file), x_as_volt=True,
                          softw_gates=[[-5000, 1000, midtof - gatewidth / 2, midtof + gatewidth / 2],
                                       [-5000, 1000, midtof - gatewidth / 2, midtof + gatewidth / 2]])
        xml.preProc(os.path.join(folder, 'Ca_CEC_collected.sqlite'))
        trs = xml.time_res[0]  # array of dimensions tracks, pmts, steps, bins
        t_proj = xml.t_proj[0]
        t_proj_err = np.sqrt(t_proj)
        v_proj = xml.cts[0]
        v_proj_err = xml.err[0]
        x_axis = -np.array(xml.x[0]) + xml.accVolt
        t_axis = xml.t[0]-t_axis_offset
        step_width = xml.stepSize[0]*xml.lineMult
        # Get sizes of arrays
        scal_data_size, x_data_size, t_data_size = np.shape(trs)
        # create mesh for time-resolved plot
        X, Y = np.meshgrid(np.arange(x_data_size), np.arange(t_data_size))
        # cts data is stored in data_array. Either per scaler [ScalerNo, Y, X]
        Z = trs[0][X, Y]
        # Z = data_arr.sum(axis=0)[X, Y]  # or summed for all scalers

        # Create plots for trs and projections
        f = plt.figure()
        widths = [0.05, 1]
        heights = [0.4, 1]
        spec = mpl.gridspec.GridSpec(nrows=2, ncols=2,
                                     width_ratios=widths, height_ratios=heights,
                                     wspace=0.03, hspace=0.05)

        ax_col = f.add_subplot(spec[1, 0])
        ax_trs = f.add_subplot(spec[1, 1])
        # ax_tpr = f.add_subplot(spec[1, 2], sharey=ax_trs)
        ax_vpr = f.add_subplot(spec[0, 1], sharex=ax_trs)

        # define output size of figure
        width, height = 1, 0.3
        # f.set_dpi(300.)
        f.set_size_inches((self.w_in * width, self.h_in * height))

        # create timeresolved plot
        im = ax_trs.pcolormesh(x_axis, t_axis, Z, cmap=self.custom_cmap)
        # im = ax_trs.imshow(Z, cmap=custom_cmap, interpolation='none', aspect='auto')  # cmap=pyl.cm.RdBu
        # # add arrows to indicate the spectroscopy levels
        # ax_trs.arrow(-25, 25, -3, -1, color=self.orange,
        #              width=0.4, length_includes_head=True, head_width=0.8, head_length=1, overhang=0)
        # work on x axis
        ax_trs.xaxis.set_ticks_position('top')
        ax_trs.axes.tick_params(axis='x', direction='out',
                                bottom=True, top=False, labelbottom=True, labeltop=False)
        ax_trs.set_xlabel('DAC scan voltage/ V')
        ax_trs.set_xlim(x_plotrange)  # (x_axis[0], x_axis[-1])
        # work on y axis
        ax_trs.set_ylim(t_plotrange)
        ax_trs.set_ylabel('time / µs')
        ax_trs.yaxis.set_label_position('right')
        ax_trs.axes.tick_params(axis='y', direction='out',
                                left=False, right=True, labelleft=False, labelright=True)
        f.colorbar(im, cax=ax_col)  # create plot legend
        ax_col.axes.tick_params(axis='y', direction='in',
                                left=True, right=True, labelleft=True, labelright=False)
        ax_col.yaxis.set_label_position('left')
        ax_col.set_ylabel('cts/arb.u.')

        # create Voltage projection
        # ax_vpr.errorbar(x_axis, v_proj[0], v_proj_err[0], **self.data_style)  # x_step_projection_sc0
        ax_vpr.bar(x_axis, v_proj[0], width=step_width, color=self.grey, edgecolor=self.grey)
        ax_vpr.axes.tick_params(axis='x', direction='in',
                                top=True, bottom=True,
                                labeltop=False, labelbottom=False)
        ax_vpr.axes.tick_params(axis='y', direction='in',
                                left=True, right=False, labelleft=True)
        ax_vpr.set_ylim((3900, 6100))
        ax_vpr.set_yticks([4000, 5000, 6000])
        ax_vpr.set_yticklabels(['4k', '5k', '6k'])
        ax_vpr.set_ylabel('cts/arb.u.')

        # create time projection
        # ax_tpr.errorbar(t_proj[0], t_axis, xerr=t_proj_err[0],
        #                 **self.data_style)  # y_time_projection_sc0
        # ax_tpr.axes.tick_params(axis='x', direction='out',
        #                         top=False, bottom=True, labeltop=False, labelbottom=True)
        # plt.setp(ax_tpr.get_xticklabels(), rotation=-90)
        # ax_tpr.axes.tick_params(axis='y', direction='in',
        #                         left=True, right=True, labelleft=False, labelright=True)
        # # ax_tpr.set_xticks([1000, 2000, 3000])
        # # ax_tpr.set_xticklabels(['1k', '2k', '3k'])
        # ax_tpr.set_xlabel('cts/arb.u.')
        # ax_tpr.yaxis.set_label_position('right')
        # ax_tpr.set_ylabel('time/bins')

        # add horizontal lines for timegates

        ax_trs.axhline((midtof - gatewidth / 2)-t_axis_offset, **self.ch_dict(self.fit_style, {'ls': '--', 'lw': 4}))
        # ax_tpr.axhline((midtof - gatewidth / 2)-t_axis_offset, **self.fit_style)
        ax_trs.axhline((midtof + gatewidth / 2)-t_axis_offset, **self.ch_dict(self.fit_style, {'ls': '--', 'lw': 4}))
        # ax_tpr.axhline((midtof + gatewidth / 2)-t_axis_offset, **self.fit_style)

        # add the TILDA logo top right corner
        # logo = mpl.image.imread(os.path.join(folder, 'Tilda256.png'))
        # ax_log = f.add_subplot(spec[0, 2])
        # ax_log.axis('off')
        # ax_log.imshow(logo)

        # f.tight_layout()
        plt.savefig(folder + 'TRS_atoms' + '.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def time_res_ions(self):
        """
        Plot time resolved data from one run as an example
        :return:
        """
        folder = os.path.join(self.fig_dir, 'ChargeExchange\\Resonances\\')
        file = 'Ca_Ion_DecNoonRes_trs_run055.xml'
        # file = '40Ca_la_trs_run605.xml'

        # Import Measurement from file using Importer
        midtof = 210  #215
        gatewidth = 30
        t_axis_offset = 0  # 190
        t_plotrange = (190, 225)
        x_plotrange = (-60, 0)

        xml = XMLImporter(os.path.join(folder, file), x_as_volt=True,
                          softw_gates=[[-500, 500, midtof - gatewidth / 2, midtof + gatewidth / 2],
                                       [-500, 500, midtof - gatewidth / 2, midtof + gatewidth / 2]])
        xml.preProc(os.path.join(folder, 'Ca_CEC_collected.sqlite'))
        trs = xml.time_res[0]  # array of dimensions tracks, pmts, steps, bins
        t_proj = xml.t_proj[0]
        t_proj_err = np.sqrt(t_proj)
        v_proj = xml.cts[0]
        v_proj_err = xml.err[0]
        x_axis = -np.array(xml.x[0]) + xml.accVolt
        t_axis = xml.t[0] - t_axis_offset
        step_width = xml.stepSize[0] * 50
        # Get sizes of arrays
        scal_data_size, x_data_size, t_data_size = np.shape(trs)
        # create mesh for time-resolved plot
        X, Y = np.meshgrid(np.arange(x_data_size), np.arange(t_data_size))
        # cts data is stored in data_array. Either per scaler [ScalerNo, Y, X]
        Z = trs[0][X, Y]
        # Z = data_arr.sum(axis=0)[X, Y]  # or summed for all scalers

        # Create plots for trs and projections
        f = plt.figure()
        widths = [0.05, 1]
        heights = [0.4, 1]
        spec = mpl.gridspec.GridSpec(nrows=2, ncols=2,
                                     width_ratios=widths, height_ratios=heights,
                                     wspace=0.03, hspace=0.05)

        ax_col = f.add_subplot(spec[1, 0])
        ax_trs = f.add_subplot(spec[1, 1])
        # ax_tpr = f.add_subplot(spec[1, 2], sharey=ax_trs)
        ax_vpr = f.add_subplot(spec[0, 1], sharex=ax_trs)

        # define output size of figure
        width, height = 1, 0.3
        # f.set_dpi(300.)
        f.set_size_inches((self.w_in * width, self.h_in * height))

        # create timeresolved plot
        im = ax_trs.pcolormesh(x_axis, t_axis, Z, cmap=self.custom_cmap)
        # im = ax_trs.imshow(Z, cmap=custom_cmap, interpolation='none', aspect='auto')  # cmap=pyl.cm.RdBu
        # add arrows to indicate the spectroscopy levels
        ax_trs.arrow(-20, 220, -5, -5, color='w',
                     width=0.4, length_includes_head=True, head_width=1.4, head_length=1, overhang=0)
        # work on x axis
        ax_trs.xaxis.set_ticks_position('top')
        ax_trs.axes.tick_params(axis='x', direction='out',
                                bottom=True, top=False, labelbottom=True, labeltop=False)
        ax_trs.set_xlabel('DAC scan voltage/ V')
        ax_trs.set_xlim(x_plotrange)  # (x_axis[0], x_axis[-1])
        # work on y axis
        ax_trs.set_ylim(t_plotrange)
        ax_trs.set_ylabel('time / µs')
        ax_trs.yaxis.set_label_position('right')
        ax_trs.axes.tick_params(axis='y', direction='out',
                                left=False, right=True, labelleft=False, labelright=True)
        f.colorbar(im, cax=ax_col)  # create plot legend
        ax_col.axes.tick_params(axis='y', direction='in',
                                left=True, right=True, labelleft=True, labelright=False)
        ax_col.yaxis.set_label_position('left')
        ax_col.set_ylabel('cts/arb.u.')

        # create Voltage projection
        # ax_vpr.errorbar(x_axis, v_proj[0], v_proj_err[0], **self.data_style)  # x_step_projection_sc0
        ax_vpr.bar(x_axis, v_proj[0], width=step_width, color=self.grey, edgecolor=self.grey)
        ax_vpr.axes.tick_params(axis='x', direction='in',
                                top=True, bottom=True,
                                labeltop=False, labelbottom=False)
        ax_vpr.axes.tick_params(axis='y', direction='in',
                                left=True, right=False, labelleft=True)
        ax_vpr.set_ylim((4500, 20500))
        ax_vpr.set_yticks([5000, 10000, 15000, 20000])
        ax_vpr.set_yticklabels(['5k', '10k', '15k', '20k'])
        ax_vpr.set_ylabel('cts/arb.u.')

        # create time projection
        # ax_tpr.errorbar(t_proj[0], t_axis, xerr=t_proj_err[0],
        #                 **self.data_style)  # y_time_projection_sc0
        # ax_tpr.axes.tick_params(axis='x', direction='out',
        #                         top=False, bottom=True, labeltop=False, labelbottom=True)
        # plt.setp(ax_tpr.get_xticklabels(), rotation=-90)
        # ax_tpr.axes.tick_params(axis='y', direction='in',
        #                         left=True, right=True, labelleft=False, labelright=True)
        # # ax_tpr.set_xticks([1000, 2000, 3000])
        # # ax_tpr.set_xticklabels(['1k', '2k', '3k'])
        # ax_tpr.set_xlabel('cts/arb.u.')
        # ax_tpr.yaxis.set_label_position('right')
        # ax_tpr.set_ylabel('time/bins')

        # add horizontal lines for timegates
        ax_trs.axhline((midtof - gatewidth / 2) - t_axis_offset, **self.ch_dict(self.fit_style, {'ls': '--', 'lw': 4}))
        # ax_tpr.axhline((midtof - gatewidth / 2)-t_axis_offset, **self.fit_style)
        ax_trs.axhline((midtof + gatewidth / 2) - t_axis_offset, **self.ch_dict(self.fit_style, {'ls': '--', 'lw': 4}))
        # ax_tpr.axhline((midtof + gatewidth / 2)-t_axis_offset, **self.fit_style)

        # add the TILDA logo top right corner
        # logo = mpl.image.imread(os.path.join(folder, 'Tilda256.png'))
        # ax_log = f.add_subplot(spec[0, 2])
        # ax_log.axis('off')
        # ax_log.imshow(logo)

        # f.tight_layout()
        plt.savefig(folder + 'TRS_ions' + '.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def draw_table_of_nuclides(self):
        # Define width and height
        w = 1/100
        h = 1/100

        allRects = []

        isotopes = {'Ni': {'Z': 28, 'N': (48, 80)}}

        fig, ax = plt.subplots(1)

        for iso, props in isotopes.items():
            Z = props['Z']
            N_nums = props['N']

            for N in range(N_nums[0], N_nums[1], 1):
                # Get properties


                # Plot Rectangle
                rec = Rectangle((N/100, Z/100), w, h, facecolor=(233/255, 80/255, 62/255))
                ax.add_patch(rec)
                allRects.append(rec)

                # Plot Text
                ax.annotate(iso, (N/100, Z/100))

        # Collect all patches
        #pc = PatchCollection(allRects)


        #ax.add_collection(pc)
        ax.set_axis_off()

        plt.show()

    def plot_king_plot(self):
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\KingPlot\\')

        fig, ax = plt.subplots(1)
        # define output size of figure
        width, height = 1, 0.3
        # f.set_dpi(300.)
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        x_list = []  # list of reduced masses (x-axis)

        ''' import the data '''
        nOfElectrons = 28.  # for mass-scaling factor nuclear mass calculation
        ionizationEnergy = 41356  # in keV from NIST Ionization Energy data for very precise nuclear mass determination

        plot_these = ['BECOLA']
        king_dict = {'COLLAPS': {'F': {}, 'K': {}, 'Alpha': {}, 'color': self.red},
                     'BECOLA': {'F': {}, 'K': {}, 'Alpha': {}, 'color': self.blue},}

        king_data = pd.read_csv(glob(os.path.join(folder, 'KingLit*'))[0], delimiter='\t',
                           index_col=0, skiprows=[])
        iso_data = pd.read_csv(glob(os.path.join(folder, 'IsoData*'))[0], delimiter='\t',
                           index_col=0, skiprows=[])

        m_list = []
        m_d_list = []
        r_list = []
        r_d_list = []

        # calculate mass-scaled delta rms charge radii
        for indx, row in iso_data.iterrows():
            # get masses and errors
            m_inp = row['mass']
            m, m_d, *_ = re.split('[()]', m_inp)
            sign_digit = len(m.split('.')[1])  # get the significant digits of the value
            m_list.append(float(m)  # make nuclear mass
                          - nOfElectrons * sc.physical_constants['electron mass in u'][0]
                          + ionizationEnergy * sc.e / (sc.atomic_mass * sc.c ** 2))
            m_d_list.append(float(m_d) / 10 ** sign_digit)
            # get radius
            r_inp = row['delta_rad_FRI']
            if r_inp != '--':
                # get radii and error
                r, r_d, *_ = re.split('[()]', r_inp)
                sign_digit = len(r.split('.')[1])  # get the significant digits of the value
                r_list.append(float(r))
                r_d_list.append(float(r_d) / 10 ** sign_digit)
            else:
                r_list.append('--')
                r_d_list.append('--')

        iso_data['m'] = m_list
        iso_data['m_d'] = m_d_list
        iso_data['r'] = r_list
        iso_data['r_d'] = r_d_list

        mu_list = []
        mu_d_list = []
        for iso, row in iso_data.iterrows():
            m_iso = row['m']
            m_iso_d = row['m_d']
            m_ref = iso_data.loc['60Ni', 'm']
            m_ref_d = iso_data.loc['60Ni', 'm_d']

            mu = (m_iso - m_ref) / (m_iso * m_ref)
            mu_d = np.sqrt(np.square(m_iso_d / m_iso ** 2) + np.square(m_ref_d / m_ref ** 2))

            # add to list of x-values
            mu_list.append(mu)
            mu_d_list.append(mu_d)
        iso_data['mu'] = mu_list
        iso_data['mu_d'] = mu_d_list

        # make x-axis or iterate over items again?


        x_plotrange = (240, 580)
        x_arr = np.arange(x_plotrange[0], x_plotrange[1], 1)

        # get the isotope shifts from our analysis
        shifts = []
        thisPoint = None
        isolist = ['55Ni', '56Ni', '58Ni', '60Ni']  # ['54Ni', '55Ni', '56Ni', '58Ni', '60Ni']
        # isolist.remove(self.ref_iso)
        # for iso in isolist:
        #     m_iso, m_iso_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''',
        #                                                    (iso[:4],))
        #     m_ref, m_ref_d = self.get_iso_property_from_db('''SELECT mass, mass_d FROM Isotopes WHERE iso = ?''',
        #                                                    (self.ref_iso[:4],))
        #     mu = (m_iso - m_ref) / (m_iso * m_ref)
        #     mu_d = np.sqrt(np.square(m_iso_d / m_iso ** 2) + np.square(m_ref_d / m_ref ** 2))
        #
        #     iso_shift = self.results[iso]['final']['shift_iso-{}'.format(self.ref_iso[:2])]['vals'][0]
        #     iso_shift_d = self.results[iso]['final']['shift_iso-{}'.format(self.ref_iso[:2])]['d_stat'][0]
        #     iso_shift_d_syst = self.results[iso]['final']['shift_iso-{}'.format(self.ref_iso[:2])]['d_syst'][0]
        #     shifts.append((iso_shift / mu / 1000, iso_shift_d / mu / 1000, iso_shift_d_syst / mu / 1000, iso))
        #
        #     if iso in self.delta_lit_radii_60 and not iso == self.ref_iso:
        #         delta_rms = self.delta_lit_radii_60[iso]
        #         r = delta_rms[0] / mu
        #         r_d = np.sqrt((delta_rms[1] / mu) ** 2 + (delta_rms[0] * mu_d / mu ** 2) ** 2)
        #         s = iso_shift / mu / 1000
        #         s_d = np.sqrt(
        #             ((iso_shift_d + iso_shift_d_syst) / mu / 1000) ** 2 + ((iso_shift) * mu_d / mu ** 2 / 1000) ** 2)
        #         thisPoint = (r, r_d, s, s_d)

        # add a band for each of our online measured isotope shifts
        for indx, row in iso_data.iterrows():
            # get radius
            rad_inp = row['iso_shift_ONL']
            if rad_inp != '--':
                r, r_d, *_ = re.split('[()]', rad_inp)
                sign_digit = len(r.split('.')[1])  # get the significant digits of the value
                isomu = float(r) / iso_data.loc[indx, 'mu'] / 1000  # in GHz
                isomu_d = float(r_d) / 10 ** sign_digit / iso_data.loc[indx, 'mu'] / 1000  # in GHz
                # plot error band for this line
                plt.axhspan(isomu - isomu_d,
                            isomu + isomu_d,
                            facecolor='black', alpha=0.2)
                ax.annotate(r'$^\mathregular{{{}}}$Ni'.format(indx[:2]), (x_plotrange[0]+5, isomu - 5))

        def _kingLine(x, k, f, a):
            return (k + f * (x)) / 1000  # in GHz

        def _kingLower(x, k, k_d, f, f_d, a):
            xa = _kingLine(x, k + k_d, f + f_d, a)
            xb = _kingLine(x, k + k_d, f - f_d, a)
            xc = _kingLine(x, k - k_d, f + f_d, a)
            xd = _kingLine(x, k - k_d, f - f_d, a)
            ret_arr = np.zeros(x.shape)
            for i in range(len(x)):
                ret_arr[i] = min(xa[i], xb[i], xc[i], xd[i])
            return ret_arr

        def _kingUpper(x, k, k_d, f, f_d, a):
            xa = _kingLine(x, k + k_d, f + f_d, a)
            xb = _kingLine(x, k + k_d, f - f_d, a)
            xc = _kingLine(x, k - k_d, f + f_d, a)
            xd = _kingLine(x, k - k_d, f - f_d, a)
            ret_arr = np.zeros(x.shape)
            for i in range(len(x)):
                ret_arr[i] = max(xa[i], xb[i], xc[i], xd[i])
            return ret_arr

        annotate_iso = []
        x_annotate = []
        y_annotate = []

        # for src, item in king_dict.items():
        for src in plot_these:
            item = king_dict[src]
            # get F
            f_inp = king_data.loc[src, 'F']
            f, f_d, *_ = re.split('[()]', f_inp)
            if '.' in f:
                sign_digit = len(f.split('.')[1])  # get the significant digits of the value
            else:
                sign_digit = 0
            king_dict[src]['F']['val'] = float(f)
            king_dict[src]['F']['d'] = float(f_d) / 10 ** sign_digit
            # get Kalpha
            k_inp = king_data.loc[src, 'Kalpha']
            k, k_d, *_ = re.split('[()]', k_inp)
            if '.' in k:
                sign_digit = len(k.split('.')[1])  # get the significant digits of the value
            else:
                sign_digit = 0
            king_dict[src]['K']['val'] = 1000*float(k)
            king_dict[src]['K']['d'] = 1000*float(k_d) / 10 ** sign_digit
            # get Alpha
            a_inp = king_data.loc[src, 'Alpha']
            king_dict[src]['Alpha']['val'] = float(a_inp)


            # get factors
            alpha = king_dict[src]['Alpha']['val']
            F, F_d = king_dict[src]['F']['val'], king_dict[src]['F']['d']
            Kalpha, Kalpha_d = king_dict[src]['K']['val'], king_dict[src]['K']['d']

            # get a color
            col = item['color']

            # plot line with errors:
            plt.plot(x_arr, _kingLine(x_arr - alpha, Kalpha, F, alpha), '--', c=col, lw=2, label='Linear fit')
            # plot error band for this line
            plt.fill_between(x_arr,
                             _kingLower(x_arr - alpha, Kalpha, Kalpha_d, F, F_d, alpha),
                             _kingUpper(x_arr - alpha, Kalpha, Kalpha_d, F, F_d, alpha),
                             alpha=0.3, edgecolor=col, facecolor=col)

            # plot each reference point from this source
            r_lst = []
            r_d_lst = []
            s_lst = []
            s_d_lst = []

            for iso, row in iso_data.iterrows():
                # get isoshift
                iso_inp = row['iso_shift_{}'.format(src[:3])]
                if iso_inp != '--':
                    isos, isos_d, *_ = re.split('[()]', iso_inp)
                    sign_digit = len(isos.split('.')[1])  # get the significant digits of the value
                    isomu = float(isos) / iso_data.loc[iso, 'mu'] / 1000  # in GHz
                    isomu_d = float(isos_d) / 10 ** sign_digit / iso_data.loc[iso, 'mu'] / 1000  # in GHz
                    # add to plot items
                    s_lst.append(isomu)
                    s_d_lst.append(isomu_d)

                    # get radii
                    r = row['r']
                    r_d = row['r_d']
                    rmu = float(r) / iso_data.loc[iso, 'mu']
                    rmu_d = float(r_d) / iso_data.loc[iso, 'mu']
                    # add to plot items
                    r_lst.append(rmu)
                    r_d_lst.append(rmu_d)

                    if 'COLLAPS' in src or len(plot_these) == 1:
                        # only use Kaufmann values for the annotation:
                        annotate_iso.append(iso)
                        x_annotate.append(rmu)
                        y_annotate.append(isomu)

            plt.errorbar(r_lst, s_lst, xerr=r_d_lst, yerr=s_d_lst, fmt='o', c=self.red, elinewidth=1.5, label='Offline data')

        for i, iso in enumerate(annotate_iso):
            ax.annotate(r'$^\mathregular{{{:.0f}}}$Ni'.format(int(iso[:2])), (x_annotate[i] + 5, y_annotate[i] + 5),
                        color=self.black)

        if thisPoint is not None:
            plt.errorbar(thisPoint[0], thisPoint[2], xerr=thisPoint[1], yerr=thisPoint[3], fmt='ok', label='This Work',
                         elinewidth=1.5)

        ax.set_xlim(x_plotrange)
        ax.set_xlabel(r'$\mu^{-1} \delta \langle r_c^2 \rangle \mathregular{\,/(u\/fm)^2}$')

        ax.set_ylim((780, 1090))
        ax.set_ylabel(r'$\mu^{-1} \delta\nu\/\mathregular{/(u\,GHz)}$')
        # plt.title('King Plot Comparison')
        plt.legend(title='King Plot:', numpoints=1, loc="upper right")
        plt.margins(0.05)

        plt.savefig(folder + 'compare_kings' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def deltarad_chain_errorband(self, dash_missing_data=True):
        # define folder
        folder = os.path.join(self.fig_dir, 'Nickel\\Discussion\\dR2_NickelChain\\')

        fig, ax = plt.subplots(1)
        # define output size of figure
        width, height = 0.8, 0.4
        # f.set_dpi(300.)
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'Ni_Results2_*'))[0]
        results = self.import_results(load_results_from)


        isolist = ['54NiBec', '55Ni', '56Ni', '58Ni', '60Ni', '62NiBec', '64NiBec']  # BECOLA isotopes
        # TODO: 58 and 60 from this or from Offline?
        refiso = '60Ni'
        ref_key = refiso[:4]
        prop_key = 'delta_ms_iso-{}'.format(ref_key[:2])

        thisVals = {key: [results[key]['final'][prop_key]['vals'][0],
                          results[key]['final'][prop_key]['d_stat'][0],
                          results[key]['final'][prop_key]['d_syst'][0]]
                    for key in isolist}

        data_dict = {'BECOLA (Exp)': {'data': thisVals, 'color': self.black}}

        # plot BECOLA values
        src = 'BECOLA (Exp)'
        col = data_dict[src]['color']
        data = data_dict[src]['data']
        keyVals = sorted(data)
        x = []
        y = []
        yerr = []
        ytiltshift = []
        for i in keyVals:
            x.append(int(''.join(filter(str.isdigit, i))))
            y.append(data[i][0])
            yerr.append(data[i][1])  # take only IS contributions here
            ytiltshift.append(data[i][2])  # Fieldshift-Factor, Massshift-Factor uncertainty

        # plt.xticks(rotation=0)
        # ax = plt.gca()
        ax.set_ylabel(r'$\mathbf{\delta \langle r^2 \rangle}\mathregular{\//fm^2}$')
        ax.set_xlabel('A')

        if dash_missing_data:
            # if some isotopes are missing, this dashes the line at these isotopes
            # has no effect when plot_evens_separate is True
            split_x_list = []
            for k, g in groupby(enumerate(x), lambda a: a[0] - a[1]):
                split_x_list.append(list(map(itemgetter(1), g)))
            i = 0
            label_created = False
            for each in split_x_list:
                y_vals = y[i:i + len(each)]
                yerr_vals = yerr[i:i + len(each)]
                if not label_created:  # only create label once
                    plt.errorbar(each, y_vals, yerr_vals, fmt='o', color=col, linestyle='-', label=src)
                    label_created=True
                else:
                    plt.errorbar(each, y_vals, yerr_vals, fmt='o', color=col, linestyle='-')
                # plot dashed lines between missing values
                if len(x) > i + len(each):  # might be last value
                    x_gap = [x[i + len(each) - 1], x[i + len(each)]]
                    y_gap = [y[i + len(each) - 1], y[i + len(each)]]
                    plt.plot(x_gap, y_gap, c=col, linestyle='--')
                i = i + len(each)
        else:
            plt.errorbar(x, y, yerr, fmt='o', color=col, linestyle='-')
        # plot errorband
        ax.fill_between(x,
                         np.array(y) - np.array(ytiltshift),
                         np.array(y) + np.array(ytiltshift),
                         alpha=0.5, edgecolor=col, facecolor=col)

        # plot theory values
        theory_sets = glob(os.path.join(folder, 'data_*'))
        offsets = [0, 0.05, -0.05, 0.1, -0.1, 0.15, -0.15]
        colornum = 1
        markernum = 0
        for num, th in enumerate(theory_sets):
            file = th.split('\\')[-1]
            name = file[5:-4]  # remove data_ and .txt
            name = name.replace('slash', '/')

            data = pd.read_csv(th, delimiter=' ', index_col=0, skiprows=[])

            isos = []
            vals = []
            unc_up = []
            unc_down = []
            for i, row in data.iterrows():
                isos.append(i+offsets[num])
                vals.append(row['val'])
                if 'unc_up' in row:
                    unc_up.append(row['unc_up'])
                    unc_down.append(row['unc_down'])
                elif 'unc' in row:
                    unc_up.append(row['unc'])
                    unc_down.append(row['unc'])
                else:
                    unc_up = None

            if unc_up is not None:
                ax.errorbar(isos, vals, yerr=(unc_down, unc_up), label=name, marker=self.markerlist[markernum],
                             linestyle='', color=self.colorlist[colornum])
            else:  # no uncertainties given
                ax.plot(isos, vals, label=name, marker=self.markerlist[markernum], linestyle='',
                         color=self.colorlist[colornum])
            colornum += 1
            markernum += 1
            if markernum == self.markerlist.__len__():
                markernum = 0


        ax.set_xmargin(0.05)
        # ax.set_xlim((53, 64.5))
        # ax.set_ylim((-0.8, 0.9))

        # sort legend alphabetically but keep experiment on top
        handles, labels = ax.get_legend_handles_labels()
        expi = [labels.index(i) for i in labels if
                'Exp' in i]  # first get experiment value(s), so we can put them at the top.
        handles_print = [handles.pop(i) for i in expi]
        labels_print = [labels.pop(i) for i in expi]
        hl = sorted(zip(handles, labels),
                    key=operator.itemgetter(1))
        handles_sort, labels_sort = zip(*hl)
        handles_print += handles_sort
        labels_print += labels_sort
        plt.legend(handles_print, labels_print, bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1)

        plt.margins(0.1)

        plt.tight_layout(True)
        plt.savefig(folder + 'delta_radii_all' + self.ffe, dpi=self.dpi, bbox_inches='tight')

        ax.set_xlim((53, 60.5))
        ax.set_ylim((-0.7, 0.05))
        plt.savefig(folder + 'delta_radii' + self.ffe, dpi=self.dpi, bbox_inches='tight')

        plt.close()
        plt.clf()

    def absradii_chain_errorband(self, dash_missing_data=True):
        # define folder
        folder = os.path.join(self.fig_dir, 'Nickel\\Discussion\\R_NickelChain\\')

        fig, ax = plt.subplots(1)
        # define output size of figure
        width, height = 1, 0.3
        # f.set_dpi(300.)
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'Ni_Results2_*'))[0]
        results = self.import_results(load_results_from)

        isolist = ['54NiBec', '55Ni', '56Ni', '58Ni', '62NiBec', '64NiBec']  # BECOLA isotopes
        # TODO: 58 and 60 from this or from Offline?
        refiso = '60Ni'
        ref_key = refiso[:4]
        prop_key = 'abs_radii'

        thisVals = {key: [results[key]['final'][prop_key]['vals'][0],
                          results[key]['final'][prop_key]['d_stat'][0],
                          results[key]['final'][prop_key]['d_syst'][0]]
                    for key in isolist}

        data_dict = {'BECOLA (Exp)': {'data': thisVals, 'color': self.black}}

        # plot theory values
        theory_sets = glob(os.path.join(folder, 'data_*'))
        offsets = [0.02, -0.02, 0.04, -0.04, 0.05, 0.1, -0.05, -0.1, 0.06, -0.06 ,0.06, -0.06, 0.1, -0.1, 0.15, -0.15, 0.2, -0.2]
        offsets = [0.04, -0.04, -0.08, 0.02, 0.12, -0.02, -0.16, 0.16]
        markerlist = ['*', 'X',
                      '^', 'v', '>',
                      'p', 'h', '8']
        colorlist = [self.blue, self.blue,
                     self.orange, self.orange, self.orange,
                     self.green, self.green, self.green,
                     self.purple, self.purple, self.purple,
                     self.orange, self.red, self.purple,
                     self.dark_blue, self.dark_green, self.yellow, self.dark_purple]

        for num, th in enumerate(theory_sets):
            file = th.split('\\')[-1]
            name = file[5:-4]  # remove data_ and .txt
            name = name.replace('slash', '/')
            name = name.replace('sat', '_{sat}')
            name = name.replace('go', '_{go}')
            name = name.replace(' ', '\/')

            data = pd.read_csv(th, delimiter=' ', index_col=0, skiprows=[])

            isos = []
            vals = []
            unc_up = []
            unc_down = []
            for i, row in data.iterrows():
                isos.append(i + offsets[num])
                # isos.append(i)
                vals.append(row['val'])
                if 'unc_up' in row:
                    unc_up.append(row['unc_up'])
                    unc_down.append(row['unc_down'])
                elif 'unc' in row:
                    unc_up.append(row['unc'])
                    unc_down.append(row['unc'])
                else:
                    unc_up = None

            if unc_up is not None:
                plt.errorbar(isos, vals, yerr=(unc_down, unc_up), label=r'$%s$' % name, marker=markerlist[num],
                             linestyle='', color=colorlist[num])
            else:  # no uncertainties given
                plt.plot(isos, vals, label=r'$%s$' % name, marker=markerlist[num], linestyle='',
                         color=colorlist[num])

        # plot BECOLA values
        src = 'BECOLA (Exp)'
        col = data_dict[src]['color']
        data = data_dict[src]['data']
        keyVals = sorted(data)
        x = []
        y = []
        yerr = []
        ytiltshift = []
        for i in keyVals:
            x.append(int(''.join(filter(str.isdigit, i))))
            y.append(data[i][0])
            yerr.append(data[i][1])  # take only IS contributions here
            ytiltshift.append(data[i][2])  # Fieldshift-Factor, Massshift-Factor uncertainty

        plt.xticks(rotation=0)
        ax = plt.gca()
        ax.set_ylabel(r'$R\mathregular{_c\//fm}$')
        ax.set_xlabel('A')

        # plot errorband
        plt.fill_between(x,
                         np.array(y) - np.array(ytiltshift),
                         np.array(y) + np.array(ytiltshift),
                         alpha=0.5, edgecolor=None, facecolor=col, label=src)


        # Radii by P.G.Reinhardt
        # Fy_isofit = np.array([3.7178, 3.7097, ])  # 3.7116, 3.6998, 3.6996, 3.7144, 3.7331, 3.7395, 3.7516, 3.7751, 3.8185, 3.8278, 3.8599, 3.8651, 3.8935, 3.8975, 3.9231, 3.9253, 3.9470])  #])  #
        # xFy_isofit = np.array([24, 25, ])  # 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42])
        # xFy_isofit = np.array(xFy_isofit) + 28
        # plt.plot(xFy_isofit, Fy_isofit, c='g', marker='*', markersize=8, linestyle='', linewidth=3, label='Fayans (P.-G. R.)')
        #
        # SVmin = np.array([3.8196, 3.8022, 3.7844, 3.7762, 3.7745, 3.7683, ])  # 3.7657, 3.7752, 3.7668, 3.7743, 3.7783, 3.7872, 3.7947, 3.8114, 3.8173, 3.8294, 3.8409, 3.8528, 3.8637, 3.8753, 3.8857, 3.8971, 3.9040])
        # xSVmin = np.array([20, 21, 22, 23, 24, 25, ])  # 26, 27,  28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42])
        # xSVmin = np.array(xSVmin) + 28
        # plt.plot(xSVmin, SVmin, c='r', marker='p', markersize=5, linestyle='', linewidth=3, label='Skyrme SVmin (P.-G. R.)')

        # IMSRG = np.array([3.7923, 3.8122, 3.83758, 3.853517, 3.875522, 3.88883,
        #                   3.909071, 3.920739, 3.940103, 3.9507136])
        # xIMSRG = np.array([28, 29, 30, 31, 32, 33, 34, 35, 36, 37])
        # xIMSRG = np.array(xIMSRG) + 28
        # plt.plot(xIMSRG, IMSRG, c='orange', marker='s', linewidth=3, label='IM-SRG',
        #          markersize=6)  # TODO: Which potential?!

        ax.set_xmargin(0.05)
        # ax.set_xlim((53, 64.5))
        # ax.set_ylim((3.5, 4))

        # sort legend alphabetically but keep experiment on top
        handles, labels = ax.get_legend_handles_labels()
        expi = [labels.index(i) for i in labels if
                'Exp' in i]  # first get experiment value(s), so we can put them at the top.
        handles_print = [handles.pop(i) for i in expi]
        labels_print = [labels.pop(i) for i in expi]
        hl = sorted(zip(handles, labels),
                    key=operator.itemgetter(1))
        handles_sort, labels_sort = zip(*hl)
        handles_print += handles_sort
        labels_print += labels_sort
        #plt.legend(handles_print, labels_print, bbox_to_anchor=(1.0, 0.5), loc='center left', ncol=1)
        plt.legend(handles_print, labels_print, bbox_to_anchor=(0.5, 1.0), loc='lower center', ncol=3, fontsize='small',
                   fancybox=False, edgecolor=self.black)
        plt.margins(0.1)
        plt.gcf().set_facecolor('w')

        # plt.tight_layout(True)
        # plt.savefig(folder + 'abs_radii_all' + self.ffe, dpi=self.dpi, bbox_inches='tight')

        ax.set_xlim((53.5, 60.5))
        ax.set_ylim((3.6, 3.88))
        plt.savefig(folder + 'abs_radii' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def absradii_chain_errorband_all(self):
        # define folder
        folder = os.path.join(self.fig_dir, 'Nickel\\Discussion\\R_NickelChain\\')

        fig, ax = plt.subplots(1)
        # define output size of figure
        width, height = 1, 0.4 #1.2, 0.4
        # f.set_dpi(300.)
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'Ni_Results2_*'))[0]
        results = self.import_results(load_results_from)

        isolist = ['54NiBec', '55Ni', '56Ni', '58Ni', '62NiBec', '64NiBec']  # BECOLA isotopes
        # TODO: 58 and 60 from this or from Offline?
        refiso = '60Ni'
        ref_key = refiso[:4]
        prop_key = 'abs_radii'

        thisVals = {key: [results[key]['final'][prop_key]['vals'][0],
                          results[key]['final'][prop_key]['d_stat'][0],
                          results[key]['final'][prop_key]['d_syst'][0]]
                    for key in isolist}

        data_dict = {'BECOLA (Exp)': {'data': thisVals, 'color': self.black}}

        # plot BECOLA values
        src = 'BECOLA (Exp)'
        col = data_dict[src]['color']
        data = data_dict[src]['data']
        keyVals = sorted(data)
        x = []
        y = []
        yerr = []
        ytiltshift = []
        for i in keyVals:
            x.append(int(''.join(filter(str.isdigit, i))))
            y.append(data[i][0])
            yerr.append(data[i][1])  # take only IS contributions here
            ytiltshift.append(data[i][2])  # Fieldshift-Factor, Massshift-Factor uncertainty

        plt.xticks(rotation=0)
        ax = plt.gca()
        ax.set_ylabel(r'$R\mathregular{_c\//fm}$')
        ax.set_xlabel('A')

        # if dash_missing_data:
        #     # if some isotopes are missing, this dashes the line at these isotopes
        #     # has no effect when plot_evens_separate is True
        #     split_x_list = []
        #     for k, g in groupby(enumerate(x), lambda a: a[0] - a[1]):
        #         split_x_list.append(list(map(itemgetter(1), g)))
        #     i = 0
        #     label_created = False
        #     for each in split_x_list:
        #         y_vals = y[i:i + len(each)]
        #         yerr_vals = yerr[i:i + len(each)]
        #         if not label_created:  # only create label once
        #             plt.errorbar(each, y_vals, yerr_vals, fmt='o', color=col, linestyle='-', label=src)
        #             label_created = True
        #         else:
        #             plt.errorbar(each, y_vals, yerr_vals, fmt='o', color=col, linestyle='-')
        #         # plot dashed lines between missing values
        #         if len(x) > i + len(each):  # might be last value
        #             x_gap = [x[i + len(each) - 1], x[i + len(each)]]
        #             y_gap = [y[i + len(each) - 1], y[i + len(each)]]
        #             plt.plot(x_gap, y_gap, c=col, linestyle='--')
        #         i = i + len(each)
        # else:
        #     plt.errorbar(x, y, yerr, fmt='o', color=col, linestyle='-')
        # plot errorband
        plt.fill_between(x,
                         np.array(y) - np.array(ytiltshift),
                         np.array(y) + np.array(ytiltshift),
                         alpha=0.5, facecolor=col, label=src)

        # # exp values
        # exp_sets = glob(os.path.join(folder, 'exp_*'))
        # for num, exp in enumerate(exp_sets):
        #     file = exp.split('\\')[-1]
        #     exp_name = file.split('_')[-1][:-4]  # remove data_ and .txt
        #     exp_name = exp_name.replace('slash', '/')
        #
        #     data = pd.read_csv(exp, delimiter=' ', index_col=0, skiprows=[])
        #
        #     isos = []
        #     vals = []
        #     unc_up = []
        #     unc_down = []
        #     for i, row in data.iterrows():
        #         isos.append(i)
        #         vals.append(row['val'])
        #         if 'unc_up' in row:
        #             unc_up.append(row['unc_up'])
        #             unc_down.append(row['unc_down'])
        #         elif 'unc' in row:
        #             unc_up.append(row['unc'])
        #             unc_down.append(row['unc'])
        #         else:
        #             unc_up = None
        #
        #     if unc_up is not None:
        #         plt.errorbar(isos, vals, yerr=(unc_down, unc_up), label=exp_name, marker='*', markersize=10,
        #                      linestyle='-', color=self.black, zorder=10)
        #     else:  # no uncertainties given
        #         plt.plot(isos, vals, label=exp_name, marker='*', markersize=10, linestyle='-', color=self.black, zorder=10)


        # plot theory values
        theory_sets = glob(os.path.join(folder, 'data_*'))
        offsets = [0.05, -0.05, 0.0, 0.1, 0.0, 0.0, -0.15, -0.15, 0.01, -0.01, 0.02, -0.02, 0.03, -0.03]
        colornum = 1
        colorlist = [self.black, self.blue, self.green, self.orange, self.red, self.purple, self.purple,
                     self.dark_green, self.dark_green, self.dark_blue, self.dark_blue]
        markernum = 0
        for num, th in enumerate(theory_sets):
            file = th.split('\\')[-1]
            name = file[5:-4]  # remove data_ and .txt
            name = name.replace('slash', '/')

            data = pd.read_csv(th, delimiter=' ', index_col=0, skiprows=[])

            isos = []
            vals = []
            unc_up = []
            unc_down = []
            for i, row in data.iterrows():
                isos.append(i + offsets[num])
                vals.append(row['val'])
                if 'unc_up' in row:
                    unc_up.append(row['unc_up'])
                    unc_down.append(row['unc_down'])
                elif 'unc' in row:
                    unc_up.append(row['unc'])
                    unc_down.append(row['unc'])
                else:
                    unc_up = None

            if unc_up is not None:
                plt.errorbar(isos, vals, yerr=(unc_down, unc_up), label=name, marker=self.markerlist[markernum],
                             linestyle='', color=colorlist[colornum])
            else:  # no uncertainties given
                plt.plot(isos, vals, label=name, marker=self.markerlist[markernum], linestyle='',
                             color=colorlist[colornum])

            colornum += 1
            if colornum == colorlist.__len__():
                colornum = 0
            markernum += 1
            if markernum == self.markerlist.__len__():
                markernum = 0

        ax.set_ylabel(r'$R\mathregular{_c\//fm}$')
        ax.set_xlabel('A')
        ax.set_xmargin(0.05)
        ax.set_xlim((53.5, 59 )) #64.5))
        ax.set_ylim((3.5, 3.9))
        # sort legend alphabetically but keep experiment on top
        handles, labels = ax.get_legend_handles_labels()
        expi = [labels.index(i) for i in labels if 'Exp' in i]  # first get experiment value(s), so we can put them at the top.
        handles_print = [handles.pop(i) for i in expi]
        labels_print = [labels.pop(i) for i in expi]
        hl = sorted(zip(handles, labels),
                    key=operator.itemgetter(1))
        handles_sort, labels_sort = zip(*hl)
        handles_print += handles_sort
        labels_print += labels_sort
        plt.legend(handles_print, labels_print, bbox_to_anchor=(1, 0.5), loc='center left', ncol=1)
        plt.margins(0.1)
        plt.gcf().set_facecolor('w')

        plt.tight_layout(pad=False)
        plt.savefig(folder + 'abs_radii_all' + self.ffe, dpi=self.dpi, bbox_inches='tight')

        plt.close()
        plt.clf()

    def absradii_neighborhood(self):
        # define folder
        folder = os.path.join(self.fig_dir, 'Nickel\\Discussion\\R_Elements\\')

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
        add_second_axis = False
        if add_second_axis:
            ax2 = fig.add_axes([0.09, 0.51, 0.33, 0.4])  # add a secondary inlet axis
        # define output size of figure
        width, height = 1., 0.50
        # f.set_dpi(300.)
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        # plot theory values
        theory_sets = glob(os.path.join(folder, 'data_*'))
        colornum = 0
        for num, th in enumerate(theory_sets):
            file = th.split('\\')[-1]
            pre, Z, name = file.split('_')
            name = name[:-4]  # .txt

            data = pd.read_csv(th, delimiter='\t', index_col=0, skiprows=[0])

            isos = []
            vals = []
            refval = 0
            refunc = 0
            unc = []
            for i, row in data.iterrows():
                if '(m)' in name:
                    i = i[:-1]
                    mark = 's'
                else:
                    mark = 'o'
                N = int(i) - int(Z)
                if N == 28:
                    refval = float(row['val'])
                    refunc = float(row['unc'])
                isos.append(N)
                vals.append(row['val'])
                unc.append(row['unc'])

            col = self.colorlist[colornum]
            if 'Ni' in name:
                col = self.red

            p = ax.errorbar(isos, vals, yerr=unc, label=r'$_\mathregular{{{1}}}${0}'.format(name, Z,),
                            marker=mark, linestyle='-', color=col)
            colornum += 1
            if colornum == self.colorlist.__len__():
                colornum = 0
            if self.colorlist[colornum] == self.red:
                colornum += 1
            if refval:
                thiscol = p[0].get_color()
                power = 2  # set to 2 for differential ms charge radii
                vals = np.array(vals)**power-refval**power
                unc = np.sqrt(np.square(2*vals*unc) + np.square(2*refval*refunc))
                if add_second_axis:
                    ax2.errorbar(isos, vals, yerr=unc, label=r'{0}$_\mathregular{{{1}}}$'.format(name, Z,),
                             marker=mark, markersize=5, linestyle='-', color=thiscol)

        ax.axvline(20, color=self.grey, linestyle='--')
        ax.axvline(28, color=self.grey, linestyle='--')
        ax.axvline(40, color=self.grey, linestyle='--')
        if add_second_axis:
            ax2.axvline(28, color=self.grey, linestyle='--')

        ylabeltext = r'R$\mathregular{_c\//fm}$'
        if add_second_axis:
            textalign = 'center'
            ax.text(x=10, y=3.4, s=ylabeltext,
                    horizontalalignment='center', verticalalignment=textalign, rotation='vertical',
                    **self.ch_dict(self.text_style, {'size': 11}))
        else:
            ax.set_ylabel(ylabeltext)
            ax.yaxis.set_label_position('left')
        ax.set_xlabel('N')
        # ax.set_xmargin(0.05)
        ax.set_xticks([16, 20, 24, 28, 32, 36, 40])
        ax.set_xlim((23.5, 42.5))  # 12
        ax.set_ylim((3.4, 3.95))  # 3.3
        if add_second_axis:
            ax.set_yticks([3.3, 3.4, 3.5, 3.6])
        ax.axes.tick_params(axis='y', direction='in', left=True, right=True, labelleft=True, labelright=False)

        if add_second_axis:
            ax2.set_xlabel('N')
            ax2.xaxis.set_label_position('top')
            ax2.set_ylabel(r'$\mathbf{\delta \langle r^2 \rangle}\mathregular{\//fm^2}$')
            ax2.yaxis.set_label_position('left')
            ax2.set_xticks([26, 28, 30])
            ax2.set_xlim((25.5, 30.5))
            ax2.set_ylim((-0.1, 0.4))
            ax2.axes.tick_params(axis='x', direction='in', bottom=True, top=True, labelbottom=False, labeltop=True)
            ax2.axes.tick_params(axis='y', direction='in', left=True, right=True, labelleft=True, labelright=False)

        # reverse legend order then plot
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], bbox_to_anchor=(1, 0), loc='lower right', ncol=1)
        # plt.margins(0.1)
        # plt.gcf().set_facecolor('w')

        # plt.tight_layout(True)
        filename = 'abs_radii_neighborhood'
        if not add_second_axis:
            filename += '_noInsert'
        plt.savefig(folder + filename + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def absradii_ni_only(self):
        # define folder
        folder = os.path.join(self.fig_dir, 'Nickel\\Discussion\\R_NickelChain\\')

        fig, ax = plt.subplots(1)
        # define output size of figure
        width, height = 0.7, 0.2
        # f.set_dpi(300.)
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        # exp values
        exp_sets = glob(os.path.join(folder, 'exp_*'))
        for num, exp in enumerate(exp_sets):
            file = exp.split('\\')[-1]
            exp_name = file.split('_')[-1][:-4]  # remove data_ and .txt
            exp_name = exp_name.replace('slash', '/')

            data = pd.read_csv(exp, delimiter=' ', index_col=0, skiprows=[])

            isos = []
            vals = []
            unc_up = []
            unc_down = []
            for i, row in data.iterrows():
                isos.append(i)
                vals.append(row['val'])
                if 'unc_up' in row:
                    unc_up.append(row['unc_up'])
                    unc_down.append(row['unc_down'])
                elif 'unc' in row:
                    unc_up.append(row['unc'])
                    unc_down.append(row['unc'])
                else:
                    unc_up = None

            plt.errorbar(isos, vals, unc_up, fmt='o', color=self.red, linestyle='-')
            # plot errorband
            plt.fill_between(isos,
                             np.array(vals) - np.array(unc_up),
                             np.array(vals) + np.array(unc_up),
                             alpha=0.5, edgecolor=self.red, facecolor=self.red)


        ax.set_ylabel(r'$R\mathregular{_c\//fm}$')
        ax.set_xlabel('A')
        ax.set_xmargin(0.05)
        ax.set_xlim((53.5, 70.5))
        ax.set_ylim((3.68, 3.92))

        plt.margins(0.1)
        plt.gcf().set_facecolor('w')

        plt.tight_layout(True)
        plt.savefig(folder + 'abs_radii_Ni' + self.ffe, dpi=self.dpi, bbox_inches='tight')

        plt.close()
        plt.clf()

    def absrad56(self):
        # define folder
        folder = os.path.join(self.fig_dir, 'Nickel\\Discussion\\R_Nickel56\\')

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'Ni_Results2_*'))[0]
        results = self.import_results(load_results_from)

        isolist = ['56Ni']  # BECOLA isotopes
        # TODO: 58 and 60 from this or from Offline?
        refiso = '60Ni'
        ref_key = refiso[:4]
        prop_key = 'abs_radii'

        thisVals = {key: [results[key]['final'][prop_key]['vals'][0],
                          results[key]['final'][prop_key]['d_stat'][0],
                          results[key]['final'][prop_key]['d_syst'][0]]
                    for key in isolist}

        data_dict = {'BECOLA': {'data': thisVals, 'color': self.red}}

        # get BECOLA values
        src = 'BECOLA'
        col = data_dict[src]['color']
        data = data_dict[src]['data']
        keyVals = sorted(data)
        x = []
        y = []
        yerr = []
        ytiltshift = []
        for i in keyVals:
            x.append(int(''.join(filter(str.isdigit, i))))
            y.append(data[i][0])
            yerr.append(data[i][1])  # take only IS contributions here
            ytiltshift.append(data[i][2])  # Fieldshift-Factor, Massshift-Factor uncertainty

        # get theory values
        theory_sets = glob(os.path.join(folder, '*56Ni_*'))

        # Create plots for each dataset
        f, axes = plt.subplots(nrows=1, ncols=len(theory_sets), sharex=False, sharey='row')
        f.subplots_adjust(wspace=0, hspace=0)

        # define output size of figure
        width, height = 1.0, 0.4
        # f.set_dpi(300.)
        f.set_size_inches((self.w_in * width, self.h_in * height))

        for num, th in enumerate(theory_sets):
            # create temporary variables
            theo_vals = []
            theo_unc = []
            theo_label = []
            # get name
            file = th.split('\\')[-1]
            name = file.split('_')[-1][:-4]  # remove data_ and .txt
            name = name.replace('slash', '/')
            name = name.replace('lambda', '$\Lambda$')

            data = pd.read_csv(th, delimiter='&', index_col=0, skiprows=[0])

            if not 'Bay' in name:  # non-bayesian errors
                cs = 0  # capsize of errorbars to zero if non-statistical errors
            else:
                cs = 4  # statistical interpretable errorbars

            was_point_proton = False
            def calc_rch_from_rpp(rpp, rpp_d):
                # calculation from Kaufmann et al. 2020
                rp2 = 0.7080  # rms charge radius proton /fm^2
                rp2_d = 0.0032
                rn2 = -0.106  # rms charge radius neutron /fm^2 (from Filin et al., PRL 124, 082501 (2020))
                rn2_d = 0.007
                relDarFol = 0.033  # relativistic Darwin-Foldy correction /fm^2
                relDarFol_d = 0
                # corSO = 0.13469  # spin-orbit correction /fm^2 (from Horowitz, Piekarewicz; PRC 86, 045503 (2012))
                # corSO = 0.0591  # extracted from Sonia Baccas calculations on NNLOsat
                corSO = 0
                corSO_d = 0
                rch = np.sqrt(float(rpp) ** 2 + rp2 + (28 / 28) * rn2 + relDarFol + corSO)
                # TODO: Should do similar calculation for uncertainties
                # error of rch**2 is a little more straight forward:
                rch2_d = np.sqrt((2*float(rpp)*float(rpp_d))**2 + rp2_d**2 + rn2_d**2 + relDarFol_d**2 + corSO_d**2)
                rch_d = rch2_d/(2*rch)  # now error of sqrt(A) where A=rch**2 --> A_d/(2*sqrt(A))
                return rch, rch_d

            for i, row in data.iterrows():
                theo_label.append(i)
                val_and_unc = row[' r_ch(delta)']
                if val_and_unc == ' -':
                    # no charge radius given. Try proton radius
                    val_and_unc = row[' rms_p(delta) ']
                    v_pro, v_pro_d, *_ = re.split('[()]', val_and_unc)
                    # convert to charge radius
                    val, unc = calc_rch_from_rpp(v_pro, v_pro_d)
                    was_point_proton = True
                else:
                    val, unc, *_ = re.split('[()]', val_and_unc)

                theo_vals.append(float(val))
                theo_unc.append(float(unc))

            if was_point_proton:
                name = '{}*'.format(name)  # add a star to indicate that it was calulated from point proton radius

            # Try to also import 68Ni results for comparison.
            try:
                # For these calculations from Bacca we also have values for Nickel 68
                # import them because I want to discuss trends.
                theory_68 = glob(os.path.join(folder, '68Ni_{}.txt'.format(name)))[0]
                data_68 = pd.read_csv(theory_68, delimiter='&', index_col=0, skiprows=[0])
                theo_vals_68 = []
                theo_unc_68 = []
                for i, row in data_68.iterrows():
                    val_and_unc = row[' r_ch(delta)']
                    if val_and_unc == ' -':
                        # no charge radius given. Try proton radius
                        val_and_unc = row[' rms_p(delta) ']
                        v_pro, v_pro_d, *_ = re.split('[()]', val_and_unc)
                        # convert to charge radius
                        val, unc = calc_rch_from_rpp(v_pro, v_pro_d)
                    else:
                        val, unc, *_ = re.split('[()]', val_and_unc)

                    theo_vals_68.append(float(val))
                    theo_unc_68.append(float(unc))
                # now plot
                axes[num].errorbar(np.arange(theo_label.__len__())+0.1, theo_vals_68, yerr=theo_unc_68, marker='s',
                                   linestyle='', color=self.dark_orange, capsize=cs, elinewidth=1.5, capthick=1.5)
                # make band with experimental nickel 68 value
                rc68 = 3.887, 0.003  # from Kaufmann.2020
                axes[num].axhspan(rc68[0] - rc68[1], rc68[0] + rc68[1], color=self.orange)
            except:
                print('No 68Ni values found for {}'.format(name))

            axes[num].errorbar(range(theo_label.__len__()), theo_vals, yerr=theo_unc, fmt='o', linestyle='',
                               color=self.blue, capsize=cs, elinewidth=1.5, capthick=1.5)
            axes[num].set_xticks(range(theo_label.__len__()))
            axes[num].set_xticklabels(theo_label, rotation=90)
            axes[num].set_xlim((-0.5, len(theo_label)-0.5))
            axes[num].set_xlabel(r'{}'.format(name))

            # TODO: Make band with experimental value instead
            axes[num].axhspan(y[0] - yerr[0] - ytiltshift[0], y[0] + yerr[0] + ytiltshift[0], color=col)

        axes[0].set_ylabel(r'$R\mathregular{_c\//fm}$')
        axes[0].set_ylim((3.07, 4.11))

        # make legend
        red_line = mpl.lines.Line2D([], [], color=self.red, marker='', linewidth='2', label='Experiment 56Ni')
        blue_dot = axes[0].errorbar([], [], [], color=self.blue, marker='o', markersize='5', linestyle='', label=r'Theorie $^{56}$Ni')
        da1, la1 = axes[0].get_legend_handles_labels()
        orange_line = mpl.lines.Line2D([], [], color=self.orange, marker='', linewidth='2', label='Experiment 68Ni')
        orange_square = axes[1].errorbar([], [], [], color=self.dark_orange, marker='s', markersize='5', linestyle='', label=r'Theorie $^{68}$Ni')
        da2, la2 = axes[1].get_legend_handles_labels()
        axes[-1].legend(handles=[red_line, da1[0][0], orange_line, da2[0][0]], labels=[r'Experiment $^{56}$Ni', la1[0], R'Experiment $^{68}$Ni', la2[0]],
                   bbox_to_anchor=(1, 1), loc='upper left', ncol=1)

        plt.savefig(folder + 'abs_radius_56' + '.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def three_point_indicator(self):
        # define folder
        folder = os.path.join(self.fig_dir, 'Nickel\\Discussion\\R_NickelChain\\')

        fig, ax = plt.subplots(1)
        # define output size of figure
        width, height = 0.5, 0.4
        # f.set_dpi(300.)
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'Ni_Results2_*'))[0]
        results = self.import_results(load_results_from)

        isolist = ['54NiBec', '56Ni', '58Ni']  # BECOLA isotopes
        prop_key = 'abs_radii'

        thisVals = {key: [results[key]['final'][prop_key]['vals'][0],
                          results[key]['final'][prop_key]['d_stat'][0],
                          results[key]['final'][prop_key]['d_syst'][0]]
                    for key in isolist}

        data_dict = {'BECOLA (Exp)': {'data': thisVals, 'color': self.black}}

        # collect three point indicators
        source = []  # lists to gather the result names
        x_pos_list = []
        three_p_i = []  # lists to gather the results
        three_p_i_dup = []  # lists to gather the upper uncertainties
        three_p_i_dlo = []  # lists to gather the lower uncertainties

        def calc_tpi(rplu2, r, rmin2):
            return 0.5*(rplu2 - 2*r + rmin2)

        def calc_tpi_d(rplu2_d, r_d, rmin2_d):
            return np.sqrt((0.5*rplu2_d)**2 + r_d**2 + (0.5*rmin2_d)**2)

        # plot theory values
        theory_sets = glob(os.path.join(folder, 'data_*'))
        colornum = 1
        x_pos = 0
        for num, th in enumerate(theory_sets):
            file = th.split('\\')[-1]
            name = file[5:-4]  # remove data_ and .txt
            name = name.replace('slash', '/')

            data = pd.read_csv(th, delimiter=' ', index_col=0, skiprows=[])

            try:  # will fail if one of the values is missing
                tpi = calc_tpi(data.loc[58, 'val'], data.loc[56, 'val'], data.loc[54, 'val'])
                three_p_i += [tpi]
                try:
                    tpi_d_up = calc_tpi_d(data.loc[58, 'unc_up'], data.loc[56, 'unc_up'], data.loc[54, 'unc_up'])
                    tpi_d_down = calc_tpi_d(data.loc[58, 'unc_down'], data.loc[56, 'unc_down'], data.loc[54, 'unc_down'])
                    three_p_i_dup += [tpi_d_up]
                    three_p_i_dlo += [tpi_d_down]
                except:
                    tpi_d = calc_tpi_d(data.loc[58, 'unc'], data.loc[56, 'unc'], data.loc[54, 'unc'])
                    three_p_i_dup += [tpi_d]
                    three_p_i_dlo += [tpi_d]
                source += [name]
                x_pos_list += [x_pos]
                ax.bar(x_pos, tpi, align='center', color=self.colorlist[colornum])
                textpos = tpi/2
                textcol = 'w'
                textalign = 'center'
                if tpi < 0.015:
                    textpos = tpi+0.001
                    textcol = self.black
                    textalign = 'bottom'
                ax.text(x_pos, textpos, name,
                           horizontalalignment='center', verticalalignment=textalign, rotation='vertical', color=textcol, fontweight='bold',
                           **self.ch_dict(self.text_style, {'size': 11})
                           )
                x_pos += 1
            except:
                try:
                    if 'pf)' in name:
                        # below and above the shell closure we were using different valence spaces...
                        file_pf5g9 = file.replace('pf).txt', 'pf5g9*')
                        theory_pf5g9 = glob(os.path.join(folder, file_pf5g9))
                        data_pf5g9 = pd.read_csv(theory_pf5g9[0], delimiter=' ', index_col=0, skiprows=[])
                        tpi = calc_tpi(data_pf5g9.loc[58, 'val'],
                                       0.5*(data.loc[56, 'val']+data_pf5g9.loc[56, 'val']),
                                       data.loc[54, 'val'])
                        three_p_i += [tpi]
                        try:
                            tpi_d_up = calc_tpi_d(data_pf5g9.loc[58, 'unc_up'],
                                                  0.5*(data.loc[56, 'unc_up']+data_pf5g9.loc[56, 'unc_up']),
                                                  data.loc[54, 'unc_up'])
                            tpi_d_down = calc_tpi_d(data_pf5g9.loc[58, 'unc_down'],
                                                    0.5*(data.loc[56, 'unc_down']+data_pf5g9.loc[56, 'unc_down']),
                                                    data.loc[54, 'unc_down'])
                            three_p_i_dup += [tpi_d_up]
                            three_p_i_dlo += [tpi_d_down]
                        except:
                            tpi_d = calc_tpi_d(data_pf5g9.loc[58, 'unc'],
                                               0.5*(data.loc[56, 'unc']+data_pf5g9.loc[56, 'unc']),
                                               data.loc[54, 'unc'])
                            three_p_i_dup += [tpi_d]
                            three_p_i_dlo += [tpi_d]
                        source += [name+'(cross)']  # denote that this was done across valence spaces
                        x_pos_list += [x_pos]
                        ax.bar(x_pos, tpi, align='center', color=self.colorlist[colornum])
                        textpos = tpi / 2
                        textcol = 'w'
                        textalign = 'center'
                        if tpi < 0.015:
                            textpos = tpi + 0.001
                            textcol = self.black
                            textalign = 'bottom'
                        ax.text(x_pos, textpos, name.replace(' pf', '')+r'$^\bigstar$',
                                horizontalalignment='center', verticalalignment=textalign, rotation='vertical',
                                color=textcol, fontweight='bold',
                                **self.ch_dict(self.text_style, {'size': 11})
                                )
                        x_pos += 1
                except:
                    # try:  # as a last resort, use 59-57-55
                    #     tpi = calc_tpi(data.loc[59, 'val'], data.loc[57, 'val'], data.loc[55, 'val'])
                    #     three_p_i += [tpi]
                    #     source += [name+'(odd)']  # denote that this was done on the odd
                    #     ax.bar(x_pos, tpi, align='center', color=self.colorlist[colornum])
                    #     textpos = tpi / 2
                    #     textcol = 'w'
                    #     textalign = 'center'
                    #     if tpi < 0.015:
                    #         textpos = tpi + 0.001
                    #         textcol = self.black
                    #         textalign = 'bottom'
                    #     ax.text(x_pos, textpos, name+r'$^\bigstar$',
                    #             horizontalalignment='center', verticalalignment=textalign, rotation='vertical',
                    #             color=textcol, fontweight='bold',
                    #             **self.ch_dict(self.text_style, {'size': 11})
                    #             )
                    #     x_pos += 1
                    # except:
                    print('Could not calculate 3-point indicator for {}'.format(th))

            colornum += 1
            if colornum == self.colorlist.__len__():
                colornum = 0

        # ax.bar(np.arange(len(source)), three_p_i, align='center')

        # add for experiment
        tpi_exp = calc_tpi(thisVals['58Ni'][0], thisVals['56Ni'][0], thisVals['54NiBec'][0])
        tpi_exp_d = calc_tpi_d(thisVals['58Ni'][1], thisVals['56Ni'][1], thisVals['54NiBec'][1])
        tpi_exp_d_syst = calc_tpi_d(thisVals['58Ni'][1], thisVals['56Ni'][1], thisVals['54NiBec'][1])
        tpi_exp_dtot = np.sqrt(tpi_exp_d**2+tpi_exp_d_syst**2)

        # ax.axhline(tpi_exp, color=self.black)
        ax.axhspan(tpi_exp-tpi_exp_dtot, tpi_exp+tpi_exp_dtot, color=self.grey)
        ax.text(0.7, tpi_exp+2*tpi_exp_dtot, 'Exp',
                horizontalalignment='left', verticalalignment='bottom', rotation='horizontal',
                color=self.grey, fontweight='bold',
                **self.ch_dict(self.text_style, {'size': 13})
                )
        ax.errorbar(x_pos_list, three_p_i, [three_p_i_dlo, three_p_i_dup])

        ax.set_xticks([])
        ax.axes.tick_params(axis='y', direction='out', right=False)
        ax.set_ylabel(r'$\Delta_{2n}^{(3)}R_\mathregular{c}$ /fm')

        plt.savefig(folder + 'three_point_ind' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

        plt.axhspan(tpi_exp-tpi_exp_dtot, tpi_exp+tpi_exp_dtot, color=self.grey)
        plt.errorbar(x_pos_list, three_p_i, [three_p_i_dlo, three_p_i_dup])
        plt.show()

    def three_point_indicator_hardcoded(self):
        """
        Version where the datapoints are given directly.
        """
        ''' DATA (NAME, VALUE, ERROR-, ERROR+, color)'''
        data_exp = ('Exp', 0.03045, 0.00259, 0.00259, self.grey)
        data_exp_pp = ('Exp', 0.0312, 0.00259, 0.00259, self.grey)
        data_svmin = ('DFT SVmin', 0.00520, 0.00938, 0.00621, self.green)
        data_fayans = ('DFT Fayans', 0.02275, 0.00672, 0.02756, self.blue)
        data_em1820 = ('VS-IMSRG EM1.8/2.0', 0.02420, 0.00011, 0.00011, self.purple)
        data_imsrg = ('mRef-IMSRG/NCSM(N4LO\')', 0.022455, 0.006675, 0.006675, self.orange)
        data_imsrg4 = ('N4LO\'', 0.022455, 0.006675, 0.006675, self.orange)
        data_imsrg3 = ('N3LO', 0.017078, 0.006675, 0.006675, self.dark_orange)
        data_imsrg2 = ('N2LO', 0.016111, 0.013039, 0.013039, self.red)
        data_imsrg1 = ('NLO', 0.036244, 0.04257, 0.04257, self.dark_red)
        data_imsrg4_pp = ('N4LO\'', 0.02300, 0.0067, 0.0067, self.orange)
        data_imsrg3_pp = ('N3LO', 0.01750, 0.0067, 0.0067, self.dark_orange)
        data_imsrg2_pp = ('N2LO', 0.01650, 0.0133, 0.0133, self.red)
        data_imsrg1_pp = ('NLO', 0.0375, 0.044, 0.044, self.dark_red)
        data_imsrg4_pp_d = ('N4LO\'', 0.023, 0.012, 0.012, self.orange)
        data_imsrg3_pp_d = ('N3LO', 0.018, 0.012, 0.012, self.dark_orange)
        data_imsrg2_pp_d = ('N2LO', 0.016, 0.012, 0.012, self.red)
        data_imsrg1_pp_d = ('NLO', 0.037, 0.040, 0.040, self.dark_red)

        all_theo_data = [data_svmin, data_fayans, data_em1820, data_imsrg]
        # all_theo_data = [data_imsrg1_pp, data_imsrg2_pp, data_imsrg3_pp, data_imsrg4_pp,
        #                  data_imsrg1_pp_d, data_imsrg2_pp_d, data_imsrg3_pp_d, data_imsrg4_pp_d]

        ''' PREPARE PLOT '''
        folder = os.path.join(self.fig_dir, 'Nickel\\Discussion\\3PointIndicator\\')
        fig, ax = plt.subplots(1)
        # define output size of figure
        width, height = 1, 0.25
        # f.set_dpi(300.)
        fig.set_size_inches((self.w_in * width, self.h_in * height))

        ''' PLOT '''
        # Experimental value as a band
        ax.axhspan(data_exp[1]-data_exp[2], data_exp[1]+data_exp[3], color=data_exp[4])
        ax.text(-1.4, data_exp[1], data_exp[0],
                horizontalalignment='left', verticalalignment='center', rotation='horizontal',
                color=self.black, fontweight='bold',
                **self.text_style
                )
        # Theory values as errorbars
        for num, data_set in enumerate(all_theo_data):
            ax.errorbar(num, data_set[1], [[data_set[2]], [data_set[3]]],
                        **self.ch_dict(self.data_style, {'color': data_set[4]}))
            ax.text(num+0.1, data_set[1], data_set[0],
                    horizontalalignment='left', verticalalignment='center', rotation='vertical',
                    color=data_set[4], fontweight='bold',
                    **self.text_style
                    )
        # horizontal line at zero
        ax.axhline(y=0, color=self.grey, ls='--')

        ax.set_xticks([])
        ax.axes.tick_params(axis='y', direction='out', right=False)
        ax.set_ylabel(r'$\Delta_{2n}^{(3)}R_\mathregular{c}$ /fm')
        ax.set_xlim([-1.5, len(all_theo_data)-0.5])

        plt.savefig(folder + 'three_point_ind_hardcode' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def mu_nickel55(self):
        """

        :return:
        """
        # define folder
        folder = os.path.join(self.fig_dir, 'Nickel\\Discussion\\mu_Nickel55\\')

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'Ni_Results2_*'))[0]
        results = self.import_results(load_results_from)

        isolist = ['55Ni']  # BECOLA isotopes

        thisVals = {key: [results[key]['final']['moments']['mu_avg']['vals'][0],
                          results[key]['final']['moments']['mu_avg']['d_stat'][0],
                          results[key]['final']['moments']['mu_avg']['d_syst'][0]]
                    for key in isolist}

        data_dict = {'BECOLA': {'data': thisVals, 'color': self.red}}

        # get BECOLA values
        src = 'BECOLA'
        col = data_dict[src]['color']
        data = data_dict[src]['data']
        keyVals = sorted(data)
        x = []
        y = []
        yerr = []
        ysyst = []
        for i in keyVals:
            x.append(int(''.join(filter(str.isdigit, i))))
            y.append(data[i][0])
            yerr.append(data[i][1])
            ysyst.append(data[i][2])

        # get theory values
        theory_sets = glob(os.path.join(folder, '*Nickel_*'))

        # Create plots for each dataset
        f, axes = plt.subplots(nrows=1, ncols=len(theory_sets), sharex=False, sharey='row')
        f.subplots_adjust(wspace=0, hspace=0)

        # define output size of figure
        width, height = 1, 0.25
        # f.set_dpi(300.)
        f.set_size_inches((self.w_in * width, self.h_in * height))

        for num, th in enumerate(theory_sets):
            # create temporary variables
            theo_vals = []
            theo_unc = []
            theo_label = []
            # get name
            file = th.split('\\')[-1]
            name = file.split('_')[-1][:-4]  # remove Nickel_ and .txt
            name = name.replace('slash', '/')
            name = name.replace('lambda', '$\Lambda$')

            data = pd.read_csv(th, delimiter='&', index_col=0, skiprows=[0])

            for i, row in data.iterrows():
                theo_label.append(i)
                val_and_unc = row[' mu']
                if isinstance(val_and_unc, float):
                    # no uncertainty given
                    val = val_and_unc
                    unc = 0
                elif isinstance(val_and_unc, str) and '(' in val_and_unc:
                    val, unc, *_ = re.split('[()]', val_and_unc)


                theo_vals.append(float(val))
                theo_unc.append(float(unc))


            # Try to also import cobalt results for comparison.
            try:
                theory_Cobalt = glob(os.path.join(folder, '*Cobalt_{}.txt'.format(name)))[0]
                data_Cobalt = pd.read_csv(theory_Cobalt, delimiter='&', index_col=0, skiprows=[0])
                theo_vals_Cobalt = []
                theo_unc_Cobalt = []
                for i, row in data_Cobalt.iterrows():
                    val_and_unc_Co = row[' mu']
                    if isinstance(val_and_unc_Co, float):
                        # no uncertainty given
                        valCo = val_and_unc_Co
                        uncCo = 0
                    elif isinstance(val_and_unc_Co, str) and '(' in val_and_unc_Co:
                        valCo, uncCo, *_ = re.split('[()]', val_and_unc_Co)
                    else:
                        valCo = np.nan
                        uncCo = np.nan

                    theo_vals_Cobalt.append(float(valCo))
                    theo_unc_Cobalt.append(float(uncCo))

                # create twin axis
                ax_Co = axes[num].twinx()
                cs = 0
                if np.any(theo_unc):
                    cs = 4
                ax_Co.errorbar(np.arange(theo_label.__len__())+0.1, theo_vals_Cobalt, yerr=theo_unc_Cobalt, marker='s',
                               linestyle='', color=self.dark_orange, capsize=cs, elinewidth=1.5, capthick=1.5)
                # make band with experimental nickel 68 value
                muCobalt = 4.822, 0.003  # from Callaghan.1973
                ax_Co.axhspan(muCobalt[0] - muCobalt[1], muCobalt[0] + muCobalt[1], color=self.orange)
                ax_Co.set_ylim((4.5, 5.7))
                ax_Co.axes.tick_params(axis='y', direction='in', left=False, right=True, labelleft=False, labelright=False)
            except:
                print('No Cobalt values found for {}'.format(name))

            if 'Experiment' in name:
                color = self.red
            else:
                color = self.blue

            cs = 0
            if np.any(theo_unc):
                cs = 4
            axes[num].errorbar(range(theo_label.__len__()), theo_vals, yerr=theo_unc, fmt='o', linestyle='', color=color,
                               capsize=cs, elinewidth=1.5, capthick=1.5, markersize=3.5)
            axes[num].set_xticks(range(theo_label.__len__()))
            axes[num].set_xticklabels(theo_label, rotation=90, va='top')
            axes[num].set_xlim((-0.5, len(theo_label) - 0.5))
            axes[num].set_xlabel(r'{}'.format(name))
            axes[num].axes.tick_params(axis='y', direction='in', left=True, right=False)

            # Make band with experimental value
            axes[num].axhspan(y[0] - yerr[0] - ysyst[0], y[0] + yerr[0] + ysyst[0], color=col)

        # Plot the single particle Schmidt Value as dotted Line
        schmidt_07gs = -1.339
        for ax in axes:
            ax.axhline(y=schmidt_07gs, color=self.grey, ls='--')

        axes[0].set_ylabel(r'$\mu(^{55}$Ni$) /\mu_N$')
        # axes[0].set_ylim((-1.95, -0.75))

        # create twin axis
        # ax_Co = axes[-1].twinx()
        # ax_Co.set_ylim((4.5, 5.7))
        # ax_Co.axes.tick_params(axis='y', direction='in', left=False, right=True, labelleft=False, labelright=True, labelcolor=self.dark_orange)
        # ax_Co.set_ylabel(r'$\mu(^{55}$Co$) \mu_N$', color=self.dark_orange)

        # make legend
        red_line = mpl.lines.Line2D([], [], color=self.red, marker='', linewidth='2', label='Experiment 56Ni')
        blue_dot = axes[0].errorbar([], [], [], color=self.blue, marker='o', markersize='5', linestyle='', label=r'Theory $^{55}$Ni')
        grey_line = mpl.lines.Line2D([], [], color=self.grey, ls='--', marker='', linewidth='1', label='Schmidt ($0.7g_s$)')
        da1, la1 = axes[0].get_legend_handles_labels()
        # orange_line = mpl.lines.Line2D([], [], color=self.orange, marker='', linewidth='2', label='Experiment 68Ni')
        # orange_square = axes[1].errorbar([], [], [], color=self.dark_orange, marker='s', markersize='5', linestyle='', label=r'Theorie $^{55}$Co')
        # da2, la2 = axes[1].get_legend_handles_labels()
        f.legend(handles=[red_line, da1[0][0], grey_line], labels=[r'Laserspec. $^{55}$Ni', la1[0], 'Schmidt ($0.7g_s$)'],  # , orange_line, da2[0][0] ... , R'Experiment $^{55}$Co', la2[0]
                        bbox_to_anchor=(0.15, 0.85), loc='upper left', ncol=1, framealpha=1)

        plt.savefig(folder + 'mu_55' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def Q_nickel55(self):
        """

        :return:
        """
        # define folder
        folder = os.path.join(self.fig_dir, 'Nickel\\Discussion\\Q_Nickel55\\')

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'Ni_Results2_*'))[0]
        results = self.import_results(load_results_from)

        isolist = ['55Ni']  # BECOLA isotopes

        thisVals = {key: [results[key]['final']['moments']['Q']['vals'][0],
                          results[key]['final']['moments']['Q']['d_stat'][0],
                          results[key]['final']['moments']['Q']['d_syst'][0]]
                    for key in isolist}

        data_dict = {'BECOLA': {'data': thisVals, 'color': self.red}}

        # get BECOLA values
        src = 'BECOLA'
        col = data_dict[src]['color']
        data = data_dict[src]['data']
        keyVals = sorted(data)
        x = []
        y = []
        yerr = []
        ysyst = []
        for i in keyVals:
            x.append(int(''.join(filter(str.isdigit, i))))
            y.append(data[i][0]*100)  # values in b, transform into efm^2
            yerr.append(data[i][1]*100)  # values in b, transform into efm^2
            ysyst.append(data[i][2]*100)  # values in b, transform into efm^2

        # get theory values
        theory_sets = glob(os.path.join(folder, '*Nickel_*'))

        # Create plots for each dataset
        f, axes = plt.subplots(nrows=1, ncols=len(theory_sets), sharex=False, sharey='row')
        f.subplots_adjust(wspace=0, hspace=0)

        # define output size of figure
        width, height = 0.5, 0.3
        # f.set_dpi(300.)
        f.set_size_inches((self.w_in * width, self.h_in * height))

        for num, th in enumerate(theory_sets):
            # create temporary variables
            theo_vals = []
            theo_unc = []
            theo_label = []
            # get name
            file = th.split('\\')[-1]
            name = file.split('_')[-1][:-4]  # remove Nickel_ and .txt
            name = name.replace('slash', '/')
            name = name.replace('lambda', '$\Lambda$')

            data = pd.read_csv(th, delimiter='&', index_col=0, skiprows=[0])

            for i, row in data.iterrows():
                theo_label.append(i)
                val_and_unc = row[' Q']
                if isinstance(val_and_unc, float):
                    # no uncertainty given
                    val = val_and_unc
                    unc = 0
                elif isinstance(val_and_unc, str) and '(' in val_and_unc:
                    val, unc, *_ = re.split('[()]', val_and_unc)


                theo_vals.append(float(val))
                theo_unc.append(float(unc))

            if 'Experiment' in name:
                color = self.red
            else:
                color = self.blue
            axes[num].errorbar(range(theo_label.__len__()), theo_vals, yerr=theo_unc, fmt='o', linestyle='', color=color)
            axes[num].set_xticks(range(theo_label.__len__()))
            axes[num].set_xticklabels([r'{}'.format(i) for i in theo_label], rotation=90)
            axes[num].set_xlim((-0.5, len(theo_label) - 0.5))
            axes[num].set_xlabel(r'{}'.format(name))
            axes[num].axes.tick_params(axis='y', direction='in', left=True, right=False)

            # Make band with experimental value
            axes[num].axhline(y[0], color=col)
            axes[num].axhspan(y[0] - yerr[0] - ysyst[0], y[0] + yerr[0] + ysyst[0], color=col, alpha=0.5)
            # also add zero line
            axes[num].axhline(0, color=self.black)

        axes[0].set_ylabel(r'$\mathregular{Q(^{55}Ni) \//efm^2}$')


        # make legend
        red_line = mpl.lines.Line2D([], [], color=self.red, marker='', linewidth='2', label='Exp 56Ni')
        blue_dot = axes[0].errorbar([], [], [], color=self.blue, marker='o', markersize='5', linestyle='', label=r'Theory $^{55}$Ni')
        da1, la1 = axes[0].get_legend_handles_labels()

        axes[0].legend(handles=[red_line, da1[0][0]], labels=[r'Exp $^{55}$Ni', la1[0]],
                        bbox_to_anchor=(0, 1.05), loc='lower left', ncol=2)

        plt.savefig(folder + 'Q_55' + self.ffe, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    """ Helper Functions """
    def ch_dict(self, orig, changes):
        """
        Change a few parameters of the original dictionary and return new
        :param orig: the base dictionary
        :param changes: dictionary with changed and/or added items:values
        :return: new dictionary
        """
        new = orig.copy()
        new.update(changes)
        return new

    def _line(self, _x, _m, _b):
        return _m * _x + _b

    def lin_regression(self, x, y, pars):
        """
        Get a dataset and return a linear regression
        :return: slope, offset
        """
        # start parameters
        p0 = pars
        # do the fitting
        popt, pcov = curve_fit(self._line, x, y, p0)
        slope, offset = popt
        perr = np.sqrt(np.diag(pcov))  # TODO: use this somewhere?

        return slope, offset

    def calc_weighted_avg(self, values, uncertainties):
        """
        Based on 'Bevington - Data Reduction and Error Analysis for the Physical Sciences'
        :param values:
        :param uncertainties:
        :return:
        """
        x = np.asarray(values)
        n = x.__len__()
        x_d = np.asarray(uncertainties)
        # calculate weights inversely proportional to the square of the uncertainties
        if not any(x_d == 0):
            w = 1 / np.square(x_d)
        else:
            logging.warning('ZERO value in uncertainties found during weighted average calculation. '
                            'Calculating mean and standard deviation instead of weighting!')
            return x.mean(), x.std(), x.std(), x.std(), x.std() / np.sqrt(n)

        if n > 1:  # only makes sense for more than one data point. n=1 will also lead to div0 error
            # calculate weighted average and sum of weights:
            wavg = np.sum(x * w) / np.sum(w)  # (Bevington 4.17)
            # calculate the uncertainty of the weighted mean
            wavg_d = np.sqrt(1 / np.sum(w))  # (Bevington 4.19)

            # calculate weighted average variance
            wvar = np.sum(w * np.square(x - wavg)) / np.sum(w) * n / (n - 1)  # (Bevington 4.22)
            # calculate weighted standard deviations
            wstd = np.sqrt(wvar / n)  # (Bevington 4.23)

            # calculate (non weighted) standard deviations from the weighted mean
            std = np.sqrt(np.sum(np.square(x - wavg)) / (n - 1))
            # calculate the standard deviation of the average
            std_avg = std / np.sqrt(n)

        else:  # for only one value, return that value
            wavg = x[0]
            # use the single value uncertainty for all error estimates
            wavg_d = x_d[0]
            wstd = x_d[0]
            std = x_d[0]
            std_avg = x_d[0]

        return wavg, wavg_d, wstd, std, std_avg

    def import_results(self, results_path):
        """
        Taken from Ni_StandardAnalysis. Modified to take an absolute path as input and return a fresh dict.
        :param results_path:
        :param is_gate_analysis:
        :return:
        """
        ele = TiTs.load_xml(results_path)
        res_dict = TiTs.xml_get_dict_from_ele(ele)[1]
        # evaluate strings in dict
        res_dict = TiTs.evaluate_strings_in_dict(res_dict)
        # # remove 'analysis_paramters' from dict
        # del res_dict['analysis_parameters']
        # stored dict has 'i' in front of isotopes. Remove that again!
        res_dict_new = res_dict.copy()
        for keys, vals in res_dict.items():
            # xml cannot take numbers as first letter of key but dicts can
            if keys[0] == 'i':
                if vals.get('file_times', None) is not None:
                    vals['file_times'] = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in vals['file_times']]
                res_dict_new[keys[1:]] = vals
                del res_dict_new[keys]

        return res_dict_new


if __name__ == '__main__':

    graphs = PlotThesisGraphics()

    ''' Theory '''

    ''' Charge Exchange '''
    # graphs.ce_efficiency()
    # graphs.ce_simulation()
    # graphs.ce_population()
    # graphs.time_res_atoms()
    # graphs.time_res_ions()  # takes a looong time!
    #
    # ''' Experiment '''
    # graphs.level2plus_be2()
    #
    # ''' Analysis '''
    # graphs.gatewidth()
    # graphs.lineshape_compare()
    # graphs.pumping_simulation()
    # graphs.a_ratio_comparison()
    # graphs.SNR_analysis()
    # graphs.voltage_deviations()
    graphs.all_spectra()
    # graphs.calibration()
    # graphs.isotope_shifts()
    # graphs.timeres_plot()
    # graphs.tof_determination()
    # graphs.plot_king_plot()
    # graphs.absradii_ni_only()
    #
    # ''' Discussion '''
    # graphs.Q_nickel55()
    graphs.absradii_chain_errorband_all()
    # graphs.absradii_neighborhood()
    # graphs.deltarad_chain_errorband()
    graphs.absradii_chain_errorband()
    graphs.three_point_indicator()
    graphs.three_point_indicator_hardcoded()
    graphs.absrad56()
    graphs.mu_nickel55()



