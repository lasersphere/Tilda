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

from math import ceil, floor, log10
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D, proj3d
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.optimize import curve_fit
from scipy import interpolate

from Measurement.XMLImporter import XMLImporter
import TildaTools as TiTs

class PlotThesisGraphics:
    def __init__(self):
        """ Folder Locations """
        # working directory:
        user_home_folder = os.path.expanduser("~")  # get user folder to access ownCloud
        owncould_path = 'ownCloud\\User\\Felix\\IKP411_Dokumente\\BeiträgePosterVorträge\\PhDThesis\\Grafiken\\'
        self.fig_dir = os.path.join(user_home_folder, owncould_path)

        """ Colors """
        # Define global color names TODO: Fill and propagate
        self.black = (0, 0, 0)
        self.blue = (0/255, 131/255, 204/255)  # TUD2b
        self.green = (153/255, 192/255, 0/255)  # TUD4b
        self.orange = (245/255, 163/255, 0/255)  # TUD7b
        self.red = (230/255, 0/255, 26/255)  # TUD9b
        self.purple = (114/255, 16/255, 133/255)  # TUD11b

        self.pmt_colors = {'scaler_0': self.blue, 'scaler_1': self.orange, 'scaler_2': self.purple, 'scaler_c012': self.red}
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
        self.colorgradient = {-3: (36/255, 53/255, 114/255),
                              -2: (0/255, 78/255, 115/255),
                              -1: (0/255, 114/255, 94/255),
                              0: (0, 0, 0),
                              1: (106/255, 139/255, 55/255),
                              2: (153/255, 166/255, 4/255),
                              3: (174/255, 142/255, 0/255)
                              }

        """ Global Style Settings """
        # https://matplotlib.org/2.0.2/users/customizing.html
        font_plot = {'family': 'sans-serif',
                     'sans-serif': 'Verdana',
                     'weight': 'normal',
                     'stretch': 'ultra-condensed',
                     'size': 11}  # wie in Arbeit
        mpl.rc('font', **font_plot)
        mpl.rc('lines', linewidth=1)

        self.text_style = {'family': 'sans-serif',
                           'size': 20.,
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

        scl = 17  # Scale Factor. Adapt to fit purpose  17 25.4
        w_a4_in = 210/scl  # A4 format pagewidth in inch
        w_rat = 0.7  # Textwidth used within A4 ~70%
        self.w_in = w_a4_in * w_rat

        h_a4_in = 297/scl  # A4 format pageheight in inch
        h_rat = 0.62  # Textheight used within A4  ~38 lines text
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
        x_0, cts_0, res_0, cts_err_0 = np.loadtxt(
            glob(os.path.join(folder, 'Voigt\\BECOLA_6501_data__*'))[0],
            delimiter=', ', skiprows=1, unpack=True)
        x_fit_0, fit_0 = np.loadtxt(
            glob(os.path.join(folder, 'Voigt\\BECOLA_6501_fit_fullShape__*'))[0],
            delimiter=', ', skiprows=1, unpack=True)
        axes[0, 0].errorbar(x_0, cts_0, cts_err_0, **self.data_style)  # data
        axes[0, 0].plot(x_fit_0, fit_0, **self.fit_style)  # fit
        axes[1, 0].errorbar(x_0, res_0, cts_err_0, **self.data_style)  # residuals
        axes[0, 0].set_title('Voigt')

        # Second Col: SatVoigt
        x_1, cts_1, res_1, cts_err_1 = np.loadtxt(glob(os.path.join(folder, 'SatVoigt\\BECOLA_6501_data__*'))[0],
                                          delimiter=', ', skiprows=1, unpack=True)
        x_fit_1, fit_1 = np.loadtxt(glob(os.path.join(folder, 'SatVoigt\\BECOLA_6501_fit_fullShape__*'))[0],
                                delimiter=', ', skiprows=1, unpack=True)
        x_mp_1, mp_1 = np.loadtxt(glob(os.path.join(folder, 'SatVoigt\\BECOLA_6501_fit_mainPeak__*'))[0],
                                delimiter=', ', skiprows=1, unpack=True)
        x_sp_1, sp_1 = np.loadtxt(glob(os.path.join(folder, 'SatVoigt\\BECOLA_6501_fit_sidePeak0__*'))[0],
                                delimiter=', ', skiprows=1, unpack=True)
        axes[0, 1].errorbar(x_1, cts_1, cts_err_1, label='data', **self.data_style)  # data
        da, la = axes[0, 1].get_legend_handles_labels()
        fi = axes[0, 1].plot(x_fit_1, fit_1, label='fit', **self.fit_style)  # fit
        mp = axes[0, 1].plot(x_mp_1, mp_1, label='main peak', **self.ch_dict(self.fit_style, {'linestyle': '--'}))  # mainPeak
        sp = axes[0, 1].plot(x_sp_1, sp_1, label='satellite peak', **self.ch_dict(self.fit_style, {'linestyle': ':'}))  # sidePeak0
        axes[0, 1].legend(handles=[da[0], fi[0], mp[0], sp[0]], bbox_to_anchor=(0.5, 1.3), loc='center', ncol=4)
        axes[1, 1].errorbar(x_1, res_1, cts_err_1, **self.data_style)  # residuals
        axes[0, 1].set_title('SatVoigt')

        # Third Col: ExpVoigt
        x_2, cts_2, res_2, cts_err_2 = np.loadtxt(
            glob(os.path.join(folder, 'ExpVoigt\\BECOLA_6501_data__*'))[0],
            delimiter=', ', skiprows=1, unpack=True)
        x_fit_2, fit_2 = np.loadtxt(
            glob(os.path.join(folder, 'ExpVoigt\\BECOLA_6501_fit_fullShape__*'))[0],
            delimiter=', ', skiprows=1, unpack=True)
        axes[0, 2].errorbar(x_2, cts_2, cts_err_2, **self.data_style)  # data
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
        plt.savefig(folder+'Lineshapes.png', dpi=self.dpi, bbox_inches='tight')
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
        plt.savefig(folder + 'TOF.png', dpi=self.dpi, bbox_inches='tight')
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
        midtof = 5.36
        gatewidth = 0.24

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
        ax_vpr.errorbar(x_axis, v_proj[0], v_proj_err[0], **self.data_style)  # x_step_projection_sc0
        ax_vpr.axes.tick_params(axis='x', direction='in',
                                top=True, bottom=True,
                                labeltop=False, labelbottom=False)
        ax_vpr.axes.tick_params(axis='y', direction='out',
                                left=True, right=False, labelleft=True)
        ax_vpr.set_yticks([1000, 2000, 3000])
        ax_vpr.set_yticklabels(['1k','2k','3k'])
        ax_vpr.set_ylabel('cts/arb.u.')

        # create time projection
        ax_tpr.errorbar(t_proj[0], np.arange(t_data_size), xerr=t_proj_err[0], **self.data_style)  # y_time_projection_sc0
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
        plt.savefig(folder + 'TRS.png', dpi=self.dpi, bbox_inches='tight')
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
                    # plot values as dots with statistical errorbars
                    ax1.errorbar(x_ax, np.array(centers), yerr=np.array(centers_d_stat), label=plt_label,
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

                    ax1.plot([x_ax[0], x_ax[-1]], [avg_shift, avg_shift], 'red')
                    # plot error of weighted average as red shaded box around that line
                    ax1.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                             [avg_shift - avg_shift_d, avg_shift - avg_shift_d,
                              avg_shift + avg_shift_d, avg_shift + avg_shift_d], 'red',
                             alpha=0.4,
                             label=plt_label)

                    # plot systematic error as lighter red shaded box around that line
                    ax1.fill([x_ax[0], x_ax[-1], x_ax[-1], x_ax[0]],
                             [avg_shift - avg_shift_d_syst - avg_shift_d,
                              avg_shift - avg_shift_d_syst - avg_shift_d,
                              avg_shift + avg_shift_d_syst + avg_shift_d,
                              avg_shift + avg_shift_d_syst + avg_shift_d],
                             'red',
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

            plt.savefig(folder + 'iso_shifts_{}.png'.format(iso), dpi=self.dpi, bbox_inches='tight')
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

        plt.savefig(folder + 'calibration_{}.png'.format(iso), dpi=self.dpi, bbox_inches='tight')
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
        isolist = ['56Ni_cal']
        scaler = 'scaler_c012'
        parameter = 'shift_iso-{}'.format('60')

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'SoftwareGateAnalysis_*'))[0]
        results = self.import_results(load_results_from)
        # get the x-axis  --> its actually the powers of two of the gatewidth (2**x)
        xax = results['60Ni_cal_6502']['scaler_012']['full_data']['xax']
        xax = TiTs.numpy_array_from_string(xax, -1, datatytpe=np.float)  # convert str to np.array
        # get the SNR results
        SNR = results['60Ni_cal_6502']['scaler_012']['full_data']['SNR']['vals']
        SNR_d = results['60Ni_cal_6502']['scaler_012']['full_data']['SNR']['d_stat']
        SNR = TiTs.numpy_array_from_string(SNR, -1, datatytpe=np.float)  # convert str to np.array
        SNR_d = TiTs.numpy_array_from_string(SNR_d, -1, datatytpe=np.float)  # convert str to np.array
        # get the center fit results
        tof_dict = {}
        for key, item in results['60Ni_cal_6502']['scaler_012']['full_data']['tof'].items():
            newkey = float(key.split('_')[-1])
            newvals = TiTs.numpy_array_from_string(item['vals'], -1, datatytpe=np.float)  # convert str to np.array
            new_d = TiTs.numpy_array_from_string(item['d_stat'], -1, datatytpe=np.float)  # convert str to np.array
            tof_dict[newkey] = {'vals': newvals, 'd_stat': new_d}

        ''' plotting '''
        # plot the midtof data
        for key, item in sorted(tof_dict.items()):
            y = item['vals']
            y_d = item['d_stat']
            col = self.colorgradient[int(key)]
            ax1.plot(xax, y, **self.ch_dict(self.fit_style, {'linewidth': 2, 'color': col}))
            ax1.fill_between(xax, y - y_d, y + y_d, label='{0:{1}.0f} bin'.format(key, '+' if key else ' '),
                             alpha=0.4, facecolor=col, edgecolor='none',
                             )

        # plot the gatewidth used in file
        # ax.axvline(x=np.log(200 * self.tof_width_sigma * self.tof_sigma[iso]) / np.log(2), color='red')
        ax1.tick_params(axis='y', labelcolor='k')
        ax1.set_ylabel('fit centroid rel. to analysis value / MHz', color='k')
        ax1.set_xlabel('gatewidth [bins]')

        # plot SNR error band
        axSNR = ax1.twinx()
        axSNR.set_ylabel('signal-to-noise ratio', color='red')  # we already handled the x-label with ax1
        axSNR.plot(xax, SNR, label='SNR', **self.ch_dict(self.fit_style, {'linewidth': 2}))
        axSNR.tick_params(axis='y', labelcolor='red')

        import matplotlib.ticker as ticker
        # convert the x-axis to real gate widths (2**x):
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x_b2, _: '{:.0f}'.format(2 ** x_b2)))
        plt.margins(0.05)
        ax1.legend(title='Offset of midTof parameter relativ to analysis value', bbox_to_anchor=(0.5, 1.2),
                   loc="center", ncol=7, columnspacing=0.7, handletextpad=0.1)


        plt.savefig(folder + 'gatewidth.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def SNR_analysis(self):
        """
        SNR ANalysis
        :return:
        """
        folder = os.path.join(self.fig_dir, 'Nickel\\Analysis\\Gates\\')

        # get data from results file
        load_results_from = glob(os.path.join(folder, 'SoftwareGateAnalysis_*'))[0]
        results = self.import_results(load_results_from)

        # Define isotopes
        iso_list = ['54Ni', '55Ni', '56Ni', '58Ni', '60Ni', '62Ni', '64Ni',]
        file_name_ex = ['Sum54Nic_9999.xml', 'Sum55Nic_9999.xml', 'Sum56Nic_9999.xml',
                        'Sum58Nic_9999.xml', 'Sum60Nic_9999.xml', 'Sum62Nic_9999.xml', 'Sum64Nic_9999.xml']  # file_name of example file for each isotope
        global_guess = {'54Ni': 5.19, '55Ni': 5.23, '56Ni': 5.28, '58Ni': 5.36, '60Ni': 5.47, '62Ni': 5.59, '64Ni': 5.68}
        scaler = 'scaler_012'

        midSNR_per_iso = []
        midSNR_d_per_iso = []
        midTOF_per_iso = []
        midTOF_d_per_iso = []

        for num, iso in enumerate(iso_list):
            base_dir = results[iso][scaler]
            file = file_name_ex[num]
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
            rec_mid_SNR = (base_dir['bestSNR_mid']['vals'][file_indx], base_dir['bestSNR_mid']['d_fit'][file_indx])
            rec_sig_SNR = (base_dir['bestSNR_sigma']['vals'][file_indx], base_dir['bestSNR_sigma']['d_fit'][file_indx])
            rec_mid_TOF = (base_dir['fitTOF_mid']['vals'][file_indx], base_dir['fitTOF_mid']['d_fit'][file_indx])
            rec_sig_TOF = (base_dir['fitTOF_sigma']['vals'][file_indx], base_dir['fitTOF_sigma']['d_fit'][file_indx])
            #
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
            scale = 10
            # xerr = [[-scale*(np.log(rec_sig_SNR[0]*2.8-rec_sig_SNR[1])/np.log(2)-2)
            #          +scale*(np.log(rec_sig_SNR[0]*2.8)/np.log(2)-2)],
            #         [scale*(np.log(rec_sig_SNR[0]*2.8+rec_sig_SNR[1])/np.log(2)-2)
            #          -scale*(np.log(rec_sig_SNR[0]*2.8)/np.log(2)-2)]]
            xerr = [[-scale * (np.log(fit_pars[2] * 2.8 - fit_errs[2]) / np.log(2) - 2)
                     + scale * (np.log(fit_pars[2] * 2.8) / np.log(2) - 2)],
                    [scale * (np.log(fit_pars[2] * 2.8 + fit_errs[2]) / np.log(2) - 2)
                     - scale * (np.log(fit_pars[2] * 2.8) / np.log(2) - 2)]]
            ax.errorbar(x=scale*(np.log(fit_pars[2]*2.8)/np.log(2)-2),
                        y=fit_pars[1]-midtof_variation_arr[0],  #-global_guess[iso]
                        xerr=xerr, yerr=fit_errs[1], **self.ch_dict(self.data_style,
                                                                                 {'color': 'w', 'markersize': 5}))

            ax.set_xlabel('gate width / bins')
            ax.set_ylabel('gate center / bins')
            x_tick_pos = [0, 10, 20, 30, 40]
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

            plt.savefig(folder + '{}_SNR.png'.format(iso), dpi=self.dpi, bbox_inches='tight')
            plt.close()
            plt.clf()

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
        m, b = self.lin_regression(iso_ax, midSNR_per_iso, (1, 0))
        ax.plot(iso_ax, self._line(iso_ax, m, b), label='linear regression', **self.fit_style)  #label='mid-tof={:.1f}\u00B7A+{:.1f}bins'.format(m, b)

        ax.set_xlim((iso_ax[0]-0.5, iso_ax[-1]+0.5))
        ax.set_xticks(iso_ax)
        ax.set_xticklabels(iso_list)
        ax.set_xlabel('Isotope')

        ax.set_ylabel('gate center / bins')

        ax.legend(bbox_to_anchor=(0.5, 1.15), loc="center", ncol=3, columnspacing=1.2, handletextpad=0.4)  # title='Scaler', columnspacing=0.5, bbox_to_anchor=(0.5, 1.1), loc="lower right"

        plt.savefig(folder + 'gate_center.png', dpi=self.dpi, bbox_inches='tight')
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
        width, height = 0.5, 0.5
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
            cen_art = axes[num].axes.axvline(x=isoshift, label='centroid', linestyle='--', color=self.blue, lw=2)
            dat_art = axes[num].errorbar(x, cts, cts_err, label='data', **self.data_style)  # data
            fit_art = axes[num].plot(x_fit, fit, label='fit', **self.fit_style)  # fit
            # create legend
            # axes[num].legend((dat_art, fit_art, cen_art), ('data', 'fit', 'centroid'),
            #                  bbox_to_anchor=(0.5, 1.3), loc='center', ncol=3)
            # Place isotope name in top right corner
            axes[num].text(0.85, 0.9, iso,
                           horizontalalignment='left', verticalalignment='top',
                           transform=axes[num].transAxes,
                           **self.ch_dict(self.text_style, {'size': 11})
                           )

        custom_ticks = False
        if custom_ticks:
            # set y-axes:
            f.text(-0.02, 0.5, 'cts / arb.u.', ha='center', va='center',
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
            f.text(-0.08, 0.5, 'Counts / arb.u.', ha='center', va='center',
                   rotation='vertical')  # common label for all y-axes
            for ax in range(len(iso_list)):
                axes[ax].axes.tick_params(axis='y', direction='in',
                                          left=True, right=True,  # ticks left and right
                                          labelleft=False, labelright=False)  # no ticklabels anywhere. Because they look cluttered
        # set x-axis
        axes[-1].set_xlabel(r'Frequency relative to $\nu_0^{60}$ / '+'{}'.format(x_unit))
        axes[-1].set_xlim(-2.5, 1.5)
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

        plt.savefig(folder + 'all_spectra.png', dpi=self.dpi, bbox_inches='tight')
        plt.close()
        plt.clf()

    def level2plus_be2(self):
        """

        :return:
        """
        # Specify in which folder input and output should be found
        folder = os.path.join(self.fig_dir, 'Nickel\\General\\LevelE_BE2\\')

        fig, ax1 = plt.subplots()
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
        ax1.bar(masses_e2p, e2plus_data, **self.bar_style)
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
        ax1.set_ylabel('2+ Energy Level / eV', color=self.blue)
        ax1.tick_params(axis='y', labelcolor=self.blue, color=self.blue, which='both')
        ax1.set_ylim((0, 4500))
        ax1.set_yticks([0, 1000, 2000, 3000, 4000])
        ax2.set_ylabel('B(E2) / (e$^2$b$^2$)', color=self.red)
        ax2.set_ylim((0.005, 0.12))
        ax2.tick_params(axis='y', labelcolor=self.red, color=self.red, which='both')

        plt.savefig(folder + 'E2.png', dpi=self.dpi, bbox_inches='tight')
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


        ax.set_ylabel('scan number', labelpad=0)
        #ax.set_zlabel('Voltage deviation / V')

        # ax.set_axis_off()
        ax.set_xlabel('DAC set value / V', labelpad=0)
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

        f.savefig(folder + 'volt_dev_3D.png', dpi=self.dpi, bbox_inches='tight', pad_inches=0.2)
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
        plt.savefig(folder + 'volt_dev_2D.png', dpi=self.dpi, bbox_inches='tight')
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
        for keys, vals in res_dict.items():
            # xml cannot take numbers as first letter of key but dicts can
            if keys[0] == 'i':
                vals['file_times'] = [datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in vals['file_times']]
                res_dict[keys[1:]] = vals
                del res_dict[keys]

        return res_dict


if __name__ == '__main__':

    graphs = PlotThesisGraphics()

    graphs.SNR_analysis()
    # graphs.voltage_deviations()
    # graphs.level2plus_be2()
    # graphs.all_spectra()
    # graphs.gatewidth()
    # graphs.calibration()
    # graphs.isotope_shifts()
    # graphs.timeres_plot()
    # graphs.lineshape_compare()
    # graphs.tof_determination()



