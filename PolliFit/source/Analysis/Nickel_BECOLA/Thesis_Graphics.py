"""
Created on 2021-02-12

@author: fsommer

Module Description:  Plotting of all Graphics for the Thesis
"""

import os
from glob import glob

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from scipy.optimize import curve_fit

from Measurement.XMLImporter import XMLImporter

class PlotThesisGraphics:
    def __init__(self):
        """ Folder Locations """
        # working directory:
        user_home_folder = os.path.expanduser("~")  # get user folder to access ownCloud
        owncould_path = 'ownCloud\\User\\Felix\\IKP411_Dokumente\\BeiträgePosterVorträge\\PhDThesis\\Grafiken\\'
        self.fig_dir = os.path.join(user_home_folder, owncould_path)

        """ Global Style Settings """
        # https://matplotlib.org/2.0.2/users/customizing.html
        font_plot = {'family': 'sans-serif',
                     'sans-serif': 'Verdana',
                     'weight': 'normal',
                     'stretch': 'ultra-condensed',
                     'size': 11}
        mpl.rc('font', **font_plot)
        mpl.rc('lines', linewidth=1)

        self.point_style = {'linestyle': '',
                           'marker': 'o',
                           'markersize': 2,
                           'color': 'black'}

        self.data_style = self.ch_dict(self.point_style, {'capsize': 2})

        self.fit_style = {'linestyle': '-',
                          'linewidth': 1,
                          'marker': '',
                          'color': 'red'}

        """ Global Size Settings """
        # Thesis document proportions
        self.dpi = 300  # resolution of Graphics

        scl = 17
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

        # Define Color map TODO: Define global
        c_dict = {  # TUD 2b-10b
                    'red'  :  ((0., 0/255, 0/255),     (1/8, 0/255, 0/255),     (2/8, 153/255, 153/255), (3/8, 201/255, 201/255), (4/8, 253/255, 253/255), (5/8, 245/255, 245/255), (6/8, 236/255, 236/255), (7/8, 230/255, 230/255), (1, 166/255, 166/255)),
                    'green':  ((0., 131/255, 131/255), (1/8, 157/255, 157/255), (2/8, 192/255, 192/255), (3/8, 212/255, 212/255), (4/8, 202/255, 202/255), (5/8, 163/255, 163/255), (6/8, 101/255, 101/255), (7/8, 0/255, 0/255),     (1, 0/255, 0/255)),
                    'blue' :  ((0., 204/255, 204/255), (1/8, 129/255, 129/255), (2/8, 0/255, 0/255),     (3/8, 0/255, 0/255),     (4/8, 0/255, 0/255),     (5/8, 0/255, 0/255),     (6/8, 0/255, 0/255),     (7/8, 26/255, 26/255),   (1, 132/255, 132/255))
                    }
        custom_cmap = mpl.colors.LinearSegmentedColormap('my_colormap', c_dict, 1024)

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
        im =ax_trs.pcolormesh(x_axis, np.arange(t_data_size), Z, cmap=custom_cmap)
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


if __name__ == '__main__':

    graphs = PlotThesisGraphics()

    graphs.timeres_plot()
    graphs.lineshape_compare()
    graphs.tof_determination()



