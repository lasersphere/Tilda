"""
Created on 2021-02-23

@author: fsommer

Module Description:  Plotting Spectra of all Isotopes in a vertical grid with the same x-axis. This is a nice view to
    compare shifts and often used in presentations/publications.
INPUT: The data and fit should be exported from PolliFit's Interactive fit "Save Fit as ASCII" option. This produces
    (at least) one file with "data" in its name and one with "fullShape". These are the keywords this script needs.
    Also the isotope name used in the isotope list for this script must be in these filenames BEFORE the keyword.
    That may already be the case since the export naming scheme is "isotope_type__datetime.txt" but you might have
    named your isotopes fancy. In that case just add the isotope manually to the (beginning) of all filenames.
CUSTOMIZE: You can do a lot of customization inside the script. I tried to document the most things, so it shouldn't be
    too hard to adapt the output to your liking.
"""

import os
from glob import glob

from math import ceil, floor, log10
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


class PlotIsoSpectraStacked:
    def __init__(self, folder, isotopes):
        """ Folder Locations """
        # working directory: ADAPT THIS at function call on bottom of this script!!
        self.fig_dir = folder
        self.iso_list = isotopes

        """ Colors """
        # Define some global color names based on "Das Bild der TU Darmstadt"
        # https://www.intern.tu-darmstadt.de/media/medien_stabsstelle_km/services/medien_cd/das_bild_der_tu_darmstadt.pdf
        self.black = (0, 0, 0)
        self.blue = (0 / 255, 131 / 255, 204 / 255)  # TUD2b
        self.green = (153 / 255, 192 / 255, 0 / 255)  # TUD4b
        self.orange = (245 / 255, 163 / 255, 0 / 255)  # TUD7b
        self.red = (230 / 255, 0 / 255, 26 / 255)  # TUD9b
        self.purple = (114 / 255, 16 / 255, 133 / 255)  # TUD11b

        """ Global Style Settings """
        # https://matplotlib.org/2.0.2/users/customizing.html
        font_plot = {'family': 'sans-serif',
                     'sans-serif': 'Verdana',
                     'weight': 'normal',
                     'stretch': 'ultra-condensed',
                     'size': 11}
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

        """ Global Size Settings """
        # You can scale the size of the output graphics on many different places within this script. See what works best
        # Thesis document proportions
        self.dpi = 300  # resolution of Graphics

        scl = 17  # Scale Factor. Adapt to fit purpose
        w_a4_in = 210 / scl  # A4 format page width in inch
        w_rat = 0.7  # Text width used within A4 ~70% (will depend on your LATEX/Word whatever)
        self.w_in = w_a4_in * w_rat  # final assumed page width in inches

        h_a4_in = 297 / scl  # A4 format pageheight in inch
        h_rat = 0.62  # Text height used within A4  ~38 lines text (will depend on your LATEX/Word whatever)
        self.h_in = h_a4_in * h_rat  # final assumed page height in inches

    def all_spectra(self):
        """
        Plot the spectra for all isotopes on top of each other together with a line for the centroid.
        :return:
        """
        # Get global input
        folder = self.fig_dir
        iso_list = self.iso_list

        # Create plots for all isotopes
        widths = [1]  # only one column, relative width = 1
        heights = [1] * iso_list.__len__()  # as many rows as isotopes. All same relative height.
        gs_kw = dict(width_ratios=widths, height_ratios=heights)  # property dict for gridspec
        f, axes = plt.subplots(nrows=iso_list.__len__(), ncols=1,
                               sharex=True, sharey=False,  # Share x-axes between all plots. Leave Y-axis free
                               gridspec_kw=gs_kw)  # gridspec to arrange the subplots

        # define output size of figure
        width, height = 0.7, 0.8  # relative to page width defined in init above.
        f.set_size_inches((self.w_in * width, self.h_in * height))

        for num, iso in enumerate(sorted(iso_list)):
            ''' Get data '''
            # sorted and enumerated list to put each isotope at the right location
            x, cts, res, cts_err = np.loadtxt(  # get the data. "data" is the keyword in the files
                glob(os.path.join(folder, '*{}*data*'.format(iso)))[0],  # name of isotope must be in filename
                delimiter=', ', skiprows=1, unpack=True)  # skip first row (header), comma-separated, unpack into vars
            x_fit, fit = np.loadtxt(  # get the fit. "fullShape" is the keyword in the files
                glob(os.path.join(folder, '*{}*fullShape*'.format(iso)))[0],  # name of isotope must be in filename
                delimiter=', ', skiprows=1, unpack=True)  # skip first row (header), comma-separated, unpack into vars

            # transform x-axis to GHz because that usually makes a nicer axis.
            x_unit = 'GHz'  # will be used for x-axis label. Set to 'MHz' if you prefer that.
            x = x / 1000  # original input is in MHz so scale this to GHz. For MHz replace with /1
            x_fit = x_fit / 1000  # original input is in MHz so scale this to GHz. For MHz replace with /1

            ''' Do the plotting '''
            axes[num].errorbar(x, cts, cts_err, label='data', **self.data_style)  # plot data with errorbars
            axes[num].plot(x_fit, fit, label='fit', **self.fit_style)  # plot fit
            # Place isotope name in top left corner
            axes[num].text(0.05, 0.9, iso,  # position settings are kind of arbitrary. See what works best for you.
                           horizontalalignment='left', verticalalignment='top',
                           transform=axes[num].transAxes,
                           **self.ch_dict(self.text_style, {'size': 11})
                           )

        ''' Adapt the axes '''
        # set y-axes:
        f.text(-0.02, 0.5, 'cts / arb.u.', ha='center', va='center', rotation='vertical')  # common label for all y-axes
        for ax in range(len(iso_list)):
            yti = axes[ax].get_yticks()
            # determine a few nice ticks
            n_ticks = 3  # number of ticks we aim for on the new axis (rounding stuff below can change it a little)
            significant_num = 1000  # cut at 1000 cts? If you want full numbers set to 1 (or 10, 100, whatever)
            sig_symb = 'k'  # symbol to be attached to numbers. e.g. 1000 -> 1k. Set empty string if not used.
            axrange = yti.max() - yti.min()  # get the range currently spanned by the ticks
            newspacing = round(axrange // (n_ticks), -int(floor(
                log10(abs(axrange // (n_ticks))))))  # A lot of rounding to get some reasonable spacing with nice numbers
            newspacing = ceil(
                newspacing / significant_num) * significant_num  # adapt spacing to the above set significant number if it is below.
            if yti.min() > 0:  # log10 doesnt't work for 0 of course
                newmin = round(yti.min(),
                               -int(floor(log10(abs(yti.min())))) + 1)  # A starting value for the lowest tick
                newmin = round(newmin, -int(log10(significant_num)))  # again adapt to significant number
            else:  # if yti.min=0, then the axis should start at 0
                newmin = 0
            newticks = np.arange(newmin, yti.max(), newspacing)
            # set the new labels
            axes[ax].set_yticks(newticks)
            axes[ax].set_yticklabels(['{:.0f}{}'.format(i // significant_num, sig_symb) for i in newticks])
            axes[ax].axes.tick_params(axis='y', direction='in',
                                      left=True, right=True,  # ticks left and right
                                      labelleft=True,
                                      labelright=False)  # no ticklabels anywhere. Because they look cluttered
        # set x-axis
        axes[-1].set_xlabel('relative frequency / {}'.format(x_unit))
        # axes[-1].set_xticks([-1000, -700, -500, -300, -100])  # for custom ticks

        # set label
        axes[0].legend(bbox_to_anchor=(0.5, 1.3), loc='center', ncol=2)

        plt.savefig(folder + 'all_iso_spectra.png', dpi=self.dpi, bbox_inches='tight')
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


if __name__ == '__main__':

    ''' Define in which folder the input files are found (and the output will be saved) '''
    user_home_folder = os.path.expanduser("~")  # get user folder to access ownCloud
    owncould_path = 'ownCloud\\User\\Felix\\IKP411_Dokumente\\BeiträgePosterVorträge\\PhDThesis\\Grafiken\\Nickel\\Analysis\\AllSpectra\\'
    in_out_folder = os.path.join(user_home_folder, owncould_path)

    ''' Define which Isotopes you want stacked. Each iso needs key words "data" and "fullShape" files. '''
    isolist = ['55Ni', '56Ni', '58Ni', '60Ni']  # These EXACT isotope names must be in the file-names BEFORE the keyword

    ''' now run the script '''
    graphs = PlotIsoSpectraStacked(in_out_folder, isolist)
    graphs.all_spectra()
