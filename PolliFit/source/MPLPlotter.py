'''
Created on 29.04.2014

@author: hammen
'''
import datetime
import os

import matplotlib
import matplotlib.figure as Figure
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
from matplotlib.widgets import Button
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from Spectra.AsymmetricVoigt import AsymmetricVoigt
from SPFitter import SPFitter
from Spectra.FullSpec import FullSpec
from copy import deepcopy

import Physics
import random

matplotlib.use('Qt5Agg')

def colAcolPlot(x_data, plotdata, error):
    plt.errorbar(x_data, plotdata, yerr=error, fmt='o', linestyle='-')
    plt.ylabel('transition frequency / MHz')
    plt.xlabel('measurement number')
    plt.axis([0, len(plotdata)+1, min(plotdata)-max(error)*1.2, max(plotdata)+max(error)*1.2])

def AlivePlot(x_data, plotdata, error, refData):

    arr = np.asarray
    y_err = arr(error).T
    n=0
    for data in plotdata:

        plt.errorbar(x_data, data, yerr=y_err, fmt='o', linestyle ='-', label=refData[n][0])
        n=n+1
        #plt.errorbar(x_data, data, fmt='o', linestyle ='-')
    #plt.ylabel('relative discrepancy [ppm]')
    plt.ylabel('voltage [V]')
    plt.xlabel('measurement number')
    #Ab hier zum Plotten der Legende
    legend=plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3, fancybox=True, shadow=True)
    frame = legend.get_frame()
    frame.set_facecolor('0.90')
    for label in legend.get_texts():
        label.set_fontsize('large')
    for label in legend.get_lines():
        label.set_linewidth(1.5)  # the legend line width


def plot(*args):
    for a in args:
        plt.plot(a[0], a[1])
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel('Intensity [cts]')
    plt.xlabel('Frequency [MHz]')


def plotFit(fit, color='-r', x_in_freq=True, plot_residuals=True, fontsize_ticks=12,
            plot_data=True, add_label='', plot_side_peaks=True, data_fmt='k.', save_plot=False, save_path='C:\\'):
    kepco = False
    if fit.meas.type == 'Kepco':
        x_in_freq = False
        kepco = True
    try:
        if fit.meas.seq_type == 'kepco':
            x_in_freq = False
            kepco = True
    except Exception:
        pass
    if x_in_freq:
        data = fit.meas.getArithSpec(*fit.st)
        for i, e in enumerate(data[0]):
            v = Physics.relVelocity(Physics.qe * e, fit.spec.iso.mass * Physics.u)
            v = -v if fit.meas.col else v

            f = Physics.relDoppler(fit.meas.laserFreq, v) - fit.spec.iso.freq
            data[0][i] = f
        plotdat = fit.spec.toPlot(fit.par)
    else:
        data = fit.meas.getArithSpec(*fit.st)
        plotdat = fit.spec.toPlotE(fit.meas.laserFreq, fit.meas.col, fit.par)
    shape = None
    try:
        shape = fit.spec.shape
    except Exception as e:
        print('warning, spectra has no shape maybe kepco fit? Than its ok. error msg: %s' % e)
    main_peaks_plot_data = []
    all_side_peaks_plot_data = []

    if isinstance(shape, AsymmetricVoigt):
        main_peaks = deepcopy(fit)
        main_peaks.spec.iso.shape['name'] = 'Voigt'
        main_full_spec = FullSpec(main_peaks.spec.iso)
        main_fit = SPFitter(main_full_spec, main_peaks.meas, main_peaks.st)
        for i, par in enumerate(main_peaks.npar):  # pass fit results to new plot
            if par in main_fit.npar:
                main_fit.par[main_fit.npar.index(par)] = main_peaks.par[i]
        # create a list of side peaks with the pars from the main peak
        side_peaks = [deepcopy(main_peaks) for i in range(main_peaks.par[main_peaks.npar.index('nOfPeaks')])]
        if x_in_freq:
            main_peaks_plot_data = main_fit.spec.toPlot(main_fit.par)
        else:
            main_peaks_plot_data = main_fit.spec.toPlotE(main_fit.meas.laserFreq, main_fit.meas.col, main_fit.par)

        # now plot side peaks:
        for side_peak_num, side_peak in enumerate(side_peaks):
            side_peaks_spec = FullSpec(side_peak.spec.iso)  # create FullSpec for each side peak
            side_peaks_fit = SPFitter(side_peaks_spec, side_peak.meas, side_peak.st)  # .. aand fit
            # calc intensity for this peak:
            asym_intensity = side_peak.par[side_peak.npar.index('IntAsym')] / (2 ** side_peak_num)
            asym_center_energy = side_peak.par[side_peak.npar.index('centerAsym')]  # eV or MHz

            diff_doppl_MHz = shape.diff_doppl

            side_peak_freq = asym_center_energy * diff_doppl_MHz * (side_peak_num + 1)

            # copy relevant parameters from asymmetric to normal voigt:
            for i, par in enumerate(side_peak.npar):
                if par in side_peaks_fit.npar:
                    new_par = side_peak.par[i]
                    if par == 'center':   # shift center of this plot
                        new_par += side_peak_freq
                    elif 'Int' in par:  # calc intensity of this plot
                        new_par *= asym_intensity
                    side_peaks_fit.par[side_peaks_fit.npar.index(par)] = new_par
            if x_in_freq:
                side_peaks_plot_data = side_peaks_fit.spec.toPlot(side_peaks_fit.par)
            else:
                side_peaks_plot_data = side_peaks_fit.spec.toPlotE(
                    side_peak.meas.laserFreq, side_peak.meas.col, side_peaks_fit.par)
            all_side_peaks_plot_data.append(side_peaks_plot_data)
        # color = '-b'

    fig = plt.figure(1, (8, 8))
    fig.patch.set_facecolor('white')

    # save plot data as ASCII
    path_clear = False
    if save_plot:
        x = "Relative frequency / MHz" if x_in_freq else "Ion kinetic energy / eV"
        if not os.path.exists(save_path):
            try:
                os.makedirs(save_path)
                path_clear = True
            except Exception as e:
                print('saving directory has not been created. Writing permission in DB directory? error msg: %s' % e)
        else:
            path_clear = True


    ax1 = plt.axes([0.15, 0.35, 0.8, 0.50])
    data_line = None
    if plot_data:
        data_line = plt.errorbar(data[0], data[1], yerr=data[2], fmt=data_fmt, label=fit.meas.file)
        # data_line.set_label(fit.meas.file)
    plt_label = 'straight' if kepco else str(fit.spec.iso.shape['name'])
    main_plot = plt.plot(plotdat[0], plotdat[1], color, label=plt_label + add_label)
    main_plot_color = main_plot[0].get_color()
    side_peak_lines = []
    if plot_side_peaks:
        if len(main_peaks_plot_data):
            # plot main peak dotted but in same color as host line
            side_peak = plt.plot(main_peaks_plot_data[0], main_peaks_plot_data[1],
                                 color=main_plot_color, label='main peak' + add_label, linestyle=':')
            side_peak_lines += side_peak[0],
            if save_plot and path_clear:
                p = os.path.join(save_path, os.path.splitext(fit.meas.file)[0] + "_fit_mainPeak_" + datetime.datetime.today().strftime('_%Y-%m-%d_%H-%M-%S.txt'))
                f = open(p, 'w')
                f.write(x + ", Main Peak cts / a.u.\n")
                for i in range(len(main_peaks_plot_data[0])):
                    f.write(str(main_peaks_plot_data[0][i]) + ", " + str(main_peaks_plot_data[1][i]) + "\n")
                f.close()
                print("Saved to file ", p)
        for side_peak_num, side_peaks_plot_data in enumerate(all_side_peaks_plot_data):
            # plot side peaks dashed / dashdot alternating with number of side peaks in same color as main peak
            line_style = '--' if side_peak_num % 2 == 0 else '-.'
            side_peak = plt.plot(side_peaks_plot_data[0], side_peaks_plot_data[1],
                                 linestyle=line_style, color=main_plot_color,
                                 label='satellite peak #%d' % (side_peak_num + 1) + add_label)
            side_peak_lines += side_peak[0],
            if save_plot and path_clear:
                p = os.path.join(save_path, os.path.splitext(fit.meas.file)[0] + "_fit_sidePeak" + str(side_peak_num) + "_" + datetime.datetime.today().strftime('_%Y-%m-%d_%H-%M-%S.txt'))
                f = open(p, 'w')
                f.write(x + ", SidePeak " + str(side_peak_num) + " cts / a.u.\n")
                for i in range(len(side_peaks_plot_data[0])):
                    f.write(str(side_peaks_plot_data[0][i]) + ", " + str(side_peaks_plot_data[1][i]) + "\n")
                f.close()
                print("Saved to file ", p)

    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    if plot_residuals:
        ax2 = plt.axes([0.15, 0.1, 0.8, 0.2], sharex=ax1)
        plt.errorbar(data[0], fit.calcRes(), yerr=data[2], fmt='k.')
        ax2.get_xaxis().get_major_formatter().set_useOffset(False)
        ax2.locator_params(axis='y', nbins=5)

    plt.ylabel('residuals / a.u.', fontsize=fontsize_ticks)
    ax1.set_ylabel('cts / a.u.', fontsize=fontsize_ticks)
    if x_in_freq:
        plt.xlabel('relative frequency / MHz', fontsize=fontsize_ticks, labelpad=fontsize_ticks/2)
    elif kepco:
        plt.xlabel('Line Voltage / V', fontsize=fontsize_ticks, labelpad=fontsize_ticks)
    else:
        plt.xlabel('Ion kinetic energy / eV', fontsize=fontsize_ticks)
    # print(plotdat[0][-2000:-100])
    # print(plotdat[1][-2000:-100])
    # np.set_printoptions(threshold=2000)
    # print(data[1])

    if save_plot and path_clear:
        p = os.path.join(save_path, os.path.splitext(fit.meas.file)[0] + "_data_" + datetime.datetime.today().strftime('_%Y-%m-%d_%H-%M-%S.txt'))
        f = open(p, 'w')
        f.write(x + ", Data cts / a.u., Fit residuals cts / a.u., Data uncertainty cts / a.u.\n")
        res = fit.calcRes()
        for i in range(len(data[0])):
            f.write(str(data[0][i]) + ", " + str(data[1][i]) + ", " + str(res[i]) + ", " + str(data[2][i]) + "\n")
        f.close()
        print("Saved to file ", p)
        p = os.path.join(save_path, os.path.splitext(fit.meas.file)[0] + "_fit_fullShape_" + datetime.datetime.today().strftime('_%Y-%m-%d_%H-%M-%S.txt'))
        f = open(p, 'w')
        f.write(x + ", Full fit cts / a.u.\n")
        for i in range(len(plotdat[0])):
            f.write(str(plotdat[0][i]) + ", " + str(plotdat[1][i]) + "\n")
        f.close()
        print("Saved to file ", p)

    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    if data_line is None:
        lines = [main_plot[0]] + side_peak_lines
    else:
        lines = [data_line, main_plot[0]] + side_peak_lines
    labels = [each.get_label() for each in lines]
    fig.legend(lines, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.15, 0.8, 0.8, 0.2), mode='expand',
               fontsize=fontsize_ticks+2, numpoints=1)


def plotMoments(cts, q=True,fontsize_ticks=10):
    if q:
        fig = plt.figure(1, (8, 6))
        fig.clear()
        fig.patch.set_facecolor('white')
        ax1 = plt.axes([0.1, 0.1, 0.85, 0.85])
        ax = plt.gca()
        ax.set_ylabel('Q (b) ')
    else:
        fig2 = plt.figure(2, (8, 6))
        fig2.clear()
        fig2.patch.set_facecolor('white')
        ax1 = plt.axes([0.1, 0.1, 0.85, 0.85])
        ax = plt.gca()
        ax.set_ylabel(r'$\mu$ ($\mu_N$) ')

    markerlist = ['s', 'o',  '>', '<', 'v', 'h', 'd', '*', 'p']
    x = 0
    for i in cts.keys():
        plt.errorbar(cts[i][0], cts[i][1], cts[i][2], label=str(i), linestyle='dotted',
                     marker=markerlist[x], ms=10, elinewidth=2)
        x += 1
        if x > 8:
            x = 0
    plt.legend()
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    plt.xticks(rotation=25)

    ax.set_xlabel('mass number A')
    plt.gcf().set_facecolor('white')

    ax.set_xmargin(0.05)
    plt.show()


def plotAverage(date, cts, errs, avg, stat_err, syst_err, forms=('k.', 'r'), showing = False, save_path='', ylabel=''):
    # avg, stat_err, sys_err = Analyzer.combineRes(iso, par, run, db, print_extracted=False)
    # val, errs, date = Analyzer.extract(iso, par, run, db, prin=False)
    try:
        fig = plt.figure(1, (8, 8))
        fig.patch.set_facecolor('white')

        ax = plt.axes()
        date = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in date]
        plt.subplots_adjust(bottom=0.2)
        plt.xticks(rotation=25)
        xfmt = DateFormatter('%Y-%m-%d %H:%M:%S')
        ax.xaxis.set_major_formatter(xfmt)
        ax.set_ylabel(ylabel)
        ax.set_xmargin(0.05)
        ax.ticklabel_format(useOffset=False, axis='y')

        plt.errorbar(date, cts, yerr=errs, fmt=forms[0], axes=ax)

        # plot the mean value and the errorband:
        err_p = avg + abs(stat_err) + abs(syst_err)
        err_m = avg - abs(stat_err) - abs(syst_err)
        err_p_l = np.full((2,), err_p)
        err_m_l = np.full((2,), err_m)
        if len(date) == 1:
            date = [date[0] - datetime.timedelta(seconds=0.5),
                    date[0] + datetime.timedelta(seconds=0.5)]
        x = (sorted(date)[0], sorted(date)[-1])
        y = (avg, avg)
        plt.plot(x, y, forms[1],
                 label='%s: %.5f +/- %.5f' % (ylabel, avg, abs(stat_err) + abs(syst_err)))
        plt.legend()
        plt.fill_between(x, err_p_l, err_m_l, alpha=0.5)

        if save_path:
            d = os.path.dirname(save_path)
            if not os.path.exists(d):
                os.makedirs(d)
            print('saving combined plot to: %s' % save_path)
            save(save_path)
        if showing:
            show()
    except Exception as e:
        print('error while plotting average: %s' % e)
    return ax


def show(block=True):
    plt.show(block=block)


def ion():
    plt.ion()


def save(file):
    plt.savefig(file, dpi=100)


def clear():
    plt.clf()


def clear_ax(axes):
    axes.cla()


def draw():
    plt.draw()


def pause(time):
    plt.pause(time)


def close_fig(fig=None):
    if fig is None:
        fig = plt.gcf()
    if fig is not None:
        plt.close(fig)


def close_all_figs():
    for i in plt.get_fignums():
        plt.figure(i)
        close_fig()


def plt_axes(axes, plotlist):
    # axes.clear()   # really really time consuming!!
    axes.plot(*plotlist)


def line2d(x_data, y_data, line_color):
    return Line2D(x_data, y_data, color=line_color, drawstyle='steps-mid')


def get_current_axes():
    return plt.gca()


def get_current_figure():
    return plt.gcf()


def setup_image_figure(facecolor='white'):
    fig = plt.figure(facecolor=facecolor)
    axes = [[0, 0, 0], [0, 0, 0, 0], [0]]

    axes[0][0] = fig.add_subplot(111)
    divider = make_axes_locatable(axes[0][0])
    axes[0][1] = divider.append_axes("right", size="5%", pad=0.05)
    axes[0][2] = divider.append_axes("right", 2, pad=0.35, sharey=axes[0][0])
    axes[1][0] = divider.append_axes("bottom", 2, pad=0.1, sharex=axes[0][0])
    axes[1][1] = plt.axes([0.6, 0.2, 0.15, 0.15], axisbg='white')
    axes[1][2] = plt.axes([0.8, 0.2, 0.15, 0.15], axisbg='white')
    axes[1][3] = plt.axes([0.75, 0.12, 0.15, 0.05], axisbg='white')
    axes[2][0] = plt.axes([0.75, 0.05, 0.15, 0.05], axisbg='white')  # slider

    return fig, axes


def image_plot(fig, axes, cbax, image_data, extent, aspect='equal'):
    img = axes.imshow(image_data, extent=extent, origin='lower',
                      aspect=aspect, interpolation='none')
    axes.set_ylabel('time [ns]')
    axes.set_xlabel('DAC voltage [V]')
    cb = fig.colorbar(img, cax=cbax)
    # cb = None
    # draw()  # removed because otherwise a figure might be created.
    return img, cb


def configure_image_plot(fig, im_ax, cb_ax, pipeData, volt_array_tr, time_array_tr, pmt_num, track_name):
    filename = os.path.split(pipeData['pipeInternals']['activeXmlFilePath'])[1]
    fig.canvas.set_window_title('plot %s track: %s pmt: %s' % (filename, track_name, str(pmt_num)))
    steps = volt_array_tr.shape[0]
    bins = time_array_tr.shape[0]
    v_min = volt_array_tr[0]
    v_max = volt_array_tr[-1]
    t_min = time_array_tr[0] - abs(time_array_tr[1] - time_array_tr[0]) / 2
    t_max = time_array_tr[-1] + abs(time_array_tr[1] - time_array_tr[0]) / 2
    # -5 due to resolution of 10ns so events with timestamp e.g. 10 (= 100ns) will be plotted @ 95 to 105 ns

    extent = [v_min, v_max, t_min, t_max]
    initial_2d_arr = np.zeros((steps, bins), dtype=np.uint32)
    image, colorbar = image_plot(fig, im_ax, cb_ax, np.transpose(initial_2d_arr), extent, 'auto')
    im_ax.xaxis.set_ticks_position('top')
    im_ax.xaxis.set_label_position('top')
    return image, colorbar


def setup_projection(axes, volt_array_tr, time_array_tr):
    tproj_ax = axes[0][2]
    vproj_ax = axes[1][0]
    t_cts = np.zeros(time_array_tr.shape)
    v_cts = np.zeros(volt_array_tr.shape)
    v_min = min(volt_array_tr)
    v_max = max(volt_array_tr)
    vproj_line = vproj_ax.add_line(line2d(volt_array_tr, v_cts, 'r'))
    vproj_ax.set_xlim(v_min, v_max)
    vproj_ax.autoscale(enable=True, axis='y', tight=True)

    t_min = min(time_array_tr)
    t_max = max(time_array_tr)
    tproj_line = tproj_ax.add_line(line2d(t_cts, time_array_tr, 'r'))
    tproj_line.set_drawstyle('default')
    tproj_ax.set_ylim(t_min, t_max)
    tproj_ax.autoscale(enable=True, axis='x', tight=True)
    tproj_ax.set_xlabel('cts')
    tproj_ax.xaxis.set_label_position('top')
    tproj_ax.xaxis.set_ticks_position('top')
    tproj_ax.yaxis.set_ticks_position('right')
    vproj_ax.set_ylabel('cts')
    vproj_ax.set_xlabel('DAC voltage [V]')
    return vproj_line, tproj_line


def add_patch(axes, extent):
    """
    adds a patch to given axes
    extent = [x, y, width, height]
    :return: patch
    """
    patch = axes.add_patch(patches.Rectangle((extent[0], extent[1]), extent[2], extent[3],
                                              fill=False, ec='white'))
    return patch


def add_rect_select(axes, con_func, minspanx, minspany):
    rect_selector = RectangleSelector(axes, con_func, drawtype='box',
                                      useblit=True, button=[1, 3],
                                      minspanx=minspanx,
                                      minspany=minspany,
                                      spancoords='data')
    return rect_selector


def add_radio_buttons(axes, labels, active, con_func):
    radio_but = RadioButtons(axes, labels, active=active)
    radio_con = radio_but.on_clicked(con_func)
    return radio_but, radio_con


def add_button(axes, label, con_func):
    button = Button(axes, label)
    button_con = button.on_clicked(con_func)
    return button, button_con


def add_slider(axes, label, valmin, valmax, confunc, valinit=0.5,
               valfmt=u'%1.2f', closedmin=True, closedmax=True,
               slidermin=None, slidermax=None, dragging=True, **kwargs):
    slider = Slider(axes, label, valmin, valmax, valinit=valinit,
                    valfmt=valfmt, closedmin=closedmin, closedmax=closedmax,
                    slidermin=slidermin, slidermax=slidermax, dragging=dragging, **kwargs)
    slider_con = slider.on_changed(confunc)
    return slider, slider_con


def create_figure_widget(parent_widget, facecolor='white'):
    """
    function to include a simple matplotlib plotting widget within a parent widget.
    :param parent_widget:
    :param facecolor:
    :return: fig, canvas, toolbar
    """
    fig = Figure.Figure(facecolor=facecolor)
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, parent_widget)

    return fig, canvas, toolbar


def create_canvas_and_toolbar_to_figure(fig, parent_widget):
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, parent_widget)
    return canvas, toolbar


def setup_image_widget(parent_widget, facecolor='white'):
    fig = Figure.Figure(facecolor=facecolor)
    canvas = FigureCanvas(fig)
    toolbar = NavigationToolbar(canvas, parent_widget)
    axes = {'image': fig.add_subplot(111)}
    divider = make_axes_locatable(axes['image'])
    axes['colorbar'] = divider.append_axes("right", size="5%", pad=0.05)
    axes['t_proj'] = divider.append_axes("right", 2, pad=0.35, sharey=axes['image'])
    axes['v_proj'] = divider.append_axes("bottom", 2, pad=0.1, sharex=axes['image'])
    axes['t_proj'].yaxis.set_ticks_position('right')
    axes['sum_proj'] = axes['v_proj'].twinx()
    axes['sum_proj'].set_axes_locator(axes['v_proj'].get_axes_locator())

    return fig, axes, canvas, toolbar


def configure_image_plot_widget(fig, im_ax, cb_ax, volt_array_tr, time_array_tr):
    steps = volt_array_tr.shape[0]
    bins = time_array_tr.shape[0]
    v_min = volt_array_tr[0]
    v_max = volt_array_tr[-1]
    t_min = time_array_tr[0] - abs(time_array_tr[1] - time_array_tr[0]) / 2
    t_max = time_array_tr[-1] + abs(time_array_tr[1] - time_array_tr[0]) / 2
    # -5 due to resolution of 10ns so events with timestamp e.g. 10 (= 100ns) will be plotted @ 95 to 105 ns

    extent = [v_min, v_max, t_min, t_max]
    initial_2d_arr = np.zeros((steps, bins), dtype=np.uint32)
    image, colorbar = image_plot(fig, im_ax, cb_ax, np.transpose(initial_2d_arr), extent, 'auto')
    im_ax.xaxis.set_ticks_position('top')
    im_ax.xaxis.set_label_position('top')
    return image, colorbar


def setup_projection_widget(axes, time_array_tr, volt_array_tr, x_label='DAC voltage [V]'):
    tproj_ax = axes['t_proj']
    vproj_ax = axes['v_proj']
    t_cts = np.zeros(time_array_tr.shape)
    v_cts = np.zeros(volt_array_tr.shape)
    v_min = min(volt_array_tr)
    v_max = max(volt_array_tr)

    vproj_line = vproj_ax.add_line(line2d(volt_array_tr, v_cts, 'r'))
    vproj_ax.set_xlim(v_min, v_max)
    vproj_ax.autoscale(enable=True, axis='y', tight=True)

    t_min = min(time_array_tr)
    t_max = max(time_array_tr)
    tproj_line = tproj_ax.add_line(line2d(t_cts, time_array_tr, 'r'))
    tproj_line.set_drawstyle('default')
    tproj_ax.set_ylim(t_min, t_max)
    tproj_ax.autoscale(enable=True, axis='x', tight=True)
    tproj_ax.set_xlabel('cts')
    tproj_ax.xaxis.set_label_position('top')
    tproj_ax.xaxis.set_ticks_position('top')
    tproj_ax.yaxis.set_ticks_position('right')
    vproj_ax.set_ylabel('cts')
    vproj_ax.set_xlabel(x_label)

    axes['sum_proj'].yaxis.label.set_color('blue')
    axes['sum_proj'].spines['right'].set_color('blue')
    axes['sum_proj'].tick_params(axis='y', colors='blue')
    axes['sum_proj'].autoscale_view()
    axes['sum_proj'].set_ylabel('sum_cts')

    return vproj_line, tproj_line


def tight_layout():
    plt.tight_layout()


def plot_par_from_combined(db, runs_to_plot, isotopes,
                           par, plot_runs_seperate=False, show_pl=True,
                           literature_run=None, literature_name='lit. values',
                           save_path='', use_syst_err_only=False, comments=None, markers=None, colors=None,
                           legend_loc=2, start_offset=-0.3, use_full_error=True,
                           lit_color='b', lit_marker='o', fontsize_ticks=12):
    import Tools
    compl_x = []
    compl_y = []
    compl_y_err = []
    if comments is None:
        # should be a list with a comment for each run
        comments = [''] * len(runs_to_plot)
    if markers is None:
        # should be a list with a marker for each run
        markers = ['o'] * len(runs_to_plot)
    if colors is None:
        colors = ['g', 'r', 'c', 'k'] * max((len(runs_to_plot) // 4), 1)
    lit_y = None
    lit_y_err = None
    val_statErr_rChi_shift_dict = Tools.extract_from_combined(runs_to_plot, db, isotopes, par, print_extracted=True)
    lit_val_statErr_rChi_shift_dict = None
    if literature_run is not None:
        lit_val_statErr_rChi_shift_dict = Tools.extract_from_combined(
            [literature_run], db, isotopes, par, print_extracted=True).get(literature_run, {})
        if len(lit_val_statErr_rChi_shift_dict) == 0:
            lit_val_statErr_rChi_shift_dict = None  # set it to None
    literarture_has_been_plotted = False
    err_index = 2 if use_syst_err_only else 1
    if use_full_error:
        err_index = -1
    lit_exists = 0 if literature_run is None else 1
    offset = start_offset
    offset_per_run = abs(offset) * 2 / (len(runs_to_plot) + lit_exists - 1)

    fig = plt.figure(1, (8, 8))
    fig.patch.set_facecolor('w')
    ax = plt.axes([0.15, 0.1, 0.8, 0.75])

    for ind, each in enumerate(runs_to_plot):
        try:
            if each:
                if lit_val_statErr_rChi_shift_dict is not None:
                    # try to get the literature values and substract experiment Values from it
                    # key of isotope should begin with mass number followed by '_' and isotope name
                    # vals [(mass_int, exp_value_float, lit_val_float), ...]
                    vals = [(int(key_pl.split('_')[0]), val_pl[0],
                             lit_val_statErr_rChi_shift_dict.get(key_pl, [0])[0]) for key_pl, val_pl in
                            sorted(val_statErr_rChi_shift_dict[each].items())]
                    if err_index > 0:
                        errs = [(int(key_pl2.split('_')[0]), val_pl2[err_index],
                                 lit_val_statErr_rChi_shift_dict.get(key_pl2, [0, 0])[err_index]) for key_pl2, val_pl2 in
                                sorted(val_statErr_rChi_shift_dict[each].items())]
                    else:  # use the full error with gaussian error prop
                        errs = [(int(key_pl2.split('_')[0]),
                                 np.sqrt(
                                     val_pl2[1] ** 2 + val_pl2[2] ** 2
                                 ),
                                 np.sqrt(
                                     lit_val_statErr_rChi_shift_dict.get(key_pl2, [0, 0])[1] ** 2 +
                                     lit_val_statErr_rChi_shift_dict.get(key_pl2, [0, 0])[2] ** 2
                                 )
                                 ) for key_pl2, val_pl2
                                in
                                sorted(val_statErr_rChi_shift_dict[each].items())]
                    x = [valo[0] + offset for valo in vals]
                    # exp_y = [val[1] for val in vals]
                    # exp_y_err = [val[1] for val in errs]
                    # maybe in future:
                    # lit_y = [val[2] for val in vals]
                    # lit_y_err = [val[2] for val in errs]
                    exp_y = [valo[1] - valo[2] for valo in vals]
                    exp_y_err = [valo[1] for valo in errs]
                    lit_y = [0 for valo in vals]
                    lit_y_err = [valo[2] for valo in errs]
                else:
                    x_y_err = [(int(iso[:2]), val[0], np.sqrt(val[1] ** 2 + val[2] ** 2))
                               for iso, val in sorted(val_statErr_rChi_shift_dict[each].items())]
                    x = [each[0] + offset for each in x_y_err]
                    exp_y = [each[1] for each in x_y_err]
                    exp_y_err = [each[2] for each in x_y_err]
                if plot_runs_seperate:
                    if lit_y is not None and not literarture_has_been_plotted:
                        offset += offset_per_run
                        x_lit = [valo[0] + offset for valo in vals]
                        lit_name = literature_run + ' (ref)' if literature_name == '' else literature_name + ' (ref)'
                        plt.errorbar(x_lit, lit_y, lit_y_err,
                                     label=lit_name, linestyle='None',
                                     marker=lit_marker, color=lit_color)
                        literarture_has_been_plotted = True
                        compl_x += x_lit
                        compl_y += lit_y
                        compl_y_err += lit_y_err
                    plt_name = each if comments[ind] == '' else comments[ind]
                    plt.errorbar(x, exp_y, exp_y_err, label='%s' % plt_name,
                                 linestyle='None', marker=markers[ind], color=colors[ind])

                compl_x += x
                compl_y += exp_y
                compl_y_err += exp_y_err

        except Exception as err:
            print('error while plotting: %s' % err)
        offset += offset_per_run

    if not plot_runs_seperate:
        plt.errorbar(compl_x, compl_y, compl_y_err, label='runs: ' + str(sorted(val_statErr_rChi_shift_dict.keys())),
                     linestyle='None', marker="o")
        if lit_y is not None:
            plt.errorbar(compl_x, lit_y, lit_y_err, label=literature_name,
                         linestyle='None', marker="o")

    plt.legend(loc='upper center', ncol=2,
               bbox_to_anchor=(0., 0.98, 1, 0.2), mode='expand',
               fontsize=fontsize_ticks+2, numpoints=1)
    # print for origin etc.:
    print('x\tval\tval_err')
    for i, each in enumerate(compl_x):
        print('%.2f\t%.8f\t%.8f' % (each, compl_y[i], compl_y_err[i]))
    plt.margins(0.25)
    ax.set_ylabel('%s [MHz]' % par)
    ax.set_xlabel('A')
    if save_path:
        d = os.path.dirname(save_path)
        if not os.path.exists(d):
            os.makedirs(d)
        save(save_path)
    if show_pl:
        show(True)
    return compl_x, compl_y, compl_y_err


def plot_iso_shift_time_dep(
        ref_dates_date_time, ref_dates_date_time_float, ref_centers, ref_errs, ref,
        iso_dates_datetime, iso_dates_datetime_float, iso_centers, iso_errs, iso,
        slope, offset, plt_label, shift_result_tuple, file_name='', show_plot=True,
        fig_name='shift', par_name='center [MHz]', font_size=12):
    """ function to plot the isotope shift along with the references versus timestamp of the files """
    fig = plt.figure('%s %s' % (fig_name, iso), figsize=(16, 9))
    fig.set_facecolor('w')
    main_ax = fig.add_axes([0.1, 0.2, 0.7, 0.6])
    first_ref = np.min(ref_dates_date_time_float)
    ref_line = main_ax.errorbar(ref_dates_date_time, ref_centers, yerr=ref_errs, fmt='ko', label='ref center %s' % ref)
    min_t_abs = min(np.min(ref_dates_date_time_float), np.min(iso_dates_datetime_float))
    max_t_abs = max(np.max(ref_dates_date_time_float), np.max(iso_dates_datetime_float))
    padding = max((max_t_abs - min_t_abs) / 100 * 5, 10)
    fit_plot_data_x_datetime = [datetime.datetime.fromtimestamp(min_t_abs - padding),
                                datetime.datetime.fromtimestamp(max_t_abs + padding)]
    fit_plot_data_x = [each.timestamp() - first_ref for each in fit_plot_data_x_datetime]
    fit_plot_data_y = [x * slope + offset for x in fit_plot_data_x]
    fit_line = plt.plot(fit_plot_data_x_datetime, fit_plot_data_y, label=plt_label, color='r')[0]
    plt.xticks(rotation=25)
    xfmt = DateFormatter('%Y-%m-%d %H:%M:%S')
    main_ax.xaxis.set_major_formatter(xfmt)
    main_ax.set_ylabel('ref %s %s' % (ref, par_name), fontsize=font_size)
    main_ax.tick_params(labelsize=font_size)
    twinx = plt.twinx(main_ax)
    iso_line = twinx.errorbar(iso_dates_datetime, iso_centers, yerr=iso_errs, fmt='bs', label='center %s' % iso)
    twinx.set_ylabel('%s %s' % (iso, par_name), color='b', fontsize=font_size)
    twinx.tick_params('y', colors='b', labelsize=font_size)
    lines = [ref_line, fit_line, iso_line]
    # shift_result_tuple should be a tuple of ([shift_run0, shift_run1, ...], [err_shift_run0, err_shift_run1, ...])
    shift_result_str = 'shift ' + str(
        ['%.1f +/- %.1f MHz' % (each, shift_result_tuple[1][i]) for i, each in enumerate(shift_result_tuple[0])])
    line_lables = [l.get_label() for l in lines] + [shift_result_str]
    lines += [patches.Rectangle((0, 0), 1, 1, fc="w", fill=False, edgecolor='none', linewidth=0)]
    fig.legend(lines, line_lables, loc='upper center', ncol=2,
               bbox_to_anchor=(0.1, 0.8, 0.7, 0.2), mode='expand', fontsize=font_size+2, numpoints=1)
    twinx.ticklabel_format(axis='y', useOffset=False)
    if file_name:
        if not os.path.isdir(os.path.dirname(file_name)):
            os.mkdir(os.path.dirname(file_name))
        print('saving to: %s' % file_name)
        save(file_name)
    if show_plot:
        plt.show(True)
    clear()
    plt.close(fig)
