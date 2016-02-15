'''
Created on 29.04.2014

@author: hammen
'''

from matplotlib.dates import DateFormatter
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.axes_divider import AxesDivider
from matplotlib import patches as patches
from matplotlib.widgets import RectangleSelector
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Button
from matplotlib.widgets import Slider

import datetime
import matplotlib.pyplot as plt
import numpy as np


def plot(*args):
    for a in args:
        plt.plot(a[0], a[1])
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel('Intensity [cts]')
    plt.xlabel('Frequency [MHz]')


def plotFit(fit):
    data = fit.meas.getArithSpec(*fit.st)
    plotdat = fit.spec.toPlotE(fit.meas.laserFreq, fit.meas.col, fit.par)

    fig = plt.figure(1, (8, 8))
    fig.patch.set_facecolor('white')

    ax1 = plt.axes([0.15, 0.35, 0.8, 0.6])
    plt.errorbar(data[0], data[1], yerr=data[2], fmt='k.')
    plt.plot(plotdat[0], plotdat[1], 'r-')
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)

    ax2 = plt.axes([0.15, 0.1, 0.8, 0.2])
    plt.errorbar(data[0], fit.calcRes(), yerr=data[2], fmt='k.')
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.ylabel('Intensity / a.u.')
    plt.xlabel('Ion kinetic energy / eV')


def plotAverage(date, cts, errs, avg, stat_err, syst_err, forms=('k.', 'r')):
    # avg, stat_err, sys_err = Analyzer.combineRes(iso, par, run, db, print_extracted=False)
    # val, errs, date = Analyzer.extract(iso, par, run, db, prin=False)
    date = [datetime.datetime.strptime(d, '%Y-%m-%d %H:%M:%S') for d in date]
    plt.subplots_adjust(bottom=0.2)
    plt.xticks(rotation=25)
    ax = plt.gca()
    xfmt = DateFormatter('%Y-%m-%d %H:%M:%S')
    ax.xaxis.set_major_formatter(xfmt)
    plt.errorbar(date, cts, yerr=errs, fmt=forms[0])
    err_p = avg + abs(stat_err) + abs(syst_err)
    err_m = avg - abs(stat_err) - abs(syst_err)
    err_p_l = np.full((2,), err_p)
    err_m_l = np.full((2,), err_m)
    x = (sorted(date)[0], sorted(date)[-1])
    y = (avg, avg)
    plt.plot(x, y, forms[1])
    plt.fill_between(x, err_p_l, err_m_l, alpha=0.5)


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


def close_fig(fig):
    plt.close(fig)


def plt_axes(axes, plotlist):
    # axes.clear()   # really really time consuming!!
    axes.plot(*plotlist)


def line2d(x_data, y_data, line_color):
    return Line2D(x_data, y_data, color=line_color, drawstyle='steps-mid')


def get_current_axes():
    return plt.gca()


def get_current_figure():
    return plt.gcf()


def setup_image_figure():
    fig = plt.figure()
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


def image_plot(fig, axes, cbax, image_date, extent, aspect='equal'):
    img = axes.imshow(image_date, extent=extent, origin='lower',
                      aspect=aspect, interpolation='none')
    axes.set_ylabel('time [ns]')
    axes.set_xlabel('DAC voltage [V]')
    cb = fig.colorbar(img, cax=cbax)
    # cb = None
    draw()
    return img, cb


def configure_image_plot(fig, im_ax, cb_ax, pipeData, volt_array_tr, time_array_tr, pmt_num, track_name):
    iso = pipeData['isotopeData']['isotope']
    type = pipeData['isotopeData']['type']
    fig.canvas.set_window_title('%s_%s_%s_pmt%s' % (iso, type, track_name, str(pmt_num)))
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
