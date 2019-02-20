"""
Created on 

@author: simkaufm

Module Description:  meant to estimate what correction in voltage is needed to apply in 2016 or 2017
"""

import os
import sqlite3
import numpy as np

from matplotlib import pyplot as plt
from pyqtgraph.Qt import QtCore, QtGui
import pyqtgraph as pg


import Physics
import Tools
import TildaTools
import Analyzer
from DBIsotope import DBIsotope


''' working directory: '''

workdir17 = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\' \
          'Measurement_and_Analysis_Simon\\Ni_workspace2017\\Ni_2017'

datafolder17 = os.path.join(workdir17, 'sums')

db17 = os.path.join(workdir17, 'Ni_2017.sqlite')
Tools.add_missing_columns(db17)

final_2017_run = '2017_Experiment'  # this will be the joined analysis of hot CEC and normal run!

isotopes17 = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni',
              '65_Ni', '66_Ni', '67_Ni', '68_Ni', '70_Ni']
x_17 = [58, 60, 61, 62, 64,
        65, 66, 67, 68, 70]
isos17 = {iso: DBIsotope(db17, iso, lineVar='tisa_60_asym_final') for iso in isotopes17}


line_freq17 = 850344226.10401
laser_freq17 = 851336076.2983379  # normal CEC !
laser_freq17_hotCEC = 851336376.0907958  # not so relevant
acc_volt_17 = 7.98528824855 * 5001.39  # 39937.54079341549  from the mean of all files in 2017

workdir16 = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder16 = os.path.join(workdir16, 'Ni_April16_mcp')

db16 = os.path.join(workdir16, 'Ni_workspace.sqlite')
runs16 = ['wide_gate_asym', 'wide_gate_asym_67_Ni']
final_2016_run = 'wide_gate_asym'

isotopes16 = ['58_Ni', '59_Ni', '60_Ni', '61_Ni', '62_Ni', '63_Ni',
              '64_Ni', '65_Ni', '66_Ni', '67_Ni', '68_Ni', '70_Ni']
isos16 = {iso: DBIsotope(db16, iso, lineVar='tisa_60_asym_wide') for iso in isotopes16}

line_freq16 = 850343816.10401
laser_freq16 = 851200725.9994   # all files
acc_volt_16 = 29.9610343474 * 1000.05  # 29962.53239911737  from the mean of all files in 2016


'''  get shifts '''

shifts16 = Tools.extract_from_combined([final_2016_run], db16, isotopes=isotopes16,
                                       par='shift', print_extracted=True)[final_2016_run]
shifts17 = Tools.extract_from_combined([final_2017_run], db17, isotopes=isotopes17,
                                       par='shift', print_extracted=True)[final_2017_run]

offsets_volts_16 = {}
for iso16 in isotopes16:
    con = sqlite3.connect(db16)
    cur = con.cursor()
    cur.execute('''SELECT offset FROM Files WHERE type = ?''', (iso16,))
    ret = cur.fetchall()
    con.close()
    ret = [each[0] * 999.985 for each in ret]  # chose agilent
    ret_err = [offs * 1.5 * 10 ** -4 for offs in ret]
    offsets_volts_16[iso16] = [Analyzer.weightedAverage(ret, ret_err), ret]
    
    
offsets_volts_17 = {}
for iso17 in isotopes17:
    con = sqlite3.connect(db17)
    cur = con.cursor()
    cur.execute('''SELECT offset FROM Files WHERE type = ?''', (iso17,))
    ret = cur.fetchall()
    con.close()
    ret = [eval(each[0])[0] * 1000.022 for each in ret]
    ret_err = [offs * 1.5 * 10 ** -4 for offs in ret]
    offsets_volts_17[iso17] = [Analyzer.weightedAverage(ret, ret_err), ret]

''' calculate differential doppler shift for each isotope '''
# TildaTools.print_dict_pretty(offsets_volts_17)
# dif_dopl_16 = Physics.diffDoppler(line_freq16, acc_volt_16, )
dif_dopler_16 = {}
for iso16 in isotopes16:
    dif_dopler_16[iso16] = Physics.diffDoppler(
        line_freq16, acc_volt_16 - offsets_volts_16[iso16][0][0], isos16[iso16].mass)
    
dif_dopler_17 = {}
for iso17 in isotopes17:
    dif_dopler_17[iso17] = Physics.diffDoppler(
        line_freq17, acc_volt_17 - offsets_volts_17[iso17][0][0], isos17[iso17].mass)

TildaTools.print_dict_pretty(dif_dopler_16)
TildaTools.print_dict_pretty(dif_dopler_17)

''' shown effect on the isotope shift: '''
TildaTools.print_dict_pretty(shifts16)
# isotope_shift = iso - ref
# add_voltage = 0  # V

add_voltages = [-15, -10, -5, 0, 5, 10]
shifts_add_volt_16 = []
for ind_add_v, add_voltage in enumerate(add_voltages):
    shifts_add_volt_16 += {},
    ni_60_center_shifted = Physics.addEnergyToFrequencyPoint(
        shifts16['60_Ni'][0], add_voltage, isos16['60_Ni'], laser_freq16, True)
    for iso16 in isotopes16:
        is16 = isos16[iso16]
        is_center_shifted = Physics.addEnergyToFrequencyPoint(
            shifts16[iso16][0], add_voltage, is16, laser_freq16, True)
        # print('add_volts: ', add_voltage, ' shift %s-60: %.3f' % (iso16, is_center_shifted - ni_60_center_shifted))
        shifts_add_volt_16[ind_add_v][iso16] = is_center_shifted - ni_60_center_shifted
print(shifts_add_volt_16)
print(shifts17)
y_17 = [0 for iso in isotopes17]
y_err_17 = [shifts17[iso][1] for iso in isotopes17]
y_16 = []
y_err_16 = []
for ind_add_v, add_voltage in enumerate(add_voltages):
    y_16 += [],
    y_err_16 += [],
    for iso_ind, iso17 in enumerate(isotopes17):
        y_16[ind_add_v] += shifts_add_volt_16[ind_add_v][iso17] - shifts17[iso17][0],
        y_err_16[ind_add_v] += np.sqrt(shifts16[iso17][1] ** 2 + shifts17[iso17][1] ** 2),


print('iso' + '\t%s' * len(add_voltages) % tuple(add_voltages))
for is_ind, is17 in enumerate(isotopes17):
    to_pr = is17
    for ind_add_v, add_voltage in enumerate(add_voltages):
        to_pr += '\t%.3f' % y_16[ind_add_v][is_ind]
    print(to_pr)

# font_s = 18
# fig = plt.figure(1, (15, 10), facecolor='w')
# ax = plt.gca()
# plt.errorbar(x_17, y_17, y_err_17, label='2017 (ref)', marker='o')
# for ind_add_v, add_voltage in enumerate(add_voltages):
#     plt.errorbar(x_17, y_16[ind_add_v], y_err_16[ind_add_v], label='2016 %+d V' % add_voltage, marker='d')
#
# ax.set_xlim(57.5, 70.5)
# ax.set_ylim(-39, 39)
# ax.set_ylabel('shift - 2017shift [MHz]', fontdict={'size': font_s + 2})
# ax.set_xlabel('A', fontdict={'size': font_s + 2})
# plt.xticks(np.arange(min(x_17), max(x_17), 1.0), fontsize=font_s)
# plt.yticks(fontsize=font_s)
# plt.legend(loc=2, fontsize=font_s + 2, numpoints=1)
# store_to = os.path.join(workdir17, 'comparison_plots_2016_vs_2017/2016_2017_voltage_16_shifted.pdf')
# plt.savefig(store_to)
# plt.show(True)
# plot show best results around -5V -> will perform red. chi^2 optimisation for both beam times.


''' perform chi square reduction of both simultaneously and find mean value '''


def isotope_shifts_shifted_by_add_volt(add_volt, isotopes, exp_16=True, print_shifts=False):
    """
    give the isotope shifts when a add_volt is added to the total voltage
    :param add_volt: float, additional voltage
    :param isotopes: list, list of isotopes
    :param exp_16: bool, True -> 2016 False -> 2017
    :return: (shifts, errs)
    """
    ref = '60_Ni'
    if exp_16:
        shifts = shifts16
        db_isos = isos16
        laser_freq = laser_freq16
    else:
        shifts = shifts17
        db_isos = isos17
        laser_freq = laser_freq17

    ref_center_shifted = Physics.addEnergyToFrequencyPoint(
        shifts[ref][0], add_volt, db_isos[ref], laser_freq, True)
    shift_list = []
    shift_err_list = []
    for iso in isotopes:
        is_shifted = Physics.addEnergyToFrequencyPoint(
            shifts[iso][0], add_volt, db_isos[iso], laser_freq, True)
        # print('add_volts: ', add_voltage, ' shift %s-60: %.3f' % (iso16, is_center_shifted - ni_60_center_shifted))
        shift_list += is_shifted - ref_center_shifted,
        shift_err_list += shifts[iso][1],
    if print_shifts:
        print('# added volt: %.2f' % add_volt)
        print('#iso\tshift\tshiftErr')
        for i, iso in enumerate(isotopes):
            print('%s\t%s\t%s' % (iso, shift_list[i], shift_err_list[i]))

    return shift_list, shift_err_list


def chi_sq_finder(add_volts_16, add_volts_17):
    isotopes = isotopes17
    chi_sq_ret_array = np.zeros((add_volts_16.size, add_volts_17.size), dtype=np.float)
    for i_16, add_volt_16 in enumerate(add_volts_16):
        calc_shifts16, err_calc_shifts16 = isotope_shifts_shifted_by_add_volt(add_volt_16, isotopes, exp_16=True)
        for i_17, add_volt_17 in enumerate(add_volts_17):
            calc_shifts17, err_calc_shifts17 = isotope_shifts_shifted_by_add_volt(add_volt_17, isotopes, exp_16=False)
            chi_sq = 0.0
            for i_is, iso in enumerate(isotopes):
                err_iso = max(np.sqrt(err_calc_shifts16[i_is] ** 2 + err_calc_shifts17[i_is] ** 2), 1)
                rchi_sq_iso = ((calc_shifts16[i_is] - calc_shifts17[i_is]) ** 2 / err_iso)
                chi_sq += rchi_sq_iso
            chi_sq_ret_array[i_16][i_17] = chi_sq
    return chi_sq_ret_array


def plot_with_add_volt(add_volt_16, add_volt_17):
    add_v_shifts_16, err_add_v_shifts_16 = isotope_shifts_shifted_by_add_volt(
        add_volt_16, isotopes17, exp_16=True, print_shifts=True)
    add_v_shifts_17, err_add_v_shifts_17 = isotope_shifts_shifted_by_add_volt(
        add_volt_17, isotopes17, exp_16=False, print_shifts=True)

    y_17 = [0 for iso in isotopes17]
    y_err_17 = err_add_v_shifts_17
    y_16 = []
    y_err_16 = err_add_v_shifts_16
    for iso_ind, sh16 in enumerate(add_v_shifts_16):
        y_16 += sh16 - add_v_shifts_17[iso_ind],

    font_s = 18
    fig = plt.figure(2, (15, 10), facecolor='w')
    ax = plt.gca()
    plt.errorbar(x_17, y_17, y_err_17, label='2017 (ref) %+.2f' % add_volt_17, marker='o')
    plt.errorbar(x_17, y_16, y_err_16, label='2016 %+.2f V' % add_volt_16, marker='d')
    ax.set_xlim(57.5, 70.5)
    # ax.set_ylim(-39, 39)
    ax.set_ylabel('shift - 2017shift [MHz]', fontdict={'size': font_s + 2})
    ax.set_xlabel('A', fontdict={'size': font_s + 2})
    plt.xticks(np.arange(min(x_17), max(x_17), 1.0), fontsize=font_s)
    plt.yticks(fontsize=font_s)
    plt.legend(loc=2, fontsize=font_s + 2, numpoints=1)
    store_to = os.path.join(
        workdir17,
        'comparison_plots_2016_vs_2017/2016_2017_v_add_2016_%+d_2017_%+d.pdf' % (add_volt_16, add_volt_17))
    plt.savefig(store_to)
    plt.show(True)


start = -10
stop = 10.5
intv = 0.5
additional_volts_16 = np.arange(start, stop, intv)
additional_volts_17 = np.arange(start, stop, intv)
chi_sq_result = chi_sq_finder(additional_volts_16, additional_volts_17)

minimum = np.min(chi_sq_result)
argmin = np.where(chi_sq_result == minimum)
print(minimum, argmin, additional_volts_16[argmin[0][0]], additional_volts_17[argmin[1][0]])
# imv = pg.ImageView()
# imv.show()
# pg.show()
plot_with_add_volt(additional_volts_16[argmin[0][0]], additional_volts_17[argmin[1][0]])

pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'k')

app = QtGui.QApplication([])
win = QtGui.QMainWindow()
win.resize(800, 800)

plt_item = pg.PlotItem()
imv_widget = pg.ImageView(view=plt_item)
plt_item.invertY(False)
plt_item.showAxis('top')
plt_item.showAxis('right')
plt_item.showLabel('bottom', False)
plt_item.showLabel('right', False)
plt_item.getAxis('right').setStyle(showValues=False)
plt_item.getAxis('bottom').setStyle(showValues=False)
plt_item.setLabel('left', 'add_Volt 2017')
plt_item.setLabel('top', 'add_Volt 2016')
colors = [
    (255, 255, 255),
    (0, 0, 255),
    (0, 255, 255),
    (0, 255, 0),
    (255, 255, 0),
    (255, 0, 0),
]
color = pg.ColorMap(pos=np.linspace(0.0, 1.0, len(colors)), color=colors)
imv_widget.setColorMap(color)

x_range = (additional_volts_16[0], additional_volts_16[-1])
x_scale = np.mean(np.ediff1d(additional_volts_16))
y_range = (np.min(additional_volts_17), np.max(additional_volts_17))
y_scale = np.mean(np.ediff1d(additional_volts_17))
plt_item.setAspectLocked(False)
imv_widget.setImage(chi_sq_result, autoRange=True,
                    pos=[x_range[0],
                         y_range[0] - abs(0.5 * y_scale)],
                    scale=[x_scale, y_scale],
                    )
plt_item.setRange(xRange=(additional_volts_16[0], additional_volts_16[-1]),
                  yRange=(additional_volts_17[0], additional_volts_17[-1]),
                  padding=0, update=True)


win.setCentralWidget(imv_widget)
win.show()
# input('anything to quit')

if __name__ == '__main__':
    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
