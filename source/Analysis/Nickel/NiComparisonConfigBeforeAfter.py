"""
Created on 

@author: simkaufm

Module Description:
"""

import sqlite3
import os
import ast
import Analyzer
import numpy as np
import MPLPlotter

workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder = os.path.join(workdir, 'Ni_April2016_mcp')

db = os.path.join(workdir, 'Ni_workspace.sqlite')

reference_run = 'wide_gate_asym'
ref_run = 'wide_gate_asym_by_time'
# compare_runs = ['ref_comp_before_and_after', 'ref_comp_before', 'ref_comp_after']
compare_runs = ['wide_gate_asym_by_time']


isotopes = ['%s_Ni' % i for i in range(58, 71)]
isotopes.remove('69_Ni')

# get all configs from existing runs:
configs = {}
for iso in isotopes:
    if iso != '60_Ni':
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute(
            '''SELECT config, val, statErr, rChi FROM Combined WHERE iso = ? AND run = ? AND parname = ? ''',
            (iso, reference_run, 'shift'))
        data = cur.fetchall()
        con.close()
        if len(data):
            config, val, statErr, rChi = data[0]
            # print('%s \t %s \t %s \t %s \t %s \n %s' % (run, iso, val, statErr, rChi, config))
            configs[iso] = ast.literal_eval(config)

# # write new configs to db:
for iso, cfg in sorted(configs.items()):
    all_before_and_after = [each for each in cfg if len(each[0]) != 0 and len(each[2]) != 0]
    print('%s\t%s' % (iso, all_before_and_after))
    par = 'shift'
    for compare_run in compare_runs:
        run_cfg = []
        if 'ref_comp_before' == compare_run:
            run_cfg = [(each[0], each[1], []) for each in cfg if len(each[0]) != 0 and len(each[2]) != 0]
        elif 'ref_comp_after' == compare_run:
            run_cfg = [([], each[1], each[2]) for each in cfg if len(each[0]) != 0 and len(each[2]) != 0]
        elif 'ref_comp_before_and_after' == compare_run:
            run_cfg = [each for each in all_before_and_after]
        elif 'wide_gate_asym_by_time' == compare_run:
            run_cfg = cfg
        syst_err_form = 'systE(accVolt_d=1.5 * 10 ** -4, offset_d=1.5 * 10 ** -4)'
        stat_err_form = 'applyChi(err, rChi)'
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute(
            '''INSERT OR IGNORE INTO Combined (
            iso, parname, run, config, statErrForm, systErrForm) VALUES (?, ?, ?, ?, ?, ?)''',
            (iso, par, compare_run, str(run_cfg), stat_err_form, syst_err_form))
        con.commit()
        con.close()

# combine isotope shifts:
for iso, cfg in sorted(configs.items()):
    if iso != '60_Ni' and iso != '60_Ni':
        for compare_run in compare_runs:
            con = sqlite3.connect(db)
            cur = con.cursor()
            cur.execute(
                ''' UPDATE FitRes SET run = ? WHERE iso = ?''',
                (compare_run, iso))
            con.commit()
            con.close()
            Analyzer.combineShiftByTime(iso, compare_run, db, show_plot=False)
# raise Exception('wohooo')
# print resulting shifts:
iso_shift_plot_data_x = list(range(58, 71))
iso_shift_plot_data_x.remove(67)
iso_shift_plot_data_x.remove(60)
iso_shift_plot_data_x.remove(69)
iso_shift_plot_data_y = {}
for compare_run in compare_runs:
    iso_shift_plot_data_y[compare_run] = {}
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(''' SELECT iso, val, statErr, systErr, rChi From Combined WHERE parname = ? AND run = ? ORDER BY iso''',
                ('shift', compare_run))
    data = cur.fetchall()
    con.close()
    if data:
        print('--------------------------- % s ---------------------------' % compare_run)
        print('iso\tshift [MHz]\tChi^2')
        for iso in data:
            if iso[0] != '67_Ni' or True:
                # err = np.sqrt(iso[2] ** 2 + iso[3] ** 2)
                err = iso[2]  # only interrested in statistical error here
                iso_shift_plot_data_y[compare_run][iso[0]] = (iso[1], err)
                # print('%s\t%.1f(%.0f)[%.0f]\t%.3f' % (iso[0], iso[1], iso[2] * 10, iso[3] * 10, iso[4]))
                for_excel = '%s\t%.1f\t%.1f\t%.1f' % (iso[0], iso[1], err, iso[4])
                for_excel = for_excel.replace('.', ',')
                print(for_excel)
print(iso_shift_plot_data_x)
print(iso_shift_plot_data_y)


compare_runs.append('Liang')
# [58, 59, 61, 62, 63, 64, 65, 66, 68, 70]
iso_shift_plot_data_y['Liang'] = {
    '61_Ni': (281.8, 1.2), '68_Ni': (2004.7, 1.2), '62_Ni': (502.5, 0.9), '58_Ni': (-509.2, 1.8),
    '59_Ni': (-215.2, 1.8), '65_Ni': (1323.9, 0.6), '70_Ni': (2381.3, 4.0), '66_Ni': (1532.4, 0.7),
    '63_Ni': (785.8, 0.3), '64_Ni': (1028.2, 1.2), '67_Ni': (1794.0, 1.0)
}
#
compare_runs.append('div_ratios_datasheet_all_files')
iso_shift_plot_data_y['div_ratios_datasheet_all_files'] = {
    '61_Ni': (284.4, 1.0), '68_Ni': (2000.6, 1.3), '62_Ni': (505, 0.9), '58_Ni': (-512.0, 0.7),
    '59_Ni': (-215.4, 1.5), '65_Ni': (1326.8, 1.9), '70_Ni': (2382.3, 1.6), '66_Ni': (1534.3, 1.1),
    '63_Ni': (786.9, 1.0), '64_Ni': (1029.8, 0.5), '67_Ni': (1797.2, 1.3)}

compare_runs.append('literature')
# adjust x-axis!
iso_shift_plot_data_y['literature'] = {
    '64_Ni': (1020.3313880423849, 6.4178958048428294), '62_Ni': (506.95, 3.6),
    '58_Ni': (-507.85, 2.7), '61_Ni': (274.61, 3.0)}

# rename Fit results back:
for iso, cfg in sorted(configs.items()):
    if iso != '60_Ni' and iso != '60_Ni':
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute(
            ''' UPDATE FitRes SET run = ? WHERE iso = ?''',
            (reference_run, iso))
        con.commit()
        con.close()

to_plot = compare_runs
# to_plot = ['literature', 'div_ratios_datasheet_all_files']

offset = -0.1
for compare_run in compare_runs:
    if compare_run in to_plot:
        run_x_data = []
        run_y_data = []
        run_y_err_data = []
        for iso in range(58, 71):
            iso_n = '%d_Ni' % iso
            if iso_shift_plot_data_y[compare_run].get(iso_n, False):
                run_x_data += iso,
                run_y_data += iso_shift_plot_data_y[ref_run][iso_n][0] - iso_shift_plot_data_y[compare_run][iso_n][0],
                run_y_err_data += iso_shift_plot_data_y[compare_run][iso_n][1],
            elif iso_n == '60_Ni':
                run_x_data += iso,
                run_y_data += 0,
                run_y_err_data += 0,
        x = np.array(run_x_data) + offset
        y = np.array(run_y_data)
        err = np.array(run_y_err_data)
        if len(x):
            MPLPlotter.plt.errorbar(x, y, err, fmt='o', label=compare_run)
            MPLPlotter.get_current_figure().set_facecolor('w')
            offset += 0.05
MPLPlotter.plt.ylabel('shift(%s) - shift(run) [MHz]' % ref_run)
# MPLPlotter.plt.ylabel('shift [MHz]')
MPLPlotter.plt.xlabel('A')
MPLPlotter.plt.legend(loc=1)
MPLPlotter.show(True)

