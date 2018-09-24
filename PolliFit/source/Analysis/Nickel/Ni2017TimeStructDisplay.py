"""
Created on 

@author: simkaufm

Module Description:
"""
import logging
import sys
import os
import ast

import TildaTools as TiTs

''' folders etc. '''

workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\' \
          'Measurement_and_Analysis_Simon\\Ni_workspace2017\\Ni_2017'

datafolder = os.path.join(workdir, 'sums')

db = os.path.join(workdir, 'Ni_2017.sqlite')

runs = ['AsymVoigt', 'AsymExpVoigt', 'AsymVoigtFree', '2016Experiment']

isotopes = ['%s_Ni' % i for i in range(58, 71)]
isotopes.remove('69_Ni')
isotopes.remove('59_Ni')  # not measured in 2017
isotopes.remove('63_Ni')  # not measured in 2017
odd_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2]
even_isotopes = [iso for iso in isotopes if int(iso[:2]) % 2 == 0]
stables = ['58_Ni', '60_Ni', '61_Ni', '62_Ni', '64_Ni']


''' get relevant files '''

configs = {}
# isotopes = []
for iso in isotopes:
    ret = TiTs.select_from_db(db, 'config', 'Combined', [['iso', 'run', 'parname'], [iso, runs[0], 'shift']])
    if ret is not None:
        ret = ret[0][0]
        print(ret)
        configs[iso] = ast.literal_eval(ret)

flat_configs = {}  # each file only once, just to check time structure!
for iso_name, conf in sorted(configs.items()):
    flat_configs[iso_name] = []
    for each in conf:
        for file in each:
            for tipa_f in file:
                if tipa_f not in flat_configs[iso_name]:
                    flat_configs[iso_name] += tipa_f,
    # print(iso_name, flat_configs[iso_name])

all_files = TiTs.select_from_db(db, 'file', 'Files', addCond='ORDER BY date')
flat_configs['all'] = []
for each in all_files:
    if '.xml' in each[0] and 'epco' not in each[0]:
        flat_configs['all'] += each[0],

''' plot the time structure and save it as a .jpg '''
from Interface.LiveDataPlottingUi.LiveDataPlottingUi import TRSLivePlotWindowUi
from Service.AnalysisAndDataHandling.DisplayData import DisplayData

from PyQt5 import QtWidgets, QtGui
from Measurement.XMLImporter import XMLImporter
from PyQt5.QtCore import *
import time


app = QtWidgets.QApplication(sys.argv)
ui = QtWidgets.QMainWindow()
label = QtWidgets.QLabel()
ui.setCentralWidget(label)
ui.show()
# copy = flat_configs_tipa
# flat_configs_tipa = {}
# flat_configs_tipa['58_Ni'] = copy['58_Ni']
storage_path = os.path.join(workdir, 'TimeStructureAnalysis')
if not os.path.isdir(storage_path):
    os.mkdir(storage_path)

win = []  # all windows need to be stored when wanting to keep them open.
# file_names = []
for iso, tilda_files in sorted(flat_configs.items()):
    # if iso in ['58_Ni', '59_Ni', '60_Ni', '61_Ni', '62_Ni', '63_Ni',
    #            '64_Ni', '65_Ni', '66_Ni', '68_Ni', '70_Ni', 'all']:
    if iso in ['65_Ni']:
        print('---------------  started working on iso: %s  ------------' % iso)
        cur_dir = os.path.join(storage_path, iso)
        if not os.path.isdir(cur_dir):
            os.mkdir(cur_dir)
        # files = [files[-1]]
        for tilda_file in tilda_files:
            logging.info('working on: %s' % tilda_file)
            label.setText('working on: %s' % tilda_file)
            try:
                run_num = int(tilda_file.split('.')[0][-3:])
                if run_num >= 0:
                    full_file_name = os.path.join(datafolder, tilda_file)
                    cur_win = TRSLivePlotWindowUi(full_file_name, subscribe_as_live_plot=False, application=app)
                    spec = XMLImporter(full_file_name, softw_gates=(db, runs[0]))
                    spec.softBinWidth_ns = [100]
                    disp_data = DisplayData(full_file_name, cur_win, x_as_volt=True, loaded_spec=spec)
                    cur_win.set_time_range(padding=0.2)
                    # win += cur_win,   # if one wants to see all the windows comment this in.
                    f_name = tilda_file
                    f_name = f_name.split('.')[0]
                    f_name += '.jpg'
                    f_name_full = os.path.join(cur_dir, f_name)
                    # file_names += f_name_full,

                    cur_win.export_screen_shot(f_name_full, quality=100)
                    cur_win.close()
                    time.sleep(2)
                    logging.info('done working on %s' % tilda_file)

            except Exception as e:
                logging.error('------------------ failed working on %s error is: %s' % (tilda_file, e), exc_info=True)
label.setText('Done!')
app.deleteLater()
app.exec_()
app.setQuitOnLastWindowClosed(True)
app.closeAllWindows()
