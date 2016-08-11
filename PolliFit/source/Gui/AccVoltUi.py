'''
Created on 06.06.2014

@author: hammen, chgorges
'''

import datetime
import os
import sqlite3

from PyQt5 import QtWidgets, QtCore

import Analyzer
import MPLPlotter as plot
from Gui.Ui_AccVolt import Ui_AccVolt


class AccVoltUi(QtWidgets.QWidget, Ui_AccVolt):
    def __init__(self):
        super(AccVoltUi, self).__init__()
        self.setupUi(self)

        self.pushButton.clicked.connect(self.plot_and_save)
        self.read_error = 10 ** -4

        self.lineEdit_readErr.setText('%e' % self.read_error)
        self.lineEdit_readErr.textChanged.connect(self.read_error_changed)

        self.dateTimeEdit_end.setDateTime(QtCore.QDateTime.currentDateTime())

        self.dateTimeEdit_end.dateTimeChanged.connect(self.average_acc_volt)
        self.dateTimeEdit_start.dateTimeChanged.connect(self.average_acc_volt)


        self.dbpath = None

        self.show()

    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def plot_and_save(self):
        dates, accVolts, err_accVolts, avg, avg_err = self.average_acc_volt()
        # print('avg, avg_err, rchi', avg, avg_err, rchi)
        filename = 'accVolt_from_%s_to_%s.png' % (dates[0], dates[-1])
        filename = filename.replace(':', '_').replace(' ', '_')
        path = os.path.join(os.path.split(self.dbpath)[0], 'combined_plots', filename)
        try:
            plot.close_all_figs()
            plot.plotAverage(dates, accVolts, err_accVolts, avg, avg_err, 0,
                             showing=True, ylabel='accVolt [V]', save_path=path)
            self.lineEdit_savedto.setText(path)
        except Exception as e:
            print(e)

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.average_acc_volt()

    def get_date_and_acc_volt(self, console_print=True):
        dates = []
        accVolts = []
        con = sqlite3.connect(self.dbpath)
        cur = con.cursor()
        cur.execute('''SELECT date, accVolt from Files''')
        dates_accVolt_tpl_list = cur.fetchall()
        for dates_accVolt_tpl in dates_accVolt_tpl_list:
            if isinstance(dates_accVolt_tpl[0], str) and isinstance(dates_accVolt_tpl[1], float) \
                    and dates_accVolt_tpl[1] > 0:
                date = datetime.datetime.strptime(dates_accVolt_tpl[0], '%Y-%m-%d %H:%M:%S')
                try:
                    if self.dateTimeEdit_start.dateTime().toPyDateTime() <= date\
                            <= self.dateTimeEdit_end.dateTime().toPyDateTime():
                        dates.append(dates_accVolt_tpl[0])
                        accVolts.append(dates_accVolt_tpl[1])
                        if console_print:
                            print('%s\t%s' % dates_accVolt_tpl)
                except Exception as e:
                    print('error while comparing: ', e)
        return dates, accVolts

    def average_acc_volt(self):
        dates, accVolts = self.get_date_and_acc_volt()
        if any(dates):
            err_accVolts = [accVolt * self.read_error for accVolt in accVolts]
            avg, avg_err, rchi = Analyzer.weightedAverage(accVolts, err_accVolts)
            self.lineEdit_avg.setText(str(avg))
            self.lineEdit_rChi2.setText(str(rchi))
            self.lineEdit_statErr.setText(str(avg_err))
            return dates, accVolts, err_accVolts, avg, avg_err

    def read_error_changed(self, text):
        try:
            read = float(text)
            self.read_error = read
            self.average_acc_volt()
        except Exception as e:
            print('this is not a float')

