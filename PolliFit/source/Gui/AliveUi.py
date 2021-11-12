"""
Created on 29.07.2016, (edited on 19.10.2021)

@author: K. Koenig, (P. Mueller)
"""

import ast
import copy
from datetime import datetime

import numpy as np
from PyQt5 import QtWidgets, QtCore

import AliveTools
import TildaTools as TiTs
import Analyzer
import MPLPlotter as Plot
from Gui.Ui_Alive import Ui_Alive


class AliveUi(QtWidgets.QWidget, Ui_Alive):
    def __init__(self):
        super(AliveUi, self).__init__()
        self.setupUi(self)

        self.runSelect.currentTextChanged.connect(self.loadIsos)
        self.isoSelect.currentIndexChanged.connect(self.loadFiles)
        self.isoSelect_2.currentIndexChanged.connect(self.loadFiles)

        self.fileList.itemChanged.connect(self.recalc)
        self.fileList_2.itemChanged.connect(self.recalc)

        self.pB_compareAuto.clicked.connect(self.compare_auto)
        self.pB_compareIndividual.clicked.connect(self.compare_individual)

        self.dbpath = None
        self.vals = []
        self.errs = []
        self.vals_2 = []
        self.errs_2 = []

        self.show()

    # noinspection PyAttributeOutsideInit
    def compare_auto(self):
        self.recalc()
        if not self.hv_files or not self.ref_files:
            print('No hv/ref-measurement pairs available.')
            return

        self.plotdata = []
        self.Error = []
        self.ref_x_data = []
        self.ref_time_data = []
        self.x_data = []
        self.time_data = []
        self.offset_volt = []
        for i in range(1):  # Add higher comparison "orders" in the future.
            self.plotdata.append([])
            self.Error.append([])
            self.ref_x_data.append([])
            self.ref_time_data.append([])
            self.x_data.append([])
            self.time_data.append([])
            self.offset_volt.append([])
            for hv_file, ref_file in zip(self.hv_files, self.ref_files):
                hv = AliveTools.calculateVoltage(self.dbpath, hv_file['filename'], self.run)
                ref = AliveTools.calculateVoltage(self.dbpath, ref_file['filename'], self.run)

                self.ref_x_data[-1].append(ref_file['number'])
                self.ref_time_data[-1].append(ref_file['date'])
                self.x_data[-1].append(hv_file['number'])
                self.time_data[-1].append(hv_file['date'])
                self.offset_volt[-1].append(AliveTools.get_real_offsetVolt_from_db(
                    self.dbpath, hv_file['filename'], self.lDivOffset.value(), self.lAgiGain.value(),
                    self.lDivRatio.value()))

                self.plotdata[-1].append((hv[0] - ref[0]))
                delta_max = (hv[1] - ref[2]) - (hv[0] - ref[0])
                delta_min = (hv[0] - ref[0]) - (hv[2] - ref[1])
                self.Error[-1].append([delta_min, delta_max])

        self.saving()

        """
        |   Old code to compare with prev. and next reference. Does not work anymore.                  |
        v   To implement other automatic comparison methods modify the outer loop of the code above.   v
        """
        # for file in hv_files:
        #     hv = AliveTools.calculateVoltage(self.dbpath, file['filename'], self.run)
        # 
        #     ref_Files = AliveTools.find_ref_files(file, all_files, self.isoSelect.currentText())
        #     if not ref_Files:
        #         print('No reference files in database.')
        # 
        #     errorData = []
        #     Delta = []
        #     ref_x_data = []
        #     ref_time_data = []
        #     x_data = []
        #     time_data = []
        #     offset_volt = []
        # 
        #     for element in ref_Files:
        #         ref = AliveTools.calculateVoltage(self.dbpath, element['filename'], self.run)
        #         ref_x_data = ref_x_data + [element['number']]
        #         ref_time_data = ref_time_data + [element['date']]
        #         x_data = x_data + [file['number']]
        #         time_data = time_data + [file['date']]
        #         offset_volt = offset_volt + [AliveTools.get_real_offsetVolt_from_db(
        #             self.dbpath, file['filename'], self.lDivOffset.value(), self.lAgiGain.value(),
        #             self.lDivRatio.value())]
        # 
        #         Delta = Delta + [(hv[0] - ref[0])]
        #         Delta_max = (hv[1] - ref[2]) - (hv[0] - ref[0])
        #         Delta_min = (hv[0] - ref[0]) - (hv[2] - ref[1])
        #         errorData = errorData + [[Delta_min, Delta_max]]
        # 
        #     plot_data = plot_data + [Delta]
        #     Error_data = Error_data + [errorData]
        #     ref_xData = ref_xData + [ref_x_data]
        #     ref_timeData = ref_timeData + [ref_time_data]
        #     xData = xData + [x_data]
        #     timeData = timeData + [time_data]
        #     offsetVolt = offsetVolt + [offset_volt]
        #
        # self.plotdata = AliveTools.changeListFormat(plot_data)
        # self.Error = AliveTools.changeListFormat(error_data)
        # self.ref_x_data = AliveTools.changeListFormat(ref_x_data)
        # self.x_data = AliveTools.changeListFormat(xData)
        # self.ref_time_data = AliveTools.changeListFormat(ref_timeData)
        # self.time_data = AliveTools.changeListFormat(timeData)
        # self.offset_volt = AliveTools.changeListFormat(offsetVolt)

    # noinspection PyAttributeOutsideInit
    def compare_individual(self):
        # TODO: Optimize the naming and code of this function.
        self.recalc()

        self.numberOfRef = len(self.chosenFiles)
        self.numberOfHV = len(self.chosenFiles_2)

        if self.numberOfRef > 0:
            if self.numberOfHV > 0:

                self.plotdata = []
                self.Error = []
                self.ref_x_data = []
                self.ref_time_data = []
                self.x_data = []
                self.time_data = []
                self.offset_volt = []
                
                ref_volt = []
                hv_volt = []

                for ref_file in self.ref_files:
                    if ref_file['filename'] in self.chosenFiles:
                        ref_volt.append(AliveTools.calculateVoltage(self.dbpath, ref_file['filename'], self.run))
                        self.ref_x_data.append(ref_file['number'])
                        self.ref_time_data.append(ref_file['date'])

                for hv_file in self.hv_files:
                    if hv_file['filename'] in self.chosenFiles_2:
                        hv_volt.append(AliveTools.calculateVoltage(self.dbpath, hv_file['filename'], self.run))
                        self.x_data.append(hv_file['number'])
                        self.time_data.append(hv_file['date'])
                        self.offset_volt.append(AliveTools.get_real_offsetVolt_from_db(
                            self.dbpath, hv_file['filename'], self.lDivOffset.value(), self.lAgiGain.value(), self.lDivRatio.value()))

                for a in range(len(hv_volt)):
                    mean = []
                    mean_delta_max = []
                    mean_delta_min = []
                    for b in range(len(ref_volt)):
                        mean.append(-(hv_volt[a][0] - ref_volt[b][0]))
                        mean_delta_max.append(-(hv_volt[a][1] - ref_volt[b][2]))
                        mean_delta_min.append(-(hv_volt[a][2] - ref_volt[b][1]))

                    # Delta = (hv_measurement[0] - ref_measurement + hv_measurement[1])/hv_measurement[1]*1000000
                    self.plotdata.append(np.mean(mean))  # Bisher wurde immer betrag der Spannung ausgegeben.
                    # Auf Wunsch jetzt negativ.
                    self.Error.append([-(np.mean(mean) - np.mean(mean_delta_min)),
                                       -(np.mean(mean_delta_max) - np.mean(mean))])

                self.plotdata = [self.plotdata]
                self.Error = [self.Error]
                self.ref_x_data = [self.ref_x_data]
                self.ref_time_data = [self.ref_time_data]
                self.x_data = [self.x_data]
                self.time_data = [self.time_data]
                self.offset_volt = [self.offset_volt]

                self.saving_mean()
            else:
                print('Please select HV measurement.')
        else:
            print('Please select reference measurement.')

    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def loadIsos(self, run):
        self.isoSelect.clear()
        self.isoSelect_2.clear()

        it = TiTs.select_from_db(self.dbpath, 'DISTINCT iso', 'FitRes', [['run'], [run]], 'ORDER BY iso',
                                 caller_name=__name__)
        if it is not None:
            for i, e in enumerate(it):
                self.isoSelect.insertItem(i, e[0])
                self.isoSelect_2.insertItem(i, e[0])

    def loadRuns(self):
        self.runSelect.clear()
        runit = TiTs.select_from_db(self.dbpath, 'run', 'Runs', caller_name=__name__)
        if runit is not None:
            for i, r in enumerate(runit):
                self.runSelect.insertItem(i, r[0])

    def loadFiles(self):
        num = ['', '_2']
        for i in num:
            getattr(self, 'fileList' + str(i)).clear()
        try:
            self.iso = getattr(self, 'isoSelect' + str(num[0])).currentText()
            self.iso_2 = getattr(self, 'isoSelect' + str(num[1])).currentText()
            self.run = self.runSelect.currentText()
            self.par = 'center'

            # self.files = Analyzer.getFiles(self.iso, self.run, self.dbpath, files)

            self.vals, self.errs, self.dates, self.files = Analyzer.extract(
                getattr(self, 'iso' + str(num[0])), self.par, self.run, self.dbpath, prin=False)
            self.vals_2, self.errs_2, self.dates_2, self.files_2 = Analyzer.extract(
                getattr(self, 'iso' + str(num[1])), self.par, self.run, self.dbpath, prin=False)
            for j in num:
                r = TiTs.select_from_db(self.dbpath, 'config, statErrForm, systErrForm', 'Combined',
                                        [['iso', 'parname', 'run'],
                                         [getattr(self, 'iso' + str(j)), self.par, self.run]],
                                        caller_name=__name__)
                select = [True] * len(getattr(self, 'files' + str(j)))
                if r is not None:
                    if j == '':
                        self.statErrForm = r[0][1]
                        self.systErrForm = r[0][2]
                    else:
                        self.statErrForm_2 = r[0][1]
                        self.systErrForm_2 = r[0][2]
                    cfg = ast.literal_eval(r[0][0])
                    for i, f in enumerate(self.files):
                        if cfg:
                            select[i] = True
                        elif f not in cfg:
                            select[i] = False
                else:
                    if j == '':
                        self.statErrForm = 0
                        self.systErrForm = 0
                    else:
                        self.statErrForm_2 = 0
                        self.systErrForm_2 = 0
                getattr(self, 'fileList' + str(j)).blockSignals(True)

                for f, s in zip(getattr(self, 'files' + str(j)), select):
                    w = QtWidgets.QListWidgetItem(f)
                    w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                    if s:
                        w.setCheckState(QtCore.Qt.Checked)
                    else:
                        w.setCheckState(QtCore.Qt.Unchecked)
                    getattr(self, 'fileList' + str(j)).addItem(w)

                getattr(self, 'fileList' + str(j)).blockSignals(False)

            self.recalc()

        except Exception as e:
            print('error: while loading files in AliveUi: %s' % e)

    # noinspection PyAttributeOutsideInit
    def recalc(self):
        self.all_files = AliveTools.get_list_all_from_db(self.dbpath)
        self.ref_files = [f for f in self.all_files if f['type'] == self.isoSelect.currentText()]
        self.hv_files = [f for f in self.all_files if f['type'] == self.isoSelect_2.currentText()]

        select = []
        self.chosenFiles = []
        self.chosenVals = []
        self.chosenErrs = []
        self.chosenDates = []
        self.val = 0
        self.err = 0
        self.redChi = 0
        self.systeErr = 0

        for index in range(self.fileList.count()):
            if self.fileList.item(index).checkState() != QtCore.Qt.Checked:
                select.append(index)
        if len(self.vals) > 0 and len(self.errs) > 0:
            self.chosenVals = np.delete(copy.deepcopy(self.vals), select)
            self.chosenErrs = np.delete(copy.deepcopy(self.errs), select)
            self.chosenDates = np.delete(copy.deepcopy(self.dates), select)
            self.chosenFiles = np.delete(copy.deepcopy(self.files), select)

        select_2 = []
        self.chosenFiles_2 = []
        self.chosenVals_2 = []
        self.chosenErrs_2 = []
        self.chosenDates_2 = []
        self.val_2 = 0
        self.err_2 = 0
        self.redChi_2 = 0
        self.systeErr_2 = 0

        for index in range(self.fileList_2.count()):
            if self.fileList_2.item(index).checkState() != QtCore.Qt.Checked:
                select_2.append(index)
        if len(self.vals_2) > 0 and len(self.errs_2) > 0:
            self.chosenVals_2 = np.delete(copy.deepcopy(self.vals_2), select_2)
            self.chosenErrs_2 = np.delete(copy.deepcopy(self.errs_2), select_2)
            self.chosenDates_2 = np.delete(copy.deepcopy(self.dates_2), select_2)
            self.chosenFiles_2 = np.delete(copy.deepcopy(self.files_2), select_2)

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()  # might still cause some problems

    def saving(self):
        Plot.clear()
        Plot.AlivePlot(self.x_data, self.plotdata, self.Error)
        Plot.show(True)
        # print(self.x_data)
        # print(self.plotdata)
        # print(self.Error)
        # print(self.ref_x_data)
        # print(self.ref_time_data)
        # print(self.time_data)
        # print(self.offset_volt)

        # Zeitstempel erzeugen
        t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        ind = self.dbpath.rfind('/')
        ind_2 = self.dbpath.rfind('.')
        save_path = self.dbpath[:ind]
        name = self.dbpath[ind + 1:ind_2]

        with open(save_path + '/Results_' + name + '_' + t + '.txt', 'w+') as out_file:
            out_string = 'number, time, Voltage, neg. Error, pos. Error, HV div. volt, Ref. number, Ref. time \n'
            for a in range(len(self.plotdata)):
                for b in range(len(self.plotdata[a])):
                    out_string += str(self.x_data[a][b])
                    out_string += ', '
                    out_string += self.time_data[a][b]
                    out_string += ', '
                    out_string += str(self.plotdata[a][b])
                    out_string += ', '
                    out_string += str(-self.Error[a][b][0])
                    out_string += ', '
                    out_string += str(self.Error[a][b][1])
                    out_string += ', '
                    out_string += str(self.offset_volt[a][b])
                    out_string += ', '
                    out_string += str(self.ref_x_data[a][b])
                    out_string += ', '
                    out_string += str(self.ref_time_data[a][b])
                    out_string += '\n'

                out_string += '\n'
            out_file.write(out_string)

    def saving_mean(self):
        Plot.clear()
        Plot.AlivePlot(self.x_data, self.plotdata, self.Error)
        Plot.show(True)
        # print(self.x_data)
        # print(self.plotdata)
        # print(self.Error)
        # print(self.ref_x_data)
        # print(self.ref_time_data)
        # print(self.time_data)
        # print(self.offset_volt)

        # Zeitstempel erzeugen
        t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        ind = self.dbpath.rfind('/')
        ind_2 = self.dbpath.rfind('.')
        save_path = self.dbpath[:ind]
        name = self.dbpath[ind + 1:ind_2]

        with open(save_path + '/Results_' + name + '_' + t + '.txt', 'w+') as out_file:
            out_string = 'Div. Offset: ' + str(self.lDivOffset.value()) + '\nDiv. Ratio: ' + str(
                self.lDivRatio.value()) + '\nGain: ' + str(self.lAgiGain.value()) + '\n'
            out_string += 'number, time, Voltage, neg. Error, pos. Error, HV div. volt, Ref. number, Ref. time, ' \
                          'ALIVE - VoltDiv / V, Alive - VoltDiv / ppm \n'
            for a in range(len(self.plotdata)):
                for b in range(len(self.plotdata[a])):
                    out_string += str(self.x_data[a][b])
                    out_string += ', '
                    out_string += self.time_data[a][b]
                    out_string += ', '
                    out_string += str(self.plotdata[a][b])
                    out_string += ', '
                    out_string += str(-self.Error[a][b][0])
                    out_string += ', '
                    out_string += str(self.Error[a][b][1])
                    out_string += ', '
                    out_string += str(self.offset_volt[a][b])
                    out_string += ', '
                    for x in self.ref_x_data[a]:
                        out_string += str(x) + '; '
                    out_string += ', '
                    for x in self.ref_time_data[a]:
                        out_string += str(x) + '; '
                    out_string += ', '
                    out_string += str(self.plotdata[a][b] - self.offset_volt[a][b])
                    out_string += ', '
                    out_string += str((self.plotdata[a][b] - self.offset_volt[a][b]) / self.offset_volt[a][b] * 10 ** 6)
                    out_string += '\n'
                out_string += '\n'
            out_file.write(out_string)
