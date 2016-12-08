'''
Created on 29.07.2016

@author: K. Koenig
'''

import ast
import copy
import sqlite3
import time

import numpy as np
from PyQt5 import QtWidgets, QtCore

import AliveTools
import Physics
import Analyzer
import MPLPlotter as plot
from Gui.Ui_Alive import Ui_Alive


class AliveUi(QtWidgets.QWidget, Ui_Alive):
    def __init__(self):
        super(AliveUi, self).__init__()
        self.setupUi(self)

        self.runSelect.currentTextChanged.connect(self.loadIsos)
        self.isoSelect.currentIndexChanged.connect(self.loadFiles)
        self.isoSelect_2.currentIndexChanged.connect(self.loadFiles)

        # self.fileList.itemChanged.connect(self.recalc)
        # self.fileList_2.itemChanged.connect(self.recalc)

        self.pB_compareAuto.clicked.connect(self.compareAuto)
        self.pB_compareIndividual.clicked.connect(self.compareIndividual)

        self.dbpath = None

        self.show()

    def compareAuto(self):
        self.recalc()

        speedOfLight = Physics.c
        electronCharge = Physics.qe
        atomicMassUnit = Physics.u

        f0 = AliveTools.get_transitionFreq_from_db(self.dbpath, self.chosenFiles[0], self.run)
        fL = AliveTools.get_laserFreq_from_db(self.dbpath, self.chosenFiles[0])
        center = 100
        mass = AliveTools.get_mass_from_db(self.dbpath, self.chosenFiles[0])[0] * atomicMassUnit
        v = Physics.invRelDoppler(fL, f0 + center)
        voltTotal = mass * speedOfLight ** 2 * ((1 - (v / speedOfLight) ** 2) ** (-1 / 2) - 1) / electronCharge
        print(voltTotal)

        print(f0)
        f1 = Physics.relDoppler(fL, v)
        print(f1)



    def compareIndividual(self):
        self.recalc()
        self.plotdata = []

        if len(self.chosenFiles) > 0:
            if len(self.chosenFiles2) > 0:

                self.numberOfRef = len(self.chosenFiles)
                self.numberOfHV = len(self.chosenFiles2)

                self.refVolt = []
                self.hvVolt = []
                self.x_data =[]

                for file in self.chosenFiles:
                    ref = AliveTools.calculateVoltage(self.dbpath, file, self.run)
                    self.refVolt = self.refVolt + [ref]

                for file in self.chosenFiles2:
                    hv = AliveTools.calculateVoltage(self.dbpath, file, self.run)
                    self.x_data = self.x_data +[AliveTools.get_nameNumber(file)]
                    offset=AliveTools.get_offsetVolt_from_db(self.dbpath, file)
                    self.hvVolt = self.hvVolt + [[hv,offset]]

                for ref_measurement in self.refVolt:
                    data = []
                    for hv_measurement in self.hvVolt:

                        Delta = (hv_measurement[0] - ref_measurement + hv_measurement[1])/hv_measurement[1]*1000000
                        data = data + [Delta]


                    self.plotdata = self.plotdata + [data]

                self.saving()

            else:
                print('select HV measurement')
        else:
            print('select reference measurement')

    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def loadIsos(self, run):
        self.isoSelect.clear()
        con = sqlite3.connect(self.dbpath)
        for i, e in enumerate(con.execute('''SELECT DISTINCT iso FROM FitRes WHERE run = ? ORDER BY iso''', (run,))):
            self.isoSelect.insertItem(i, e[0])
        con.close()

        self.isoSelect_2.clear()
        con = sqlite3.connect(self.dbpath)
        for i, e in enumerate(con.execute('''SELECT DISTINCT iso FROM FitRes WHERE run = ? ORDER BY iso''', (run,))):
            self.isoSelect_2.insertItem(i, e[0])
        con.close()

    def loadRuns(self):
        self.runSelect.clear()
        con = sqlite3.connect(self.dbpath)
        for i, r in enumerate(con.execute('''SELECT run FROM Runs''')):
            self.runSelect.insertItem(i, r[0])
        con.close()

    # better split this up into two functions!
    def loadFiles(self):
        self.fileList.clear()
        try:
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()

            self.iso = self.isoSelect.currentText()
            self.run = self.runSelect.currentText()
            self.par = 'center'

            self.files = Analyzer.getFiles(self.iso, self.run, self.dbpath)
            self.vals, self.errs, self.dates = Analyzer.extract(self.iso, self.par, self.run, self.dbpath, prin=False)

            cur.execute(
                '''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ?''',
                (self.iso, self.par, self.run))
            r = cur.fetchall()
            con.close()

            select = [True] * len(self.files)
            self.statErrForm = 0
            self.systErrForm = 0

            if len(r) > 0:
                self.statErrForm = r[0][1]
                self.systErrForm = r[0][2]
                cfg = ast.literal_eval(r[0][0])
                for i, f in enumerate(self.files):
                    if cfg == []:
                        select[i] = True
                    elif f not in cfg:
                        select[i] = False

            self.fileList.blockSignals(True)
            for f, s in zip(self.files, select):
                w = QtWidgets.QListWidgetItem(f)
                w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                if s:
                    w.setCheckState(QtCore.Qt.Checked)
                else:
                    w.setCheckState(QtCore.Qt.Unchecked)
                self.fileList.addItem(w)

            self.fileList.blockSignals(False)

            # self.recalc()
        except Exception as e:
            print(str(e))


            #########################################  FILELIST_2
        self.fileList_2.clear()
        try:
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()

            self.iso2 = self.isoSelect_2.currentText()
            self.run = self.runSelect.currentText()
            self.par = 'center'

            self.files2 = Analyzer.getFiles(self.iso2, self.run, self.dbpath)
            self.vals2, self.errs2, self.dates2 = Analyzer.extract(self.iso2, self.par, self.run, self.dbpath,
                                                                   prin=False)

            cur.execute(
                '''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ?''',
                (self.iso2, self.par, self.run))
            r = cur.fetchall()
            con.close()

            select2 = [True] * len(self.files2)
            self.statErrForm2 = 0
            self.systErrForm2 = 0

            if len(r) > 0:
                self.statErrForm2 = r[0][1]
                self.systErrForm2 = r[0][2]
                cfg2 = ast.literal_eval(r[0][0])
                for i, f in enumerate(self.files2):
                    if cfg2 == []:
                        select2[i] = True
                    elif f not in cfg2:
                        select2[i] = False

            self.fileList_2.blockSignals(True)
            for f, s in zip(self.files2, select2):
                w = QtWidgets.QListWidgetItem(f)
                w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                if s:
                    w.setCheckState(QtCore.Qt.Checked)
                else:
                    w.setCheckState(QtCore.Qt.Unchecked)
                self.fileList_2.addItem(w)

            self.fileList_2.blockSignals(False)

            self.recalc()
        except Exception as e:
            print('error: while loading files in AliveUi: %s' % e)

    def recalc(self):
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

        select2 = []
        self.chosenFiles2 = []
        self.chosenVals2 = []
        self.chosenErrs2 = []
        self.chosenDates2 = []
        self.val2 = 0
        self.err2 = 0
        self.redChi2 = 0
        self.systeErr2 = 0

        for index in range(self.fileList_2.count()):
            if self.fileList_2.item(index).checkState() != QtCore.Qt.Checked:
                select2.append(index)
        if len(self.vals2) > 0 and len(self.errs2) > 0:
            self.chosenVals2 = np.delete(copy.deepcopy(self.vals2), select2)
            self.chosenErrs2 = np.delete(copy.deepcopy(self.errs2), select2)
            self.chosenDates2 = np.delete(copy.deepcopy(self.dates2), select2)
            self.chosenFiles2 = np.delete(copy.deepcopy(self.files2), select2)

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()  # might still cause some problems

    def saving(self):
        plot.clear()
        plot.AlivePlot(self.x_data, self.plotdata)
        plot.show(True)
