'''
Created on 06.06.2014

@author: hammen, chgorges
'''

import ast
import copy
import sqlite3

import numpy as np
from PyQt5 import QtWidgets, QtCore

import Analyzer
from Gui.Ui_Averager import Ui_Averager


class AveragerUi(QtWidgets.QWidget, Ui_Averager):


    def __init__(self):
        super(AveragerUi, self).__init__()
        self.setupUi(self)

        self.runSelect.currentTextChanged.connect(self.loadIsos)
        self.isoSelect.currentIndexChanged.connect(self.loadParams)
        self.parameter.currentIndexChanged.connect(self.loadFiles)
        self.fileList.itemChanged.connect(self.recalc)
        self.bsave.clicked.connect(self.saving)
        self.pushButton_select_all.clicked.connect(self.select_all)

        self.select_all_state = True
        self.chosenFiles = []  # list of chosen Files (will be defined in self.recalc() due to checkboxes in gui)

        self.dbpath = None
        
        self.show()
        
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)
    
    def loadIsos(self, run):
        self.isoSelect.clear()
        con = sqlite3.connect(self.dbpath)
        for i, e in enumerate(con.execute('''SELECT DISTINCT iso FROM FitRes WHERE run = ? ORDER BY iso''',
                                          (run,))):
            self.isoSelect.insertItem(i, e[0])
        con.close()

    def loadRuns(self):
        self.runSelect.clear()
        con = sqlite3.connect(self.dbpath)        
        for i, r in enumerate(con.execute('''SELECT run FROM Runs''')):
            self.runSelect.insertItem(i, r[0])
        con.close()
        
    def loadParams(self):
        self.parameter.clear()
        con = sqlite3.connect(self.dbpath)
        cur = con.cursor()
        cur.execute('''SELECT pars FROM FitRes WHERE run = ? AND iso = ?''',
                    (self.runSelect.currentText(), self.isoSelect.currentText()))
        r = cur.fetchone()
        con.close()

        try:
            for e in sorted(ast.literal_eval(r[0]).keys()):
                self.parameter.addItem(e)
        except Exception as e:
            print(e)

    def loadFiles(self):
        self.fileList.clear()
        try:
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()

            self.iso = self.isoSelect.currentText()
            self.run = self.runSelect.currentText()
            self.par = self.parameter.currentText()

            # self.files = Analyzer.getFiles(self.iso, self.run, self.dbpath)
            self.vals, self.errs, self.dates, self.files = Analyzer.extract(
                self.iso, self.par, self.run, self.dbpath, prin=False)

            # check if a config exists and if so check only the files within this config
            cur.execute(
                '''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ?''',
                (self.iso, self.par, self.run))
            r = cur.fetchall()
            con.close()

            select = [True] * len(self.files)

            if len(r) > 0:
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

            self.recalc()
        except Exception as e:
            print('error while loading files: %s' % e)

    def select_all(self):
        self.fileList.clear()
        self.select_all_state = not self.select_all_state
        select = [self.select_all_state] * len(self.files)

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
        self.recalc()

    def recalc(self):
        select = []
        self.chosenFiles = []
        self.val = 0
        self.err = 0
        self.redChi = 0
        self.systeErr = 0

        for index in range(self.fileList.count()):
            if self.fileList.item(index).checkState() != QtCore.Qt.Checked:
                select.append(index)
        if len(self.vals) > 0 and len(self.errs) > 0:
            self.chosenFiles = np.delete(copy.deepcopy(self.files), select)
            if len(self.chosenFiles) > 0:
                self.val, self.err, self.systeErr, self.redChi, plotdata, ax = Analyzer.combineRes(
                    self.iso, self.par, self.run, self.dbpath,
                    show_plot=False, only_this_files=self.chosenFiles, write_to_db=False)
        self.result.setText(str(self.val))
        self.rChi.setText(str(self.redChi))
        self.statErr.setText(str(self.err))
        self.systErr.setText(str(self.systeErr))

    def saving(self):
        if len(self.chosenFiles):
            Analyzer.combineRes(self.iso, self.par, self.run, self.dbpath,
                                show_plot=True, only_this_files=self.chosenFiles)
        else:
            print('nothing to save!!!')

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()  # might still cause some problems
