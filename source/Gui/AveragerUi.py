'''
Created on 06.06.2014

@author: hammen, chgorges
'''

import ast
import copy
import os
import sqlite3

import numpy as np
from PyQt5 import QtWidgets, QtCore

import Analyzer
import MPLPlotter as plot
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

        self.dbpath = None
        
        self.show()
        
    
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)
    
        
    def loadIsos(self, run):
        self.isoSelect.clear()
        con = sqlite3.connect(self.dbpath)
        for i, e in enumerate(con.execute('''SELECT DISTINCT iso FROM FitRes WHERE run = ? ORDER BY iso''', (run,))):
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
        cur.execute('''SELECT pars FROM FitRes WHERE run = ? AND iso = ?''', (self.runSelect.currentText(), self.isoSelect.currentText()))
        r = cur.fetchone()
        try:
            for e in sorted(ast.literal_eval(r[0]).keys()):
                self.parameter.addItem(e)
        except Exception as e:
            print(e)
        con.close()
        
        
    def loadFiles(self):
        self.fileList.clear()
        try:
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()

            self.iso = self.isoSelect.currentText()
            self.run = self.runSelect.currentText()
            self.par = self.parameter.currentText()

            self.files = Analyzer.getFiles(self.iso, self.run, self.dbpath)
            self.vals, self.errs, self.dates = Analyzer.extract(self.iso, self.par, self.run, self.dbpath, prin=False)

            cur.execute('''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ?''', (self.iso, self.par, self.run))
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

            self.recalc()
        except Exception as e:
            print(str(e))

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
            if len(self.chosenVals) > 0 and len(self.chosenErrs > 0) and np.count_nonzero(self.chosenErrs) == len(self.chosenErrs):
                self.val, self.err, self.redChi = Analyzer.weightedAverage(self.chosenVals, self.chosenErrs)
        self.err = self.err * self.redChi  # does the user want it like this?
        self.result.setText(str(self.val))
        self.rChi.setText(str(self.redChi))
        self.statErr.setText(str(self.err))
        self.systErr.setText(str(self.systeErr))

    def saving(self):
        if np.any(self.chosenVals):
            filename = '%s_%s_%s.png' % (self.iso, self.run, self.par)
            path = os.path.join(os.path.split(self.dbpath)[0], 'combined_plots', filename)
            print(path)
            plot.close_all_figs()
            plot.plotAverage(self.chosenDates, self.chosenVals, self.chosenErrs,
                             self.val, self.err, self.systeErr, showing=True, save_path=path,
                             ylabel=str(filename.split(sep='.')[0]) + ' [MHz]')
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()
            cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''',
                        (self.iso, self.par, self.run))
            con.commit()
            con.close()
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()
            cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, rChi = ? WHERE iso = ?
                AND parname = ? AND run = ?''', (self.val, self.err, self.systeErr, self.redChi, self.iso, self.par, self.run))
            con.commit()
            con.close()

            print('date \t file \t val \t err')
            for i, dt in enumerate(self.chosenDates):
                print(dt, '\t', self.chosenFiles[i], '\t', self.chosenVals[i], '\t', self.chosenErrs[i])
        else:
            print('nothing to save!!!')

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()  # might still cause some problems
