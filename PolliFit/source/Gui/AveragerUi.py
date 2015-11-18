'''
Created on 06.06.2014

@author: hammen
'''

import sqlite3
import ast
import itertools
import numpy as np

from PyQt5 import QtWidgets, QtCore

from Gui.Ui_Averager import Ui_Averager
import Analyzer
import MPLPlotter as plot

class AveragerUi(QtWidgets.QWidget, Ui_Averager):


    def __init__(self):
        super(AveragerUi, self).__init__()
        self.setupUi(self)

        self.runSelect.currentTextChanged.connect(self.loadIsos)
        self.isoSelect.currentIndexChanged.connect(self.loadParams)
        self.parameter.currentIndexChanged.connect(self.loadFiles)
        self.fileList.itemChanged.connect(self.recalc)
        
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

            iso = self.isoSelect.currentText()
            run = self.runSelect.currentText()
            par = self.parameter.currentText()

            self.files = Analyzer.getFiles(iso, run, self.dbpath)
            self.vals, self.errs = Analyzer.extract(iso, par, run, self.dbpath)

            cur.execute('''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ?''', (iso, par, run))
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
                    if f not in cfg:
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


    def recalc(self):
        select = []
        for i in range(self.fileList.count()):
            select.append(self.fileList.item(i).checkState() == QtCore.Qt.Checked)
        if len(self.vals) > 0 and len(self.errs) > 0:
            vals = np.delete(self.vals, select, 0)
            errs = np.delete(self.errs, select, 0)
            val, err, rChi = Analyzer.weightedAverage(vals, errs)

            self.result.setText(str(val))
            self.rChi.setText(str(rChi))
        
        # plot.plotAverage(self.vals, self.errs, val, self.statErr)
        
    
    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()  # might still cause some problems
