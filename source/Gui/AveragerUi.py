'''
Created on 06.06.2014

@author: hammen
'''

import sqlite3
import ast

from PyQt5 import QtWidgets, QtCore

from Gui.Ui_Averager import Ui_Averager
import Analyzer

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

        for e in sorted(ast.literal_eval(r[0]).keys()):
            self.parameter.addItem(e)
        con.close()
        
        
    def loadFiles(self):
        self.fileList.clear()
        con = sqlite3.connect(self.dbpath)
        cur = con.cursor()
        
        cur.execute('''SELECT file, pars FROM FitRes WHERE iso = ? AND run = ?''', (self.isoSelect.currentText(), self.runSelect.currentText()))
        e = cur.fetchall()
        files = [f[0] for f in e]

        cur.execute('''SELECT config FROM Combined WHERE iso = ? AND parname = ? AND run = ?''', (self.isoSelect.currentText(), self.parameter.currentText(), self.runSelect.currentText()))
        r = cur.fetchall()
        con.close()
        
        select = [True] * len(files)
        
        if len(r) > 0:
            for i, f in enumerate(files):
                if f not in select:
                    select[i] = False
                    
        self.fileList.blockSignals(True)
        for f, s in zip(files, select):
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
        files = []
        select = []
        for i in range(self.fileList.count()):
            files.append(self.fileList.item(i).text())
            select.append(self.fileList.item(i).checkState() == QtCore.Qt.Checked)
        
        lin = [f for f, s in zip(files, select) if s == True]
        lout = [f for f, s in zip(files, select) if s == False]
        
        val, errorprop, rChi = Analyzer.weightedAverage(*Analyzer.extract(self.isoSelect.currentText(), self.parameter.currentText(), self.runSelect.currentText(), self.dbpath, lin))
        self.result.setText(str(val))
        self.rChi.setText(str(rChi))
        
    
    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()
        
