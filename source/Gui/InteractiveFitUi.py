'''
Created on 06.06.2014

@author: hammen
'''

import ast
import sqlite3

from PyQt5 import QtWidgets, QtCore, QtGui

from Gui.Ui_InteractiveFit import Ui_InteractiveFit
from InteractiveFit import InteractiveFit


class InteractiveFitUi(QtWidgets.QWidget, Ui_InteractiveFit):


    def __init__(self):
        super(InteractiveFitUi, self).__init__()
        self.setupUi(self)

        self.bLoad.clicked.connect(self.load)
        self.bFit.clicked.connect(self.fit)
        self.bReset.clicked.connect(self.reset)
        self.isoFilter.currentIndexChanged.connect(self.loadFiles)
        self.bFontSize.valueChanged.connect(self.fontSize)
        self.parTable.cellChanged.connect(self.setPar)

        """ add shortcuts """
        QtWidgets.QShortcut(QtGui.QKeySequence("L"), self, self.load)
        QtWidgets.QShortcut(QtGui.QKeySequence("F"), self, self.fit)
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, self.reset)

        self.dbpath = None
        
        self.show()
        
    
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)
    
    
    def load(self):
        if self.fileList.currentItem() is not None:
            iso = self.fileList.currentItem().text()
            if iso:
                self.intFit = InteractiveFit(iso, self.dbpath, self.runSelect.currentText())
                self.loadPars()
        
        
    def fit(self):
        self.intFit.fit()
        self.loadPars()
    
    
    def reset(self):
        self.intFit.reset()
        self.loadPars()
        
    def loadPars(self):
        self.parTable.blockSignals(True)
        self.parTable.setRowCount(len(self.intFit.fitter.par))
        for i, (n, v, f) in enumerate(self.intFit.getPars()):
            w = QtWidgets.QTableWidgetItem(n)
            w.setFlags(QtCore.Qt.ItemIsEnabled)
            self.parTable.setItem(i, 0, w)
            
            w = QtWidgets.QTableWidgetItem(str(v))
            self.parTable.setItem(i, 1, w)
            
            w = QtWidgets.QTableWidgetItem(str(f))
            self.parTable.setItem(i, 2, w)
            
        self.parTable.blockSignals(False)
                
        
    def loadIsos(self):
        self.isoFilter.clear()
        con = sqlite3.connect(self.dbpath)
        for i, e in enumerate(con.execute('''SELECT DISTINCT type FROM Files ORDER BY type''')):
            self.isoFilter.insertItem(i, e[0])
        
        con.close()
    
    
    def loadRuns(self):
        self.runSelect.clear()
        con = sqlite3.connect(self.dbpath)        
        for i, r in enumerate(con.execute('''SELECT run FROM Runs''')):
            self.runSelect.insertItem(i, r[0])
        con.close()
        
        
    def loadFiles(self):
        self.fileList.clear()
        con = sqlite3.connect(self.dbpath)        
        for r in con.execute('''SELECT file FROM Files WHERE type = ? ORDER BY date''', (self.isoFilter.currentText(),)):
            self.fileList.addItem(r[0])
        con.close()
        
    
    def setPar(self, i, j):
        try:
            val = ast.literal_eval(self.parTable.item(i, j).text())
        except SyntaxError as e:
            val = 0.0
        if j == 1:
            self.intFit.setPar(i, val)
        else:
            self.intFit.setFix(i, val)
    
    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()
        self.loadIsos()

    def fontSize(self):
        if self.fileList.currentItem() is not None:
            iso = self.fileList.currentItem().text()
            if iso:
                self.intFit = InteractiveFit(iso, self.dbpath, self.runSelect.currentText(), fontSize=self.bFontSize.value())
                self.loadPars()

