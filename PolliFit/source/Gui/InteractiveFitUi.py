'''
Created on 06.06.2014

@author: hammen
'''

import ast

from PyQt5 import QtWidgets, QtCore, QtGui

import TildaTools as TiTs
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
        try:
            self.intFit.fit()
            self.loadPars()
        except Exception as e:
            print('error while fitting: %s' % e)

    
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
        it = TiTs.select_from_db(self.dbpath, 'DISTINCT type', 'Files', addCond='ORDER BY type', caller_name=__name__)
        if it is not None:
            for i, e in enumerate(it):
                self.isoFilter.insertItem(i, e[0])
    
    
    def loadRuns(self):
        self.runSelect.clear()
        it = TiTs.select_from_db(self.dbpath, 'run', 'Runs', caller_name=__name__)
        if it is not None:
            for i, r in enumerate(it):
                self.runSelect.insertItem(i, r[0])
        
        
    def loadFiles(self):
        self.fileList.clear()
        it = TiTs.select_from_db(self.dbpath, 'file', 'Files',
                                     [['type'], [self.isoFilter.currentText()]], 'ORDER BY date', caller_name=__name__)
        if it is not None:
            for r in it:
                self.fileList.addItem(r[0])
    
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

