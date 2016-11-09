'''
Created on 05.10.2016

@author: chgorges
'''

import ast
import copy
import os
import sqlite3

import numpy as np
from PyQt5 import QtWidgets, QtCore

import MPLPlotter as plot
from Gui.Ui_KingFitter import Ui_KingFitter
from KingFitter import KingFitter


class KingFitUi(QtWidgets.QWidget, Ui_KingFitter):


    def __init__(self):
        super(KingFitUi, self).__init__()
        self.setupUi(self)

        self.runSelect.currentIndexChanged.connect(self.loadIsos)

        self.bKingFit.clicked.connect(self.kingFit)
        self.bCalcChargeRadii.clicked.connect(self.calcChargeRadii)
        self.pushButton_select_all.clicked.connect(self.select_all)

        self.allRuns.stateChanged.connect(self.changeRun)

        self.select_all_state = True


        self.run = -1
        self.isotopes = []
        self.dbpath = None
        
        self.show()
        
    
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)
    
        
    def loadIsos(self):
        print('loading isos!')
        self.isoList.clear()
        self.isotopes = []
        con = sqlite3.connect(self.dbpath)
        if self.run == -1:
            for i, e in enumerate(con.execute('''SELECT DISTINCT iso FROM Combined WHERE parname ="shift"  ORDER BY iso''')):
                self.isotopes.append(e[0])
        else:
            for i, e in enumerate(con.execute('''SELECT DISTINCT iso FROM Combined WHERE parname ="shift" AND run = ?  ORDER BY iso''', (self.run,))):
                self.isotopes.append(e[0])
        con.close()

        select = [True] * len(self.isotopes)
        self.isoList.blockSignals(True)
        for f, s in zip(self.isotopes, select):
            w = QtWidgets.QListWidgetItem(f)
            w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            if s:
                w.setCheckState(QtCore.Qt.Checked)
            else:
                w.setCheckState(QtCore.Qt.Unchecked)
            self.isoList.addItem(w)

        self.isoList.blockSignals(False)


    def loadRuns(self):
        self.runSelect.clear()
        con = sqlite3.connect(self.dbpath)
        for i, r in enumerate(con.execute('''SELECT run FROM Runs''')):
            self.runSelect.insertItem(i, r[0])
        con.close()

    def select_all(self):
        self.select_all_state = not self.select_all_state
        select = [self.select_all_state] * len(self.isotopes)
        self.isoList.clear()
        self.isoList.blockSignals(True)
        for f, s in zip(self.isotopes, select):
            w = QtWidgets.QListWidgetItem(f)
            w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            if s:
                w.setCheckState(QtCore.Qt.Checked)
            else:
                w.setCheckState(QtCore.Qt.Unchecked)
            self.isoList.addItem(w)

        self.isoList.blockSignals(False)

    def kingFit(self):
        self.king.kingFit(run=self.run,alpha=self.sAlpha.value(),findBestAlpha=self.alphaTrue.isChecked())

    def calcChargeRadii(self):
        isotopeIndex = []
        isotopes = []
        for index in range(self.isoList.count()):
            if self.isoList.item(index).checkState() == QtCore.Qt.Checked:
                isotopeIndex.append(index)
        for i, j in enumerate(self.isotopes):
            if i in isotopeIndex:
                isotopes.append(j)
        self.king.calcChargeRadii(isotopes=isotopes, run=self.run)

    def changeRun(self):
        if self.allRuns.isChecked():
            self.run = -1
        else:
            self.run = self.runSelect.itemText()

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.king = KingFitter(dbpath,showing=True)
        self.loadRuns()  # might still cause some problems
