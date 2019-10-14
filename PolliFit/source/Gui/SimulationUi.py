'''
Created on 17.10.2017

@author: P. Imgram
'''

import ast
import copy
import sqlite3
from datetime import datetime
import os

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

import AliveTools
import Physics
import functools
import TildaTools as TiTs
import Tools
import ColAcolTools as cTo
import Analyzer
import MPLPlotter as plot
from Gui.Ui_Simulation import Ui_Simulation


class SimulationUi(QtWidgets.QWidget, Ui_Simulation):
    def __init__(self):
        super(SimulationUi, self).__init__()
        self.setupUi(self)
        self.pShow.clicked.connect(self.showSpec)
        self.dbpath = None
        self.refresh()



    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)


    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.refresh()

    def refresh(self):
        self.loadIsotopes()
        self.loadLines()

    def loadIsotopes(self):
        self.listIsotopes.blockSignals(True)
        self.listIsotopes.clear()
        r = TiTs.select_from_db(self.dbpath, 'DISTINCT iso', 'Isotopes', [], 'ORDER BY iso', caller_name=__name__)
        if r is not None:
            for i, each in enumerate(r):
                self.listIsotopes.insertItem(i, each[0])

        self.listIsotopes.blockSignals(False)

    def loadLines(self):
        self.coLineVar.blockSignals(True)
        self.coLineVar.clear()
        r = TiTs.select_from_db(self.dbpath, 'DISTINCT lineVar', 'Lines', [], '', caller_name=__name__)
        if r is not None:
            for i, each in enumerate(r):
                self.coLineVar.insertItem(i, each[0])

        self.coLineVar.blockSignals(False)

    def showSpec(self):
        isotopes = self.listIsotopes.selectedItems()
        linevar = self.coLineVar.currentText()
        laserFreqCol = self.dColFreq.value()
        laserFreqAcol = self.dAcolFreq.value()
        colChecked = self.cColFreq.isChecked()
        acolChecked = self.cAcolFreq.isChecked()
        inFreq = self.cInFreq.isChecked()

        for iso in isotopes[:-1]:
            if colChecked:
                Tools.isoPlot(self.dbpath, iso.text(), linevar=linevar, col=True, laserfreq=laserFreqCol, show=False,
                          as_freq=inFreq)

            if acolChecked:
                Tools.isoPlot(self.dbpath, iso.text(), linevar=linevar, col=False, laserfreq=laserFreqAcol, show=False,
                              as_freq=inFreq)

        if colChecked and acolChecked:
            Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=True, laserfreq=laserFreqCol, show=False,
                          as_freq=inFreq)
            Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=False, laserfreq=laserFreqAcol, show=True,
                          as_freq=inFreq)
        else:
            if colChecked:
                Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=True, laserfreq=laserFreqCol,
                              show=True,
                              as_freq=inFreq)
            else:
                Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=False, laserfreq=laserFreqAcol,
                              show=True,
                              as_freq=inFreq)