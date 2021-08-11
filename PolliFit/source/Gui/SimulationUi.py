"""
Created on 17.10.2017

@author: P. Imgram
edited by P. Mueller
"""

import ast
import copy
import sqlite3
from datetime import datetime
import os

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

import logging
import TildaTools as TiTs
import Tools
import Physics
from DBIsotope import DBIsotope
from Gui.Ui_Simulation import Ui_Simulation


class SimulationUi(QtWidgets.QWidget, Ui_Simulation):
    def __init__(self):
        super(SimulationUi, self).__init__()
        self.setupUi(self)
        self.dbpath = None
        self.refresh()

        self.cAmplifier.stateChanged[int].connect(self.toggle_amplifier)
        self.dAmpSlope.valueChanged.connect(self.calcFreq)
        self.dAmpOff.valueChanged.connect(self.calcFreq)
        self.dAccVolt.valueChanged.connect(self.calcFreq)
        self.dAmpSlope.valueChanged.connect(self.update_dIsoAmp)
        self.dAmpOff.valueChanged.connect(self.update_dIsoAmp)
        self.dAccVolt.valueChanged.connect(self.update_dIsoAmp)

        self.cIso.stateChanged[int].connect(self.toggle_iso)
        self.comboIso.currentIndexChanged.connect(self.calcFreq)
        self.dIso.valueChanged[float].connect(self.calcVolt)
        self.dIso.valueChanged.connect(self.calcFreq)
        self.dIsoAmp.valueChanged[float].connect(self.calcVolt)

        self.pShow.clicked.connect(self.showSpec)

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
        self.comboIso.blockSignals(True)
        self.comboIso.clear()
        r = TiTs.select_from_db(self.dbpath, 'DISTINCT iso', 'Isotopes', [], 'ORDER BY iso', caller_name=__name__)
        if r is not None:
            for i, each in enumerate(r):
                self.listIsotopes.insertItem(i, each[0])
                self.comboIso.insertItem(i, each[0])
        self.listIsotopes.blockSignals(False)
        self.comboIso.blockSignals(False)

    def loadLines(self):
        self.coLineVar.blockSignals(True)
        self.coLineVar.clear()
        r = TiTs.select_from_db(self.dbpath, 'DISTINCT lineVar', 'Lines', [], '', caller_name=__name__)
        if r is not None:
            for i, each in enumerate(r):
                self.coLineVar.insertItem(i, each[0])
        self.coLineVar.blockSignals(False)

    def toggle_iso(self, state):
        if state == 0:
            self.dColFreq.setReadOnly(False)
            self.dAcolFreq.setReadOnly(False)
            self.dIso.setReadOnly(True)
            self.dIsoAmp.setReadOnly(True)
        if state == 2:
            self.dColFreq.setReadOnly(True)
            self.dAcolFreq.setReadOnly(True)
            self.dIso.setReadOnly(False)
            if self.cAmplifier.isChecked():
                self.dIsoAmp.setReadOnly(False)
            self.calcFreq()

    def update_dIsoAmp(self):
        self.dIso.valueChanged.emit(self.dIso.value())

    def toggle_amplifier(self, state):
        if state == 0:
            self.dIsoAmp.setReadOnly(True)
            self.dAmpSlope.setReadOnly(True)
            self.dAmpOff.setReadOnly(True)
            self.dAccVolt.setReadOnly(True)
        if state == 2:
            self.dAmpSlope.setReadOnly(False)
            self.dAmpOff.setReadOnly(False)
            self.dAccVolt.setReadOnly(False)
            if self.cIso.isChecked():
                self.dIsoAmp.setReadOnly(False)
                self.update_dIsoAmp()

    def calcFreq(self):
        linevar = self.coLineVar.currentText()
        if self.cIso.isChecked() and linevar != '':
            u = self.dIso.value()
            iso_name = self.comboIso.currentText()
            if iso_name is None:
                return
            iso = DBIsotope(self.dbpath, iso_name, lineVar=linevar)
            v = Physics.relVelocity(u * Physics.qe, iso.mass * Physics.u)
            self.dColFreq.setValue(Physics.relDoppler(iso.center + iso.freq, v))
            self.dAcolFreq.setValue(Physics.relDoppler(iso.center + iso.freq, -v))

    def calcVolt(self, val):
        sender = self.sender()
        if sender is self.dIso:
            if self.cAmplifier.isChecked():
                self.dIsoAmp.setValue(-(val - self.dAccVolt.value() + self.dAmpOff.value()) / self.dAmpSlope.value())
        elif sender is self.dIsoAmp:
            if self.cAmplifier.isChecked():
                self.dIso.setValue(self.dAccVolt.value() - self.dAmpSlope.value() * val - self.dAmpOff.value())

    def get_x_transform(self):
        if self.cInFreq.isChecked() or not self.cAmplifier.isChecked():
            return lambda x: x
        else:
            return lambda x: -(x - self.dAccVolt.value() + self.dAmpOff.value()) / self.dAmpSlope.value()

    def showSpec(self):
        isotopes = self.listIsotopes.selectedItems()
        linevar = self.coLineVar.currentText()
        laserFreqCol = self.dColFreq.value()
        laserFreqAcol = self.dAcolFreq.value()
        colChecked = self.cColFreq.isChecked()
        acolChecked = self.cAcolFreq.isChecked()
        inFreq = self.cInFreq.isChecked()
        x_transform = self.get_x_transform()
        x_label = 'Amplifier voltage [V]' if self.cAmplifier.isChecked() and not self.cInFreq.isChecked() else None

        if len(isotopes) == 0:
            logging.error("No isotopes have been selected!")
        elif not colChecked and not acolChecked:
            logging.error("No direction has been selected!")
        elif (laserFreqCol <= 0 and colChecked) or (laserFreqAcol <= 0 and acolChecked):
            logging.error("No propper laserFreq has been entered!")
        else:
            for iso in isotopes[:-1]:
                if colChecked:
                    Tools.isoPlot(self.dbpath, iso.text(), linevar=linevar, col=True, laserfreq=laserFreqCol,
                                  show=False, as_freq=inFreq, x_transform=x_transform, x_label=x_label)

                if acolChecked:
                    Tools.isoPlot(self.dbpath, iso.text(), linevar=linevar, col=False, laserfreq=laserFreqAcol,
                                  show=False, as_freq=inFreq, x_transform=x_transform, x_label=x_label)

            if colChecked and acolChecked:
                Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=True, laserfreq=laserFreqCol,
                              show=False, as_freq=inFreq, x_transform=x_transform, x_label=x_label)
                Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=False, laserfreq=laserFreqAcol,
                              show=True, as_freq=inFreq, x_transform=x_transform, x_label=x_label)
            else:
                if colChecked:
                    Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=True, laserfreq=laserFreqCol,
                                  show=True, as_freq=inFreq, x_transform=x_transform, x_label=x_label)
                else:
                    Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=False, laserfreq=laserFreqAcol,
                                  show=True, as_freq=inFreq, x_transform=x_transform, x_label=x_label)
