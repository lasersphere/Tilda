"""
Created on 17.10.2017

@author: P. Imgram
edited by P. Mueller
"""

from PyQt5 import QtCore, QtWidgets

import logging
from importlib import import_module
from operator import attrgetter
import ast
import sqlite3
import TildaTools as TiTs
import Tools
import Physics
from DBIsotope import DBIsotope
from Gui.Ui_Simulation import Ui_Simulation


# noinspection PyPep8Naming
class SimulationUi(QtWidgets.QWidget, Ui_Simulation):
    def __init__(self):
        super(SimulationUi, self).__init__()
        self.setupUi(self)
        self.dbpath = None
        self.iso = None
        self.line = None
        self.refresh()

        self.coLineVar.currentIndexChanged.connect(self.load_parameters)

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

        self.cInFreq.stateChanged[int].connect(self.toggle_units)
        self.sCharge.valueChanged[int].connect(self.toggle_charge_state)

        self.pShow.clicked.connect(self.showSpec)

        self.cAutosave.stateChanged[int].connect(self.toggle_autosave)
        self.bSavePars.clicked.connect(self.save_parameters)

        self.enable_iso_gui(False)
        self.enable_amplifier_gui(False)

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
        self.load_parameters()

    def load_parameters(self):
        line = self.coLineVar.currentText()
        iso = self.comboIso.currentText()
        if line and iso:
            self.iso = DBIsotope(self.dbpath, iso, lineVar=line)
            self.line = self.iso.shape['name']
            model = import_module('Spectra.{}'.format(self.line))
            spec = attrgetter(self.line)(model)(self.iso)
            # spec = HyperfineN(iso, spec)
            par_names = spec.getParNames()
            pars = spec.getPars()
            self.parTable.blockSignals(True)
            self.parTable.setRowCount(len(par_names))
            for i, (n, p) in enumerate(zip(par_names, pars)):
                # self.parTable.insertRow(i)

                w = QtWidgets.QTableWidgetItem(n)
                w.setFlags(QtCore.Qt.ItemIsEnabled)
                self.parTable.setItem(i, 0, w)

                w = QtWidgets.QTableWidgetItem(str(p))
                w.setFlags(QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsEditable)

                self.parTable.setItem(i, 1, w)
            self.parTable.blockSignals(False)

    def save_parameters(self):
        shape = ast.literal_eval(TiTs.select_from_db(
            self.dbpath, 'shape', 'Lines', [['lineVar'], [self.iso.lineVar]])[0][0])
        shape.update({self.parTable.item(i, 0).text(): ast.literal_eval(self.parTable.item(i, 1).text())
                      for i in range(self.parTable.rowCount())})
        shape.update({'name': self.line})
        con = sqlite3.connect(self.dbpath)
        cur = con.cursor()
        try:
            cur.execute('UPDATE Lines SET shape = ? WHERE lineVar = ?', (str(shape), self.iso.lineVar))
            con.commit()
            print('Saved line pars in Lines!')
        except sqlite3.Error:
            print('error: Couldn\'t save line pars. All values correct?')
        con.close()

    def toggle_autosave(self, state):
        if state == 0:
            self.bSavePars.setEnabled(True)
            self.parTable.itemChanged.disconnect(self.save_parameters)
        if state == 2:
            self.bSavePars.setEnabled(False)
            self.save_parameters()
            self.parTable.itemChanged.connect(self.save_parameters)
    
    def enable_iso_gui(self, enable):
        self.comboIso.setEnabled(enable)
        self.label_7.setEnabled(enable)
        self.dIso.setEnabled(enable)
        if self.cAmplifier.isChecked():
            self.dIsoAmp.setEnabled(enable)
        else:
            self.sCharge.setEnabled(enable)
    
    def toggle_iso(self, state):
        if state == 0:
            self.dColFreq.setReadOnly(False)
            self.dAcolFreq.setReadOnly(False)
            self.enable_iso_gui(False)
        if state == 2:
            self.dColFreq.setReadOnly(True)
            self.dAcolFreq.setReadOnly(True)
            self.enable_iso_gui(True)
            self.calcFreq()

    def update_dIsoAmp(self):
        self.dIso.valueChanged.emit(self.dIso.value())

    def enable_amplifier_gui(self, enable):
        self.dAmpSlope.setEnabled(enable)
        self.dAmpOff.setEnabled(enable)
        self.dAccVolt.setEnabled(enable)
        self.label_4.setEnabled(enable)
        self.label_5.setEnabled(enable)
        self.label_8.setEnabled(enable)
        self.label_9.setEnabled(enable)
        if self.cIso.isChecked():
            self.dIsoAmp.setEnabled(enable)
            self.update_dIsoAmp()
        else:
            self.sCharge.setEnabled(enable)

    def toggle_amplifier(self, state):
        self.toggle_units(self.cInFreq.checkState())
        if state == 0:
            self.enable_amplifier_gui(False)
        if state == 2:
            self.enable_amplifier_gui(True)

    def toggle_units(self, state):
        if state == 2:
            self.label_13.setText('[MHz]')
        else:
            self.label_13.setText('[V]')

    def toggle_charge_state(self):
        if self.cIso.isChecked():
            self.calcFreq()
            if self.sCharge.value() == 0:
                self.label_12.setText('Voltage has no effect on neutral atoms.')
            else:
                self.label_12.setText('')

    def calcFreq(self):
        linevar = self.coLineVar.currentText()
        if self.cIso.isChecked() and linevar != '':
            u = self.dIso.value()
            iso_name = self.comboIso.currentText()
            if iso_name is None:
                return
            self.iso = DBIsotope(self.dbpath, iso_name, lineVar=linevar)
            v = Physics.relVelocity(u * Physics.qe * self.sCharge.value(), self.iso.mass * Physics.u)
            self.dColFreq.setValue(Physics.relDoppler(self.iso.center + self.iso.freq, v))
            self.dAcolFreq.setValue(Physics.relDoppler(self.iso.center + self.iso.freq, -v))

    def calcVolt(self, val):
        sender = self.sender()
        if sender is self.dIso:
            if self.cAmplifier.isChecked():
                self.dIsoAmp.setValue(-(val - self.dAccVolt.value() + self.dAmpOff.value()) / self.dAmpSlope.value())
        elif sender is self.dIsoAmp:
            if self.cAmplifier.isChecked():
                self.dIso.setValue(self.dAccVolt.value() - self.dAmpSlope.value() * val - self.dAmpOff.value())

    def get_x_transform(self):
        if self.cInFreq.isChecked():
            return lambda x: x
        elif not self.cAmplifier.isChecked():
            return lambda x: x / self.sCharge.value()
        else:
            return lambda x: -(x / self.sCharge.value() - self.dAccVolt.value() + self.dAmpOff.value()) \
                             / self.dAmpSlope.value()

    def showSpec(self):
        isotopes = self.listIsotopes.selectedItems()
        linevar = self.coLineVar.currentText()
        laserFreqCol = self.dColFreq.value()
        laserFreqAcol = self.dAcolFreq.value()
        colChecked = self.cColFreq.isChecked()
        acolChecked = self.cAcolFreq.isChecked()
        inFreq = self.cInFreq.isChecked()
        x_transform = self.get_x_transform()
        x_label = None
        if not inFreq:
            if self.cAmplifier.isChecked():
                x_label = 'Amplifier voltage [V]'
            else:
                x_label = 'Acceleration voltage [V]'

        if len(isotopes) == 0:
            logging.error('No isotopes have been selected!')
        elif not colChecked and not acolChecked:
            logging.error('No direction has been selected!')
        elif (laserFreqCol <= 0 and colChecked) or (laserFreqAcol <= 0 and acolChecked):
            logging.error('No proper laserFreq has been entered!')
        else:
            for iso in isotopes[:-1]:
                if colChecked:
                    Tools.isoPlot(self.dbpath, iso.text(), linevar=linevar, col=True, laserfreq=laserFreqCol,
                                  show=False, as_freq=inFreq, x_transform=x_transform, x_label=x_label,
                                  norm=self.cNorm.isChecked())

                if acolChecked:
                    Tools.isoPlot(self.dbpath, iso.text(), linevar=linevar, col=False, laserfreq=laserFreqAcol,
                                  show=False, as_freq=inFreq, x_transform=x_transform, x_label=x_label,
                                  norm=self.cNorm.isChecked())

            if colChecked and acolChecked:
                Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=True, laserfreq=laserFreqCol,
                              show=False, as_freq=inFreq, x_transform=x_transform, x_label=x_label,
                              norm=self.cNorm.isChecked())
                Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=False, laserfreq=laserFreqAcol,
                              show=True, as_freq=inFreq, x_transform=x_transform, x_label=x_label,
                              norm=self.cNorm.isChecked())
            else:
                if colChecked:
                    Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=True, laserfreq=laserFreqCol,
                                  show=True, as_freq=inFreq, x_transform=x_transform, x_label=x_label,
                                  norm=self.cNorm.isChecked())
                else:
                    Tools.isoPlot(self.dbpath, isotopes[-1].text(), linevar=linevar, col=False, laserfreq=laserFreqAcol,
                                  show=True, as_freq=inFreq, x_transform=x_transform, x_label=x_label,
                                  norm=self.cNorm.isChecked())
