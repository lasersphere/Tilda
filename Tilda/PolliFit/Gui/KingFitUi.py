"""
Created on 05.10.2016

@author: chgorges
"""

from PyQt5 import QtWidgets, QtCore

from Tilda.PolliFit.Gui.Ui_KingFitter import Ui_KingFitter
from Tilda.PolliFit.KingFitter import KingFitter
from Tilda.PolliFit import TildaTools as TiTs


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
        self.setEnabled(False)
    
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def loadIsos(self):
        self.isoList.clear()
        self.isotopes = []
        if self.run == -1:
            isoiter = TiTs.select_from_db(self.dbpath, 'DISTINCT iso', 'Combined',
                                          [['parname'], ['shift']], 'ORDER BY iso', caller_name=__name__)
            if isoiter is not None:
                for e in isoiter:
                    self.isotopes.append(e[0])
        else:
            isoiter = TiTs.select_from_db(self.dbpath, 'DISTINCT iso', 'Combined',
                                          [['parname', 'run'], ['shift', self.run]], 'ORDER BY iso',
                                          caller_name=__name__)
            if isoiter is not None:
                for e in isoiter:
                    self.isotopes.append(e[0])
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
        runIter = TiTs.select_from_db(self.dbpath, 'run', 'Runs', caller_name=__name__)
        if runIter is not None:
            for i, r in enumerate(runIter):
                self.runSelect.insertItem(i, r[0])

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
        self.king.kingFit(run=self.run, alpha=self.sAlpha.value(), findBestAlpha=self.alphaTrue.isChecked())

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
            print(self.runSelect.currentText())

            self.run = self.runSelect.currentText()

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.king = KingFitter(dbpath, showing=True)
        self.loadRuns()
