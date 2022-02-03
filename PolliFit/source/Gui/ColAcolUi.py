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
import ColAcolTools as cTo
import Analyzer
import MPLPlotter as plot
from Gui.Ui_ColAcol import Ui_ColAcol


class ColAcolUi(QtWidgets.QWidget, Ui_ColAcol):
    def __init__(self):
        super(ColAcolUi, self).__init__()
        self.setupUi(self)

        self.dbpath = None

        # Old UI Design
        # self.bCol.clicked.connect(self.shiftToCol)
        # self.bAcol.clicked.connect(self.shiftToAcol)
        # self.bReset.clicked.connect(self.shiftToAll)
        self.bRemove.clicked.connect(self.remove)
        self.bStart.clicked.connect(self.startEval)
        self.coAcolIso.currentTextChanged.connect(self.loadRunsAcol)
        self.coColIso.currentTextChanged.connect(self.loadRunsCol)
        self.coAcolRun.currentTextChanged.connect(self.loadFits)
        self.coColRun.currentTextChanged.connect(self.loadFits)

        # QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+A"), self, self.shiftToAcol)
        # QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+C"), self, self.shiftToCol)



    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)


    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadIsotopes()
        self.setPath()

    def loadIsotopes(self):
        self.coColIso.blockSignals(True)
        self.coAcolIso.blockSignals(True)
        self.coAcolIso.clear()
        self.coColIso.clear()
        r = TiTs.select_from_db(self.dbpath, 'DISTINCT iso', 'FitRes', [], 'ORDER BY iso', caller_name=__name__)
        if r is not None:
            for i, each in enumerate(r):
                if each[0].find('_col') != -1 or each[0].find('Col') != -1:
                    self.coColIso.insertItem(i, each[0])
                elif each[0].find('_acol') != -1 or each[0].find('Acol') != -1:
                    self.coAcolIso.insertItem(i, each[0])
                else:
                    self.coAcolIso.insertItem(i, each[0])
                    self.coColIso.insertItem(i, each[0])
        self.coColIso.blockSignals(False)
        self.coAcolIso.blockSignals(False)

        self.loadRunsAcol()
        self.loadRunsCol()


    def loadRunsAcol(self):
        self.coAcolRun.blockSignals(True)
        self.coAcolRun.clear()
        r_acol = TiTs.select_from_db(self.dbpath, 'DISTINCT run', 'FitRes',
                                     [['iso'], [str(self.coAcolIso.currentText())]], 'ORDER BY run',
                                     caller_name=__name__)
        if r_acol is not None:
            for i, each in enumerate(r_acol):
                self.coAcolRun.insertItem(i, each[0])

        self.coAcolRun.blockSignals(False)
        self.loadFits()

    def loadRunsCol(self):
        self.coColRun.blockSignals(True)
        self.coColRun.clear()
        r_col = TiTs.select_from_db(self.dbpath, 'DISTINCT run', 'FitRes',
                                    [['iso'], [str(self.coColIso.currentText())]], 'ORDER BY run',
                                    caller_name=__name__)
        if r_col is not None:
            for i, each in enumerate(r_col):
                self.coColRun.insertItem(i, each[0])
        self.coColRun.blockSignals(False)
        self.loadFits()

    # Old UI Design
    # def addToAll(self, label):
    #     w = QtWidgets.QListWidgetItem(label)
    #     w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
    #     w.setCheckState(QtCore.Qt.Unchecked)
    #     self.lAll.addItem(w)

    def addToCol(self, label):
        w = QtWidgets.QListWidgetItem(label)
        w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        w.setCheckState(QtCore.Qt.Unchecked)
        self.lCol.addItem(w)

    def addToAcol(self, label):
        w = QtWidgets.QListWidgetItem(label)
        w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        w.setCheckState(QtCore.Qt.Unchecked)
        self.lAcol.addItem(w)

    def loadFits(self):
        # self.lAll.clear()
        self.lAcol.clear()
        self.lCol.clear()
        db = self.dbpath
        r_acol = TiTs.select_from_db(db, 'file', 'FitRes', [['iso', 'run'], [str(self.coAcolIso.currentText()),
                                                                             str(self.coAcolRun.currentText())]],
                                     'ORDER BY file', caller_name=__name__)
        r_col = TiTs.select_from_db(db, 'file', 'FitRes', [['iso', 'run'], [str(self.coColIso.currentText()),
                                                                            str(self.coColRun.currentText())]],
                                    'ORDER BY file', caller_name=__name__)

        if r_acol is not None:
            for each in r_acol:
                # self.addToAll(each[0] + " - Run: " + each[1])
                col = TiTs.select_from_db(db, 'colDirTrue', 'Files', [['file'], [each[0]]], caller_name=__name__)[0][0]
                if not col:
                    self.addToAcol(each[0])

        if r_col is not None:
            for each in r_col:
                # self.addToAll(each[0] + " - Run: " + each[1])
                col = TiTs.select_from_db(db, 'colDirTrue', 'Files', [['file'], [each[0]]], caller_name=__name__)[0][0]
                if col:
                    self.addToCol(each[0])

    def remove(self):
        for index in range(self.lCol.count()):
            i = self.lCol.takeItem(0)
            if i.checkState() == QtCore.Qt.Checked:
                self.lCol.removeItemWidget(i)
            else:
                self.lCol.addItem(i)

        for index in range(self.lAcol.count()):
            i = self.lAcol.takeItem(0)
            if i.checkState() == QtCore.Qt.Checked:
                self.lAcol.removeItemWidget(i)
            else:
                self.lAcol.addItem(i)

    # def shiftToCol(self):
    #     for index in range(self.lAll.count()):
    #         i = self.lAll.takeItem(0)
    #         if i.checkState() == QtCore.Qt.Checked:
    #             self.lCol.addItem(i)
    #         else:
    #             self.lAll.addItem(i)
    #
    # def shiftToAcol(self):
    #     for index in range(self.lAll.count()):
    #         i = self.lAll.takeItem(0)
    #         if i.checkState() == QtCore.Qt.Checked:
    #             self.lAcol.addItem(i)
    #         else:
    #             self.lAll.addItem(i)
    #
    # def shiftToAll(self):
    #     for index in range(self.lCol.count()):
    #         i = self.lCol.takeItem(0)
    #         if i.checkState() == QtCore.Qt.Checked:
    #             self.lAll.addItem(i)
    #         else:
    #             self.lCol.addItem(i)
    #
    #     for index in range(self.lAcol.count()):
    #         i = self.lAcol.takeItem(0)
    #         if i.checkState() == QtCore.Qt.Checked:
    #             self.lAll.addItem(i)
    #         else:
    #             self.lAcol.addItem(i)

    def setPath(self):
        projectPath, dbname = os.path.split(self.dbpath)
        self.iFile.setText(datetime.today().strftime('result_%Y-%m-%d_%H-%M-%S.txt'))
        self.iPath.setText(str(projectPath))

    def wToList(self):
        list = [] #[file1, run1, file2, run2]
        for index in range(self.lCol.count()):
            list.append([str(self.lCol.item(index).text()), str(self.coColRun.currentText()),
                         str(self.lAcol.item(index).text()), str(self.coAcolRun.currentText())])

        return list

    def startEval(self):
        if self.lCol.count() != self.lAcol.count():
            print('Equal amount of collinear and anti-collinear measurements needed.')
        elif self.lCol.count() == 0:
            print('No files selected.')
        else:
            p = os.path.join(os.path.normpath(self.iPath.text()), self.iFile.text())
            l = self.wToList()
            cTo.files_to_csv(self.dbpath, l, p)
            self.setPath()