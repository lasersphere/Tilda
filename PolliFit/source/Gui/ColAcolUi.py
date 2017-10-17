'''
Created on 29.07.2016

@author: K. Koenig
'''

import ast
import copy
import sqlite3
from datetime import datetime
import os

import numpy as np
from PyQt5 import QtWidgets, QtCore

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

        self.bCol.clicked.connect(self.shiftToCol)
        self.bAcol.clicked.connect(self.shiftToAcol)
        self.bReset.clicked.connect(self.shiftToAll)
        self.bStart.clicked.connect(self.startEval)

    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)


    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadFits()
        self.setPath()


    def addToAll(self, label):
        w = QtWidgets.QListWidgetItem(label)
        w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        w.setCheckState(QtCore.Qt.Unchecked)
        self.lAll.addItem(w)

    def addToCol(self, label):
        w = QtWidgets.QListWidgetItem(label)
        w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        w.setCheckState(QtCore.Qt.Checked)
        self.lAll.addItem(w)

    def addToAcol(self, label):
        w = QtWidgets.QListWidgetItem(label)
        w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
        w.setCheckState(QtCore.Qt.Checked)
        self.lAll.addItem(w)

    def loadFits(self):
        self.lAll.clear()
        self.lAcol.clear()
        self.lCol.clear()
        db = self.dbpath
        r = TiTs.select_from_db(db, 'file', 'FitRes')

        for each in r:
            self.addToAll(each[0])

    def shiftToCol(self):
        for index in range(self.lAll.count()):
            i = self.lAll.takeItem(0)
            if i.checkState() == QtCore.Qt.Checked:
                self.lCol.addItem(i)
            else:
                self.lAll.addItem(i)

    def shiftToAcol(self):
        for index in range(self.lAll.count()):
            i = self.lAll.takeItem(0)
            if i.checkState() == QtCore.Qt.Checked:
                self.lAcol.addItem(i)
            else:
                self.lAll.addItem(i)

    def shiftToAll(self):
        for index in range(self.lCol.count()):
            i = self.lCol.takeItem(0)
            if i.checkState() == QtCore.Qt.Checked:
                self.lAll.addItem(i)
            else:
                self.lCol.addItem(i)

        for index in range(self.lAcol.count()):
            i = self.lAcol.takeItem(0)
            if i.checkState() == QtCore.Qt.Checked:
                self.lAll.addItem(i)
            else:
                self.lAcol.addItem(i)

    def setPath(self):
        projectPath, dbname = os.path.split(self.dbpath)
        self.iFile.setText(datetime.today().strftime('result_%Y-%m-%d_%H-%M-%S.txt'))
        self.iPath.setText(str(projectPath))

    def wToList(self):
        list = []
        for index in range(self.lCol.count()):
            list.append([str(self.lCol.item(index).text()), str(self.lAcol.item(index).text())])

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