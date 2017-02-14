'''
Created on 06.06.2014

@author: chgorges
'''

import copy
import sqlite3

import numpy as np
from PyQt5 import QtWidgets, QtCore

import BatchFit
from Gui.Ui_Batchfitter import Ui_Batchfitter
import TildaTools as TiTs


class BatchfitterUi(QtWidgets.QWidget, Ui_Batchfitter):


    def __init__(self):
        super(BatchfitterUi, self).__init__()
        self.setupUi(self)

        self.runSelect.currentTextChanged.connect(self.loadIsos)
        self.isoSelect.currentIndexChanged.connect(self.loadFiles)
        self.fileList.itemChanged.connect(self.recalc)
        self.bfit.clicked.connect(self.fitting)

        self.pushButton_select_all.clicked.connect(self.select_all)

        self.select_all_state = True

        self.dbpath = None

        self.chosenFiles = []
        
        self.show()
        
    
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def loadIsos(self):
        self.isoSelect.clear()
        it = TiTs.select_from_db(self.dbpath, 'DISTINCT type', 'Files', addCond='ORDER BY type', caller_name=__name__)
        if it:
            for i, e in enumerate(it):
                self.isoSelect.insertItem(i, e[0])

    
    def loadRuns(self):
        self.runSelect.clear()
        it = TiTs.select_from_db(self.dbpath, 'run', 'Runs', caller_name=__name__)
        if it:
            for i, r in enumerate(it):
                self.runSelect.insertItem(i, r[0])

    def loadFiles(self):
        self.fileList.clear()
        try:
            self.iso = self.isoSelect.currentText()
            self.run = self.runSelect.currentText()
            self.files = [f[0] for f in TiTs.select_from_db(self.dbpath, 'file', 'Files',
                                        [['type'], [self.iso]], 'ORDER BY type', caller_name=__name__)]
            select = [False] * len(self.files)

            self.fileList.blockSignals(True)
            for f, s in zip(self.files, select):
                w = QtWidgets.QListWidgetItem(f)
                w.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
                if s:
                    w.setCheckState(QtCore.Qt.Checked)
                else:
                    w.setCheckState(QtCore.Qt.Unchecked)
                self.fileList.addItem(w)

            self.fileList.blockSignals(False)

            self.recalc()
        except Exception as e:
            print(str(e))

    def select_all(self):
        self.fileList.clear()
        self.select_all_state = not self.select_all_state
        select = [self.select_all_state] * len(self.files)

        self.fileList.blockSignals(True)
        for f, s in zip(self.files, select):
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
        select = []
        self.chosenFiles = []

        for index in range(self.fileList.count()):
            if self.fileList.item(index).checkState() != QtCore.Qt.Checked:
                select.append(index)
        if len(self.files) > 0:
            self.chosenFiles = np.delete(copy.deepcopy(self.files), select)


    def fitting(self):
        print('chosen files: ', self.chosenFiles)
        if len(self.chosenFiles):
            BatchFit.batchFit(self.chosenFiles, self.dbpath, run=self.run)
        else:
            print('nothing to fit!!!')

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()  # might still cause some problems
