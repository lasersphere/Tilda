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
        
        self.show()
        
    
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

        
    def loadIsos(self):
        self.isoSelect.clear()
        con = sqlite3.connect(self.dbpath)
        for i, e in enumerate(con.execute('''SELECT DISTINCT type FROM Files ORDER BY type''')):
            self.isoSelect.insertItem(i, e[0])
        con.close()
    
    
    def loadRuns(self):
        self.runSelect.clear()
        con = sqlite3.connect(self.dbpath)        
        for i, r in enumerate(con.execute('''SELECT run FROM Runs''')):
            self.runSelect.insertItem(i, r[0])
        con.close()
        
    def loadFiles(self):
        self.fileList.clear()
        try:
            self.iso = self.isoSelect.currentText()
            self.run = self.runSelect.currentText()
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()
            cur.execute('''SELECT file FROM Files WHERE type = ? ORDER BY date''', (self.iso,))
            self.files = [f[0] for f in cur.fetchall()]
            con.close()

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
        if self.chosenFiles != []:
            BatchFit.batchFit(self.chosenFiles, self.dbpath, run=self.run)
        else:
            print('nothing to fit!!!')

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()  # might still cause some problems
