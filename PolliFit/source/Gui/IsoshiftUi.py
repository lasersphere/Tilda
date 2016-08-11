'''
Created on 29.07.2016

@author: chgorges
'''

import sqlite3
import ast
import itertools
import numpy as np
import copy
import time

from PyQt5 import QtWidgets, QtCore

from Gui.Ui_Isoshift import Ui_Isoshift
import Analyzer
import MPLPlotter as plot

class IsoshiftUi(QtWidgets.QWidget, Ui_Isoshift):


    def __init__(self):
        super(IsoshiftUi, self).__init__()
        self.setupUi(self)

        self.runSelect.currentTextChanged.connect(self.loadIsos)
        self.isoSelect.currentIndexChanged.connect(self.loadFiles)
        self.fileList.itemChanged.connect(self.recalc)
        self.bsave.clicked.connect(self.saving)

        self.dbpath = None

        self.show()

        
    
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)
    
        
    def loadIsos(self, run):
        self.isoSelect.clear()
        con = sqlite3.connect(self.dbpath)
        for i, e in enumerate(con.execute('''SELECT DISTINCT iso FROM FitRes WHERE run = ? ORDER BY iso''', (run,))):
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
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()

            self.iso = self.isoSelect.currentText()
            self.run = self.runSelect.currentText()

            self.files = Analyzer.getFiles(self.iso, self.run, self.dbpath)

            cur.execute('''SELECT date FROM Files WHERE type = ?''', (self.iso,))
            r = cur.fetchall()
            self.dates = []
            for i in r:
                self.dates.append(i[0])

            cur.execute('''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ?''', (self.iso, 'shift', self.run))
            r = cur.fetchall()
            con.close()
            select = [False] * len(self.files)
            self.statErrForm = 0
            self.systErrForm = 0
            if len(r) > 0:
                self.statErrForm = r[0][1]
                self.systErrForm = r[0][2]
                cfg = ast.literal_eval(r[0][0])
                for i, f in enumerate(self.files):
                    if cfg == []:
                        select[i] = False
                    elif f not in cfg:
                        select[i] = False
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


    def recalc(self):
        select = []
        self.chosenFiles = []
        self.chosenDates = []
        self.val = 0
        self.err = 0
        self.redChi = 0
        self.systeErr = 0
        config = []
        for index in range(self.fileList.count()):
            if self.fileList.item(index).checkState() != QtCore.Qt.Checked:
                select.append(index)
        self.chosenDates = np.delete(copy.deepcopy(self.dates), select)
        self.chosenFiles = np.delete(copy.deepcopy(self.files), select)
        print(self.chosenFiles)
        if len(self.chosenFiles) > 0:
            for index, file in enumerate(self.chosenFiles):
                config.append(self.getConfig(file, index))
            print(config)
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()
            cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run, config) VALUES (?, ?, ?, ?)''', (self.iso, 'shift', self.run, str(config)))
            con.commit()
            cur.execute('''UPDATE Combined SET config = ? WHERE iso = ? AND parname = ? AND run = ?''', (str(config), self.iso, 'shift', self.run))
            con.commit()
            con.close()


            self.shifts, self.shiftErrors, self.val, self.err, self.systeErr, self.redChi = Analyzer.combineShift(self.iso, self.run, self.dbpath, show_plot=False)
            self.result.setText(str(self.val))
            self.rChi.setText(str(self.redChi))
            self.statErr.setText(str(self.err))
            self.systErr.setText(str(self.systeErr))

    def saving(self):
        if self.iso:
            plot.close_all_figs()
            Analyzer.combineShift(self.iso, self.run, self.dbpath, show_plot=True)
        else:
            print('nothing to save!!!')

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()  # might still cause some problems
        con = sqlite3.connect(self.dbpath)
        cur = con.cursor()
        cur.execute('''SELECT reference, refRun FROM Lines''')
        r = cur.fetchall()
        con.close()
        if r:
            self.reference = r[0][0]
            self.refRun = r[0][1]
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()
            cur.execute('''SELECT file, date FROM Files WHERE type = ?''', (self.reference,))
            r = cur.fetchall()
            con.close()
            self.referenceList = []
            self.referenceDates = []
            for i in r:
                self.referenceList.append(i[0])
                self.referenceDates.append(time.strptime(i[1], '%Y-%m-%d %H:%M:%S'))

    def getConfig(self, file, index):
        date = self.chosenDates[index]
        datesafter = {}
        datesbefore = {}
        date = time.strptime(date, '%Y-%m-%d %H:%M:%S')
        date = time.mktime(date)
        for i in self.referenceDates:
            secs = time.mktime(i)
            if date - secs < 0:
                datesafter[np.abs(date-secs)] = i
            else:
                datesbefore[date-secs] = i
        if datesafter:
            afterkey = sorted(datesafter.keys())[0]
            after = datesafter[afterkey]
        else:
            afterkey = -1
            after = None
        if datesbefore:
            beforekey = sorted(datesbefore.keys())[0]
            before = datesbefore[beforekey]
        else:
            beforekey = -1
            before = None
        for i, j in enumerate(self.referenceDates):
            if j == before:
                indexBefore = i
            elif j == after:
                indexAfter = i
        if beforekey == -1 or beforekey > 5*afterkey:
            fileBefore = []
            fileAfter = [self.referenceList[indexAfter]]
        elif afterkey == -1 or afterkey > 5*beforekey:
            fileBefore = [self.referenceList[indexBefore]]
            fileAfter = []
        else:
            fileBefore = [self.referenceList[indexBefore]]
            fileAfter = [self.referenceList[indexAfter]]

        file = [file]
        return (fileBefore,file, fileAfter)

