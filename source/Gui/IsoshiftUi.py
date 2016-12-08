'''
Created on 29.07.2016

@author: chgorges
'''

import ast
import copy
import sqlite3
import time

import numpy as np
from PyQt5 import QtWidgets, QtCore

import Analyzer
import MPLPlotter as plot
from Gui.Ui_Isoshift import Ui_Isoshift


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

            self.dates = []
            for file in self.files:
                cur.execute('''SELECT date FROM Files WHERE file = ?''', (file,))
                r = cur.fetchall()
                self.dates.append(r[0])

            cur.execute('''SELECT config, statErrForm, systErrForm FROM Combined WHERE iso = ? AND parname = ? AND run = ?''', (self.iso, 'shift', self.run))
            r = cur.fetchall()
            con.close()
            select = [True] * len(self.files)
            self.statErrForm = 0
            self.systErrForm = 0
            if len(r) > 0:
                self.statErrForm = r[0][1]
                self.systErrForm = r[0][2]
                cfg = ast.literal_eval(r[0][0])
                cfg_files = [each[1][0] for each in cfg]
                for i, f in enumerate(self.files):
                    if cfg == []:
                        select[i] = False
                    elif f not in cfg_files:
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
        print('chosen Files for Shift are: %s \n and dates are: %s' % (self.chosenFiles, self.chosenDates))
        if len(self.chosenFiles) > 0:
            for index, file in enumerate(self.chosenFiles):
                config.append(self.getConfig(index))
            print('config for isotope shift is: %s' % config)
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()
            cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run, config) VALUES (?, ?, ?, ?)''',
                        (self.iso, 'shift', self.run, str(config)))
            con.commit()
            cur.execute('''UPDATE Combined SET config = ? WHERE iso = ? AND parname = ? AND run = ?''',
                        (str(config), self.iso, 'shift', self.run))
            con.commit()
            con.close()
            self.shifts, self.shiftErrors, self.val, self.err, self.systeErr, self.redChi = Analyzer.combineShift(
                self.iso, self.run, self.dbpath, show_plot=False)
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
            cur.execute('''SELECT file FROM FitRes WHERE iso = ?''', (self.reference,))
            fitres = cur.fetchall()
            con.close()
            self.referenceList = []
            self.referenceDates = []
            fitres = [item for sublist in fitres for item in sublist]
            # print('reference files are: %s' % fitres)
            for i in r:
                if i[0] in fitres:
                    self.referenceList.append(i[0])
                    self.referenceDates.append(time.strptime(i[1], '%Y-%m-%d %H:%M:%S'))
                else:
                    print('Warning! While creating list of reference files,'
                          ' the reference File: %s could not be found in the fit results!' % i[0])

    def getConfig(self, index):
        if not len(self.referenceList):
            return [''], [''], ['']
        # factor that determines if a reference file before or after is ignored
        # because it is to old compared to the file before or after:
        newer_ref_factor = 20
        indexAfter = 0
        indexBefore = 0
        date = self.chosenDates[index]
        file = self.chosenFiles[index]
        date = time.strptime(date, '%Y-%m-%d %H:%M:%S')
        date = time.mktime(date)
        datesafter = {}
        datesbefore = {}
        for ref_date in self.referenceDates:
            secs = time.mktime(ref_date)
            if date - secs < 0:
                datesafter[np.abs(date - secs)] = ref_date
            else:
                datesbefore[date - secs] = ref_date
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
        for ref_date, j in enumerate(self.referenceDates):
            if j == before:
                indexBefore = ref_date
            elif j == after:
                indexAfter = ref_date
        if beforekey == -1 or beforekey > newer_ref_factor * afterkey:
            # if no reference found before or reference is more than 5 times older than refence after,
            # do not take a file before and vice versa
            fileBefore = []
            fileAfter = [self.referenceList[indexAfter]]
        elif afterkey == -1 or afterkey > newer_ref_factor * beforekey:
            fileBefore = [self.referenceList[indexBefore]]
            fileAfter = []
        else:
            fileBefore = [self.referenceList[indexBefore]]
            fileAfter = [self.referenceList[indexAfter]]

        file = [file]
        return fileBefore, file, fileAfter
