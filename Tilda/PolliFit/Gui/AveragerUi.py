"""
Created on 06.06.2014

@author: hammen, chgorges
"""

import ast
import copy

import numpy as np
from PyQt5 import QtWidgets, QtCore

from Tilda.PolliFit.Gui.Ui_Averager import Ui_Averager
from Tilda.PolliFit import TildaTools as TiTs, Analyzer


class AveragerUi(QtWidgets.QWidget, Ui_Averager):

    def __init__(self):
        super(AveragerUi, self).__init__()
        self.setupUi(self)

        self.runSelect.currentTextChanged.connect(self.loadIsos)
        self.isoSelect.currentIndexChanged.connect(self.loadParams)
        self.parameter.currentIndexChanged.connect(self.loadFiles)
        self.fileList.itemChanged.connect(self.recalc)
        self.cWeighted.stateChanged.connect(self.recalc)
        self.cEstimated.stateChanged.connect(self.recalc)
        self.bsave.clicked.connect(self.saving)
        self.pushButton_select_all.clicked.connect(self.select_all)

        self.select_all_state = True
        self.chosenFiles = []  # list of chosen Files (will be defined in self.recalc() due to checkboxes in gui)

        self.dbpath = None
        
        self.show()
        
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)
    
    def loadIsos(self, run):
        self.isoSelect.clear()
        print('run is: ', run)
        it = TiTs.select_from_db(self.dbpath, 'DISTINCT iso', 'FitRes', [['run'], [run]], 'ORDER BY iso',
                                 caller_name=__name__)
        if it is not None:
            for i, e in enumerate(it):
                self.isoSelect.insertItem(i, e[0])
        self.isoSelect.insertItem(-1, 'all')

    def loadRuns(self):
        self.runSelect.clear()
        it = TiTs.select_from_db(self.dbpath, 'run', 'Runs', caller_name=__name__)
        if it is not None:
            for i, r in enumerate(it):
                self.runSelect.insertItem(i, r[0])
        
    def loadParams(self):
        self.parameter.clear()
        runselect, isoSelect = self.runSelect.currentText(), self.isoSelect.currentText()
        if isoSelect == 'all':
            isoSelect = self.isoSelect.itemText(0)  # just use the parameters from iso at first position
        r = None
        if runselect and isoSelect:
            if isoSelect == 'all':
                r = TiTs.select_from_db(self.dbpath, 'pars', 'FitRes', [['run'], [runselect]], caller_name=__name__)
            else:
                r = TiTs.select_from_db(self.dbpath, 'pars', 'FitRes', [['run', 'iso'],
                                        [runselect, isoSelect]], caller_name=__name__)

        if r is not None:
            r = r[0]
            try:
                for e in sorted(ast.literal_eval(r[0]).keys()):
                    self.parameter.addItem(e)
            except Exception as e:
                print(e)

    def loadFiles(self):
        self.fileList.clear()
        try:
            self.iso = self.isoSelect.currentText()
            self.run = self.runSelect.currentText()
            self.par = self.parameter.currentText()
            # self.files = Analyzer.getFiles(self.iso, self.run, self.dbpath)
            self.vals, self.errs, self.dates, self.files = Analyzer.extract(
                self.iso, self.par, self.run, self.dbpath, prin=False)
            # check if a config exists and if so check only the files within this config
            if self.iso == 'all':
                r = TiTs.select_from_db(self.dbpath, 'config, statErrForm, systErrForm', 'Combined',
                                        [['parname', 'run'], [self.par, self.run]],
                                        caller_name=__name__)
            else:
                r = TiTs.select_from_db(self.dbpath, 'config, statErrForm, systErrForm', 'Combined',
                                        [['iso', 'parname', 'run'], [self.iso, self.par, self.run]], caller_name=__name__)
            select = [True] * len(self.files)
            if r is not None:
                cfg = ast.literal_eval(r[0][0])
                for i, f in enumerate(self.files):
                    if not cfg:
                        select[i] = True
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
        except Exception as e:
            print('error while loading files: %s' % e)

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
        print('selected all!')
        self.fileList.blockSignals(False)
        self.recalc()

    def recalc(self):
        select = []
        self.chosenFiles = []
        self.val = 0
        self.err = 0
        self.redChi = 0
        self.systeErr = 0
        print('recalculating!')
        for index in range(self.fileList.count()):
            if self.fileList.item(index).checkState() != QtCore.Qt.Checked:
                select.append(index)
        if len(self.vals) > 0 and len(self.errs) > 0:
            self.chosenFiles = np.delete(copy.deepcopy(self.files), select)
            print('self.chosenFiles: ', self.chosenFiles)
            if len(self.chosenFiles) > 0:
                self.val, self.err, self.systeErr, self.redChi, plotdata, ax = Analyzer.combineRes(
                    self.iso, self.par, self.run, self.dbpath, weighted=bool(self.cWeighted.checkState()),
                    show_plot=False, only_this_files=self.chosenFiles, write_to_db=False,
                    estimate_err=bool(self.cEstimated.checkState()))
                print('values: ', self.val, self.err, self.systeErr, self.redChi, plotdata, ax)
        self.result.setText(str(self.val))
        self.rChi.setText(str(self.redChi))
        self.statErr.setText(str(self.err))
        self.systErr.setText(str(self.systeErr))

    def saving(self):
        if len(self.chosenFiles):
            Analyzer.combineRes(self.iso, self.par, self.run, self.dbpath, weighted=bool(self.cWeighted.checkState()),
                                show_plot=True, only_this_files=self.chosenFiles,
                                estimate_err=bool(self.cEstimated.checkState()))
        else:
            print('nothing to save!!!')

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()  # might still cause some problems
