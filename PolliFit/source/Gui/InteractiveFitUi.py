'''
Created on 06.06.2014

@author: hammen
'''

import ast
import os
from copy import deepcopy

from PyQt5 import QtWidgets, QtCore, QtGui

import TildaTools as TiTs
import Physics
from Gui.Ui_InteractiveFit import Ui_InteractiveFit
from InteractiveFit import InteractiveFit


class InteractiveFitUi(QtWidgets.QWidget, Ui_InteractiveFit):

    def __init__(self):
        super(InteractiveFitUi, self).__init__()
        self.setupUi(self)

        self.bLoad.clicked.connect(self.load)
        self.bFit.clicked.connect(self.fit)
        self.pushButton_check_softw_gates.clicked.connect(self.open_softw_gates)
        self.bReset.clicked.connect(self.reset)
        self.isoFilter.currentIndexChanged.connect(self.loadFiles)
        self.bFontSize.valueChanged.connect(self.fontSize)
        self.parTable.cellChanged.connect(self.setPar)
        self.cX_in_freq.stateChanged.connect(self.load)
        self.cSave_fit.stateChanged.connect(self.load)
        self.bParsToDB.clicked.connect(self.parsToDB)

        """ add shortcuts """
        QtWidgets.QShortcut(QtGui.QKeySequence("L"), self, self.load)
        QtWidgets.QShortcut(QtGui.QKeySequence("F"), self, self.fit)
        QtWidgets.QShortcut(QtGui.QKeySequence("R"), self, self.reset)

        self.dbpath = None
        self.main_tilda_gui = None
        
        self.show()

    def con_main_tilda_gui(self, main_tilda_gui):
        self.main_tilda_gui = main_tilda_gui

    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def load(self):
        if self.fileList.currentItem() is not None:
            iso = self.fileList.currentItem().text()
            if iso:
                self.intFit = InteractiveFit(iso, self.dbpath, self.runSelect.currentText(), plot_in_freq=self.cX_in_freq.isChecked(), save_plot=self.cSave_fit.isChecked())
                self.loadPars()
        
    def fit(self):
        try:
            self.intFit.printPars()
            self.intFit.fit()
            self.loadPars()
        except Exception as e:
            print('error while fitting: %s' % e)

    def reset(self):
        self.intFit.reset()
        self.loadPars()
        
    def loadPars(self):
        self.parTable.blockSignals(True)
        self.parTable.setRowCount(len(self.intFit.fitter.par))
        if self.cX_in_freq.isChecked():
            pars = self.intFit.getPars()
        else:
            pars = self.intFit.getParsE()

        for i, (n, v, f) in enumerate(pars):
            w = QtWidgets.QTableWidgetItem(n)
            w.setFlags(QtCore.Qt.ItemIsEnabled)
            self.parTable.setItem(i, 0, w)
            
            w = QtWidgets.QTableWidgetItem(str(v))
            self.parTable.setItem(i, 1, w)
            
            w = QtWidgets.QTableWidgetItem(str(f))
            self.parTable.setItem(i, 2, w)
        self.parTable.blockSignals(False)
                
    def loadIsos(self):
        self.isoFilter.clear()
        it = TiTs.select_from_db(self.dbpath, 'DISTINCT type', 'Files', addCond='ORDER BY type', caller_name=__name__)
        if it is not None:
            for i, e in enumerate(it):
                self.isoFilter.insertItem(i, e[0])
    
    def loadRuns(self):
        self.runSelect.clear()
        it = TiTs.select_from_db(self.dbpath, 'run', 'Runs', caller_name=__name__)
        if it is not None:
            for i, r in enumerate(it):
                self.runSelect.insertItem(i, r[0])
        
    def loadFiles(self):
        self.fileList.clear()
        it = TiTs.select_from_db(self.dbpath, 'file', 'Files',
                                     [['type'], [self.isoFilter.currentText()]], 'ORDER BY date', caller_name=__name__)
        if it is not None:
            for r in it:
                self.fileList.addItem(r[0])

    def setPar(self, i, j):
        val = ''
        try:
            val = ast.literal_eval(self.parTable.item(i, j).text())
        except Exception as e:
            print('error: %s while converting your typed value: %s using 0.0 / False instead' % (e, val))
            val = 0.0 if j == 1 else False
        if isinstance(val, float) or isinstance(val, int)\
                or isinstance(val, bool) or isinstance(val, list):
            val = val
        else:
            e = 'val is not a float / int / bool / list'
            print('error: %s while converting your typed value: %s of type: %s using 0.0 / False instead' % (
                e, type(val), val))
            val = 0.0 if j == 1 else False
        if j == 1:
            self.intFit.setPar(i, val)
            if self.intFit.fitter.npar[i] in ['softwGatesWidth', 'softwGatesDelayList', 'midTof']:
                if self.main_tilda_gui:
                    cur_itm = self.fileList.currentItem()
                    file_path = TiTs.select_from_db(self.dbpath, 'filePath', 'Files',
                                                    [['file'], [cur_itm.text()]], 'ORDER BY date',
                                                    caller_name=__name__)
                    if file_path is None:
                        return None
                    file_n = os.path.join(os.path.dirname(self.dbpath), file_path[0][0])
                    f_win = self.main_tilda_gui.file_plot_wins.get(file_n, False)
                    if f_win:
                        f_win.gate_data(self.intFit.fitter.meas)
        else:
            self.intFit.setFix(i, val)
        self.parTable.blockSignals(True)
        self.parTable.item(i, j).setText(str(val))
        self.parTable.blockSignals(False)



    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadRuns()
        self.loadIsos()

    def fontSize(self):
        if self.fileList.currentItem() is not None:
            iso = self.fileList.currentItem().text()
            if iso:
                self.intFit = InteractiveFit(iso, self.dbpath, self.runSelect.currentText(), fontSize=self.bFontSize.value(), plot_in_freq=self.cX_in_freq.isChecked(), save_plot=self.cSave_fit.isChecked())
                self.loadPars()

    def softw_gates_from_time_res_plot(self, softw_g, tr_ind, soft_b_width, plot_bool):
        """
        this should be connected to the pyqtsignal in the time resolved plotting window.
        ->  This will emit new software gates once they are changed there.
        -> update teh parameter table accordingly.
        :param softw_g: list: software gates [[[tr0_sc0_vMin, tr0_sc0_vMax, tr0_sc0_tMin, tr0_sc0_tMax], [tr0_sc1_...
        :param tr_ind: int: track_index to rebin -1 for all
        :param soft_b_width: list: software bin width in ns for each track
        :param plot_bool: bool: plot bool to force a plotting even if nothing has changed.
        """
        run_gates_width, del_list, iso_mid_tof = TiTs.calc_db_pars_from_software_gate(softw_g[tr_ind])
        self.parTable.blockSignals(True)
        parns = ['softwGatesWidth', 'softwGatesDelayList', 'midTof']
        vals = [run_gates_width, del_list, iso_mid_tof]
        for par, val in zip(parns, vals):
            item_found = self.parTable.findItems(par, QtCore.Qt.MatchExactly)
            if len(item_found):
                item_row = self.parTable.row(item_found[0])
                print('item row:', item_found, item_row)
                self.parTable.item(item_row, 1).setText(str(val))
                # still need to set it in the fitter:
                # cannot call self.setPar(...) since otherwise this would cause a loop,
                # because the tres gui would gate again...
                if par in parns[:-1]:
                    # pass on par to the fitter
                    # do not gate the data since not all pars have been passed on to the fitter yet
                    self.intFit.fitter.setPar(item_row, val)
                elif par == parns[-1]:
                    # now all pars have been passed on to the fitter -> gate the data
                    self.intFit.setPar(item_row, val)
        self.parTable.blockSignals(False)

    def open_softw_gates(self):
        cur_itm = self.fileList.currentItem()
        if cur_itm is not None:
            if '.xml' in cur_itm.text():
                if self.main_tilda_gui:
                    file_path = TiTs.select_from_db(self.dbpath, 'filePath', 'Files',
                                                    [['file'], [cur_itm.text()]], 'ORDER BY date',
                                                    caller_name=__name__)
                    if file_path is None:
                        return None
                    file_n = os.path.join(os.path.dirname(self.dbpath), file_path[0][0])
                    spec_to_plot = deepcopy(self.intFit.fitter.meas)
                    x_in_freq = TiTs.convert_fit_volt_axis_to_freq(self.intFit.fitter)
                    spec_to_plot.x = x_in_freq
                    spec_to_plot.x_units = spec_to_plot.x_units_enums.frequency_mhz
                    self.main_tilda_gui.load_spectra(file_n, spec_to_plot, sum_sc_tr=self.intFit.fitter.st)  # -> will be loaded to the main of Tilda
                    f_win = self.main_tilda_gui.file_plot_wins.get(file_n, False)
                    if f_win:
                        # connect to the pyqtsignal which is emitted when the liveplot window receives new softw_gates
                        # -> use this to update the pars table in gui here.
                        # connect to the callback signal when new gates are set in the live plot.
                        # list: software gates [[[tr0_sc0_vMin, tr0_sc0_vMax, tr0_sc0_tMin, tr0_sc0_tMax], [tr0_sc1_...
                        # int: track_index to rebin -1 for all
                        # list: software bin width in ns for each track
                        # bool: plot bool to force a plotting even if nothing has changed.
                        # new_gate_or_soft_bin_width = QtCore.pyqtSignal(list, int, list, bool)
                        f_win.new_gate_or_soft_bin_width.connect(self.softw_gates_from_time_res_plot)
                else:
                    print('get TILDA for full support')
                print('jup this is an xml file')

    def parsToDB(self):
        try:
            self.intFit.parsToDB(self.dbpath)
        except Exception as e:
            print("error: no file loaded!")


