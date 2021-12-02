"""
Created on 23.03.2017

@author: S. Kaufmann
edited by P. Mueller

Gui for converting specdata to an ascii file.

"""

import ast
import os
import platform
import subprocess

from PyQt5 import QtWidgets, Qt

import Tools
import TildaTools as TiTs
from Gui.Ui_AsciiConv import Ui_AsciiConv


class AsciiConvUi(QtWidgets.QWidget, Ui_AsciiConv):
    def __init__(self):
        super(AsciiConvUi, self).__init__()
        self.setupUi(self)
        self.show()
        self.dbpath = ''
        self.output_dir = None
        self.scalers = []
        self.softw_gates = []
        self.x_in_freq = False

        self.pushButton_sel_and_conv.clicked.connect(self.sel_files)
        self.pushButton_choose_output.clicked.connect(self.output_dir_change)
        self.pushButton_open_out_dir.clicked.connect(self.open_dir)

        self.lineEdit_scalers.editingFinished.connect(self.scalers_changed)
        self.sc_minus.clicked.connect(self.decrease_n_scaler)
        self.sc_plus.clicked.connect(self.increase_n_scaler)
        self.lineEdit_softw_gates.editingFinished.connect(self.softw_gates_changed)
        self.checkBox_x_axis_in_freq.stateChanged.connect(self.check_box_changed)

        self.scalers = [0, 1, 2, 3]
        self.lineEdit_scalers.setText(str(self.scalers))
        self.softw_gates = [[[-10, 10, 0, 99], [-10, 10, 0, 99], [-10, 10, 0, 99], [-10, 10, 0, 99]]]
        self.lineEdit_softw_gates.setText(str(self.softw_gates))

    def sel_files(self):
        """ open file selection dialog and convert selected t ascii """
        file_filter = "XML (*.xml);;MCP (*.mcp)"
        file_name = QtWidgets.QFileDialog()
        file_name.setFileMode(Qt.QFileDialog.ExistingFiles)
        files = file_name.getOpenFileNames(self, "Open files", os.path.dirname(self.dbpath), file_filter)
        if len(files[0]):
            self.convert_files_to_ascii(files[0])

    def convert_files_to_ascii(self, files):
        """ convert a list of files to ascii using the function in Tools """
        print('converting files: ', files)
        sc = self.scalers
        tr = self.lineEdit_tracks.value()
        if self.output_dir is None:
            self.output_dir_change()
        x_in_freq = self.x_in_freq
        line_var = self.lineEdit_lineVar.currentText()
        for file in files:
            add_name = '_' + self.lineEdit_add_name.text() if self.lineEdit_add_name.text() else ''
            save_to = os.path.join(
                self.output_dir, os.path.split(file)[1].split('.')[0] + add_name + '.txt')
            Tools.extract_file_as_ascii(
                self.dbpath, file, sc, tr, x_in_freq=x_in_freq,
                line_var=line_var, save_to=save_to, softw_gates=self.softw_gates)

    def dbChange(self, dbpath):
        """ ... """
        self.dbpath = dbpath
        self.load_lines()

    def conSig(self, dbSig):
        """ dbSig comes from the MainUi """
        dbSig.connect(self.dbChange)

    def output_dir_change(self):
        """ dialog for choosing an existing dir for output of the ascii files """
        parent_widg = QtWidgets.QFileDialog(self)
        start_dir = self.output_dir if self.output_dir is not None else os.path.dirname(self.dbpath)
        dirname = QtWidgets.QFileDialog.getExistingDirectory(
            parent_widg, 'choose output directory', start_dir)
        if os.path.isdir(dirname):
            self.output_dir = dirname
        if self.output_dir is not None:
            self.label_current_output.setText(self.output_dir)

    def open_dir(self):
        path = self.output_dir
        if self.output_dir is not None:
            if os.path.exists(path):
                if platform.system() == "Windows":
                    os.startfile(path)
                elif platform.system() == "Darwin":
                    subprocess.Popen(["open", path])
                else:
                    subprocess.Popen(["xdg-open", path])
            else:
                self.output_dir_change()
                self.open_dir()
        else:
            self.output_dir_change()
            self.open_dir()

    def load_lines(self):
        self.lineEdit_lineVar.blockSignals(True)
        self.lineEdit_lineVar.clear()
        r = TiTs.select_from_db(self.dbpath, 'DISTINCT lineVar', 'Lines', [], '', caller_name=__name__)
        if r is not None:
            for i, each in enumerate(r):
                self.lineEdit_lineVar.insertItem(i, each[0])
        self.lineEdit_lineVar.blockSignals(False)

    def scalers_changed(self):
        """ lineedit in scaler finished, check if it is a list. """
        text = self.lineEdit_scalers.text()
        try:
            text = ast.literal_eval(text)
        except Exception as e:
            return
        if isinstance(text, list):
            text = [abs(int(t)) for t in text]
            self.lineEdit_scalers.setText(str(text))
            self.scalers = text

    def decrease_n_scaler(self):
        if len(self.scalers) > 1:
            self.scalers = self.scalers[:-1]
            self.lineEdit_scalers.setText(str(self.scalers))
            # Do not decrease the list of software gates, so that the user does not delete their input.
            # self.softw_gates = [tr[:-1] for tr in self.softw_gates]
            # self.lineEdit_softw_gates.setText(str(self.softw_gates))

    def increase_n_scaler(self):
        self.scalers.append(self.scalers[-1] + 1)
        self.lineEdit_scalers.setText(str(self.scalers))
        self.softw_gates = [tr + [[gate for gate in tr[-1]], ] if len(self.scalers) > len(tr) else tr
                            for tr in self.softw_gates]
        self.lineEdit_softw_gates.setText(str(self.softw_gates))

    def softw_gates_changed(self):
        """
        When lineedit is finished, check if it is a list or it is empty,
        then set self.softw_gates either to this list or set it to None
        """
        text = self.lineEdit_softw_gates.text()
        if text == '':  # deleting everything from line edit -> use gates from file.
            self.softw_gates = None
        else:
            try:
                text = ast.literal_eval(text)
            except Exception as e:
                return
            if isinstance(text, list):
                text = [t for t in text]
                self.softw_gates = text

    def check_box_changed(self, state):
        """ checkbox for output of data in volt or MHz """
        text = 'x in frequency' if state else 'x in line volts'
        self.x_in_freq = 2 == state
        self.checkBox_x_axis_in_freq.setText(text)
