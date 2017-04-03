"""
Created on 23.03.2017

@author: S. Kaufmann

Gui for converting specdata to an ascii file.

"""

import ast
import os

from PyQt5 import QtWidgets, Qt

import Tools
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

        self.lineEdit_scalers.editingFinished.connect(self.scalers_changed)
        self.lineEdit_softw_gates.editingFinished.connect(self.softw_gates_changed)
        self.checkBox_x_axis_in_freq.stateChanged.connect(self.check_box_changed)

        self.lineEdit_tracks.setText('-1')
        self.lineEdit_softw_gates.setText('[[[-10, 10, 0.5, 99], [-10, 10, 0.5, 99]]]')
        self.lineEdit_scalers.setText('[0, 1]')
        self.checkBox_x_axis_in_freq.setChecked(False)

    def sel_files(self):
        """ open file selection dialog and convert selected t ascii """
        filter = "XML (*.xml);;MCP (*.mcp)"
        file_name = QtWidgets.QFileDialog()
        file_name.setFileMode(Qt.QFileDialog.ExistingFiles)
        files = file_name.getOpenFileNames(self, "Open files", os.path.dirname(self.dbpath), filter)
        if len(files[0]):
            self.convert_files_to_ascii(files[0])

    def convert_files_to_ascii(self, files):
        """ convert a list of files to ascii using the function in Tools """
        print('converting files: ', files)
        sc = self.scalers
        tr = int(self.lineEdit_tracks.text())
        if self.output_dir is None:
            self.output_dir_change()
        x_in_freq = self.x_in_freq
        line_var = self.lineEdit_lineVar.text()
        for file in files:
            add_name = '_' + self.lineEdit_add_name.text() if self.lineEdit_add_name.text() else ''
            save_to = os.path.join(
                self.output_dir, os.path.split(file)[1].split('.')[0] + add_name + '.txt')
            # TODO fix when selecting frequency...
            Tools.extract_file_as_ascii(
                self.dbpath, file, sc, tr, x_in_freq=x_in_freq,
                line_var=line_var, save_to=save_to, softw_gates=self.softw_gates)

    def dbChange(self, dbpath):
        """ ... """
        self.dbpath = dbpath

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

    def scalers_changed(self):
        """ lineedit in scaler finished, check if it is a list. """
        text = self.lineEdit_scalers.text()
        try:
            text = ast.literal_eval(text)
        except Exception as e:
            return
        if isinstance(text, list):
            self.scalers = text

    def softw_gates_changed(self):
        """ when lineedit is finished, check if it is a list or it is empty,
         then set self.softw_gates either to this list or set it to None"""
        text = self.lineEdit_softw_gates.text()
        if text == '':  # deleting everything from line edit -> use gates from file.
            self.softw_gates = None
        else:
            try:
                text = ast.literal_eval(text)
            except Exception as e:
                return
            if isinstance(text, list):
                self.softw_gates = text

    def check_box_changed(self, state):
        """ checkbox for output of data in volt or MHz """
        text = 'x in frequency' if state else 'x in line volts'
        self.x_in_freq = 2 == state
        self.checkBox_x_axis_in_freq.setText(text)
