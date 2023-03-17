"""
Created on 09.11.2016

@author: kaufmann
"""

import functools
import os

import numpy as np
from PyQt5 import QtWidgets, QtGui

from Tilda.PolliFit import TildaTools as TiTs
import Tilda.PolliFit.Measurement.MeasLoad as Meas
from Tilda.PolliFit.Gui.Ui_AddFiles import Ui_AddFiles


class AddFilesUi(QtWidgets.QWidget, Ui_AddFiles):

    def __init__(self):
        super(AddFilesUi, self).__init__()
        self.setupUi(self)

        self.host_file = None  # str, filename
        self.host_file_meas = None  # Measurment object as in XMLImporter etc
        self.files_to_add = []  # list of tuples, [(+/-1, meas), ... ]

        self.buttons_active_host_file(False)
        self.check_if_saving_possible()

        ''' connect buttons '''
        self.isoFilter.currentIndexChanged.connect(self.loadFiles)
        self.pushButton_choose_host_file.clicked.connect(self.choose_host_file)
        self.pushButton_clear.clicked.connect(self.clear_host_file)
        self.pushButton_save.clicked.connect(self.save)
        self.pushButton_add_file.clicked.connect(functools.partial(self.add_substract_file, 1))
        self.pushButton_substract_file.clicked.connect(functools.partial(self.add_substract_file, -1))
        self.pushButton_remove_file.clicked.connect(self.remove_file_from_add_list)

        ''' add shortcuts '''
        QtWidgets.QShortcut(QtGui.QKeySequence("h"), self, self.choose_host_file)
        QtWidgets.QShortcut(QtGui.QKeySequence("ESC"), self, self.clear_host_file)
        QtWidgets.QShortcut(QtGui.QKeySequence("+"), self, functools.partial(self.add_substract_file, 1))
        QtWidgets.QShortcut(QtGui.QKeySequence("-"), self, functools.partial(self.add_substract_file, -1))

        self.dbpath = None
        
        self.show()
        
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def loadIsos(self):
        self.isoFilter.clear()
        it = TiTs.select_from_db(self.dbpath, 'DISTINCT type', 'Files', addCond='ORDER BY type', caller_name=__name__)
        if it:
            for i, e in enumerate(it):
                self.isoFilter.insertItem(i, e[0])
        self.isoFilter.insertItem(-1, 'all')

    def loadFiles(self):
        self.clear_host_file()
        self.fileList.clear()
        iso = self.isoFilter.currentText()
        if iso != 'all':
            it = TiTs.select_from_db(self.dbpath, 'file', 'Files', [['type'], [iso]], 'ORDER BY date',
                                     caller_name=__name__)
            if it:
                for r in it:
                    self.fileList.addItem(r[0])
        else:
            it = TiTs.select_from_db(self.dbpath, 'file', 'Files', addCond='ORDER BY date', caller_name=__name__)
            if it:
                for r in it:
                    self.fileList.addItem(r[0])

    def clear_host_file(self):
        self.label_host_file_set.setText('None')
        self.host_file = None
        self.host_file_meas = None
        self.files_to_add = []
        self.listWidget_added_files.clear()
        self.buttons_active_host_file(False)
        self.check_if_saving_possible()

    def choose_host_file(self):
        if self.fileList.currentItem() is not None:
            self.host_file = self.fileList.currentItem().text()
            full_path = self.get_full_file_path(self.host_file)
            if os.path.isfile(full_path):
                # do not call preproc! raw x axis will be compared!
                self.host_file_meas = Meas.load(full_path, self.dbpath, raw=True)
                self.buttons_active_host_file(True)
                self.label_host_file_set.setText(self.host_file)
            else:
                self.host_file = None

    def add_substract_file(self, mult_factor):
        curr_item = self.fileList.currentItem()
        if curr_item is not None:
            file = curr_item.text()
            full_path = self.get_full_file_path(file)
            if os.path.isfile(full_path):
                meas = Meas.load(full_path, self.dbpath, raw=True)
                if self.compare_x_axis_of_files(self.host_file_meas, meas):
                    self.files_to_add.append((mult_factor, meas))
                    self.listWidget_added_files.addItem(curr_item.text())
                    self.check_if_saving_possible()

    def remove_file_from_add_list(self):
        curr_item = self.listWidget_added_files.currentItem()
        if curr_item is not None:
            curr_ind = self.listWidget_added_files.currentRow()
            self.files_to_add.pop(curr_ind)
            self.listWidget_added_files.takeItem(self.listWidget_added_files.row(curr_item))
            self.check_if_saving_possible()

    def check_if_saving_possible(self):
        possible = True if len(self.files_to_add) else False
        self.pushButton_save.setEnabled(possible)

    def buttons_active_host_file(self, hostfile_available):
        """ this will de-/activate all buttons when no hostfile is available """
        self.pushButton_choose_host_file.setDisabled(hostfile_available)

        self.pushButton_add_file.setEnabled(hostfile_available)
        self.pushButton_substract_file.setEnabled(hostfile_available)
        self.pushButton_clear.setEnabled(hostfile_available)

    def save(self):
        start_path = os.path.join(os.path.dirname(self.dbpath), self.host_file[:-4] + '_sum')
        path, ending = QtWidgets.QFileDialog.getSaveFileName(
            QtWidgets.QFileDialog(), 'save files as .xml file to', start_path, '*.xml')
        save_dir, file = os.path.split(path)
        if file != '':
            spec, files, save_name = TiTs.add_specdata(self.host_file_meas, self.files_to_add, save_dir, file, self.dbpath)
            self.label_last_saved.setText(os.path.join(save_dir, save_name))

    def get_full_file_path(self, file):
        data = TiTs.select_from_db(self.dbpath, 'filePath', 'Files', [['file'], [file]], caller_name=__name__)

        if len(data):
            rel_path = data[0][0]
            db_dir = os.path.dirname(self.dbpath)
            full_path = os.path.normpath(os.path.join(db_dir, rel_path))
            return full_path
        else:
            return ''

    def compare_x_axis_of_files(self, parent_specdata, to_check_spec_data):
        if parent_specdata.nrTracks != to_check_spec_data.nrTracks:
            print('Files do not have the same number of tracks, cannot add')
            return False
        if len(parent_specdata.cts[0]) != len(to_check_spec_data.cts[0]):
            print('Files do not have the same number of scalers, cannot add')
            return False
        for tr_ind, tr in enumerate(parent_specdata.cts):
            if len(parent_specdata.x[tr_ind]) != len(to_check_spec_data.x[tr_ind]):
                print('Files do not have the same number of steps in track%s' % tr_ind)
                return False
        for tr_ind, tr in enumerate(parent_specdata.cts):
            # check if the x-axis of the two specdata are equal:
            if np.allclose(parent_specdata.x[tr_ind], to_check_spec_data.x[tr_ind], rtol=1 ** -5):
                print('x-axis are matching for track%s' % tr_ind)
                pass
            else:
                return False
        return True

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.loadIsos()
