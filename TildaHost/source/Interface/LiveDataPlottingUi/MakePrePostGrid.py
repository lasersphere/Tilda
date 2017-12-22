"""
Created on 19.12.2017

@author: fsommer

Module Description:
"""
import ast
import functools
import logging
from copy import deepcopy
from datetime import datetime

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

import Application.Config as Cfg
import PyQtGraphPlotter as Pg
from Interface.LiveDataPlottingUi.Ui_LiveDataPlotting import Ui_MainWindow_LiveDataPlotting
from Interface.ScanProgressUi.ScanProgressUi import ScanProgressUi
from Measurement.XMLImporter import XMLImporter



class PrePostGridWidget(QtWidgets.QWidget):
    def __init__(self, pre_dur_post_str, pre_post_meas_dict_this_track):
        QtWidgets.QWidget.__init__(self)

        # copy the dict
        self.track_data_dict = deepcopy(pre_post_meas_dict_this_track)

        self.gridLayout_devices = QtWidgets.QGridLayout(self)
        self.gridLayout_devices.setContentsMargins(0, -1, -1, -1)

        self.dmm_list = []
        self.dmm_dict = self.track_data_dict.get('measureVoltPars', {}).get(pre_dur_post_str, {}).get('dmms', {})

        for dmms in self.dmm_dict:
            self.dmm_list.append(dmms)
        self.triton_list = []
        self.triton_dict = self.track_data_dict.get('triton', {}).get(pre_dur_post_str, {})
        for devs in self.triton_dict:
            self.triton_list.append(devs)

        self.index = 0
        for dmm in self.dmm_list:
            self.setup_device_grid(self.index, dmm, self.dmm_dict[dmm])
            self.index += 1

        for dev in self.triton_list:
            # triton devices have data per channel
            for channel in self.triton_dict[dev]:
                dev_plus_channel_name = dev + ':' + channel
                self.setup_device_grid(self.index, dev_plus_channel_name, self.triton_dict[dev][channel])
                self.index += 1

        #if self.track_data_dict_selected == {}:
        if self.track_data_dict == {}:
            self.not_setup_label = QtWidgets.QLabel()
            self.not_setup_label.setText('No measurements available for ' + pre_dur_post_str + ' (yet)!')
            self.gridLayout_devices.addWidget(self.not_setup_label, 0, 0, 1, 1)


    def setup_device_grid(self, index, device_name, device_dict):
        # insert check box
        cb_plot_dev = QtWidgets.QCheckBox(self)
        self.gridLayout_devices.addWidget(cb_plot_dev, index, 0, 1, 1)
        # set name of device
        label_name_dev = QtWidgets.QLabel(self)
        label_name_dev.setText(device_name)
        self.gridLayout_devices.addWidget(label_name_dev, index, 1, 1, 1)
        # display data of device
        label_data_dev = QtWidgets.QLabel(self)
        label_data_dev.setFrameShape(QtWidgets.QFrame.StyledPanel)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        label_data_dev.setSizePolicy(size_policy)
        try:
            device_data = device_dict['data']
        except:
            device_data = device_dict['readings']
        try:
            device_data_required = device_dict['required']
        except:
            device_data_required = device_dict['sampleCount']
        if device_data.__len__() > 10:
            device_data = device_data[-2:]
        label_data_dev.setText(str(device_data))
        print('dev_data: ' + str(device_data))
        self.gridLayout_devices.addWidget(label_data_dev, index, 2, 1, 1)
        # insert progress bar
        if device_data_required is 0:
            label_cont_data = QtWidgets.QLabel()
            label_cont_data.setText('continuous')
            self.gridLayout_devices.addWidget(label_cont_data, index, 3, 1, 1)
        else:
            progressBar_dev = QtWidgets.QProgressBar(self)
            progressBar_dev.setProperty("value", 100*device_data.__len__()/device_data_required)
            progressBar_dev.setMaximumSize(100, 40)
            self.gridLayout_devices.addWidget(progressBar_dev, index, 3, 1, 1)
