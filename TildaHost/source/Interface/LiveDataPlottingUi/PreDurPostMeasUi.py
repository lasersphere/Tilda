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
from Interface.LiveDataPlottingUi.MakePrePostGrid import PrePostGridWidget
from Measurement.XMLImporter import XMLImporter


class PrePostTabWidget(QtWidgets.QTabWidget):
    def __init__(self, pre_post_meas_dict):
        QtWidgets.QTabWidget.__init__(self)

        self.data_dict = pre_post_meas_dict

        self.track_list = []
        for key in self.data_dict:
            # make sure we only create tabs for tracks. There is also 'isotopeData', 'pipeInternals' etc. in pipeData!
            if 'track' in key:
                self.track_list.append(key)

        for track_name in self.track_list:
            self.tab = self.create_new_tab(track_name)
            self.addTab(self.tab, track_name)

    def create_new_tab(self, track_name):
        new_tab = QtWidgets.QTabWidget()
        self.set_widget_layout(new_tab, track_name)
        return new_tab

    def set_widget_layout(self, parent_Widget, track_name):
        # Split the tab in two widgets, one for the table of data and one for plotting
        widget_layout = QtWidgets.QHBoxLayout(parent_Widget)
        # add the data window to the layout
        pre_post_data_widget = QtWidgets.QWidget()
        widget_layout.addWidget(pre_post_data_widget)
        pre_post_data_widget.setStyleSheet('background-color: yellow')# just for testing
        plot_pre_post_widget = QtWidgets.QWidget()
        # add the plot window to the layout
        widget_layout.addWidget(plot_pre_post_widget)
        plot_pre_post_widget.setStyleSheet('background-color: red')# just for testing
        plot_layout = QtWidgets.QHBoxLayout(plot_pre_post_widget)
        plot_layout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))

        # The pre/during/post data shall be presented in a Form Layout
        data_layout = QtWidgets.QFormLayout(pre_post_data_widget)

        self.index = 1
        self.act_track = track_name

        for pre_dur_post_str in ['preScan', 'duringScan', 'postScan']:
            self.add_pre_dur_post_form(pre_dur_post_str, data_layout)

    def add_pre_dur_post_form(self, pre_dur_post_str, parent_layout):
        self.index += 1 # add new line to the Form
        # add preScan to the Form Layout
        dev_label = QtWidgets.QLabel()
        dev_label.setText(pre_dur_post_str)
        parent_layout.setWidget(self.index, QtWidgets.QFormLayout.LabelRole, dev_label)
        # add dev_content to the preScan field
        dev_content = PrePostGridWidget(pre_dur_post_str, self.data_dict[self.act_track])
        parent_layout.setWidget(self.index, QtWidgets.QFormLayout.FieldRole, dev_content)
        # ---------- add divider line ---------
        self.index += 1 # add new line to the Form
        line_hor_divide = QtWidgets.QFrame()
        line_hor_divide.setFrameShape(QtWidgets.QFrame.HLine)
        line_hor_divide.setFrameShadow(QtWidgets.QFrame.Sunken)
        parent_layout.setWidget(self.index, QtWidgets.QFormLayout.FieldRole, line_hor_divide)

