"""
Created on 19.12.2017

@author: fsommer

Module Description:
"""
from copy import deepcopy

from PyQt5 import QtWidgets

from Interface.LiveDataPlottingUi.MakePrePostGrid import PrePostGridWidget


class PrePostTabWidget(QtWidgets.QTabWidget):
    def __init__(self, pre_post_meas_dict):
        """
        will cretae a tab for each track
        :param pre_post_meas_dict:
        """
        QtWidgets.QTabWidget.__init__(self)

        self.data_dict = pre_post_meas_dict
        self.pre_post_during_grid_widget = {}  # {'track0': [preScan, duringScan, postScan], ... } -
        # Grid widgets that will be created on first call. for each track!

        self.track_list = []
        self.track_tabs_dict = {}  # key is track name
        for key in sorted(self.data_dict):
            # make sure we only create tabs for tracks.
            # here is also 'isotopeData', 'pipeInternals' etc. in pipeData!
            if 'track' in key:
                self.track_list.append(key)

        for track_name in self.track_list:
            self.track_tabs_dict[track_name] = self.create_new_tab(track_name)
            self.addTab(self.track_tabs_dict[track_name], track_name)

    def create_new_tab(self, track_name):
        new_tab = QtWidgets.QTabWidget()
        self.set_widget_layout(new_tab, track_name)
        return new_tab

    def set_widget_layout(self, parent_Widget, track_name):
        """
        initial empty layout of the tab

        trackname
            pre scan      |
            ------------  |
            during scan   |   plot area
            ------------  |
            post scan     |

        :param parent_Widget: Qwidget, parent tab or so
        :param track_name: str, 'track0' , ....
        """
        # Split the tab in two widgets, one for the table of data and one for plotting
        main_layout = QtWidgets.QHBoxLayout(parent_Widget)
        splitter = QtWidgets.QSplitter()  # splitter for adjusting plot / data display size
        main_layout.addWidget(splitter)
        # add the data window to the layout
        pre_post_data_widget = QtWidgets.QWidget()
        splitter.addWidget(pre_post_data_widget)
        pre_post_data_widget.setStyleSheet('background-color: yellow')  # TODO just for testing
        plot_pre_post_widget = QtWidgets.QWidget()
        # add the plot window to the layout
        splitter.addWidget(plot_pre_post_widget)
        plot_pre_post_widget.setStyleSheet('background-color: red')  # TODO just for testing
        plot_layout = QtWidgets.QHBoxLayout(plot_pre_post_widget)
        plot_layout.addItem(QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum))

        # The pre/during/post data shall be presented in a Form Layout
        data_layout = QtWidgets.QFormLayout(pre_post_data_widget)

        ind = 1

        self.pre_post_during_grid_widget[track_name] = []
        for pre_dur_post_str in ['preScan', 'duringScan', 'postScan']:
            ind, pre_post_during_grid_widget = self.add_pre_dur_post_form(
                pre_dur_post_str, data_layout, ind, track_name)
            self.pre_post_during_grid_widget[track_name].append(pre_post_during_grid_widget)

    def add_pre_dur_post_form(self, pre_dur_post_str, parent_layout, index, tr_name):
        index += 1  # add new line to the Form
        # add preScan to the Form Layout
        pre_post_label = QtWidgets.QLabel()
        pre_post_label.setText(pre_dur_post_str)
        parent_layout.setWidget(index, QtWidgets.QFormLayout.LabelRole, pre_post_label)
        # add dev_content to the preScan field -> all exisitng devises of the preScan are in this widget.
        pre_post_during_grid_widget = PrePostGridWidget(pre_dur_post_str, self.data_dict[tr_name])
        parent_layout.setWidget(index, QtWidgets.QFormLayout.FieldRole, pre_post_during_grid_widget)
        # ---------- add divider line ---------
        index += 1  # add new line to the Form
        line_hor_divide = QtWidgets.QFrame()
        line_hor_divide.setFrameShape(QtWidgets.QFrame.HLine)
        line_hor_divide.setFrameShadow(QtWidgets.QFrame.Sunken)
        parent_layout.setWidget(index, QtWidgets.QFormLayout.FieldRole, line_hor_divide)
        index += 1
        return index, pre_post_during_grid_widget

    def update_data(self, pre_post_meas_dict):
        """
        if the tab was already created, update the existing data and add maybe missing devices.
        :param pre_post_meas_dict: dict, as self.data_dict
        :return:
        """
        self.data_dict = deepcopy(pre_post_meas_dict)  # contains all tracks
        for tr_name in self.track_list:
            for ind, pre_dur_post_str in enumerate(['preScan', 'duringScan', 'postScan']):
                self.pre_post_during_grid_widget[tr_name][ind].update_data(
                    pre_dur_post_str, self.data_dict[tr_name])

