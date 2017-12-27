"""
Created on 19.12.2017

@author: fsommer

Module Description:
"""
from copy import deepcopy

from PyQt5 import QtWidgets, QtGui


class PrePostGridWidget(QtWidgets.QWidget):
    def __init__(self, pre_dur_post_str, pre_post_meas_dict_this_track):
        """
        displays all values for all devices for pre_du_post scan
        returns a widget which can be used to update those values.

        :param pre_dur_post_str:
        :param pre_post_meas_dict_this_track: dict
        """
        QtWidgets.QWidget.__init__(self)

        # copy the dict
        self.track_data_dict = deepcopy(pre_post_meas_dict_this_track)

        self.gridLayout_devices = QtWidgets.QGridLayout(self)
        self.gridLayout_devices.setContentsMargins(0, -1, -1, -1)

        self.dmm_widget_dicts = {}  # dict to store the created widgets

        self.dmm_dict = self.track_data_dict.get('measureVoltPars', {}).get(pre_dur_post_str, {}).get('dmms', {})

        self.triton_widget_dicts = {}  # dict to store the created widgets
        # each triton dev has channels therefore key is "dev_plus_channel_name" = dev + ':' + channel
        self.triton_dict = self.track_data_dict.get('triton', {}).get(pre_dur_post_str, {})

        self.index = 0

        self.update_data(pre_dur_post_str, pre_post_meas_dict_this_track)

    def setup_device_grid(self, index, device_name, device_dict):
        """
        create one line for one dev containing:
        qCheckBox (plot or not) | QLabel (name of dev) | QTextEdit
        :param index:
        :param device_name:
        :param device_dict:
        :return:
        """
        # insert check box
        cb_plot_dev = QtWidgets.QCheckBox(self)
        self.gridLayout_devices.addWidget(cb_plot_dev, index, 0, 1, 1)
        # set name of device
        label_name_dev = QtWidgets.QLabel(self)
        self.gridLayout_devices.addWidget(label_name_dev, index, 1, 1, 1)
        # data of device
        label_data_dev = QtWidgets.QTextEdit(self)  # for scrolling
        label_data_dev.setReadOnly(True)
        label_data_dev.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        label_data_dev.setFixedHeight(58)
        label_data_dev.setFrameShape(QtWidgets.QFrame.StyledPanel)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        label_data_dev.setSizePolicy(size_policy)
        self.gridLayout_devices.addWidget(label_data_dev, index, 2, 1, 1)

        # insert progress bar
        progressBar_dev = QtWidgets.QProgressBar(self)
        progressBar_dev.setMaximumSize(100, 40)
        self.gridLayout_devices.addWidget(progressBar_dev, index, 3, 1, 1)
        self.update_dev(cb_plot_dev, label_name_dev, label_data_dev, progressBar_dev, device_name, device_dict)
        return cb_plot_dev, label_name_dev, label_data_dev, progressBar_dev

    def update_dev(self, cb_plot_dev, label_name_dev, label_data_dev, progressBar_dev, device_name, device_dict):
        """
        update an existing device
        :param cb_plot_dev: qcheckbox, for plotting or not, currently not used
        :param label_name_dev: qlabel, name of dev
        :param label_data_dev: qTextEdit, for displaying the data
        :param progressBar_dev: qProgressBar, for displaying the progress
        :param device_name: str, name of device to be dipslayed
        :param device_dict: dict, contains the data/readings of a device
        :return:
        """
        label_name_dev.setText(device_name)
        try:
            device_data = device_dict['data']
        except:
            device_data = device_dict['readings']
        try:
            device_data_required = device_dict['required']
        except:
            device_data_required = device_dict['sampleCount']
        # if device_data.__len__() > 10:
        #     device_data = device_data[-2:]
        label_data_dev.setText(str(device_data))
        label_data_dev.moveCursor(QtGui.QTextCursor.End)

        if device_data_required is 0:
            progressBar_dev.setValue(100)
            progressBar_dev.setFormat('continuous')
        else:
            progressBar_dev.setProperty("value", 100 * len(device_data) / device_data_required)

    def update_dev_from_dict(self, dev_widget_dict, dev_name, new_device_dict):
        """ simple wrapper for update dev """
        self.update_dev(dev_widget_dict['plotCb'],
                        dev_widget_dict['nameLabel'],
                        dev_widget_dict['dataLabel'],
                        dev_widget_dict['progressBar'],
                        dev_name,
                        new_device_dict
                        )

    def update_data(self, pre_dur_post_str, pre_post_meas_dict_this_track):
        """
        this will update the existing grid widget for each device with the new data for this track.
        If device was not created yet it will be created here.
        :param pre_dur_post_str: str, 'preScan' / 'postScan' / 'duringScan' needed to get
        the  data from pre_post_meas_dict_this_track
        :param pre_post_meas_dict_this_track: dict, header dict of this scan containing all data
        :return:
        """
        # copy the dict
        self.track_data_dict = deepcopy(pre_post_meas_dict_this_track)

        self.dmm_dict = self.track_data_dict.get('measureVoltPars', {}).get(pre_dur_post_str, {}).get('dmms', {})
        # Todo delete unnecessary widgets here
        # -> devices that are in self.dmm_widget_dicts or
        # self.triton_widget_dicts but not in self.track_data_dict anymore.

        for dmm in sorted(self.dmm_dict.keys()):  # sorting is always preferable when displaying
            if dmm not in self.dmm_widget_dicts.keys():  # add dmm, was not existing yet
                check_box, name_label, data_label, progress_bar = self.setup_device_grid(
                    self.index, dmm, self.dmm_dict[dmm])
                self.dmm_widget_dicts[dmm] = {'plotCb': check_box,
                                              'nameLabel': name_label,
                                              'dataLabel': data_label,
                                              'progressBar': progress_bar,
                                              'index': deepcopy(self.index)
                                              }
                self.index += 1
            self.update_dev_from_dict(self.dmm_widget_dicts[dmm], dmm, self.dmm_dict[dmm])

        self.triton_dict = self.track_data_dict.get('triton', {}).get(pre_dur_post_str, {})
        for dev in sorted(self.triton_dict.keys()):
            for channel in self.triton_dict[dev]:
                dev_plus_channel_name = dev + ':' + channel
                if dev_plus_channel_name not in self.triton_widget_dicts:
                    # create if not existing.
                    check_box, name_label, data_label, progress_bar = self.setup_device_grid(
                        self.index, dev_plus_channel_name, self.triton_dict[dev][channel])
                    self.triton_widget_dicts[dev_plus_channel_name] = {'plotCb': check_box,
                                                                       'nameLabel': name_label,
                                                                       'dataLabel': data_label,
                                                                       'progressBar': progress_bar,
                                                                       'index': deepcopy(self.index)
                                                                       }
                    self.index += 1
                self.update_dev_from_dict(self.triton_widget_dicts[dev_plus_channel_name], dev_plus_channel_name,
                                          self.triton_dict[dev][channel])

    def check_unnecessary(self):
        """ compare devices self.track_data_dict with the widget dicts and delete unnessecary widgets """
        # TODO
        pass
