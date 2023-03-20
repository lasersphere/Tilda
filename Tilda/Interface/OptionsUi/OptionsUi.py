"""

Created on '10.09.2019'

@author:'fsommer'

"""

import logging
import os
import platform
import subprocess

from PyQt5 import QtWidgets

import Tilda.Application.Config as Cfg
from Tilda.Interface.OptionsUi.Ui_Options import Ui_Dialog_Options


class OptionsUi(QtWidgets.QDialog, Ui_Dialog_Options):

    def __init__(self, main_gui):
        super(OptionsUi, self).__init__()

        self.main_gui = main_gui
        self.main = Cfg._main_instance
        self.setupUi(self)

        ''' Option functionality '''
        self.setup_all_options()

        self.show()

    ''' setup '''
    def setup_all_options(self):
        """
        Function to encapsulate the setup of connectivity (connect) and load the current values from the options.
        """
        # BUTTON BOX
        self.buttonBox_okCancel.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_default)
        # GENERAL TAB
        # PRE SCAN
        self.doubleSpinBox_preScanTimeout.setValue(self.get_setting_from_options('SCAN:pre_scan_timeout'))
        self.doubleSpinBox_preScanTimeout.valueChanged.connect(
            lambda: self.change_set_value(self.doubleSpinBox_preScanTimeout, 'SCAN:pre_scan_timeout'))
        # CONNECT
        self.link_openFpgaConfig.setText(self.create_folder_str(self.get_setting_from_options('FPGA:config_file')))
        self.link_openFpgaConfig.clicked.connect(lambda: self.open_file_or_folder(self.link_openFpgaConfig.text()))

        self.link_openTritonConfig.setText(self.create_folder_str(self.get_setting_from_options('TRITON:config_file')))
        self.link_openTritonConfig.clicked.connect(lambda: self.open_file_or_folder(self.link_openTritonConfig.text()))

        self.spinBox_tritonReadInterval.setValue(self.get_setting_from_options('TRITON:read_interval_ms'))
        self.spinBox_tritonReadInterval.valueChanged.connect(
            lambda: self.change_set_value(self.spinBox_tritonReadInterval, 'TRITON:read_interval_ms'))

        self.checkBox_disableTritonLink.setChecked(self.get_setting_from_options('TRITON:is_local'))
        self.checkBox_disableTritonLink.clicked.connect(
            lambda: self.toggle_option(self.checkBox_disableTritonLink, 'TRITON:is_local'))
        # SCAN FINISHED WIN
        self.groupBox_scanFinished.setChecked(self.get_setting_from_options('SCAN:show_scan_finished'))
        self.groupBox_scanFinished.clicked.connect(
            lambda: self.toggle_option(self.groupBox_scanFinished, 'SCAN:show_scan_finished'))

        self.checkBox_playSound.setChecked(self.get_setting_from_options('SOUND:is_on'))
        self.checkBox_playSound.clicked.connect(
            lambda: self.toggle_option(self.checkBox_playSound, 'SOUND:is_on'))

        self.link_openSoundsFolder.clicked.connect(lambda: self.open_file_or_folder(self.link_openSoundsFolder.text()))
        self.pushButton_chooseSoundsFolder.setText(
            self.create_folder_str(self.get_setting_from_options('SOUND:folder')))
        self.pushButton_chooseSoundsFolder.clicked.connect(
            lambda: self.choose_folder(self.pushButton_chooseSoundsFolder, 'SOUND:folder'))

    ''' general '''

    def accept(self):
        self.main.save_options()
        self.main.load_options()  # make sure that all new settings get transported to their corresponding variables
        self.close()

    def reject(self):
        self.main.load_options()
        self.close()

    def restore_default(self):
        """
        Will load from options_default.yaml and therefore reset all locally stored options to their default
        """
        dial = QtWidgets.QMessageBox(self)
        dial.setIcon(QtWidgets.QMessageBox.Warning)
        dial.setDefaultButton(QtWidgets.QMessageBox.No)
        dial.setText("Do you really want to reset all options to their default setting?")
        dial.setInformativeText("All current settings will be overwritten once you 'Save All'!")
        dial.setWindowTitle("WARNING")
        dial.setDetailedText("Will load all options from 'options_default.yaml' as if this were a fresh install.\n"
                             "If you are unsure, store a copy of your current options.yaml file.")
        dial.setStandardButtons(QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.Cancel)
        # Get the return value from the QMessagebBox:
        ret = dial.exec_()
        if ret == QtWidgets.QMessageBox.Yes:
            self.main.load_options(reset_to_default=True)
            self.setup_all_options()

    ''' options related '''
    def get_setting_from_options(self, setting_address):
        """
        Check the local options file for the current state of options and set them all in the GUI
        """
        return self.main.get_option(setting_address)

    def toggle_option(self, passed_checkbox, setting_address):
        """
        Change the option according to current checkbox status
        :param passed_checkbox: QCheckBox: Pointer to the specific checkbox
        :param setting_address: str: Address of the setting in the options. Format: CATEGORY:SUBCATEGORY:setting
        """
        is_checked = passed_checkbox.isChecked()
        self.main.set_option(setting_address, is_checked)

    def change_set_value(self, passed_spinbox, setting_address):
        """
        Change the setting according to the current spinBox value
        :param passed_spinbox: QSpinBox: Pointer to the specific Spin Box
        :param setting_address: str: Address of the setting in the options. Format: CATEGORY:SUBCATEGORY:setting
        """
        val = passed_spinbox.value()
        self.main.set_option(setting_address, val)

    def choose_folder(self, passed_object, setting_address):
        """ will open a modal file dialog and return the chosen folder """
        current = passed_object.text()
        start_path = self.get_folder_from_str(current)
        folder = QtWidgets.QFileDialog.getExistingDirectory(QtWidgets.QFileDialog(), 'choose a folder', start_path)
        return folder

    def create_folder_str(self, folder):
        """

        :param folder:
        :return:
        """
        source_path = os.path.abspath(os.curdir)
        if source_path in folder:
            # folder is within source, but full path is given
            folder = folder.replace(source_path, '...')
        return folder

    @staticmethod
    def get_folder_from_str(passed_str):
        """
        Convert the given string to a good folder path
        :param passed_str:
        :return:
        """
        if os.path.exists(passed_str):
            path = passed_str
        elif '...' in passed_str:
            path = passed_str.replace('...', Cfg.config_dir)
        else:
            path = Cfg.config_dir
        return path

    def open_file_or_folder(self, path_to):
        """
        Click to open the given directory or file in the OS filesystem.
        """
        path_to = self.get_folder_from_str(path_to)
        if os.path.exists(path_to):
            if platform.system() == "Windows":
                if os.path.isfile(path_to):
                    # open as file with notepad (instead of e.g. pycharm which is super slow)
                    subprocess.Popen(["notepad.exe", path_to])
                else:
                    os.startfile(path_to)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", path_to])
            else:
                subprocess.Popen(["xdg-open", path_to])
        else:
            logging.info('Path {} does not exist. Can not open'.format(path_to))

    ''' window related '''

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        if self.main_gui is not None:
            # tell main window that this window is closed.
            self.main_gui.close_options_win()
