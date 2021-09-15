"""

Created on '10.09.2019'

@author:'fsommer'

"""


from PyQt5 import QtWidgets, QtCore, QtGui
import Service.FileOperations.FolderAndFileHandling as FileHandler

import Application.Config as Cfg

from Interface.OptionsUi.Ui_Options import Ui_Dialog_Options


class OptionsUi(QtWidgets.QWidget, Ui_Dialog_Options):

    def __init__(self, main_gui):
        super(OptionsUi, self).__init__()

        self.main_gui = main_gui
        self.main = Cfg._main_instance
        self.setupUi(self)

        ''' Option functionality '''
        self.load_settings_from_options()
        self.checkBox_playSound.clicked.connect(self.toggle_sound_onoff)
        self.buttonBox_okCancel.button(QtWidgets.QDialogButtonBox.RestoreDefaults).clicked.connect(self.restore_default)

        self.show()

    ''' general '''

    def accept(self):
        self.main.save_options()
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
        dial.setText("This is a message box")
        dial.setInformativeText("This is additional information")
        dial.setWindowTitle("MessageBox demo")
        dial.setDetailedText("The details are as follows:")
        dial.setStandardButtons(QtWidgets.QMessageBox.Ok | QtWidgets.QMessageBox.Cancel)

        ret = dial.exec_()
        # ret = QtWidgets.QMessageBox.question(dial, 'Reset Options',
        #                                     'Do you really want to reset all options to their default?\n'
        #                                     'Local changes will be overwritten once you click "Save all"!'
        #                                     )
        if ret==QtWidgets.QMessageBox.Yes:
            self.main.load_options(reset_to_default=True)

    ''' options related '''
    def load_settings_from_options(self):
        """
        Check the local options file for the current state of options and set them all in the GUI
        """
        pass  # TODO: put functionality. May be some work...

    def toggle_sound_onoff(self):
        """
        If checked, no sound will be played when a scan finished successfully
        """
        is_checked = self.checkBox_playSound.isChecked()
        self.main.set_option("SOUND:is_on", is_checked)

    ''' window related '''

    def closeEvent(self, *args, **kwargs):
        """ overwrite the close event """
        if self.main_gui is not None:
            # tell main window that this window is closed.
            self.main_gui.close_options_win()
