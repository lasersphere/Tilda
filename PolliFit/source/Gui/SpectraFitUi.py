"""
Created on 18.02.2022

@author: Patrick Mueller
"""


from PyQt5 import QtWidgets

from Gui.Ui_SpectraFit import Ui_SpectraFit
from SpectraFit import SpectraFit


class SpectraFitUi(QtWidgets.QWidget, Ui_SpectraFit):

    def __init__(self):
        super(SpectraFitUi, self).__init__()
        self.setupUi(self)
        self.main_tilda_gui = None
        self.dbpath = None

        self.spectra_fit = None

        self.show()

    def con_main_tilda_gui(self, main_tilda_gui):
        self.main_tilda_gui = main_tilda_gui

    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)

    def dbChange(self, dbpath):
        self.dbpath = dbpath

    def gen_fit_kwargs(self):
        return dict(guess_offset=self.check_guess_offset.isChecked(),
                    x_in_freq=self.check_x_as_freq.isChecked(),
                    save_ascii=self.check_save_ascii.isChecked(),
                    fmt=self.edit_fmt.text(),
                    font_size=self.s_fontsize.value())

    def load(self):
        files = [f.text for f in self.list_files.selectedItems()]
        self.spectra_fit = SpectraFit(files, self.dbpath, self.c_run.currentText(), **self.gen_fit_kwargs())


