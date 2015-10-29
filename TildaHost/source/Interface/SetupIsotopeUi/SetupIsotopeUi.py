"""

Created on '29.10.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import logging

from Interface.SetupIsotopeUi.Ui_setupIsotope import Ui_SetupIsotope

class SetupIsotopeUi(QtWidgets.QDialog, Ui_SetupIsotope):
    def __init__(self):
        super(SetupIsotopeUi, self).__init__()
        self.setupUi(self)
        self.iso = None
        self.sequencer = None

        """Buttons"""
        self.pushButton_add_new_to_db.clicked.connect(self.add_new_to_db)
        self.pushButton_init_sequencer.clicked.connect(self.init_seq)
        self.pushButton_ok.clicked.connect(self.ok)
        self.pushButton_cancel.clicked.connect(self.cancel)

        """ComboBoxes"""
        self.comboBox_isotope.currentTextChanged.connect(self.iso_select)
        self.comboBox_sequencer_select.currentTextChanged.connect(self.sequencer_select)

        self.show()

    def load_existing_isotopes_from_db(self, database):
        pass

    def init_seq(self):
        logging.debug('initializing sequencer...')

    def add_new_to_db(self):
        logging.debug('adding new isotope to database')

    def iso_select(self, iso_str):
        logging.debug('selected isotope: ' + iso_str)

    def sequencer_select(self, seq_str):
        logging.debug('selected sequencer: ' + seq_str)

    def ok(self):
        pass

    def cancel(self):
        pass