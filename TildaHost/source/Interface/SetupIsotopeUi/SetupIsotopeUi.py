"""

Created on '29.10.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import logging

from Interface.SetupIsotopeUi.Ui_setupIsotope import Ui_SetupIsotope
import Service.DatabaseOperations.DatabaseOperations as DbOp
import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as Dft


class SetupIsotopeUi(QtWidgets.QDialog, Ui_SetupIsotope):
    def __init__(self, main, default_scan_dict):
        """
        Dialog which will set the
        :param main:
        :param scan_ctrl:
        :return:
        """
        super(SetupIsotopeUi, self).__init__()
        self.setupUi(self)
        self.iso = None
        self.sequencer = None
        self.main = main
        self.db = main.database
        self.default_scan_dict = default_scan_dict
        self.new_scan_dict = {}

        """Buttons"""
        self.pushButton_add_new_to_db.clicked.connect(self.add_new_iso_to_db)
        self.pushButton_init_sequencer.clicked.connect(self.init_seq)
        self.pushButton_ok.clicked.connect(self.ok)
        self.pushButton_cancel.clicked.connect(self.cancel)

        """ComboBoxes"""
        self.comboBox_isotope.currentTextChanged.connect(self.iso_select)
        self.comboBox_sequencer_select.currentTextChanged.connect(self.sequencer_select)

        self.comboBox_sequencer_select.addItems(Dft.sequencer_types_list)

        self.exec()

    def init_seq(self):
        logging.debug('initializing sequencer...')

    def add_new_iso_to_db(self):
        """ connect to the db and add a new isotope if this has not yet been added """
        iso = self.lineEdit_new_isotope.text()
        sctype = self.comboBox_sequencer_select.currentText()
        DbOp.add_new_iso(self.db, iso, sctype)
        logging.debug('added ' + iso + ' (' + sctype + ') to database')
        self.update_isos()

    def iso_select(self, iso_str):
        logging.debug('selected isotope: ' + iso_str)

    def sequencer_select(self, seq_str):
        logging.debug('selected sequencer: ' + seq_str)
        self.update_isos()

    def update_isos(self):
        """ update the items in the isotope combobox by connecting to the sqlite db
         and check for isotopes for this sequencer type """
        self.comboBox_isotope.clear()
        sequencer = self.comboBox_sequencer_select.currentText()
        isos = DbOp.check_for_existing_isos(self.db, sequencer)
        self.comboBox_isotope.addItems(isos)
        return isos

    def ok(self):
        self.new_scan_dict = SdOp.merge_dicts(
            self.default_scan_dict, DbOp.extract_all_tracks_from_db(
                self.db, self.comboBox_isotope.currentText(),
                self.comboBox_sequencer_select.currentText()))
        self.close()

    def cancel(self):
        self.new_scan_dict = self.default_scan_dict
        self.close()
