"""

Created on '29.10.2015'

@author:'simkaufm'

"""

from PyQt5 import QtWidgets
import logging

from Interface.SetupIsotopeUi.Ui_setupIsotope import Ui_SetupIsotope
import Service.Scan.ScanDictionaryOperations as SdOp
import Service.Scan.draftScanParameters as Dft
import Application.Config as Cfg


class SetupIsotopeUi(QtWidgets.QDialog, Ui_SetupIsotope):
    def __init__(self, scan_ctrl_win):
        """
        Modal Dialog which will connect to the database and read the stored parameters.
        by clicking ok, the data will be stored in self.new_scan_dict.
        """
        super(SetupIsotopeUi, self).__init__()
        self.setupUi(self)
        self.iso = None
        self.sequencer = None
        self.main = Cfg._main_instance
        self.scan_ctrl_win = scan_ctrl_win
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

        self.exec_()

    def init_seq(self):
        """ this starts the sequencer, if it has not been started yet """
        logging.debug('not implemented yet')

    def add_new_iso_to_db(self):
        """ connect to the db and add a new isotope if this has not yet been added """
        iso = self.lineEdit_new_isotope.text()
        seq_type = self.comboBox_sequencer_select.currentText()
        Cfg._main_instance.add_new_iso_to_db(iso, seq_type)
        self.update_isos()

    def iso_select(self, iso_str):
        """ is called when something changed in the comboBox for the isotope
        iso_str is always the string in the comboBox. """
        logging.debug('selected isotope: ' + iso_str)

    def sequencer_select(self, seq_str):
        """ is called when something changed in the comboBox for the sequencer
        seq_str is always the string in the comboBox. """
        logging.debug('selected sequencer: ' + seq_str)
        self.update_isos()

    def update_isos(self):
        """ update the items in the isotope combobox by connecting to the sqlite db
         and check for isotopes for this sequencer type """
        self.comboBox_isotope.clear()
        sequencer = self.comboBox_sequencer_select.currentText()
        isos = Cfg._main_instance.get_available_isos_from_db(sequencer)
        self.comboBox_isotope.addItems(isos)
        return isos

    def ok(self):
        """ by a given track in the database, this will read all scan values and
         then merge them with the default dictionary """
        iso_text = self.comboBox_isotope.currentText()
        if iso_text:
            self.scan_ctrl_win.active_iso = Cfg._main_instance.add_iso_to_scan_pars(iso_text,
                                                    self.comboBox_sequencer_select.currentText())
        self.close()

    def cancel(self):
        """ by clicking cancel, the new_scan_dict will be set to the default scan dict """
        self.close()
