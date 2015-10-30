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
    def __init__(self, main):
        super(SetupIsotopeUi, self).__init__()
        self.setupUi(self)
        self.iso = None
        self.sequencer = None
        self.main = main
        self.db = main.global_scanpars.get('db_loc')

        """Buttons"""
        self.pushButton_add_new_to_db.clicked.connect(self.add_new_iso_to_db)
        self.pushButton_init_sequencer.clicked.connect(self.init_seq)
        self.pushButton_ok.clicked.connect(self.ok)
        self.pushButton_cancel.clicked.connect(self.cancel)

        """ComboBoxes"""
        self.comboBox_isotope.currentTextChanged.connect(self.iso_select)
        self.comboBox_sequencer_select.currentTextChanged.connect(self.sequencer_select)

        self.comboBox_sequencer_select.addItems(Dft.sequencer_types_list)

        self.show()

    def load_existing_isotopes_from_db(self, database):
        pass

    def init_seq(self):
        logging.debug('initializing sequencer...')

    def add_new_iso_to_db(self):
        iso = self.lineEdit_new_isotope.text()
        already_exist = [self.comboBox_isotope.itemText(i)
                         for i in range(self.comboBox_isotope.count())]
        print(already_exist)
        if iso in already_exist and len(iso):
            logging.info('isotope ' + iso + ' already created, will not be added')
            return None
        scand = SdOp.init_empty_scan_dict()
        scand['isotopeData']['isotope'] = iso
        type = self.comboBox_sequencer_select.currentText()
        scand['isotopeData']['type'] = type
        scand['pipeInternals']['activeTrackNumber'] = 0
        DbOp.add_track_dict_to_db(self.db, scand)
        logging.debug('added ' + iso + ' ('  + type +  ') to database')
        self.update_isos()

    def iso_select(self, iso_str):
        logging.debug('selected isotope: ' + iso_str)

    def sequencer_select(self, seq_str):
        logging.debug('selected sequencer: ' + seq_str)
        self.update_isos()

    def update_isos(self):
        self.comboBox_isotope.clear()
        sequencer = self.comboBox_sequencer_select.currentText()
        isos = DbOp.check_for_existing_isos(self.db, sequencer)
        self.comboBox_isotope.addItems(isos)
        return isos

    def ok(self):
        # pass new set variables here
        self.destroy()

    def cancel(self):
        self.destroy()
