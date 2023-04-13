"""
Created on 

@author: simkaufm

Module Description: Dialog for setting up an Isotope in the table Isotopes.
"""

import logging
from Tilda.Interface.SetupIsotopeUi.Ui_setupIsotopeDatabase import Ui_setupIsotopeDatabase

from PyQt5 import QtWidgets, QtCore
import Tilda.Application.Config as Cfg


class SetupIsotopeDatabase(QtWidgets.QDialog, Ui_setupIsotopeDatabase):
    def __init__(self, iso, parent):
        super(SetupIsotopeDatabase, self).__init__(parent)
        self.setupUi(self)

        self.pars = ['iso', 'mass', 'mass_d', 'I', 'center', 'Al', 'Bl', 'Au', 'Bu',
                     'fixedArat', 'fixedBrat', 'intScale', 'fixedInt', 'relInt', 'm', 'midTof']
        self.vals = [iso, 0, 0, 0, 0, 0, 0, 0, 0,
                     '0', '0', 5000, '0', '', '', 10]
        if Cfg._main_instance is not None:
            iso_set = Cfg._main_instance.get_isotope_settings_from_db(iso)
            if iso_set is not None:
                self.pars, self.vals = iso_set

        self.tableWidget_iso_pars.setColumnCount(2)
        self.tableWidget_iso_pars.setHorizontalHeaderLabels(['par', 'val'])
        self.add_items(self.pars, self.vals)
        self.setWindowTitle('isotope setup')

        self.buttonBox.accepted.connect(self.ok_pressed)

        self.show()

    def add_items(self, pars, vals):
        self.tableWidget_iso_pars.setRowCount(len(pars))
        for i, par in enumerate(pars):
            w = QtWidgets.QTableWidgetItem(par)
            w.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.tableWidget_iso_pars.setItem(i, 0, w)

            w = QtWidgets.QTableWidgetItem(str(vals[i]))
            if par == 'iso':
                w.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
            self.tableWidget_iso_pars.setItem(i, 1, w)

    def ok_pressed(self):
        vals = [self.tableWidget_iso_pars.item(i, 1).text() for i in range(self.tableWidget_iso_pars.rowCount())]
        iso = vals[0]
        logging.info('pressed ok -> adding isotope %s to db now' % iso)
        vals = vals[1:]
        if Cfg._main_instance is not None:
            Cfg._main_instance.update_iso_in_db(iso, vals)


if __name__ == '__main__':
    import sys
    import os
    app = QtWidgets.QApplication(sys.argv)
    workdir = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'
    db = os.path.join(workdir, 'Ni_workspace.sqlite')
    gui = SetupIsotopeDatabase(db)
    # print(len(gui.pars), len(gui.vals))
    app.exec_()

