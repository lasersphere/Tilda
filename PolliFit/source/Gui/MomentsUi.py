'''
Created on 05.10.2016

@author: chgorges
'''

import sqlite3

from PyQt5 import QtWidgets, QtCore

from Gui.Ui_Moments import Ui_Moments
from Moments import Moments

import numpy as np
import TildaTools as TiTs


class MomentsUi(QtWidgets.QWidget, Ui_Moments):


    def __init__(self):
        super(MomentsUi, self).__init__()
        self.setupUi(self)

        self.bCalcMagneticMoment.clicked.connect(self.calcMu)
        self.bCalcQ.clicked.connect(self.calcQ)
        self.saveMuRef.clicked.connect(self.saveMu)
        self.saveQRef.clicked.connect(self.saveQ)

        self.muRefVals = [0, 0]
        self.qRefVals = [0, 0]
        self.upperA = True
        self.upperB = True

        self.dbpath = None

    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)


    def calcMu(self):

        if self.radiobuttonMuRef.isChecked():
            (muVal, muErr) = TiTs.select_from_db(self.dbpath, 'val, systErr', 'Combined',
                                         [['parname'], ['muRef']])[0]
            (aVal, aErr) = TiTs.select_from_db(self.dbpath, 'val, systErr', 'Combined',
                                         [['parname'], ['ARef']])[0]
            iRef = TiTs.select_from_db(self.dbpath, 'val, systErr', 'Combined',
                                         [['parname'], ['IRef']])[0][0]

            val = float(muVal)/(float(aVal)*iRef)
            err = np.sqrt(np.square(float(muErr)/(float(aVal)*iRef))
                        + np.square(float(muVal)*float(aErr)/(np.square(float(aVal))*iRef)))
            self.ARef.setText(str(aVal))
            self.ARef_err.setText(str(aErr))
            self.muRef.setText(str(muVal))
            self.muRef_err.setText(str(muErr))
            self.spin.setValue(float(iRef))
            self.muRefVals = [val, err]
        else:
            val = float(self.muRef.text())/(float(self.ARef.text())*self.spin.value())
            err = np.sqrt(np.square(float(self.muRef_err.text())/(float(self.ARef.text())*self.spin.value()))
                        + np.square(float(self.muRef.text())*float(self.ARef_err.text())/
                                    (np.square(float(self.ARef.text()))*self.spin.value())))

            self.muRefVals = [val, err]

        self.upperA = self.UpperA.isChecked()

        self.mom.calcMu(self.muRefVals, self.upperA)

    def calcQ(self):
        if self.radiobuttonQRef.isChecked():
            [(val, err)] = TiTs.select_from_db(self.dbpath, 'val, systErr', 'Combined',
                                         [['parname'], ['eVzz']])
            self.qRefVals = [val, err]
            self.eVzz.setText(str(val))
            self.eVzz_err.setText(str(err))
        else:
            if self.BRef.text():
                val = float(float(self.BRef.text())/float(self.QRef.text()))
                err = np.sqrt(np.square(float(self.BRef_err.text())/float(self.QRef.text()))
            + np.square(float(self.BRef.text())*float(self.QRef_err.text())/np.square(float(self.QRef.text()))))
                self.eVzz.setText(str(val))
                self.eVzz_err.setText(str(err))
                self.qRefVals = [val, err]
            else:
                self.qRefVals = [float(self.eVzz.text()), float(self.eVzz_err.text())]
        self.upperB = self.UpperB.isChecked()
        self.mom.calcQ(self.qRefVals, self.upperB)

    def saveMu(self):
        con = sqlite3.connect(self.dbpath)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('all', 'muRef', '-1'))
        con.commit()
        cur.execute('''UPDATE Combined SET val = ?, systErr = ? WHERE iso = ? AND parname = ?''',
                    (self.muRef.text(), self.muRef_err.text(), 'all', 'muRef'))
        con.commit()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('all', 'ARef', '-1'))
        con.commit()
        cur.execute('''UPDATE Combined SET val = ?, systErr = ? WHERE iso = ? AND parname = ?''',
                    (self.ARef.text(), self.ARef_err.text(), 'all', 'ARef'))
        con.commit()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('all', 'IRef', '-1'))
        con.commit()
        cur.execute('''UPDATE Combined SET val = ? WHERE iso = ? AND parname = ?''',
                    (self.spin.value(), 'all', 'IRef'))
        con.commit()
        con.close()

    def saveQ(self):
        val = 0
        err = 0
        if self.BRef.text():
            val = float(float(self.BRef.text())/float(self.QRef.text()))
            err = np.sqrt(np.square(float(self.BRef_err.text())/float(self.QRef.text()))
                + np.square(float(self.BRef.text())*float(self.QRef_err.text())/np.square(float(self.QRef.text()))))
            self.eVzz.setText(str(val))
            self.eVzz_err.setText(str(err))
        else:
            val = float(self.eVzz.text())
            err = float(self.eVzz_err.text())
        con = sqlite3.connect(self.dbpath)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('all', 'eVzz', '-1'))
        con.commit()
        cur.execute('''UPDATE Combined SET val = ?, systErr = ? WHERE iso = ? AND parname = ?''',
                    (val, err, 'all', 'eVzz'))
        con.commit()
        con.close()


    def dbChange(self, dbpath):
        self.dbpath = dbpath
        self.mom = Moments(dbpath)
