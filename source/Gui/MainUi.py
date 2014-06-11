'''
Created on 06.06.2014

@author: hammen
'''
import os.path
import sqlite3

from PyQt5 import QtWidgets, QtCore

from Gui.Ui_Main import Ui_MainWindow
import Tools


class MainUi(QtWidgets.QMainWindow, Ui_MainWindow):

    dbSig = QtCore.pyqtSignal(str)

    def __init__(self):
        super(MainUi, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('PolliFit')
        
        self.crawler.conSig(self.dbSig)
        self.intfit.conSig(self.dbSig)
        self.averager.conSig(self.dbSig)
        self.bOpenDb.clicked.connect(self.openDb)
        
        self.show()
        
        
    def openDb(self):
        file = QtWidgets.QFileDialog.getSaveFileName(parent=self, caption='Choose Database', directory='', filter='*.sqlite', options = QtWidgets.QFileDialog.DontConfirmOverwrite)
        if file == '':
            return
        
        p = file[0]
        
        print('New DB: ' + p)
        
        if not os.path.isfile(p):
            Tools.createDB(p)
        
        self.dbPath = p
        self.oDbPath.setText(self.dbPath)
        self.dbSig.emit(self.dbPath)
        
    def out(self, text):
        self.oOut.appendPlainText(text)
        