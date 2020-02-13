'''
Created on 06.06.2014

@author: hammen
'''
import os.path
import sys
from datetime import datetime

from PyQt5 import QtWidgets, QtCore, QtGui

import MPLPlotter as plot
import Tools
from Gui.Ui_Main import Ui_MainWindow


class MainUi(QtWidgets.QMainWindow, Ui_MainWindow):

    dbSig = QtCore.pyqtSignal(str)

    def __init__(self, db_path, parent=None, overwrite_stdout=True):
        super(MainUi, self).__init__()

        self.setupUi(self)
        self.setWindowTitle('PolliFit')
        self.parent_win = parent
        self.crawler.conSig(self.dbSig)
        self.intfit.conSig(self.dbSig)
        self.intfit.con_main_tilda_gui(self.parent_win)
        self.averager.conSig(self.dbSig)
        self.batchfit.conSig(self.dbSig)
        self.isoshift.conSig(self.dbSig)
        self.kingfit.conSig(self.dbSig)
        self.moments.conSig(self.dbSig)
        self.accVolt_tab.conSig(self.dbSig)
        self.Alive_tab.conSig(self.dbSig)
        self.addFiles_tab.conSig(self.dbSig)
        self.asciiConv_tab.conSig(self.dbSig)
        self.ColAcol_tab.conSig(self.dbSig)
        self.Simulation_tab.conSig(self.dbSig)
        self.bOpenDb.clicked.connect(self.openDb)
        self.pushButton_refresh.clicked.connect(self.re_emit_db_path)
        if overwrite_stdout:
            sys.stdout = EmitStream(textWritten=self.out)
        else:  # do not display the output box if it is not used -> size = 0
            self.splitter.setSizes([self.size().height(), 0])

        self.openDb(db_path)
        self.show()

    def openDb(self, db_path=''):
        # print(db_path)
        if not os.path.isfile(db_path):
            file, end = QtWidgets.QFileDialog.getSaveFileName(
                parent=self, caption='Choose Database', directory='', filter='*.sqlite',
                options = QtWidgets.QFileDialog.DontConfirmOverwrite)
        else:
            file = db_path
        if file == '':
            return

        p = file
        
        print('New DB: ' + p)
        
        if not os.path.isfile(p):
            Tools.createDB(p)
        Tools.add_missing_columns(p)
        
        self.dbPath = p
        self.oDbPath.setText(self.dbPath)
        self.dbSig.emit(self.dbPath)
        self.save_db_path_to_text_file()

    def re_emit_db_path(self):
        self.dbSig.emit(self.dbPath)

    def out(self, text):
        if text and text != '\n' and text !=' ':
            date = datetime.today().strftime('%Y-%m-%d %H:%M:%S')
            text = '%s\t%s' % (date, text)
            self.oOut.appendPlainText(text)
            cursor = self.oOut.textCursor()
            cursor.movePosition(QtGui.QTextCursor.End)
            self.oOut.setTextCursor(cursor)
            self.oOut.ensureCursorVisible()

    def closeEvent(self, *args, **kwargs):
        sys.stdout = sys.__stdout__
        plot.close_all_figs()
        if self.parent_win is not None:
            self.parent_win.close_pollifit_win()

    def save_db_path_to_text_file(self):
        cur_dir = os.path.dirname(__file__)
        source_dir = os.path.join(cur_dir, os.pardir)
        txt_file = os.path.normpath(os.path.join(source_dir, 'current_db_loc.txt'))
        print('saving current db_loc to: %s' % txt_file)
        with open(txt_file, 'w') as f:
            f.write(self.dbPath + '\n')


class EmitStream(QtCore.QObject):

    textWritten = QtCore.pyqtSignal(str)

    def write(self, text):
        self.textWritten.emit(str(text))