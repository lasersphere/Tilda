'''
Created on 06.06.2014

@author: hammen
'''

import os
import sqlite3

from PyQt5 import QtWidgets

import Tools
from Gui.Ui_Crawler import Ui_Crawler


class CrawlerUi(QtWidgets.QWidget, Ui_Crawler):



    def __init__(self):
        super(CrawlerUi, self).__init__()
        self.setupUi(self)
        
        self.bcrawl.clicked.connect(self.crawl)
        self.pushButton_save_sql.clicked.connect(self.save_sql_cmd)
        self.pushButton_load_sql.clicked.connect(self.load_sql_cmd)

        
        self.dbpath = None
        
        self.show()
        
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)    
    
    def crawl(self):
        t = self.path.text()
        path = t if t is not '' else '.'
        Tools.crawl(self.dbpath, path, self.recursive.isChecked())
        if self.lineEdit_sql_cmd.text():
            con = sqlite3.connect(self.dbpath)
            cur = con.cursor()
            cur.execute(''' %s ''' % self.lineEdit_sql_cmd.text())
            con.commit()
            con.close()

    def dbChange(self, dbpath):
        self.dbpath = dbpath
        sql_file = os.path.join(os.path.split(self.dbpath)[0], 'sql_cmd.txt')
        if os.path.isfile(sql_file):
            self.load_sql_cmd(False, path=sql_file)

    def save_sql_cmd(self):
        start_path = os.path.join(os.path.split(self.dbpath)[0], 'sql_cmd.txt')
        path, ending = QtWidgets.QFileDialog.getSaveFileName(
            QtWidgets.QFileDialog(), 'save sql cmd as txt', start_path,
            '*.txt')
        if path:
            if os.path.isfile(path):
                full_path = path
            else:
                full_path = path + ending[1:]
            text_file = open(full_path, 'w')
            text_file.write(self.lineEdit_sql_cmd.text())
            text_file.close()
            print('saved sql cmd to: ', full_path)

    def load_sql_cmd(self, buttonres, path=''):
        if path == '':
            path, ending = QtWidgets.QFileDialog.getOpenFileName(
                QtWidgets.QFileDialog(), 'load sql cmd from txt', os.path.split(self.dbpath)[0],
                '*.txt')
        if path:
            text_file = open(path, 'r')
            self.lineEdit_sql_cmd.setText(text_file.readline())
            text_file.close()


