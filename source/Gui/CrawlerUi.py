'''
Created on 06.06.2014

@author: hammen
'''

from PyQt5 import QtWidgets
from Gui.Ui_Crawler import Ui_Crawler

import Tools

class CrawlerUi(QtWidgets.QWidget, Ui_Crawler):



    def __init__(self):
        super(CrawlerUi, self).__init__()
        self.setupUi(self)
        
        self.bcrawl.clicked.connect(self.crawl)
        
        self.dbpath = None
        
        self.show()
        
    
    def conSig(self, dbSig):
        dbSig.connect(self.dbChange)    
    
    def crawl(self):
        t = self.path.text()
        path = t if t is not '' else '.'
        Tools.crawl(self.dbpath, path, self.recursive.isChecked())
        
    def dbChange(self, dbpath):
        self.dbpath = dbpath
        
