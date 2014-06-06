'''
Created on 06.06.2014

@author: hammen
'''

from PyQt5 import QtWidgets
from Gui.Ui_Crawler import Ui_Crawler

class CrawlerUi(QtWidgets.QWidget, Ui_Crawler):



    def __init__(self):
        super(CrawlerUi, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('PolliFit')
        
        self.show()