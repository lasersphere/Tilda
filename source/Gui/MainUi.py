'''
Created on 06.06.2014

@author: hammen
'''

from PyQt5 import QtWidgets
from Gui.Ui_Main import Ui_MainWindow

class MainUi(QtWidgets.QMainWindow, Ui_MainWindow):



    def __init__(self):
        super(MainUi, self).__init__()
        self.setupUi(self)
        self.setWindowTitle('PolliFit')
        
        self.show()
        