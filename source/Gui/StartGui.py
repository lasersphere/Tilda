'''
Created on 06.06.2014

@author: hammen
'''

from PyQt5 import QtWidgets

from Gui.MainUi import MainUi

app = QtWidgets.QApplication([""])

ui = MainUi()


app.exec_()