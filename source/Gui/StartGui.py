'''
Created on 06.06.2014

@author: hammen
'''
import sys

from PyQt5 import QtWidgets

from Gui.MainUi import MainUi
from Gui.EmitText import EmitText

app = QtWidgets.QApplication([""])

ui = MainUi()

#emi = EmitText()
#emi.textSig.connect(ui.out)
#sys.stdout = emi

app.exec_()