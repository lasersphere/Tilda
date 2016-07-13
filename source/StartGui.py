'''
Created on 06.06.2014

@author: hammen
'''
import sys

from PyQt5 import QtWidgets

from Gui.MainUi import MainUi
from Gui.EmitText import EmitText

app = QtWidgets.QApplication([""])

ui = MainUi('V:/Projekte/COLLAPS/Sn/Measurement_and_Analysis_Christian/Sn.sqlit')

#emi = EmitText()
#emi.textSig.connect(ui.out)
#sys.stdout = emi

app.exec_()