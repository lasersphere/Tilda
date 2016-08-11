'''
Created on 06.06.2014

@author: hammen
'''

from PyQt5 import QtWidgets

from Gui.MainUi import MainUi

app = QtWidgets.QApplication([""])

ui = MainUi('C:\COLLAPS\Online_Analysis_Al\Al\Analysis\Analysis.sqlite')

#emi = EmitText()
#emi.textSig.connect(ui.out)
#sys.stdout = emi

app.exec_()