'''
Created on 06.06.2014

@author: hammen
'''

from PyQt5 import QtWidgets

from Gui.MainUi import MainUi

app = QtWidgets.QApplication([""])

ui = MainUi('C:/Workspace/PolliFit/Ca.sqlite', overwrite_stdout=False)
#ui = MainUi('V:\Projekte\COLLAPS\ROC\ROC_October/CaD2_new.sqlite', overwrite_stdout=False)
#emi = EmitText()
#emi.textSig.connect(ui.out)
#sys.stdout = emi

app.exec_()