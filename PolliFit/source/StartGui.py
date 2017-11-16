'''
Created on 06.06.2014

@author: hammen
'''

import os
from PyQt5 import QtWidgets

from Gui.MainUi import MainUi

app = QtWidgets.QApplication([""])

user_other_db = ''
db_path = user_other_db  # take db from user
assumed_db_path = ''

assumed_db_storage = os.path.normpath(os.path.join(os.path.dirname(__file__), 'current_db_loc.txt'))

print(assumed_db_storage)

if os.path.isfile(assumed_db_storage):
    with open(assumed_db_storage) as f:
        assumed_db_path = f.readline()[:-1]

if os.path.isfile(assumed_db_path):
    # overwrite db_path with teh one from the user
    db_path = assumed_db_path


ui = MainUi(db_path, overwrite_stdout=False)

app.exec_()
