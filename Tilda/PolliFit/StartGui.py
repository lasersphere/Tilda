"""
Created on 06.06.2014

@author: hammen
"""

import os
import argparse
from PyQt5 import QtWidgets

from Tilda.Application import Config as Cfg
from Tilda.Service.FileOperations.FolderAndFileHandling \
    import get_default_config_dir, make_config_dir, check_config_dir
from Gui.MainUi import MainUi

parser = argparse.ArgumentParser(description='Start PolliFit')
parser.add_argument('--config_dir', '-d', type=check_config_dir)
args = parser.parse_args()
Cfg.config_dir = args.config_dir if args.config_dir else get_default_config_dir()
make_config_dir(Cfg.config_dir)

app = QtWidgets.QApplication([""])

user_other_db = ''
db_path = user_other_db  # take db from user
assumed_db_path = ''


assumed_db_storage = os.path.normpath(os.path.join(Cfg.config_dir, 'PolliFit', 'current_db_loc.txt'))

if os.path.isfile(assumed_db_storage):
    with open(assumed_db_storage) as f:
        assumed_db_path = f.readline()[:-1]

if os.path.isfile(assumed_db_path):
    # overwrite db_path with the one from the user
    db_path = assumed_db_path


ui = MainUi(db_path, overwrite_stdout=False)

app.exec_()
