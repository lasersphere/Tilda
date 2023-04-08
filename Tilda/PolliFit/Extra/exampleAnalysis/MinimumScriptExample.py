"""
Created on 2021-03-15
@author: fsommer
Module Description:  Very minimal example on how to use PolliFit in analysis scripts
"""

import os  # necessary for dealing with folders and filepaths
import ast  # neccessary for converting strings from db to python stuff
import sqlite3  # access to the sqlite database
import numpy as np  # numpy is always good
import matplotlib.pyplot as plt  # for plotting
from Tilda.PolliFit import BatchFit

''' Define folders and database '''
user_home_folder = os.path.expanduser("~")  # OwnCloud folder is located in user home. May change depending on system.
ownCloud_path = 'ownCloud\\User\\Felix\\Measurements\\Nickel_online_Becola\\Analysis\\XML_Data'  # db and Sums folder
workdir = os.path.join(user_home_folder, ownCloud_path)  # Combine above. Here lies the database
results_dir = os.path.join(workdir, 'minimum_example\\')  # fits are stored here relative to database folder
db = os.path.join(workdir, 'Ni_Becola.sqlite')  # combine name of the database with folder

''' Which isotope do you want to fit with what line shape? '''
iso = '%56Ni%'  # Name of the isotope to fit. % is a placeholder (in case you got fancy names)
lineshape = 'Voigt'  # Name of the 'run' used to fit. The run defines the line shape.

''' Get all files of one isotope from database '''
con = sqlite3.connect(db)  # connect to the db
cur = con.cursor()  # create a cursor to navigate the db
cur.execute(  # collect files matching type: iso
    '''SELECT file FROM Files WHERE type LIKE ? ORDER BY date ''', (iso,))
file_tups = cur.fetchall()  # get collected: [(file, ), (file2,), ...] is list of tuples
con.close()  # close database
all_files_iso = np.array([f[0] for f in file_tups])  # create a list of files

''' Batchfit all measurements of this isotope '''
BatchFit.batchFit(all_files_iso, db, lineshape, guess_offset=True, save_to_folder=results_dir)

''' Retrieve results from database for each file '''
fit_centers = []
fit_centers_errors = []
for indx, files in enumerate(all_files_iso):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute(  # Query fit results for file and isotope combo
        '''SELECT pars FROM FitRes WHERE file = ? AND iso LIKE ? AND run = ?''', (files, iso, lineshape))
    pars = cur.fetchall()  # get result: [(dict, )] is list of tuples
    con.close()
    parsdict = ast.literal_eval(pars[0][0])  # stored as str in database. Convert to dictionary
    fit_centers.append(parsdict['center'][0])  # each parameter is of format 'parameter':(value, uncertainty, fixed)
    fit_centers_errors.append(parsdict['center'][1])

''' Plot all results '''
x_ax = np.arange(len(fit_centers))  # primitive x-axis
plt.errorbar(x=x_ax, y=fit_centers, yerr=fit_centers_errors)
plt.show()
