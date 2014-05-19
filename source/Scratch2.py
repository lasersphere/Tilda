'''
Created on 16.05.2014

@author: hammen
'''

from DBExperiment import DBExperiment
import datetime

path = "V:/Projekte/A2-MAINZ-EXP/TRIGA/Measurements and Analysis_Christian/Calcium Isotopieverschiebung/397nm_14_05_13/calciumD1.sqlite"
exp = DBExperiment(path)


print(datetime.datetime.now() + datetime.timedelta(days=1))
exp.con.commit()

print(exp.getAccVolt(datetime.datetime.now()))