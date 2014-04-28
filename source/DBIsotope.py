'''
Created on 25.04.2014

@author: hammen
'''

import sqlite3

class DBIsotope(object):
    '''
    classdocs
    '''


    def __init__(self, name, file):
        '''
        Constructor
        '''
        con = sqlite3.connect("iso.sqlite")
        cur = con.cursor()
        
        cur.close()
        
def createDB():
    con = sqlite3.connect("iso.sqlite")
    cur = con.cursor()
    cur.execute('CREATE TABLE "Lines" ("Line" TEXT PRIMARY KEY  NOT NULL , "Reference" TEXT, "Frequency" FLOAT, "Jl" FLOAT, "Ju" FLOAT, "shape" TEXT, "fixShape" TEXT)')
    cur.execute('CREATE TABLE "Isotopes" ("Isotope" TEXT PRIMARY KEY  NOT NULL ,"mass" FLOAT NOT NULL ,"mass_d" FLOAT NOT NULL ,"I" FLOAT,"center" FLOAT,"Al" FLOAT,"B1" FLOAT DEFAULT (null) ,"Ar" FLOAT,"Br" FLOAT,"fixedArat" BOOL,"fixedBrat" BOOL,"intScale" DOUBLE,"fixedInt" BOOL, "relInt" TEXT)')
    
    #Test data
    #cur.execute('INSERT INTO "Lines" VALUES ("Mi-D0","1_Mi","1000","0","1","{\'gau\': 50, \'lor\': 10}","{\'gau\': True, \'lor\': True}");')
    #cur.execute('INSERT INTO "Isotopes" VALUES ("1_Mi","1001","0.1","5","3","101","20","102","30","False","False","999","False",null);')
    
    con.commit()
    
    con.close()
    