'''
Created on 25.04.2014

@author: hammen
'''

import Physics

import sqlite3

class DBIsotope(object):
    '''
    A sqlite database driven version fo the isotope object
    '''


    def __init__(self, iso, line, file):
        '''Load relevant values of isotope name from database file'''
        print("Loading isotope", iso, line)
        #sqlite3.register_converter("BOOL", lambda v: bool(int(v)))
        
        con = sqlite3.connect(file)
        cur = con.cursor()
        
        cur.execute("SELECT * FROM Lines WHERE Line =?", (line,))
        data = cur.fetchall()[0]
        
        self.name = iso
        self.line = line
        self.ref = data[1]
        self.freq = data[2]
        self.Jl = data[3]
        self.Ju = data[4]
        self.shape = eval(data[5])
        self.fixShape = eval(data[6])
        elmass = data[7] * Physics.me_u
        
        cur.execute("SELECT * FROM Isotopes WHERE Isotope =?", (iso,))
        data = cur.fetchall()[0]
        
        self.mass = data[1] - elmass
        self.mass_d = data[2]
        self.I = data[3]
        self.center = data[4]
        self.Al = data[5]
        self.Bl = data [6]
        self.Au = data[7]
        self.Bu = data[8]
        self.fixArat = data[9]
        self.fixBrat = data[10]
        self.intScale = data[11]
        self.fixInt = data[12]
        if data[13] is not None:
            self.relInt = eval(data[13])
        
        if data[14] == None:
            self.m = None
        else:
            self.m = DBIsotope(data[14], line, file)

        cur.close()
        con.close()
        
def createDB():
    '''Create a sqlite database with the appropriate structure and two example entries'''
    con = sqlite3.connect("iso.sqlite")
    cur = con.cursor()
    cur.execute('CREATE TABLE "Lines" ("Line" TEXT PRIMARY KEY  NOT NULL , "Reference" TEXT, "Frequency" FLOAT, "Jl" FLOAT, "Ju" FLOAT, "shape" TEXT, "fixShape" TEXT)')
    cur.execute('CREATE TABLE "Isotopes" ("Isotope" TEXT PRIMARY KEY  NOT NULL ,"mass" FLOAT NOT NULL ,"mass_d" FLOAT NOT NULL ,"I" FLOAT,"center" FLOAT,"Al" FLOAT,"Bl" FLOAT DEFAULT (null) ,"Au" FLOAT,"Bu" FLOAT,"fixedArat" BOOL,"fixedBrat" BOOL,"intScale" DOUBLE,"fixedInt" BOOL, "relInt" TEXT, "m" TEXT)')
    
    #Test data
    cur.execute('INSERT INTO "Lines" VALUES ("Mi-D0","1_Mi","1000","0","1","{\'gau\': 50, \'lor\': 10}","{\'name\': \'Voigt\', \'gau\': True, \'lor\': True}");')
    cur.execute('INSERT INTO "Isotopes" VALUES ("1_Mi","1001","0.1","5","3","101","20","102","30",0,0,999,0,null,null);')
    
    con.commit()
    
    con.close()
    