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

    def __init__(self, iso, line, file, isovar = '', linevar = ''):
        '''Load relevant values of isotope name from database file'''
        print("Loading", line + linevar, "line of", iso + isovar)
        #sqlite3.register_converter("BOOL", lambda v: bool(int(v)))
        
        con = sqlite3.connect(file)
        cur = con.cursor()
        
        cur.execute("SELECT * FROM Lines WHERE line =?", (line,))
        try:
            data = cur.fetchall()[0]
        except:
            raise Exception("No such line: " + line)
        
        self.name = iso
        self.isovar = isovar
        self.line = line
        self.linevar = linevar
        self.ref = data[1]
        self.freq = data[2]
        self.Jl = data[3]
        self.Ju = data[4]
        self.shape = eval(data[5])
        self.fixShape = eval(data[6])
        elmass = data[7] * Physics.me_u
        
        cur.execute("SELECT * FROM Isotopes WHERE Isotope =?", (iso,))
        try:
            data = cur.fetchall()[0]
        except:
            raise Exception("No such isotope: " + iso)
        
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
        
    