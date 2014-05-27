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

    def __init__(self, db, iso, isovar = '', lineVar = ''):
        '''Load relevant values of isotope name from database file'''
        print("Loading", lineVar, "line of", iso + isovar)
        #sqlite3.register_converter("BOOL", lambda v: bool(int(v)))
        
        con = sqlite3.connect(db)
        cur = con.cursor()
        
        cur.execute('''SELECT reference, frequency, Jl, Ju, shape, fixShape, charge
            FROM Lines WHERE lineVar = ?''', (lineVar,))
        try:
            data = cur.fetchall()[0]
        except:
            raise Exception("No such line: " + lineVar)
        
        self.name = iso
        self.isovar = isovar
        self.lineVar = lineVar
        self.ref = data[0]
        self.freq = data[1]
        self.Jl = data[2]
        self.Ju = data[3]
        self.shape = eval(data[4])
        self.fixShape = eval(data[5])
        elmass = data[6] * Physics.me_u
        
        cur.execute('''SELECT mass, mass_d, I, center, Al, Bl, Au, Bu, fixedArat, fixedBrat, intScale, fixedInt, relInt, m
            FROM Isotopes WHERE iso = ?''', (iso,))
        try:
            data = cur.fetchall()[0]
        except:
            raise Exception("No such isotope: " + iso)
        
        self.mass = data[0] - elmass
        self.mass_d = data[1]
        self.I = data[2]
        self.center = data[3]
        self.Al = data[4]
        self.Bl = data [5]
        self.Au = data[6]
        self.Bu = data[7]
        self.fixArat = data[8]
        self.fixBrat = data[9]
        self.intScale = data[10]
        self.fixInt = data[11]
        if data[12] is not None:
            self.relInt = eval(data[12])
        
        if data[13] == None:
            self.m = None
        else:
            self.m = DBIsotope(db, data[13], lineVar)

        cur.close()
        con.close()
        
    