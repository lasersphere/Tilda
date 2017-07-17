'''
Created on 18.05.2017

@author: gorges

'''

import ast
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import TildaTools as TiTs
import MPLPlotter as plot


class Moments(object):
    '''

    '''

    def __init__ (self, db):

        self.db = db


    def calcMu(self, refVals, upperA, showing=True):
        a = 'Al'
        results = []
        if upperA:
            a = 'Au'
        list = TiTs.select_from_db(self.db, 'iso, val, statErr, systErr, run', 'Combined', [['parname'], [a]])
        for iterator in list:
            (spin, mass) = TiTs.select_from_db(self.db, 'I, mass', 'Isotopes', [['iso'], [iterator[0]]])[0]
            val = refVals[0]*iterator[1]*spin
            staterr = refVals[0]*iterator[2]*spin
            systerr = np.sqrt(np.square(refVals[1]*iterator[1]*spin)+np.square(refVals[0]*iterator[3]*spin))
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''',
                        (iterator[0], 'mu', iterator[4]))
            con.commit()
            cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ? WHERE iso = ? AND parname = ? AND run = ?''',
                        (val, staterr, systerr, iterator[0], 'mu', iterator[4]))
            con.commit()
            con.close()
            results.append((iterator[0], spin, mass, val, staterr, systerr))
        if showing:
            x = []
            y = []
            yerr = []
            cts = {}
            print('iso \t spin \t mass \t magnetic dipole moment')
            results = sorted(results, key=lambda x: x[2])
            for i in results:
                print(i[0], '\t', i[1], '\t', i[2], '\t', i[3], '\t', i[4], '\t', i[5])
                if i[1] in cts.keys():
                    cts[i[1]][0].append(i[2])
                    cts[i[1]][1].append(i[3])
                    cts[i[1]][2].append(i[4])
                else:
                    cts[i[1]] = [[i[2]], [i[3]], [i[4]]]

            plot.plotMoments(cts, False, fontsize_ticks=12)
        return results

    def calcQ(self, refVals, upperB, showing=True):
        b = 'Bl'
        results = []
        if upperB:
            b = 'Bu'
        list = TiTs.select_from_db(self.db, 'iso, val, statErr, systErr, run', 'Combined', [['parname'], [b]])
        for iterator in list:
            (spin, mass) = TiTs.select_from_db(self.db, 'I, mass', 'Isotopes', [['iso'], [iterator[0]]])[0]
            val = iterator[1]/refVals[0]
            staterr = iterator[2]/refVals[0]
            systerr = np.sqrt(np.square(refVals[1]*iterator[1]/np.square(refVals[0]))+np.square(iterator[3]/refVals[0]))
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''',
                        (iterator[0], 'Q', iterator[4]))
            con.commit()
            cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ? WHERE iso = ? AND parname = ? AND run = ?''',
                        (val, staterr, systerr, iterator[0], 'Q', iterator[4]))
            con.commit()
            con.close()
            results.append((iterator[0], spin, mass, val, staterr, systerr))

        if showing:
            x = []
            y = []
            yerr = []
            cts = {}
            print('iso \t spin \t mass \t magnetic dipole moment')
            results = sorted(results, key=lambda x: x[2])
            for i in results:
                print(i[0], '\t', i[1], '\t', i[2], '\t', i[3], '\t', i[4], '\t', i[5])
                if i[1] in cts.keys():
                    cts[i[1]][0].append(i[2])
                    cts[i[1]][1].append(i[3])
                    cts[i[1]][2].append(i[4])
                else:
                    cts[i[1]] = [[i[2]], [i[3]], [i[4]]]

            plot.plotMoments(cts, True, fontsize_ticks=12)

        return results

    def compareField(self, field):
       def c(l1,l2):
          if l1[field]<l2[field]:
              return l1[field]
          else:
              return l2[field]
       return c