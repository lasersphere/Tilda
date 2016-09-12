'''
Created on 23.08.2016

@author: gorges

'''

import sqlite3

import matplotlib.pyplot as plt
import numpy as np

class KingFitter(object):
    '''
    The Kingfitter needs some (at least three) charge radii as input and calculates the kingfits and new charge radii
    from the isotopeshifts in the database. The fitting routine is based on
    ['Unified equations for the slope, intercept, and standard errors of the best straight line', York et al.,
    American Journal of Physics 72, 367 (2004)]
    The variable alpha can be varied to reduce the uncertainty in the intercept and thus in the charge radii,
    this is described in e.g. Hammen PhD Thesis 2013
    '''

    def __init__ (self, db, litvals, alpha=0, findBestAlpha=True, showing=True):
        '''
        Import the litvals and initializes a KingFit
        '''
        self.showing = showing
        self.db = db
        self.a = 0
        self.b = 1
        self.c = alpha

        self.masses = []
        self.x_origin = []
        self.x = []
        self.xerr = []
        self.y = []
        self.yerr = []

        self.isotopes = []
        self.isotopeMasses = []
        self.isotopeShifts = []
        self.isotopeShiftErr = []
        self.run = []

        for i in litvals.keys():
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT mass FROM Isotopes WHERE iso = ?''', (i,))
            self.masses.append(cur.fetchall()[0][0])
            cur.execute('''SELECT val, statErr, systErr FROM Combined WHERE iso = ? AND parname="shift"''', (i,))
            y = cur.fetchall()[0]
            self.y.append(y[0])
            self.yerr.append(np.sqrt(np.square(y[1])+np.square(y[2])))
            con.close()
            self.x_origin.append(litvals[i][0])
            self.xerr.append(litvals[i][1])

        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT reference FROM Lines''')
        self.ref = cur.fetchall()[0][0]
        cur.execute('''SELECT mass FROM Isotopes WHERE iso = ?''', (self.ref,))
        self.refmass = cur.fetchall()[0][0]
        self.redmasses= [i*self.refmass/(self.refmass-i) for i in self.masses]

        self.y = [self.redmasses[i]*j for i,j in enumerate(self.y)]
        self.yerr = [self.redmasses[i]*j for i,j in enumerate(self.yerr)]
        self.xerr = [self.redmasses[i]*j for i,j in enumerate(self.xerr)]

        if findBestAlpha:
            self.findBestAlpha()
        self.x = [self.redmasses[i]*j - self.c for i,j in enumerate(self.x_origin)]
        print('performing King fit!')
        (self.a, self.b, self.aerr, self.berr) = self.fit(self.showing)
        print('King fit performed, final values:')
        print('intercept: ', self.a, '(', self.aerr, ') u MHz')
        print('slope: ', self.b, '(', self.berr, ') MHz/fm^2')


    def fit(self, showplot=True):
        i=0
        totaldiff = 1
        omega_x = [1/np.square(i) for i in self.xerr]
        omega_y = [1/np.square(i) for i in self.yerr]
        alpha = [np.sqrt(j*omega_y[i]) for i,j in enumerate(omega_x)]
        r = [0 for i in self.x]

        while totaldiff>5e-15 and i < 100:
            w = [j*omega_y[i]/(j + np.square(self.b)*omega_y[i] - 2 * self.b * alpha[i] * r[i]) for i,j in enumerate(omega_x)]
            w_x = [j*w[i] for i,j in enumerate(self.x)]
            x_bar = sum(w_x)/sum(w)
            u = [i - x_bar for i in self.x]

            x_fit = [(i-self.a)/self.b for i in self.y]
            w_x_fit = [j*w[i] for i,j in enumerate(x_fit)]
            x_fit_bar = sum(w_x_fit)/sum(w)
            u_fit = [i - x_fit_bar for i in x_fit]
            w_u_fit_square = [np.square(j)*w[i] for i,j in enumerate(u_fit)]

            w_y = [j*w[i] for i,j in enumerate(self.y)]
            y_bar = sum(w_y)/sum(w)
            v = [i - y_bar for i in self.y]

            y_fit = [self.a+self.b*i for i in self.x]
            w_y_fit = [j*w[i] for i,j in enumerate(y_fit)]
            y_fit_bar = sum(w_y_fit)/sum(w)
            v_fit = [i - y_fit_bar for i in y_fit]

            beta = [j*(u[i]/omega_y[i]+self.b*v[i]/omega_x[i] - (self.b*u[i]+v[i])*r[i]/alpha[i]) for i,j in enumerate(w)]
            betaW = [j*w[i] for i,j in enumerate(beta)]
            betaWV = [j*v[i] for i,j in enumerate(betaW)]
            betaWU = [j*u[i] for i,j in enumerate(betaW)]

            self.b = sum(betaWV)/sum(betaWU)
            self.a = y_bar - self.b*x_bar

            sigma_b_square = 1/sum(w_u_fit_square)
            sigma_a_square = 1/sum(w)+np.square(x_fit_bar)*sigma_b_square
            diff_x = np.abs(sum([w_x_fit[i] - j for i,j in enumerate(w_x)]))
            diff_y = np.abs(sum([w_y_fit[i] - j for i,j in enumerate(w_y)]))
            totaldiff = diff_x+diff_y
            i+=1
            if i == 99:
                print('King fit not succesful!')

        if showplot:
            plt.subplots_adjust(bottom=0.2)
            plt.xticks(rotation=25)
            ax = plt.gca()
            ax.set_ylabel(r' M $\Delta$ $\nu$ (u MHz) ')
            ax.set_xlabel(r'M $\Delta$ < r'+r'$^2$ > - $\alpha$ (u fm $^2$)')
            plt.errorbar(self.x, self.y, self.yerr, self.xerr, fmt='k.')
            ax.set_xmargin(0.05)
            x_king = [min(self.x) - abs(min(self.x) - max(self.x)) * 0.2,max(self.x) + abs(min(self.x) - max(self.x)) * 0.2]
            y_king = [self.a+self.b*i for i in x_king]
            plt.plot(x_king, y_king, 'r', label='King fit', )
            plt.legend()
            plt.show()
        return (self.a, self.b, np.sqrt(sigma_a_square), np.sqrt(sigma_b_square))

    def calcChargeRadii(self):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT iso, val, statErr, systErr, run FROM Combined WHERE parname="shift"''')
        vals = cur.fetchall()
        for i in vals:
            (name, val, statErr, systErr, run) = i
            if name != self.ref:
                self.isotopes.append(name)
                cur.execute('''SELECT mass FROM Isotopes WHERE iso = ?''', (name,))
                self.isotopeMasses.append(cur.fetchall()[0][0])
                self.isotopeShifts.append(val)
                self.isotopeShiftErr.append(np.sqrt(np.square(float(statErr))+np.square(float(systErr))))
                self.run.append(run)
        con.close()
        self.isotopeRedMasses = [i*self.refmass/(self.refmass-i) for i in self.isotopeMasses]

        self.chargeradii = [(-self.a/self.isotopeRedMasses[i]+j)/self.b+self.c/self.isotopeRedMasses[i]
                            for i,j in enumerate(self.isotopeShifts)]
        self.chargeradiiErrs = [np.sqrt(np.square(self.isotopeShiftErr[i]/self.berr)+
                                        np.square(self.aerr/(self.isotopeRedMasses[i]*self.b))+
                                        np.square((-self.a/self.isotopeRedMasses[i]+j)*self.berr/np.square(self.b)))
                                for i,j in enumerate(self.isotopeShifts)]
        finalVals = {}
        for i,j in enumerate(self.isotopes):
            finalVals[j] = [self.chargeradii[i],self.chargeradiiErrs[i]]
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', (j, 'delta_r_square', self.run[i]))
            con.commit()
            cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ? WHERE iso = ? AND parname = ?''',
                        (self.chargeradii[i], self.chargeradiiErrs[i], 0, j, 'delta_r_square'))
            con.commit()
            con.close()
        if self.showing:
            keyVals = sorted(finalVals)
            x = []
            y = []
            yerr = []
            for i in keyVals:
                x.append(int(str(i).split('_')[0]))
                y.append(finalVals[i][0])
                yerr.append(finalVals[i][1])
                print(i, '\t', np.round(finalVals[i][0],3), '('+str(np.round(finalVals[i][1],3))+')')
            plt.subplots_adjust(bottom=0.2)
            plt.xticks(rotation=25)
            ax = plt.gca()
            ax.set_ylabel(r'$\Delta$ < r'+r'$^2$ > (fm $^2$) ')
            ax.set_xlabel('A')
            plt.errorbar(x, y, yerr, fmt='k.')
            ax.set_xmargin(0.05)
            plt.show()


        return finalVals

    def findBestAlpha(self):
        print('searching for the best alpha...')
        self.x = [self.redmasses[i]*j - self.c for i,j in enumerate(self.x_origin)]
        (bestA, bestB, bestAerr, bestBerr) = self.fit(False)
        bestRatio = np.abs(bestAerr/bestA)
        step = 1
        best = self.c
        end = False
        up = True
        self.c += step
        while not end:
            if self.c > 20000:
                up = True
                self.c = - self.c
            self.x = [self.redmasses[i]*j - self.c for i,j in enumerate(self.x_origin)]
            (newA, newB, newAerr, newBerr) = self.fit(False)
            newRatio = np.abs(newAerr/newA)
            if newRatio < bestRatio:
                bestRatio = newRatio
                best = self.c
                if up:
                    self.c += step
                else:
                    self.c -= step
            else:
                if up:
                    up = False
                    self.c -= 2*step
                else:
                    end = True
        self.c = best
        print('best alpha is: ', self.c)
