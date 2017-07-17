'''
Created on 23.08.2016

@author: gorges

'''

import ast
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import TildaTools as TiTs


class KingFitter(object):
    '''
    The Kingfitter needs some (at least three) charge radii as input and calculates the kingfits and new charge radii
    from the isotopeshifts in the database. The fitting routine is based on
    ['Unified equations for the slope, intercept, and standard errors of the best straight line', York et al.,
    American Journal of Physics 72, 367 (2004)]
    The variable alpha can be varied to reduce the uncertainty in the intercept and thus in the charge radii,
    this is described in e.g. Hammen PhD Thesis 2013
    '''

    def __init__ (self, db, litvals={}, showing=True):
        '''
        Import the litvals and initializes a KingFit, run can be specified, for run==-1 any shift results are chosen
        '''
        self.showing = showing
        self.db = db
        self.a = 0
        self.b = 1

        self.litvals = litvals

        self.isotopes = []
        self.isotopeMasses = []
        self.massErr = []
        self.isotopeShifts = []
        self.isotopeShiftErr = []
        self.isotopeShiftStatErr = []
        self.isotopeShiftSystErr = []
        self.run = []
        try:
            self.ref = TiTs.select_from_db(self.db, 'reference', 'Lines', caller_name=__name__)[0][0]
            self.ref_mass = TiTs.select_from_db(self.db, 'mass', 'Isotopes', [['iso'], [self.ref]], caller_name=__name__)[0][0]
        except Exception as e:
            print('error: %s  \n\t-> Kingfitter could not find a reference isotope from'
                  ' Lines in database or mass of this reference Isotope in Isotopes' % e)


    def kingFit(self, run=-1, alpha=0, findBestAlpha=True, find_slope_with_statistical_error=False):
        '''
        For find_b_with_statistical_error=True:
        performs at first a KingFit with just statistical uncertainty to find out the slope, afterwards
        performing a KingFit with full error to obtain the y-intercept
        '''
        self.masses = []
        self.x_origin = []
        self.x = []
        self.xerr = []
        self.y = []
        self.yerr = []
        self.yerr_total = []

        self.c = alpha
        self.findBestAlphaTrue = findBestAlpha
        if self.litvals == {}:
            self.litvals = ast.literal_eval(TiTs.select_from_db(self.db, 'config', 'Combined',
                                               [['parname', 'run'], ['slope', run]], caller_name=__name__)[0][0])
        for i in self.litvals.keys():
            self.masses.append(TiTs.select_from_db(self.db, 'mass', 'Isotopes', [['iso'], [i]],
                                                   caller_name=__name__)[0][0])
            y = [0,0,0]
            if run == -1:
                y = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                        [['iso', 'parname'], [i, 'shift']], caller_name=__name__)[0]
            else:
                y = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                        [['iso', 'parname', 'run'], [i, 'shift', run]], caller_name=__name__)[0]
            self.y.append(y[0])
            if find_slope_with_statistical_error:
                self.yerr.append(y[1]) #statistical error
                self.yerr_total.append(np.sqrt(np.square(y[1])+np.square(y[2]))) #total error
            else:
                self.yerr.append(np.sqrt(np.square(y[1])+np.square(y[2]))) #total error
            self.x_origin.append(self.litvals[i][0])
            self.xerr.append(self.litvals[i][1])
        self.redmasses= [i*self.ref_mass/(i-self.ref_mass) for i in self.masses]
        self.y = [self.redmasses[i]*j for i,j in enumerate(self.y)]
        self.yerr = [np.abs(self.redmasses[i]*j) for i,j in enumerate(self.yerr)]
        self.yerr_total = [np.abs(self.redmasses[i]*j) for i,j in enumerate(self.yerr_total)]
        self.xerr = [np.abs(self.redmasses[i]*j) for i,j in enumerate(self.xerr)]

        if self.findBestAlphaTrue:
            self.findBestAlpha(run)
        self.x = [self.redmasses[i]*j - self.c for i,j in enumerate(self.x_origin)]
        print('performing King fit!')
        if find_slope_with_statistical_error:
            (self.a, self.b, self.aerr, self.berr) = self.fit(run, showplot=False )
            self.yerr = self.yerr_total
            (self.a, self.b, self.aerr, self.berr) = self.fit(run, showplot=self.showing, bFix=True)
        else:
            (self.a, self.b, self.aerr, self.berr) = self.fit(run, showplot=self.showing)
        print('King fit performed, final values:')
        print('intercept: ', self.a, '(', self.aerr, ') u MHz')
        print('slope: ', self.b, '(', self.berr, ') MHz/fm^2')

    def fit(self, run, showplot=True, bFix=False):
        i = 0
        totaldiff = 1
        omega_x = [1/np.square(i) for i in self.xerr]
        omega_y = [1/np.square(i) for i in self.yerr]
        alpha = [np.sqrt(j*omega_y[i]) for i,j in enumerate(omega_x)]
        r = [0 for i in self.x]

        while totaldiff > 1e-10 and i < 200:
            w = [j*omega_y[i]/(j + np.square(self.b)*omega_y[i] - 2 * self.b * alpha[i] * r[i])
                 for i, j in enumerate(omega_x)]
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

            if not bFix:
                self.b = sum(betaWV)/sum(betaWU)
            self.a = y_bar - self.b*x_bar

            sigma_b_square = 1/sum(w_u_fit_square)
            sigma_a_square = 1/sum(w)+np.square(x_fit_bar)*sigma_b_square
            diff_x = np.abs(sum([np.abs(w_x_fit[i] - j) for i,j in enumerate(w_x)]))
            diff_y = np.abs(sum([np.abs(w_y_fit[i] - j) for i,j in enumerate(w_y)]))
            totaldiff = diff_x+diff_y
            i+=1

        if showplot:
            plt.subplots_adjust(bottom=0.2)
            plt.xticks(rotation=25)
            ax = plt.gca()
            ax.set_ylabel(r' M $\delta$ $\nu$ (u MHz) ')
            if self.c == 0:
                ax.set_xlabel(r'M $\delta$ < r'+r'$^2$ > (u fm $^2$)')
            else:
                ax.set_xlabel(r'M $\delta$ < r'+r'$^2$ > - $\alpha$ (u fm $^2$)')
            plt.errorbar(self.x, self.y, self.yerr, self.xerr, fmt='k.')
            # print('x', self.x, self.xerr)
            # print('y', self.y, self.yerr)

            ax.set_xmargin(0.05)
            x_king = [min(self.x) - abs(min(self.x) - max(self.x)) * 0.2,max(self.x) + abs(min(self.x) - max(self.x)) * 0.2]
            y_king = [self.a+self.b*i for i in x_king]
            plt.plot(x_king, y_king, 'r', label='King fit', )
            plt.legend()
            plt.show()

        self.aerr = np.sqrt(sigma_a_square)
        if not bFix:
            self.berr = np.sqrt(sigma_b_square)
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('kingVal', 'intercept', run))
        con.commit()
        cur.execute('''UPDATE Combined SET val = ?, systErr = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
                    (self.a, self.aerr, str(self.litvals), 'kingVal', 'intercept', run))
        con.commit()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('kingVal', 'slope', run))
        con.commit()
        cur.execute('''UPDATE Combined SET val = ?, systErr = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
                    (self.b, self.berr, str(self.litvals), 'kingVal', 'slope', run))
        con.commit()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('kingVal', 'alpha', run))
        con.commit()
        cur.execute('''UPDATE Combined SET val = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
                    (self.c, str(self.litvals), 'kingVal', 'alpha', run))
        con.commit()
        con.close()
        return (self.a, self.b, self.aerr, self.berr)

    def calcChargeRadii(self, isotopes=[], run=-1):
        print('calculating the charge radii...')
        self.isotopes = []
        self.isotopeMasses = []
        self.massErr = []
        self.isotopeShifts = []
        self.isotopeShiftErr = []
        self.isotopeShiftStatErr = []
        self.isotopeShiftSystErr = []
        self.run = []

        (self.b, self.berr) = TiTs.select_from_db(self.db, 'val, systErr', 'Combined',
                                                  [['parname', 'run'], ['slope', run]], caller_name=__name__)[0]
        (self.a, self.aerr) = TiTs.select_from_db(self.db, 'val, systErr', 'Combined',
                                                  [['parname', 'run'], ['intercept', run]], caller_name=__name__)[0]
        (self.c,) = TiTs.select_from_db(self.db, 'val', 'Combined',
                                                  [['parname', 'run'], ['alpha', run]], caller_name=__name__)[0]
        vals = []
        if run == -1:
            vals = TiTs.select_from_db(self.db,'iso, val, statErr, systErr, run', 'Combined', [['parname'], ['shift']],
                                       caller_name=__name__)
        else:
            vals = TiTs.select_from_db(self.db,'iso, val, statErr, systErr, run', 'Combined',
                                       [['parname', 'run'], ['shift', run]], caller_name=__name__)
        for i in vals:
            (name, val, statErr, systErr, run) = i
            if name != self.ref:
                if isotopes == [] or name in isotopes:
                    self.isotopes.append(name)
                    mass = TiTs.select_from_db(self.db,'mass, mass_d', 'Isotopes', [['iso'], [name]], caller_name=__name__)[0]
                    self.isotopeMasses.append(mass[0])
                    self.massErr.append(mass[1])
                    self.isotopeShifts.append(val)
                    self.isotopeShiftStatErr.append(statErr)
                    self.isotopeShiftSystErr.append(systErr)
                    self.run.append(run)
        self.isotopeRedMasses = [i*self.ref_mass/(i-self.ref_mass) for i in self.isotopeMasses]
        self.chargeradii = [(-self.a/self.isotopeRedMasses[i]+j)/self.b+self.c/self.isotopeRedMasses[i]
                            for i,j in enumerate(self.isotopeShifts)]
        #self.chargeradiiStatErrs = [np.abs(i/self.b) for i in self.isotopeShiftStatErr]
        self.chargeradiiTotalErrs = [np.sqrt(np.square(self.isotopeShiftStatErr[i]/self.b)+
                                        np.square(self.aerr/(self.isotopeRedMasses[i]*self.b))+
                                        np.square((-self.a/self.isotopeRedMasses[i]+j)*self.berr/np.square(self.b)) +
                                        np.square((self.a/self.b+self.c)*self.massErr[i]/np.square(self.ref_mass)))
                                    for i,j in enumerate(self.isotopeShifts)]
        finalVals = {}
        for i,j in enumerate(self.isotopes):
            finalVals[j] = [self.chargeradii[i], self.chargeradiiTotalErrs[i]]
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', (j, 'delta_r_square', self.run[i]))
            con.commit()
            cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ? WHERE iso = ? AND parname = ?''',
                        (self.chargeradii[i], 0, self.chargeradiiTotalErrs[i], j, 'delta_r_square'))
            con.commit()
            con.close()
        if self.showing:
            keyVals = sorted(finalVals)
            x = []
            y = []
            yerr = []
            print('iso\t $\delta$ <r$^2$>[fm$^2$]($\Delta$ $\delta$<r$^2$>[fm$^2$])')
            for i in keyVals:
                x.append(int(str(i).split('_')[0]))
                y.append(finalVals[i][0])
                yerr.append(finalVals[i][1])
                print(i, '\t', np.round(finalVals[i][0],3), '('+str(np.round(finalVals[i][1],3))+')')
                #print("'"+str(i)+"'", ':[', np.round(finalVals[i][0],3), ','+ str(np.round(np.sqrt(finalVals[i][1]**2 + finalVals[i][2]**2),3))+'],')

            plt.subplots_adjust(bottom=0.2)
            plt.xticks(rotation=25)
            ax = plt.gca()
            ax.set_ylabel(r'$\delta$ < r'+r'$^2$ > (fm $^2$) ')
            ax.set_xlabel('A')
            plt.errorbar(x, y, yerr, fmt='k.')
            ax.set_xmargin(0.05)
            plt.show()


        return finalVals

    def findBestAlpha(self, run):
        self.x = [self.redmasses[i]*j - self.c for i,j in enumerate(self.x_origin)]
        (bestA, bestB, bestAerr, bestBerr) = self.fit(run, showplot=False)
        bestRatio = np.abs(bestAerr/bestA)
        step = 1
        best = self.c
        end = False
        up = True
        self.c += step
        while not end:
            print('searching for the best alpha... Trying alpha = ', self.c)
            if np.abs(self.c) >= 2000: #searching just between [-2000,2000]
                self.c = - self.c + step
            self.x = [self.redmasses[i]*j - self.c for i,j in enumerate(self.x_origin)]
            (newA, newB, newAerr, newBerr) = self.fit(run, showplot=False)
            newRatio = np.abs(newAerr/newA)
            if newRatio < bestRatio:
                bestRatio = newRatio
                best = self.c
                self.c += step

            else:
                if up:
                    up = False
                    step = - step
                    self.c += 2*step
                else:
                    end = True
        self.c = best
        print('best alpha is: ', self.c)
