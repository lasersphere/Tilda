'''
Created on 16.03.2018

@author: nörtershäuser based on KingFitter from gorges

'''

import ast
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
from Tilda.PolliFit import TildaTools as TiTs


class KingFitter2Lines(object):
    '''
    This procedure performs a King Plot of 2 Lines in  an isotope series
    and projects the isotope shift measured in one line onto the other line
    such that the combined information can be used for a common king plot.
    The linear regression routine takes y and x errors into account and is based on
    ['Unified equations for the slope, intercept, and standard errors of the best straight line', York et al.,
    American Journal of Physics 72, 367 (2004)]
    The variable alpha can be varied to reduce the covariance between the slope and the
    intercept and therefore treat the covariance in an elegant way when the uncertainty
    of the predicted charge radii has to be estimated. This is described in more detail
    in e.g. Hammen PhD Thesis 2013, University Mainz.
    The routine can be called either with a database in which the results of one line are available,
    while the information (IS and total uncertainty) of the second line can be transferred in
    the parameter isotope list or as a database in which information of both lines is provided
    with different run numbers.
    In the second case the isotope list should include those isotopes names
    for which information in both lines is available. Here, the equality of the
    reference isotope is checked during initialization but only a warning is given if they are different.
    In this case the code can be easily extended to calculate the respective isotope shifts provided that
    the reference isotope of the other transition has also been measured. Otherwise a different isotope
    must be chosen as reference in both lines ....
    '''

    def __init__(self, db, isotopelist={}, showing=True, plot_mhz=True, font_size=12, ref_run_y=-1, ref_run_x=-1):

        '''
        Import the isotopeList and initializes a KingFit2Lines, run can be specified,
        for ref_run_y==-1 any shift results are chosen from the database for the Y axis and
        isotope shifts for the x-axis must must be provided in isotopelist.
        If results for both isotopes are in the database, they must be referenced with ref_Run_x and ref_run_y
        ref_run_y ist the line to which data from ref_run_x is projected
        '''
        self.showing = showing
        self.fontsize = font_size  # fontsize used in plots
        self.plot_mhz = plot_mhz  # use False to plot y axis in gigahertz
        self.db = db
        self.a = 0
        self.b = 1

        self.isotopeList = isotopelist

        self.isotopes = []
        self.isotopeMasses = []
        self.massErr = []
        self.isotopeShifts = []
        self.isotopeShiftErr = []
        self.isotopeShiftStatErr = []
        self.isotopeShiftSystErr = []
        self.runX = []
        self.runY = []

        try:
            if ref_run_y == -1:
                self.ref = TiTs.select_from_db(self.db, 'reference', 'Lines', caller_name=__name__)[0][0]
            else:
                self.ref = TiTs.select_from_db(self.db, 'reference', 'Lines',
                                               [['refRun'], [ref_run_y]],
                                               caller_name=__name__)[0][0]
            self.runY = ref_run_y

            self.ref_mass = TiTs.select_from_db(self.db, 'mass', 'Isotopes',
                                                [['iso'], [self.ref]], caller_name=__name__)[0][0]
            self.ref_massErr = TiTs.select_from_db(self.db, 'mass_d', 'Isotopes',
                                                [['iso'], [self.ref]], caller_name=__name__)[0][0]
        except Exception as e:
            print('error: %s  \n\t-> KingFitter2Lines could not find a reference isotope from'
                  ' Lines in database or mass of this reference Isotope in Isotopes' % e)

        try:
            ref_2 = TiTs.select_from_db(self.db, 'reference', 'Lines',
                                           [['refRun'], [ref_run_x]],
                                           caller_name=__name__)[0][0]
            if ref_2!=self.ref:
                print('Warning: KingFitter2Lines found different reference isotopes for'
                      ' the two Lines in database. This case can currently not been treated')


            self.runX = ref_run_x
        except Exception as e:
            print('error: %s  \n\t-> KingFitter2Lines could not find a reference isotope for second line from'
                  ' Lines in database ' % e)

    def kingFit2Lines(self, run_y=-1, run_x=-1, alpha=0, findBestAlpha=True, find_slope_with_statistical_error=False):
        '''
        For find_slope_with_statistical_error=True:
        performs at first a KingFit with just statistical uncertainty to find out the slope, afterwards
        performing a KingFit with full error to obtain the y-intercept
        '''
        self.masses = []
        self.x_origin = []      # original x values before transformation (x--> x-alpha ), self.c is the parameter alpha
        self.x = []             # transformed values
        self.xerr = []          # statistical uncertainty in x
        self.xerr_total = []    # combined uncertainty (statistical and systematic added in square) in x
        self.y = []             # analog
        self.yerr = []          # analog
        self.yerr_total = []    # analog
        self.redmasses =[]


        self.c = alpha
        self.findBestAlphaTrue = findBestAlpha
        if self.isotopeList == {}:   # if no list in parameters, read list from database, don't know what happens if there is no list in config
            self.isotopeList = ast.literal_eval(TiTs.select_from_db(self.db, 'config', 'Combined',
                                               [['parname', 'run'], ['slope', run_x]], caller_name=__name__)[0][0])
        for i in self.isotopeList.keys():
            #read masses from database
            self.masses.append(TiTs.select_from_db(self.db, 'mass', 'Isotopes', [['iso'], [i]],
                                                   caller_name=__name__)[0][0])
            y = [0,0,0] # variable for the three parameters to be read from the db
            if run_y == -1: #takes all data (all runs)
                y = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                        [['iso', 'parname'], [i, 'shift']], caller_name=__name__)[0]
            else: # takes only the data of the corresponding run
                y = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                        [['iso', 'parname', 'run'], [i, 'shift', run_y]], caller_name=__name__)[0]
            self.y.append(y[0])

            if find_slope_with_statistical_error:
                self.yerr.append(y[1])  # statistical error
                self.yerr_total.append(np.sqrt(np.square(y[1])+np.square(y[2])))  # total error
            else:
                self.yerr.append(np.sqrt(np.square(y[1])+np.square(y[2])))  # total error

            x = [0,0,0] # now we do the same with x-values from the other run or from the list
            if run_x == -1: #take x-data from isotopeList
                self.x_origin.append(self.isotopeList[i][0])
                self.xerr.append(self.isotopeList[i][1])
            else:  # takes the data from the database
                x = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                        [['iso', 'parname', 'run'], [i, 'shift', run_x]], caller_name=__name__)[0]
                self.x_origin.append(x[0])
                if find_slope_with_statistical_error:
                    self.xerr.append(x[1])  # statistical error
                    self.xerr_total.append(np.sqrt(np.square(x[1])+np.square(x[2])))  # total error as 'squared sum' of stat and styst
                else:
                    self.xerr.append(np.sqrt(np.square(x[1])+np.square(x[2])))  # total error

        self.redmasses = [i*self.ref_mass/(i-self.ref_mass) for i in self.masses]
        self.y = [self.redmasses[i]*j for i,j in enumerate(self.y)]
        self.yerr = [np.abs(self.redmasses[i]*j) for i, j in enumerate(self.yerr)]
        self.yerr_total = [np.abs(self.redmasses[i]*j) for i, j in enumerate(self.yerr_total)]
        if find_slope_with_statistical_error:
            self.xerr = [np.abs(self.redmasses[i]*j) for i, j in enumerate(self.xerr)]
            self.xerr_total = [np.abs(self.redmasses[i]*j) for i, j in enumerate(self.xerr_total)]
        else:
            self.xerr = [np.abs(self.redmasses[i]*j) for i, j in enumerate(self.xerr)]

        if self.findBestAlphaTrue:
            self.findBestAlpha(self.runX) # determines optimum of self.c
                                          # for minimum correlation coefficient between a and b
        self.x = [self.redmasses[i]*j - self.c for i,j in enumerate(self.x_origin)]
        print('performing King fits!')

        final_a = final_b = slope_syst_err = intercept_syst_err = slope_stat_err = intercept_stat_err = 0

        if find_slope_with_statistical_error:
            # first fit with only statistical error in x and y
            (self.a, self.b, self.aerr, self.berr, a_b_correlation) = self.fit(run_y, showplot=False, print_corr_coeff=False)
            slope_stat_err = self.berr
            intercept_stat_err = self.aerr
            print('condition\t intercept (u MHz)\t err_int\t slope\t err_slope\t correlation coefficient' )
            print("statistical errors only\t %.0f \t %.0f \t %.3f \t %.3f \t %.2f" % (self.a, self.aerr, self.b, self.berr, a_b_correlation))

            # the following fits are performed with total error in x and y
            self.yerr = self.yerr_total
            self.xerr = self.xerr_total

            # fit with fixed slope total error in x and y, correlation = 0
            (self.a, self.b, self.aerr, self.berr, a_b_correlation) = self.fit(run_y, showplot=False, bFix=True, err_corr_xy=0, print_corr_coeff=False)
            print("fixed slope, full errors, error correlation = 0"
                  "\t %.0f \t %.0f \t %.3f \t %.3f \t %.2f"
                  % (self.a, self.aerr, self.b, self.berr, a_b_correlation))
            slope_syst_err = max(slope_syst_err,self.berr)
            intercept_syst_err = max(intercept_syst_err,self.aerr)

            # fit with fixed slope but full errors, error correlation = 1')
            (self.a, self.b, self.aerr, self.berr, a_b_correlation) = self.fit(run_y, showplot=False, bFix=True, err_corr_xy=1, print_corr_coeff=False)
            print("fixed slope, full errors, error correlation = 1"
                  "\t %.0f \t %.0f \t %.3f \t %.3f \t %.2f"
                  % (self.a, self.aerr, self.b, self.berr, a_b_correlation))
            slope_syst_err = max(slope_syst_err,self.berr)
            intercept_syst_err = max(intercept_syst_err,self.aerr)

            # fit with total error in x and y, correlation = - 1
            (self.a, self.b, self.aerr, self.berr, a_b_correlation) = self.fit(run_y, showplot=False, bFix=True, err_corr_xy=-1, print_corr_coeff=False)
            print("fixed slope, full errors, error correlation = -1"
                  "\t %.0f \t %.0f \t %.3f \t %.3f \t %.2f"
                  % (self.a, self.aerr, self.b, self.berr, a_b_correlation))
            slope_syst_err = max(slope_syst_err,self.berr)
            intercept_syst_err = max(intercept_syst_err,self.aerr)

            # fit with total error in x and y, correlation = 1
            (self.a, self.b, self.aerr, self.berr, a_b_correlation) = self.fit(run_y, showplot=False, bFix=False, err_corr_xy=1, print_corr_coeff=False)
            print("free slope, full errors, error correlation = +1"
                  "\t %.0f \t %.0f \t %.3f \t %.3f \t %.2f"
                  % (self.a, self.aerr, self.b, self.berr, a_b_correlation))
            slope_syst_err = max(slope_syst_err,self.berr)
            intercept_syst_err = max(intercept_syst_err,self.aerr)

            # fit with total error in x and y, correlation = -1
            (self.a, self.b, self.aerr, self.berr, a_b_correlation) = self.fit(run_y, showplot=False, bFix=False, err_corr_xy=-1, print_corr_coeff=False)
            print("free slope, full errors, error correlation = -1"
                  "\t %.0f \t %.0f \t %.3f \t %.3f \t %.2f"
                  % (self.a, self.aerr, self.b, self.berr, a_b_correlation))
            slope_syst_err = max(slope_syst_err,self.berr)
            intercept_syst_err = max(intercept_syst_err,self.aerr)

            # fit with slope free, total error in x and y, correlation = 0
            # the result of this fit will be carried into the database for the slope and the intercept (not the uncertainties)
            (self.a, self.b, self.aerr, self.berr, a_b_correlation) = self.fit(run_y, showplot=self.showing, bFix=False, err_corr_xy=0, print_corr_coeff=False)
            print("free slope, full errors, error correlation = 0"
                  "\t %.0f \t %.0f \t %.3f \t %.3f \t %.2f"
                  % (self.a, self.aerr, self.b, self.berr, a_b_correlation))
            slope_syst_err = max(slope_syst_err,self.berr)
            intercept_syst_err = max(intercept_syst_err,self.aerr)

            print('King fits performed, final values:')
            print('intercept: ', round(self.a), '(', round(intercept_stat_err), ') [', round(intercept_syst_err), '] u MHz', '\t percent systematic: %.2f' % (intercept_syst_err / self.a * 100))
            print('slope: ', self.b, '(', slope_stat_err, ') [', slope_syst_err , ']',  '\t percent systematic: %.2f' % (slope_syst_err / self.b * 100))

        else:
            (self.a, self.b, self.aerr, self.berr, a_b_correlation) = self.fit(run_y, showplot=self.showing)

            # errors are systematic since they include the systematics but have been handled statistically --> be carefull
            print('King fit performed with full errors only, no correlation assumed, uncertainties might be too small.\n')
            print('final values:')
            print('intercept: ', round(self.a), '[', round(self.aerr), '], u MHz', '\t percent: %.2f' % (self.aerr / self.a * 100))
            print('slope: ', self.b, '[', self.berr, '] ',  '\t percent: %.2f' % (self.berr / self.b * 100))
            slope_syst_err = self.berr
            intercept_syst_err = self.aerr

        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('kingVal', 'intercept', 'KingPlot'+str(self.runX)+str(self.runY)))
        con.commit()
        cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
                    (self.a, intercept_stat_err, intercept_syst_err, str(self.isotopeList), 'kingVal', 'intercept', 'KingPlot'+str(self.runX)+str(self.runY)))
        con.commit()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('kingVal', 'slope', 'KingPlot'+str(self.runX)+str(self.runY)))
        con.commit()
        cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
                    (self.b, slope_stat_err, slope_syst_err, str(self.isotopeList), 'kingVal', 'slope', 'KingPlot'+str(self.runX)+str(self.runY)))
        con.commit()
        cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('kingVal', 'alpha', 'KingPlot'+str(self.runX)+str(self.runY)))
        con.commit()
        cur.execute('''UPDATE Combined SET val = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
                    (self.c, str(self.isotopeList), 'kingVal', 'alpha', 'KingPlot'+str(self.runX)+str(self.runY)))
        con.commit()
        con.close()

    #currently only a single value for the correlation between x and y errors is allowed,
    # but can be expanded to an array with individual values
    def fit(self, run_y, showplot=True, bFix=False, plot_mhz=None, font_size=None, err_corr_xy=0, print_corr_coeff=True):
        if plot_mhz is None:
            plot_mhz = self.plot_mhz
        if font_size is None:
            font_size = self.fontsize
        i = 0
        totaldiff = 1
        omega_x = [1/np.square(i) for i in self.xerr]
        omega_y = [1/np.square(i) for i in self.yerr]
        alpha = [np.sqrt(j*omega_y[i]) for i,j in enumerate(omega_x)]
        #r = [0 for i in self.x]  # muonic data is not correlated to iso shift measurement results
        r = [err_corr_xy for i in self.x]

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

            # covariance
            sigma_b_tilde_square = 1 / (sum([w_i * u_fit[i] ** 2 for i, w_i in enumerate(w)]))
            sigma_a_tilde_square = 1 / (sum([w_i for w_i in w])) + x_bar ** 2 * sigma_b_tilde_square
            cov_a_b = - x_bar * sigma_b_tilde_square
            # r_ab in paper:
            a_b_correlation_coeff = - x_bar * np.sqrt(sigma_b_tilde_square) / np.sqrt(sigma_a_tilde_square)

            if not bFix:
                self.b = sum(betaWV)/sum(betaWU)
            self.a = y_bar - self.b*x_bar

            sigma_b_square = 1/sum(w_u_fit_square) if not bFix else 0
            sigma_a_square = 1/sum(w)+np.square(x_fit_bar)*sigma_b_square
            diff_x = np.abs(sum([np.abs(w_x_fit[i] - j) for i,j in enumerate(w_x)]))
            diff_y = np.abs(sum([np.abs(w_y_fit[i] - j) for i,j in enumerate(w_y)]))
            totaldiff = diff_x+diff_y
            i+=1

        if showplot:
            plt.subplots_adjust(bottom=0.2)
            plt.xticks(rotation=25)
            ax = plt.gca()
            if plot_mhz:
                unit='(u MHz) '
            else:
                unit='(u GHz) '
            ax.set_ylabel(r' M $\delta$ $\nu_\mathrm{ref}$ '+ str(unit), fontsize=font_size)
            if self.c == 0:
                ax.set_xlabel(r' M $\delta$ $\nu_2$ '+ str(unit), fontsize=font_size)
            else:
                ax.set_xlabel(r'M $\delta$ $\nu_2$ - $\alpha$ '+ str(unit), fontsize=font_size)
            if plot_mhz:
               plt.errorbar(self.x, self.y, self.yerr, self.xerr, fmt='k.')
            else:  # plot in Gigahertz
                y_ghz = [each / 1000 for each in self.y]
                y_err_ghz = [each / 1000 for each in self.yerr]
                x_ghz = [each / 1000 for each in self.x]
                x_err_ghz = [each / 1000 for each in self.xerr]
                plt.errorbar(x_ghz, y_ghz, y_err_ghz, x_err_ghz, fmt='k.')

            # print('x', self.x, self.xerr)
            # print('y', self.y, self.yerr)

            ax.set_xmargin(0.05)
            x_king = [min(self.x) - abs(min(self.x) - max(self.x)) * 0.2,
                      max(self.x) + abs(min(self.x) - max(self.x)) * 0.2]
            y_king = [self.a+self.b*i for i in x_king]
            if not plot_mhz:
                x_king = [min(x_ghz) - abs(min(x_ghz) - max(x_ghz)) * 0.2,
                          max(x_ghz) + abs(min(x_ghz) - max(x_ghz)) * 0.2]
            if plot_mhz:
                plt.plot(x_king, y_king, 'r', label='King plot')
            else:
                y_king_ghz = [each/1000 for each in y_king]
                plt.plot(x_king, y_king_ghz, 'r', label='King plot')
            plt.gcf().set_facecolor('w')
            plt.legend()
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.show()

        self.aerr = np.sqrt(sigma_a_square)
        if not bFix:
            self.berr = np.sqrt(sigma_b_square)

        ''' Not needed to write every result to the database --> moved to kingFit2Lines '''
        # con = sqlite3.connect(self.db)
        # cur = con.cursor()
        # cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('kingVal', 'intercept', 'KingPlot'+str(self.runX)+str(self.runY)))
        # con.commit()
        # cur.execute('''UPDATE Combined SET val = ?, systErr = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
        #             (self.a, self.aerr, str(self.isotopeList), 'kingVal', 'intercept', 'KingPlot'+str(self.runX)+str(self.runY)))
        # con.commit()
        # cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('kingVal', 'slope', 'KingPlot'+str(self.runX)+str(self.runY)))
        # con.commit()
        # cur.execute('''UPDATE Combined SET val = ?, systErr = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
        #             (self.b, self.berr, str(self.isotopeList), 'kingVal', 'slope', 'KingPlot'+str(self.runX)+str(self.runY)))
        # con.commit()
        # cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''', ('kingVal', 'alpha', 'KingPlot'+str(self.runX)+str(self.runY)))
        # con.commit()
        # cur.execute('''UPDATE Combined SET val = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
        #             (self.c, str(self.isotopeList), 'kingVal', 'alpha', 'KingPlot'+str(self.runX)+str(self.runY)))
        # con.commit()
        # con.close()

        if print_corr_coeff :
            print('correlation coefficient of a and b is: %.5f' % a_b_correlation_coeff)

        return (self.a, self.b, self.aerr, self.berr, a_b_correlation_coeff)


    def calcProjectedIS(self, isotopes=[], run_kp=-1, run_y=-1, run_x=-1, plot_evens_seperate=False):
        print('calculating the projected isotope shift...')
        self.isotopes = []
        self.isotopeMasses = []
        self.massErr = []
        self.isotopeShifts = []
        self.isotopeShiftErr = []
        self.isotopeShiftStatErr = []
        self.isotopeShiftSystErr = []
        self.run_x = []

        if run_kp==-1:
            print('Please specify King Plot result by Run! \n')
            return
        if run_y == -1:
            print('Please specify run for y-axis!\n')
            return
        if run_x == -1:
            print('Please specify run for x-axis!\n')
            return

        (self.b, self.bStatErr, self.berr) = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                                  [['parname', 'run'], ['slope', run_kp]], caller_name=__name__)[0]
        (self.a, self.aStatErr, self.aerr) = TiTs.select_from_db(self.db, 'val, statErr, systErr', 'Combined',
                                                  [['parname', 'run'], ['intercept', run_kp]], caller_name=__name__)[0]
        (self.c,) = TiTs.select_from_db(self.db, 'val', 'Combined',
                                                  [['parname', 'run'], ['alpha', run_kp]], caller_name=__name__)[0]
        vals = []
        # if run == -1:
        #     vals = TiTs.select_from_db(self.db, 'iso, val, statErr, systErr, run', 'Combined', [['parname'], ['shift']],
        #                                caller_name=__name__)
        # else:
        vals = TiTs.select_from_db(self.db, 'iso, val, statErr, systErr, run', 'Combined',
                                   [['parname', 'run'], ['shift', run_x]], caller_name=__name__)
        for i in vals:
            (name, val, statErr, systErr, run_x) = i
            if name != self.ref:
                if isotopes == [] or name in isotopes:
                    self.isotopes.append(name)
                    mass = TiTs.select_from_db(self.db,'mass, mass_d', 'Isotopes', [['iso'], [name]], caller_name=__name__)[0]
                    self.isotopeMasses.append(mass[0])
                    self.massErr.append(mass[1])
                    self.isotopeShifts.append(val)
                    self.isotopeShiftStatErr.append(statErr)
                    self.isotopeShiftSystErr.append(systErr)
                    self.isotopeShiftErr.append(np.sqrt(np.square(statErr) + np.square(systErr)))
                    self.run_x.append(run_x)
        self.isotopeRedMasses = [i*self.ref_mass/(i-self.ref_mass) for i in self.isotopeMasses]
        # from error prop:
        self.isotopeRedMassesErr = [
            ((iso_m_d * (self.ref_mass ** 2)) / (iso_m - self.ref_mass) ** 2) ** 2 +
            ((self.ref_massErr * (iso_m ** 2)) / (iso_m - self.ref_mass) ** 2) ** 2
            for iso_m, iso_m_d in zip(self.isotopeMasses, self.massErr)
        ]
        # self.projectedIS = [(j - self.a / self.isotopeRedMasses[i]) / self.b + self.c / self.isotopeRedMasses[i]
        #                     for i, j in enumerate(self.isotopeShifts)]
        self.projectedIS = [j*self.b + (self.a - self.b * self.c) / self.isotopeRedMasses[i]
                            for i, j in enumerate(self.isotopeShifts)]

        '''Alte Version: assumption: slope has no errors'''
        #self.projectedISStatErrs = [np.abs(i*self.b) for i in self.isotopeShiftStatErr]
        '''After the calculation of statistical and systematic errors of slope and intercept, they can be treated similarly:'''
        self.projectedISStatErrs = [np.sqrt(
            np.square(self.isotopeShiftStatErr[i]*self.b) +
            np.square((self.isotopeShifts[i]-self.c / (self.isotopeRedMasses[i])) *self.bStatErr) +
            np.square(self.aStatErr/(self.isotopeRedMasses[i])) +
            np.square(
                (self.a-self.b*self.c) * self.isotopeRedMassesErr[i]/np.square(self.isotopeRedMasses[i]))
        )
                                     for i, j in enumerate(self.isotopeShifts)]

        '''Total Errors are calculated only with the systematic error since this is the largest unceratinty'''
        self.projectedISTotalErrs = [np.sqrt(
            np.square(self.isotopeShiftErr[i]*self.b) +
            np.square((self.isotopeShifts[i]-self.c / (self.isotopeRedMasses[i])) *self.berr) +
            np.square(self.aerr/(self.isotopeRedMasses[i])) +
            np.square(
                (self.a-self.b*self.c) * self.isotopeRedMassesErr[i]/np.square(self.isotopeRedMasses[i]))
        )
                                     for i, j in enumerate(self.isotopeShifts)]
        errs_to_print = [(
                             self.isotopes[i],
                             self.projectedIS[i],
                             self.projectedISStatErrs[i],
                             self.projectedISTotalErrs[i],
                             abs(self.projectedISTotalErrs[i] / self.projectedIS[i]) * 100,
                             #abs(self.isotopeShiftStatErr[i] * self.b), # old statistical uncertainty
                             abs((self.isotopeShifts[i] - self.c / (self.isotopeRedMasses[i])) * self.berr ),
                             abs(self.aerr / self.isotopeRedMasses[i]),
                             abs((self.a - self.b * self.c) * self.isotopeRedMassesErr[i] / np.square(self.isotopeRedMasses[i]))
                         )
            for i, j in enumerate(self.isotopeShifts)
        ]
        print('iso\t IS_proj\t Delta IS_proj_Stat\t Delta IS_proj_Tot\t Delta IS_proj_Tot%'
              '\t Delta slope\t Delta intercept\t Delta M')
        for each in errs_to_print:
            print('%s\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f' % each)

        finalVals = {}
        for i,j in enumerate(self.isotopes):
            finalVals[j] = [self.projectedIS[i], self.projectedISStatErrs[i], self.projectedISTotalErrs[i]]
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''INSERT OR IGNORE INTO Combined (iso, parname, run) VALUES (?, ?, ?)''',
                        (j, 'projected IS', run_y))
            con.commit()
            cur.execute('''UPDATE Combined SET val = ?, statErr = ?, systErr = ?, config=? WHERE iso = ? AND parname = ? AND run = ?''',
                (self.projectedIS[i], self.projectedISStatErrs[i], self.projectedISTotalErrs[i], str(run_x) + str('-->') + str(run_y), j, 'projected IS', run_y))
            con.commit()
            con.close()
        if self.showing:
            font_size = self.fontsize
            finalVals[self.ref] = [0, 0, 0]
            keyVals = sorted(finalVals)
            x = []
            y = []
            yerr = []
            print('iso\t $\delta \nu_\mathrm{proj}$  [MHz]')
            for i in keyVals:
                x.append(int(str(i).split('_')[0]))
                y.append(finalVals[i][0])
                yerr.append(finalVals[i][2])
                print('%s\t%.1f(%.1f)[%.1f]\t%.1f' % (i, finalVals[i][0], finalVals[i][1], finalVals[i][2],
                                                finalVals[i][2] / max(abs(finalVals[i][0]), 0.000000000000001) * 100))
                #print("'"+str(i)+"'", ':[', np.round(finalVals[i][0],3), ','+ str(np.round(np.sqrt(finalVals[i][1]**2 + finalVals[i][2]**2),3))+'],')

            plt.subplots_adjust(bottom=0.2)
            plt.xticks(rotation=25)
            ax = plt.gca()
            ax.set_ylabel(r'$\delta \nu_\mathrm{proj}$ (MHz) ', fontsize=font_size)
            ax.set_xlabel('A', fontsize=font_size)
            if plot_evens_seperate:
                x_odd = [each for each in x if each % 2 != 0]
                y_odd = [each for i, each in enumerate(y) if x[i] % 2 != 0]
                y_odd_err = [each for i, each in enumerate(yerr) if x[i] % 2 != 0]
                x_even = [each for each in x if each % 2 == 0]
                y_even = [each for i, each in enumerate(y) if x[i] % 2 == 0]
                y_even_err = [each for i, each in enumerate(yerr) if x[i] % 2 == 0]

                plt.errorbar(x_even, y_even, y_even_err, fmt='ro', label='even', linestyle='-')
                plt.errorbar(x_odd, y_odd, y_odd_err, fmt='r^', label='odd', linestyle='--')
                plt.legend(loc=2)
            else:
                plt.errorbar(x, y, yerr, fmt='k.')
            ax.set_xmargin(0.05)
            plt.margins(0.1)
            plt.gcf().set_facecolor('w')
            plt.xticks(fontsize=font_size)
            plt.yticks(fontsize=font_size)
            plt.show()

        return finalVals

    def findBestAlpha(self, run):
        no_of_steps = 200 # number of steps over span range for finding alpha
        x_values=[]
        if self.c==-1: # if no starting value for \alpha is given, i.e. alpha=-1, we divide the whole span of x values in stepno equidistant steps
            x_values=[self.redmasses[i]*j for i,j in enumerate(self.x_origin)]
            alpha_min = min(x_values)
            alpha_max = max(x_values)
            step = (alpha_max-alpha_min)/no_of_steps
            self.c = alpha_min
        self.x = [self.redmasses[i]*                                                                                                   j - self.c for i,j in enumerate(self.x_origin)]
        (bestA, bestB, bestAerr, bestBerr, bestCorrelation) = self.fit(run, showplot=False)
        best = self.c
        end = False
        self.c += step
        while not end:
            print('searching for the best alpha... Trying alpha = ', np.round(self.c,0,))
            if self.c > alpha_max :
                end = True
            self.x = [self.redmasses[i]*j - self.c for i,j in enumerate(self.x_origin)]
            (newA, newB, newAerr, newBerr, newCorrelation) = self.fit(run, showplot=False)
            if abs(newCorrelation) < abs(bestCorrelation):
                bestCorrelation = newCorrelation
                best = self.c
            self.c += step

        self.c = best
        print('best alpha is: ', self.c)
