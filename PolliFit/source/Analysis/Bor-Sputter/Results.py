import numpy as np
import sqlite3
import os
import ast
import matplotlib.pyplot as plt


class Results:

    def __init__(self):
        # self.workingdir = 'C:\\Users\\Laura Renth\\ownCloud\\User\\Laura\\KOALA\\2021-03-Data'
        # #working dir IKP Owncloud
        self.workingdir = 'D:\\ownCloud\\User\\Laura\\KOALA\\2021-03-Data'  # working dir hp Owncloud
        self.db = os.path.join(self.workingdir, 'B-_Auswertung.sqlite')

        self.iso = '11B_D2'
        self.run = 'sym2'
        self.iso = '11B'
        self.run = 'sym1'
        self.iso = '10B_D2'
        self.run = 'sym1'

        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT pars FROM FitRes WHERE iso = ? AND run = ?''', (self.iso, self.run,))
        self.pars = cur.fetchall()
        con.close()

    def calcMean(self):
        centers = []
        errs = []
        weights = []
        for res in self.pars:
            result = ast.literal_eval(res[0])
            centers.append(ast.literal_eval(res[0])['center'][0])
            errs.append(ast.literal_eval(res[0])['center'][1])
            weights.append(1 / ast.literal_eval(res[0])['center'][1]**2)
        mean = np.average(centers, weights=weights)
        std = np.std(centers)
        return centers, errs, mean, std

    def plotRes(self, cts, errs, mean, std):
        x = list(range(0,len(cts)))
        fig = plt.figure()  # create figure object
        ax = fig.add_axes([0.12, 0.1, 0.8, 0.8])   # create axes object in figure
        #ax.plot(x, cts, 'bx')
        ax.errorbar(x, cts, yerr=errs, fmt='b.')
        ax.set_title('Center frequencies')
        ax.set_xlabel('Measurement')
        ax.set_ylabel('center in MHz')
        ax.plot([0, 5], [mean, mean], 'r-')
        ax.fill_between([0, 5], [mean-std, mean-std], [mean+std, mean+std], alpha=0.2, color='b')
        plt.show()


res = Results()
pars = res.calcMean()
res.plotRes(pars[0], pars[1], pars[2], pars[3])
