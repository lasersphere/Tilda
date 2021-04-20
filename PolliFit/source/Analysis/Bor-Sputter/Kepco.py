import numpy as np
import matplotlib.pyplot as plt
import os
from openpyxl import Workbook, load_workbook
from scipy.optimize import curve_fit


class Kepco():

    def __init__(self, files):
        # workingdir = 'C:\\Users\\Laura Renth\\Desktop\\Daten\\Promotion\\Bor\\Sputter source\\2021-03-Data' #working dir IKP
        # self.workingdir = 'C:\\Users\\Laura Renth\\ownCloud\\User\\Laura\\KOALA\\2021-03-Data'  #working dir IKP Owncloud
        self.workingdir = 'D:\\ownCloud\\User\\Laura\\KOALA\\2021-03-Data'  # working dir hp Owncloud
        self.db = os.path.join(self.workingdir, 'B-_Auswertung.sqlite')
        self.datadir = os.path.join(self.workingdir, 'kepco')
        self.datafiles = files

    def read_file(self, file):
        ### read excel-file
        try:
            self.wb = load_workbook(os.path.join(self.datadir, file))
            ws = self.wb.active
        except:
            raise Exception('File ' + file + ' could not be found!')

        ### create lists of DAC-Voltage and measured voltage
        self.dac = []
        self.volt = []
        for row in ws.rows:
            self.dac.append(row[0].value)
            self.volt.append(row[1].value)
        self.dac.pop(0)
        self.volt.pop(0)

    def plot_data(self,file):
        fig = plt.figure()  # create figure object
        ax = fig.add_subplot(1, 1, 1)   # create axes
        ax.plot(self.dac, self.volt, '.b')
        ax.set_title('Kepco-Scan ' + file)
        ax.set_xlabel('DAC-Voltage')
        ax.set_ylabel('Voltage')
        try:
            ax.plot([-8, 8], [self.linear_func(-8, self.best_vals[0], self.best_vals[1]),
                             self.linear_func(8, self.best_vals[0], self.best_vals[1])], 'r')
        except:
            plt.show()
        plt.show()

    @staticmethod
    def linear_func(x, off, m):
        return off + m * x

    def linear_fit(self):
        init_vals = [0, 50]
        self.best_vals, covar = curve_fit(self.linear_func, self.dac, self.volt, init_vals)
        print(self.best_vals, covar)
        return  self.best_vals, np.sqrt(np.diag(covar))

    def find_pars(self):
        off = []
        off_errs = []
        off_weights = []
        m =[]
        m_errs = []
        m_weights =[]
        for file in self.datafiles:
            self.read_file(file)
            pars, errs = self.linear_fit()
            self.plot_data(file)
            off.append(pars[0])
            off_errs.append(errs[0])
            off_weights.append(1/(errs[0]**2))
            m.append(pars[1])
            m_errs.append(errs[1])
            m_weights.append(1/(errs[1]**2))
        mean_off = np.average(off, weights=off_weights)
        std_off = np.std(off)
        mean_m = np.average(m, weights=m_weights)
        std_m = np.std(m)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.errorbar([1, 2], off, yerr=off_errs, fmt='b.')
        ax.set_xlim([0.8, 2.2])
        ax.plot([1, 2], [mean_off, mean_off], 'r')
        ax.fill_between([1, 2], [mean_off-std_off, mean_off-std_off], [mean_off+std_off, mean_off+std_off], alpha=0.2, color ='r')
        plt.show()
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.errorbar([1, 2], m, yerr=m_errs, fmt='b.')
        ax.plot([1, 2], [mean_m, mean_m], 'r')
        ax.fill_between([1, 2], [mean_m - std_m, mean_m - std_m], [mean_m + std_m, mean_m + std_m],
                        alpha=0.2, color='r')
        ax.set_xlim([0.8, 2.2])
        ax.set_ylim([50.481+0.001, 50.481+0.005])
        plt.show()


test = Kepco(['run350.xlsx', 'run352.xlsx'])
test.find_pars()