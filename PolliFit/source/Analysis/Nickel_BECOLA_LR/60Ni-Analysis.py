import matplotlib.dates as dates
import importlib
import math
import os
import sqlite3
from matplotlib import pyplot as plt
import ast
import numpy as np
from Measurement.XMLImporter import XMLImporter
import Physics
import BatchFit
from operator import add
from scipy.optimize import curve_fit
from datetime import datetime
from lxml import etree as et
from XmlOperations import xmlWriteDict
from openpyxl import Workbook, load_workbook


class NiAnalysis:

    def __init__(self, working_dir, db60):
        # working_dir: path of working directory where to save files
        # db: path of database
        # data_path: path where to find xml data files
        # line_var: line variables to use for fitting
        # runs60: runs to use for fitting
        # frequ_60ni: literature value of 60Ni transition frequency (Kristian)
        # laser_freq60: laser frequency for 60ni measurements
        # wb: excel workbook containing "good" files
        self.working_dir = working_dir
        self.db = os.path.join(self.working_dir, db60)
        self.data_path = os.path.join(self.working_dir, 'data')
        self.line_var = ['refLine_0', 'refLine_1', 'refLine_2']
        self.runs60 = ['AsymVoigt0', 'AsymVoigt1', 'AsymVoigt2']
        self.frequ_60ni = 850344183
        self.laser_freq60 = 851224124.8007469
        self.wb = load_workbook(os.path.join(self.data_path, 'Files.xlsx'))
        self.results = Workbook()
        self.worksheet = self.results.active
        self.worksheet['A1'] = 'Isotope'
        self.worksheet['B1'] = 'isotope shift'
        self.worksheet['C1'] = 'statistic uncertainty'
        self.worksheet['D1'] = 'method'

        # for Calibration
        # times: list of datetimes
        # volts: list of calibrated voltages
        self.times = []
        self.volts = []
        self.fig, self.ax = plt.subplots()

    def reset(self):
        # resets the data base

        # reset fixed shapes
        for run in self.runs60:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT fixShape FROM Lines WHERE refRun = ?''', (run,))
            fix_shape = ast.literal_eval(cur.fetchall()[0][0])
            fix_shape['asy'] = [0, 20]
            cur.execute('''UPDATE Lines SET fixShape = ? WHERE refRun = ?''', (str(fix_shape), run,))
            con.commit()
            con.close()

        # reset calibration
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Files SET accVolt = ?''', (29850,))
        con.commit()
        con.close()
        print('Calibration resetted')

        # reset FitResults
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''DELETE FROM FitRes''')
        con.commit()
        con.close()
        print('FitRes cleared')

    def prep_file_list(self):
        # prepares the files database based on the excle workbook

        # create list of "good" files
        ws = self.wb['Tabelle1']
        cells = []
        for i, row in enumerate(ws.rows):
            cells.append([])
            for cell in row:
                cells[i].append(cell.value)
        cells.pop(0)

        # delete "bad" files from database
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT file FROM Files''')
        files = cur.fetchall()
        for f in files:
            remove = 0
            for c in cells:
                if f[0] == c[0] and c[2] == 'y':
                    remove = 0
                    break
                else:
                    remove = 1
            if remove == 1:
                cur.execute('''DELETE FROM Files WHERE file=?''', (f[0],))
        con.commit()
        con.close()
        print('Removed bad files')

    def fit_all(self, files, run):
        # fits all files
        # files: list of files to be fitted
        # run: run to use for fitting

        for f in files:
            # Guess offset
            spec = XMLImporter(path=os.path.join(self.data_path, f))
            offset = (spec.cts[0][0][0] + spec.cts[0][0][-1]) / 2
            self.adj_offset(offset, run)

            # Guess center
            index = list(spec.cts[0][0]).index(max(spec.cts[0][0]))
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT accVolt FROM Files WHERE file = ?''', (f,))
            acc_v = cur.fetchall()[0][0]
            center_voltage = acc_v - spec.x[0][index]
            cur.execute('''SELECT laserFreq FROM Files WHERE file = ? ''', (f,))
            laser_frequ = cur.fetchall()[0][0]
            cur.execute('''SELECT type FROM Files WHERE file = ?''', (f,))
            iso = cur.fetchall()[0][0]
            if iso == '58Ni':
                mass = 58
            elif iso == '60Ni':
                mass = 60
            else:
                mass = 56
            cur.execute('''SELECT frequency FROM Lines WHERE refRun = ? ''', (run,))
            frequ = cur.fetchall()[0][0]
            v = Physics.relVelocity(Physics.qe * center_voltage, mass * Physics.u)
            v = -v
            center_frequ = Physics.relDoppler(laser_frequ, v) - frequ
            center = center_frequ - 500
            self.adj_center(center, iso)

            # Fit
            BatchFit.batchFit(np.array([f]), self.db, run)

    def adj_offset(self, offset, run):
        # sets offset in database
        # offset: value to set the offset to
        # run: run of which the offset is to be adjusted

        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT shape FROM Lines WHERE refRun LIKE ? ''', (run,))
        shape_dict = ast.literal_eval(cur.fetchall()[0][0])  # create dictionary
        shape_dict['offset'] = offset
        cur.execute('''UPDATE Lines SET shape = ? WHERE refRun = ?''', (str(shape_dict), run,))
        con.commit()
        con.close()
        print('Adjusted Offset to ', shape_dict['offset'])

    def adj_center(self, center, iso):
        # sets center in database
        # center: value to set the center to
        # iso: istope of which the center is to be adjsued

        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Isotopes SET center = ? WHERE iso = ?''', (center, iso,))
        con.commit()
        con.close()
        print('Adjusted Center to', center)

    def initialize(self):
        # first fit of reference measurements (60Ni)

        files = self.get_files('60Ni')
        for run in self.runs60:
            self.fit_all(files, run)
        plt.close()  # without this, some pyplot error is thrown...

    def asymmetry(self):
        # find fix asymmetry parameter

        con = sqlite3.connect(self.db)
        cur = con.cursor()
        files = self.get_files('60Ni')
        for run in self.runs60:
            asy = []
            unc = []
            weights = []

            # collect asymmetry parameter for each file
            for f in files:
                cur.execute('''SELECT pars FROM FitRes WHERE file = ? AND run = ?''', (f, run,))
                pars = ast.literal_eval(cur.fetchall()[0][0])['asy']    # create dictionary
                asy.append(pars[0])
                unc.append(pars[1])
                weights.append(1 / (pars[1] ** 2))

            # calculate weighted mean of all parameters
            mean = np.average(asy, weights=weights)
            std = np.std(asy)

            # plot results
            fig, ax = plt.subplots()
            ax.errorbar(list(range(0, len(asy))), asy, yerr=unc)
            ax.plot([0, len(asy)], [mean, mean], 'r-')
            ax.fill_between([0, len(asy)], [mean + std, mean + std], [mean - std, mean - std], color='r', alpha=0.2)
            ax.set_title('Run: ' + run)
            ax.set_xlabel('Messungen')
            ax.set_ylabel('Asymmetry factor')
            ax.text(20, 13, str(mean))
            fig.show()  # show plot
            fig.savefig(os.path.join(self.working_dir, 'asymmetry\\Scaler_' + str(run) + '.png'))  # save plot
            plt.close()

            # set asymmetry parameter to mean and fix it
            cur.execute('''SELECT shape FROM Lines WHERE refRun = ?''', (run,))
            shape = ast.literal_eval(cur.fetchall()[0][0])  # create dictionary
            cur.execute('''SELECT fixShape FROM Lines WHERE refRun = ?''', (run,))
            fix = ast.literal_eval(cur.fetchall()[0][0])    # create dictionary
            shape['asy'] = mean
            fix['asy'] = True
            cur.execute('''UPDATE Lines SET shape = ? WHERE refRun = ?''', (str(shape), run,))
            cur.execute('''UPDATE Lines SET fixShape = ? WHERE refRun = ?''', (str(fix), run,))
            con.commit()
        con.close()

    def get_files(self, isotope):
        # return list of all files in database of one isotope
        # isotope: istope of which to get the files of

        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT file FROM Files WHERE type LIKE ? ''', (isotope,))
        files = cur.fetchall()
        con.close()
        return [f[0] for f in files]

    def calibrate_all_ref(self):
        # Calibrate DAC voltage of all refernce files on transition frequency of 60Ni (Kristian)

        files = self.get_files('60Ni')
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        for f in files:
            cur.execute('''SELECT pars FROM FitRes WHERE file = ?''', (f,))
            pars = cur. fetchall()
            center = []
            errs = []
            weights = []

            # each file has fit results for 3 different runs (3 scalers)
            for result in pars:
                dic = ast.literal_eval(result[0])
                center.append(dic['center'][0])
                errs.append(dic['center'][1])
                weights.append(1 / (dic['center'][1] ** 2))

            # use mean of 3 scalers for calibration of each file
            mean = np.average(center, weights=weights)
            acc_volt = self.calibrate(f, mean)
            cur.execute('''UPDATE Files Set accVolt = ? WHERE file = ?''', (acc_volt, f,))
        con.commit()
        con.close()
        self.plot_calibration()

    def calibrate(self, file, delta_frequ):
        # returns calibrated voltage of a single reference file based on the mean fitresult of the center - parameter
        # file: file to calibrate
        # delta_frequ: difference of fitted frequence and literature value

        # Calculate differential Doppler shift
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt from Files WHERE file = ?''', (file,))
        acc_volt = cur.fetchall()[0][0]
        con.close()
        diff_dopp60 = Physics.diffDoppler(delta_frequ + self.frequ_60ni, acc_volt, 60)

        # Calculate calibration
        delta_u = delta_frequ / diff_dopp60
        print('Delta voltage:', delta_u)
        return acc_volt + delta_u

    def plot_calibration(self):
        # plots calibrated voltage of all reference files in the database over the time

        files = self.get_files('60Ni')
        file_times = []
        voltages = []
        for f in files:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT date FROM Files  WHERE file = ?''', (f,))
            date = cur.fetchall()[0][0]
            cur.execute('''SELECT accVolt FROM Files WHERE file = ?''', (f,))
            acc_vol = cur.fetchall()[0][0]
            con.close()
            file_times.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))  # convert string to datetime
            voltages.append(acc_vol)
        self.times = file_times
        self.volts = voltages

        plt.close()
        self.fig, self.ax = plt.subplots()
        self.ax.plot_date(file_times, voltages, fmt='-b')
        self.ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.setp(self.ax.get_xticklabels(), rotation=30, ha="right")
        self.ax.set_title('Calibrated Voltage')
        self.ax.set_ylabel('Voltage in V')

    def calibrate_all(self, files):
        # use calibration of reference measurements for calibration of all files
        # files: files to use calibration on

        times = []
        volts = []
        con = sqlite3.connect(self.db)
        cur = con.cursor()

        # select time stamp of file and use interpolation to find calibrated voltage
        for f in files:
            cur.execute('''SELECT date FROM Files WHERE file = ?''', (f,))
            time = datetime.strptime(cur.fetchall()[0][0], '%Y-%m-%d %H:%M:%S')
            times.append(time)
            if time <= self.times[0]:
                volts.append(self.volts[0])
                cur.execute('''UPDATE Files SET accVolt = ? WHERE file = ?''', (self.volts[0], f,))
                continue
            for i, t in enumerate(self.times):
                if t <= time < self.times[i + 1]:
                    y0, m = curve_fit(self.lin_func, dates.date2num(self.times[i:i+2]), self.volts[i:i+2])[0]
                    new_volt = y0 + m * dates.date2num(time)
                    volts.append(new_volt)
                    cur.execute('''UPDATE Files SET accVolt = ? WHERE file = ?''', (new_volt, f,))
                    break
        con.commit()
        con.close()
        self.ax.plot_date(times, volts, fmt='rx')
        self.fig.show()
        self.fig.savefig(os.path.join(self.working_dir, 'calibration\\Voltagecalibration.png'))

    def calib_procedure(self, isotope):
        # the whole procedure of calibration
        # isotpe: isotpe to use the calibration on

        plt.close()  # without this, some pyplot error is thrown...

        # calibration of reference files
        self.calibrate_all_ref()

        # refit with new voltage
        for run in self.runs60:
            self.fit_all(self.get_files('60Ni'), run)
        plt.close()  # without this, some pyplot error is thrown...

        # calibration of reference files (second time for higher accurancy)
        self.calibrate_all_ref()

        # calibrate all other files
        self.calibrate_all(niAna.get_files(isotope))

    @staticmethod
    def lin_func(x, y0, m):
        # linear function

        return y0 + m * x

    def center(self, isotope):
        # analysis procedure plots results of each file and returns the weighted mean and standard deviation of all

        for run in self.runs60:
            self.fit_all(self.get_files(isotope), run)
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        files = self.get_files(isotope)
        times = []
        centers = []
        errs = []
        for f in files:
            cur.execute('''SELECT pars FROM FitRes WHERE file = ?''', (f,))
            pars = cur.fetchall()
            results = []
            weihgts = []
            # there are three results for each file (three scalers)
            for run in pars:
                results.append(ast.literal_eval(run[0])['center'][0])
                weihgts.append(1 / (ast.literal_eval(run[0])['center'][1] ** 2))
            # take mean of the three scalers
            mean = np.average(results, weights=weihgts)
            std = np.std(results)
            centers.append(mean)
            errs.append(std)
            cur.execute('''SELECT date FROM Files WHERE file = ?''', (f,))
            date = datetime.strptime(cur.fetchall()[0][0], '%Y-%m-%d %H:%M:%S')
            times.append(date)
        con.close()
        plt.close()    # without this, some pyplot error is thrown...
        upper = []
        lower = []
        for i, c in enumerate(centers):
            upper.append(c + errs[i])
            lower.append(c - errs[i])
        fig, ax = plt.subplots()
        ax.plot_date(times, centers, fmt='bx', linestyle='-', )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        ax.fill_between(times, upper, lower, alpha=0.2)
        ax.set_title('Center frequency of ' + isotope)
        ax.set_ylabel('Frequency in MHz')
        fig.show()
        fig.savefig(os.path.join(self.working_dir, 'center\\'+ isotope + '.png'))
        return np.average(centers, weights=errs), np.std(centers)

    def isotope_shift(self, isotope):
        ref_center, ref_unc = self.center('60Ni')
        iso_center, iso_unc = self.center(isotope)
        isotope_shift = iso_center - ref_center
        ishift_unc = np.sqrt(np.square(iso_unc) + np.square(ref_unc))
        self.worksheet.append([isotope, isotope_shift, ishift_unc])
        print(isotope_shift, ishift_unc)


# workingDir = 'D:\\Owncloud\\User\\Laura\\Nickelauswertung'  # Path Laptop
workingDir = 'C:\\Users\\Laura Renth\\ownCloud\\User\\Laura\\Nickelauswertung'  # Path IKP
db = 'Nickel_BECOLA_60Ni.sqlite'

niAna = NiAnalysis(workingDir, db)
#niAna.reset()
#niAna.prep_file_list()
#niAna.initialize()
#niAna.asymmetry()
#for r in niAna.runs60:
    #niAna.fit_all(niAna.get_files('60Ni'), r)
#niAna.calib_procedure('58Ni')
niAna.isotope_shift('58Ni')
row = niAna.worksheet.max_row
niAna.\
results.save(os.path.join(workingDir, 'results\\Isotope_shifts.xlsx'))