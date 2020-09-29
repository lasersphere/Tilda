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
from lxml import etree as ET
from XmlOperations import xmlWriteDict
from openpyxl import Workbook, load_workbook

class NiAnalysis:

    def __init__(self, working_dir, db60):
        self.working_dir = working_dir
        self.db = os.path.join(self.working_dir, db60)
        self.data_path = os.path.join(self.working_dir, 'data')
        self.lineVar = ['refLine_0', 'refLine_1', 'refLine_2']
        self.runs60 = ['AsymVoigt0', 'AsymVoigt1', 'AsymVoigt2']
        self.frequ_60ni = 850344183
        self.laserFreq60 = 851224124.8007469
        self.wb = load_workbook(os.path.join(self.data_path, 'Files.xlsx'))

        # Calibration
        self.times = []
        self.volts = []

    def reset(self):    # resets the data base
        #reset fixed shapes
        self.reset_fixShape()

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

    def reset_fixShape(self):
        # Reset FixShape
        for run in self.runs60:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT fixShape FROM Lines WHERE refRun = ?''', (run,))
            fixShape = ast.literal_eval(cur.fetchall()[0][0])
            fixShape['asy'] = [0, 20]
            cur.execute('''UPDATE Lines SET fixShape = ? WHERE refRun = ?''', (str(fixShape), run,))
            con.commit()
            con.close()

    def prepFileList(self):
        ws = self.wb['Tabelle1']
        cells = []
        for i, row in enumerate(ws.rows):
            cells.append([])
            for cell in row:
                cells[i].append(cell.value)
        cells.pop(0)
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
            accV = cur.fetchall()[0][0]
            center_voltage = accV - spec.x[0][index]
            cur.execute('''SELECT laserFreq FROM Files WHERE file = ? ''', (f,))
            laserFrequ = cur.fetchall()[0][0]
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
            centerFrequ = Physics.relDoppler(laserFrequ, v) - frequ
            center = centerFrequ - 500
            self.adj_center(center, iso)

            # Fit
            BatchFit.batchFit(np.array([f]), self.db, run)

    def adj_offset(self, offset, run):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT shape FROM Lines WHERE refRun LIKE ? ''', (run,))
        shape_dict = ast.literal_eval(cur.fetchall()[0][0])
        shape_dict['offset'] = offset
        cur.execute('''UPDATE Lines SET shape = ? WHERE refRun = ?''', (str(shape_dict), run,))
        con.commit()
        con.close()
        print('Adjusted Offset to ', shape_dict['offset'])

    def adj_center(self, center, iso):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Isotopes SET center = ? WHERE iso = ?''', (center, iso,))
        con.commit()
        con.close()
        print('Adjusted Center to', center)

    def initialize(self):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT file FROM Files WHERE type=?''', ('60Ni',))
        fetch = cur.fetchall()
        con.close()
        files = []
        for f in fetch:
            files.append(f[0])
        for run in self.runs60:
            self.fit_all(files, run)
        plt.plot([1,1], [1,1])
        plt.show()

    def asymmetry(self):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        files = self.get_files('60Ni')
        for run in self.runs60:
            asy = []
            unc = []
            weights = []
            for f in files:
                cur.execute('''SELECT pars FROM FitRes WHERE file = ? AND run = ?''', (f, run,))
                pars = ast.literal_eval(cur.fetchall()[0][0])['asy']
                asy.append(pars[0])
                unc.append(pars[1])
                weights.append(1 / (pars[1] ** 2))
            mean = np.average(asy, weights=weights)
            std = np.std(asy)
            plt.errorbar(list(range(0, len(asy))), asy, yerr=unc)
            plt.plot([0, len(asy)], [mean, mean], 'r-')
            plt.fill_between([0, len(asy)], [mean + std, mean + std], [mean - std, mean - std], color='r', alpha=0.2)
            plt.title('Run: ' + run)
            plt.xlabel('Messungen')
            plt.ylabel('Asymmetry factor')
            plt.show()
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT shape FROM Lines WHERE refRun = ?''', (run,))
            shape = ast.literal_eval(cur.fetchall()[0][0])
            cur.execute('''SELECT fixShape FROM Lines WHERE refRun = ?''', (run,))
            fix = ast.literal_eval(cur.fetchall()[0][0])
            shape['asy'] = mean
            fix['asy'] = True
            cur.execute('''UPDATE Lines SET shape = ? WHERE refRun = ?''', (str(shape), run,))
            cur.execute('''UPDATE Lines SET fixShape = ? WHERE refRun = ?''', (str(fix), run,))
            con.commit()
        con.close()

    def get_files(self, isotope):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT file FROM Files WHERE type LIKE ? ''', (isotope,))
        files = cur.fetchall()
        con.close()
        return [f[0] for f in files]

    def calibrate_all_ref(self):    # Calibrate DAC voltage
        files = self.get_files('60Ni')
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        for f in files:
            cur.execute('''SELECT pars FROM FitRes WHERE file = ?''', (f,))
            pars = cur. fetchall()
            center = []
            errs = []
            weights = []
            for result in pars:
                dic = ast.literal_eval(result[0])
                center.append(dic['center'][0])
                errs.append(dic['center'][1])
                weights.append(1 / (dic['center'][1] ** 2))
            mean = np.average(center, weights=weights)
            acc_volt = self.calibrate(f, mean)
            cur.execute('''UPDATE Files Set accVolt = ? WHERE file = ?''', (acc_volt, f,))
        con.commit()
        con.close()
        self.plot_calibration()

    def calibrate(self, file, delta_frequ):

        # Calculate differential Doppler shift
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt from Files WHERE file = ?''', (file,))
        accVolt = cur.fetchall()[0][0]
        con.close()
        diffDopp60 = Physics.diffDoppler(delta_frequ + self.frequ_60ni, accVolt, 60)
        print('differential doppler shift:', diffDopp60)

        # Calculate calibration
        delta_u = delta_frequ / diffDopp60
        print('Delta voltage:', delta_u)
        return accVolt + delta_u

    def plot_calibration(self):
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
            file_times.append(datetime.strptime(date, '%Y-%m-%d %H:%M:%S'))
            voltages.append(acc_vol)
        self.times = file_times
        self.volts = voltages
        fig, ax = plt.subplots()
        plt.plot_date(file_times, voltages, fmt='-b')
        ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.title('Calibrated Voltage')
        plt.ylabel('Voltage in V')

    def calibrate_all(self, files):
        times = []
        volts = []
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        for f in files:
            cur.execute('''SELECT date FROM Files WHERE file = ?''', (f,))
            time = datetime.strptime(cur.fetchall()[0][0], '%Y-%m-%d %H:%M:%S')
            times.append(time)
            if time <= self.times[0]:
                volts.append(self.volts[0])
                cur.execute('''UPDATE Files SET accVolt = ? WHERE file = ?''', (self.volts[0], f,))
                continue
            for i, t in enumerate(self.times):
                if t <= time and time < self.times[i + 1]:
                    y0, m = curve_fit(self.lin_func, dates.date2num(self.times[i:i+2]), self.volts[i:i+2])[0]
                    new_volt = y0 + m * dates.date2num(time)
                    volts.append(new_volt)
                    cur.execute('''UPDATE Files SET accVolt = ? WHERE file = ?''', (new_volt, f,))
                    break
        con.commit()
        con.close()
        plt.plot_date(times, volts, fmt='rx')
        plt.show()

    def calib_procedure(self, isotope):
        self.calibrate_all_ref()
        self.calibrate_all(niAna.get_files(isotope))
        for run in self.runs60:
            self.fit_all(self.get_files('60Ni'), run)
        self.calibrate_all_ref()
        plt.show()

        # TODO: Reihnfolge
        # TODO: refit nach asym??

    def lin_func(self, x, y0, m):
        return y0 + m * x

    def analyse(self):
        for run in self.runs60:
            self.fit_all(self.get_files('60Ni'), run)
            self.fit_all(self.get_files('58Ni'), run)
            #continue
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        files = self.get_files('60Ni')
        times = []
        centers = []
        errs = []
        for f in files:
            cur.execute('''SELECT pars FROM FitRes WHERE file = ?''', (f,))
            pars = cur.fetchall()
            results = []
            weihgts = []
            for run in pars:
                results.append(ast.literal_eval(run[0])['center'][0])
                weihgts.append(1 / (ast.literal_eval(run[0])['center'][1] ** 2))
            mean = np.average(results, weights=weihgts)
            std = np.std(results)
            centers.append(mean)
            errs.append(std)
            cur.execute('''SELECT date FROM Files WHERE file = ?''', (f,))
            date = datetime.strptime(cur.fetchall()[0][0], '%Y-%m-%d %H:%M:%S')
            times.append(date)
        con.close()
        plt.plot([1, 1], [1, 1])
        plt.show()
        upper = []
        lower = []
        for i, c in enumerate(centers):
            upper.append(c + errs[i])
            lower.append(c - errs[i])
        fig, ax = plt.subplots()
        plt.plot_date(times, centers, fmt = 'bx', linestyle='-', )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        plt.fill_between(times, upper, lower, alpha=0.2)
        plt.title('Center frequency after calibration')
        plt.ylabel('Frequency in MHz')
        plt.show()



working_dir = 'D:\\Owncloud\\User\\Laura\\Nickelauswertung' # Path Laptop
#working_dir = 'C:\\Users\\Laura Renth\\ownCloud\\User\\Laura\\Nickelauswertung' # Path IKP
db = 'Nickel_BECOLA_60Ni.sqlite'

niAna = NiAnalysis(working_dir, db)
niAna.reset()
niAna.prepFileList()
niAna.initialize()
niAna.asymmetry()
niAna.calib_procedure('58Ni')
niAna.analyse()


