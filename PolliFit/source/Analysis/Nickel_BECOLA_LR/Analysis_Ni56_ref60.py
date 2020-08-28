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


class NiAnalysis:

    def __init__(self, working_dir, db56, line_vars, runs60, runs56, frequ_60ni, ref_groups, cal_groups):
        self.working_dir = working_dir
        self.db = os.path.join(self.working_dir, db56)
        self.data_path = os.path.join(self.working_dir, 'data')
        self.lineVar = line_vars
        self.runs56 = runs56
        self.runs60 = runs60
        self.frequ_60ni = frequ_60ni
        self.laserFreq56 = 8.5125386721050508 * 10 ** 8
        self.ref_groups = ref_groups
        self.cal_groups = cal_groups

    def reset(self):
        self.reset_fixShape()
        # reset calibration
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Files SET accVolt = ?''', (29850,))
        con.commit()
        con.close()
        print('---------------------------- Calibration resetted')

    def prep(self):
        # calibration of all files and fit of all reference files

        # get reference files
        files = []
        for tup in self.ref_groups:
            files.append('BECOLA_' + str(tup[0]) + '.xml')
        print('Fitting files', files)

        # fit reference files
        for run in self.runs60:
            self.fit_all(files, run)

        print('-------------------------references fitted for asymmertry factor---------------------------------------')

        self.plot_asy(files)

        print('Fitting files', files)

        # fit reference files
        for run in self.runs60:
            self.fit_all(files, run)

        print('------------------------references fitted with fix asymmetry factor------------------------------------')

        # plot uncalibrated center
        center = []
        uncert = []
        for f in files:
            file_center = []
            weights = []
            for run in self.runs60:
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''SELECT pars FROM FitRes WHERE file = ? AND run = ?''', (f, run,))
                center_pars = ast.literal_eval(cur.fetchall()[0][0])['center']
                file_center.append(center_pars[0] + self.frequ_60ni)
                print('file:', f, ':', center_pars)
                weights.append(1 / (center_pars[1] ** 2))
            mean_file_center = np.average(file_center, weights=weights)
            file_uncert = np.std(file_center)
            center.append(mean_file_center)
            uncert.append(file_uncert)
        plt.errorbar([6192, 6208, 6243, 6254, 6259], center, yerr=uncert)
        plt.title('Center uncalibrated')
        plt.show()
        print('Uncalibrated centers:', center)

        # Calibrate on absolute transition frequency of 60Ni
        self.calibrate_all()
        print('-------------------- Calibration done')

        # refit
        for run in self.runs60:
            self.fit_all(files, run)

        # plot calibrated center
        center = []
        uncert = []
        for f in files:
            file_center = []
            weights = []
            for run in self.runs60:
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''SELECT pars FROM FitRes WHERE file = ? AND run = ?''', (f, run,))
                center_pars = ast.literal_eval(cur.fetchall()[0][0])['center']
                file_center.append(center_pars[0] + self.frequ_60ni)
                print('file:', f, ':', center_pars)
                weights.append(1 / (center_pars[1] ** 2))
            mean_file_center = np.average(file_center, weights=weights)
            file_uncert = np.std(file_center)
            center.append(mean_file_center)
            uncert.append(file_uncert)
        weights = []
        for u in uncert:
            weights.append(1 / (u ** 2))
        mean = np.average(center, weights=weights)
        plt.errorbar([6192, 6208, 6243, 6254, 6259], center, yerr=uncert)
        plt.plot([6192, 6259], [mean, mean], 'r')
        plt.plot([6192, 6259], [self.frequ_60ni, self.frequ_60ni], 'g')
        plt.title('Center calibrated')
        plt.show()
        print('calibrated centers:', center)

        self.calibrate_all()
        print('-------------------- Calibration done')

        # refit
        for run in self.runs60:
            self.fit_all(files, run)

        # plot calibrated center
        center = []
        uncert = []
        for f in files:
            file_center = []
            weights = []
            for run in self.runs60:
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''SELECT pars FROM FitRes WHERE file = ? AND run = ?''', (f, run,))
                center_pars = ast.literal_eval(cur.fetchall()[0][0])['center']
                file_center.append(center_pars[0] + self.frequ_60ni)
                print('file:', f, ':', center_pars)
                weights.append(1 / (center_pars[1] ** 2))
            mean_file_center = np.average(file_center, weights=weights)
            file_uncert = np.std(file_center)
            center.append(mean_file_center)
            uncert.append(file_uncert)
        weights = []
        for u in uncert:
            weights.append(1 / (u ** 2))
        mean = np.average(center, weights=weights)
        plt.errorbar([6192, 6208, 6243, 6254, 6259], center, yerr=uncert)
        plt.plot([6192, 6259], [mean, mean], 'r')
        plt.plot([6192, 6259], [self.frequ_60ni, self.frequ_60ni], 'g')
        plt.title('Center calibrated')
        plt.show()
        print('calibrated centers:', center)

        #Assign calibration
        self.assign_cal()

    def plot_asy(self, files):
        for run in self.runs60:
            asy = []
            err = []
            weight = []
            for f in files:
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''SELECT pars FROM FitRes WHERE file = ? AND run = ?''', (f, run,))
                pars = ast.literal_eval(cur.fetchall()[0][0])
                con.close()
                asy.append(pars['asy'][0])
                err.append(pars['asy'][1])
                weight.append(1 / (pars['asy'][1] ** 2))
            mean = np.average(asy, weights=weight)
            plt.errorbar(list(range(0, len(asy))), asy, yerr=err, fmt='b.')
            plt.plot([0, len(asy)-1], [mean, mean], 'r-')
            plt.title('Asymmetry factors')
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
            print(shape, fix)

    def fit_all(self, files, run):
        for f in files:
            #for tup in self.ref_groups:
                #if 'BECOLA_' + str(tup[0]) + '.xml' == f:
                    #file60 = 'BECOLA_' + str(tup[0]) + '.xml'
                    #print(f)
                    #print(file60)
                    #con = sqlite3.connect(self.db)
                    #cur = con.cursor()
                    #cur.execute('''SELECT pars from FitRes WHERE file = ? AND run = ?''', (file60, run))
                    #pars = ast.literal_eval(cur.fetchall()[0][0])
                    #asy = pars['asy'][0]
                    #cur.execute('''SELECT shape FROM Lines WHERE refRun = ?''', (run,))
                    #setpars = ast.literal_eval(cur.fetchall()[0][0])
                    #setpars['asy'] = asy
                    #cur.execute('''UPDATE Lines SET shape = ? WHERE refRun = ?''', (str(setpars), run,))
                    #cur.execute('''SELECT fixShape FROM Lines WHERE refRun = ?''', (run,))
                    #fixShape = ast.literal_eval(cur.fetchall()[0][0])
                    #fixShape['asy'] = True
                    #cur.execute('''UPDATE Lines SET fixShape = ? WHERE refRun = ?''',(str(fixShape), run,))
                    #con.commit()
                    #con.close()
                    #print('The Asymmetry Factor of file', file60, ' and run', run, 'is', asy)

            # Guess offset
            spec = XMLImporter(path=os.path.join(self.data_path, f))
            offset = (spec.cts[0][0][0] + spec.cts[0][0][-1]) / 2
            self.adj_offset(offset, run)

            # Guess center
            index = list(spec.cts[0][0]).index(max(spec.cts[0][0]))
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            print(f)
            cur.execute('''SELECT accVolt FROM Files WHERE file = ?''', (f,))
            accV = cur.fetchall()[0][0]
            center_voltage = accV - spec.x[0][index]
            print('AccV:', accV, 'DAC:', spec.x[0][index], 'CenterV:', center_voltage)
            cur.execute('''SELECT laserFreq FROM Files WHERE file = ? ''', (f,))
            laserFrequ = cur.fetchall()[0][0]
            print('LaserFreq:', laserFrequ)
            cur.execute('''SELECT type FROM Files WHERE file = ?''', (f,))
            iso = cur.fetchall()[0][0]
            print('Iso:', iso)
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
            print('relDoppler:', Physics.relDoppler(laserFrequ, v))
            centerFrequ = Physics.relDoppler(laserFrequ, v) - frequ
            print('Dopplershifted Frequ:', centerFrequ)
            center = centerFrequ - 500
            self.adj_center(center, iso)

            # Fit
            print('Run to fit with:', run)
            BatchFit.batchFit(np.array([f]), self.db, run)

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
        cur.execute('''UPDATE Isotopes SET center = ? WHERE iso = ?''', (center, iso, ))
        con.commit()
        con.close()
        print('Adjusted Center to', center)

    def calibrate_all(self):
        files = self.get_files('60Ni')
        ref_frequ = 0
        for lv in self.lineVar:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT frequency FROM Lines WHERE lineVar = ?''', (lv,))
            ref_frequ += cur.fetchall()[0][0]
            con.close()
        ref_frequ = ref_frequ / len(self.lineVar)
        abs_frequs = []
        unc = []
        for f in files:
            center = []
            weights = []
            for run in self.runs60:
                print('Run:', run)
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''SELECT pars FROM FitRes WHERE file = ? AND run = ?''', (f, run,))
                centerpars = ast.literal_eval(cur.fetchall()[0][0])['center']
                con.close()
                center.append(centerpars[0])
                print('file:', f, ':', centerpars)
                weights.append(1 / (centerpars[1] ** 2))
            abs_frequ = np.average(center, weights=weights) + ref_frequ
            abs_frequs.append(abs_frequ)
            unc.append(np.std(center))
            accVolt = self.calibrate(f, abs_frequ)

            # write calibrated Voltage to database
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''UPDATE Files SET accVolt = ? WHERE file = ?''', (accVolt, f))
            con.commit()
            con.close()

    def get_files(self, isotope):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT file FROM Files WHERE type LIKE ? ''', (isotope,))
        files = cur.fetchall()
        con.close()
        return [f[0] for f in files]

    def calibrate(self, file, frequ):
        delta_frequ = frequ - self.frequ_60ni

        # Calculate differential Doppler shift
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT accVolt from Files WHERE file = ?''', (file,))
        accVolt = cur.fetchall()[0][0]
        con.close()
        diffDopp60 = Physics.diffDoppler(frequ, accVolt, 60)

        # Calculate calibration
        delta_u = delta_frequ / diffDopp60
        return accVolt + delta_u

    def assign_cal(self):
        # Assign calibration to 56Ni files
        for tup in self.cal_groups:
            print('Tuple to assigne to :', tup[1])
            for file in tup[1]:
                print('File:', file)
                file56 = 'BECOLA_' + str(file) + '.xml'
                file600 = 'BECOLA_' + str(tup[0][0]) + '.xml'
                file601 = 'BECOLA_' + str(tup[0][1]) + '.xml'

                # Query calibrated voltage from 60Ni
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''SELECT accVolt FROM files WHERE file LIKE ?''', (file600,))
                cal_volt = cur.fetchall()[0][0]
                cur.execute('''SELECT accVolt FROM files WHERE file LIKE ?''', (file601,))
                cal_volt = (cal_volt + cur.fetchall()[0][0]) / 2
                con.close()

                # Update 56Ni voltage to calibration
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''UPDATE Files SET accVolt = ? WHERE file LIKE ?''', (cal_volt, file56,))
                con.commit()
                con.close()

                print('Voltage updated for file', file56)

    def gauss(self, t, a, s , t0, o):
        # prams:    t: time
        #           a: cts
        #           s: sigma
        #           t0: mid of time
        #           o: offset
        return o + a / np.sqrt(2 * np.pi * s ** 2) * np.exp(-1 / 2 * np.square((t - t0) / s))

    def lorentz(self, x, a, gam, loc, o):
        lw = gam * 2
        return o + (a * 2 / (math.pi * lw)) * ((lw ** 2 / 4) / ((x - loc) ** 2 + (lw ** 2 / 4)))

    def center_ref(self,files):
        centers = []
        errs = []
        for run in self.runs60:
            for f in files:
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                print(f, run)
                cur.execute('''SELECT pars FROM FitRes WHERE file = ? AND run = ?''', (f, run,))
                center = ast.literal_eval(cur.fetchall()[0][0])
                con.close()
                centers.append(center['center'][0])
                errs.append(center['center'][1])
        weights = []
        for e in errs:
            weights.append(1 / (e ** 2))
        center_average = np.average(centers, weights=weights)
        center_sigma = np.std(centers)
        return center_average, center_sigma

    def plot_results(self, files):
        asy = []
        for f in files:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT pars FROM FitRes WHERE file = ?''', (f,))
            pars = ast.literal_eval(cur.fetchall()[0][0])
            con.close()
            asy.append(pars['asy'])
        print(asy)

    def calc_iso(self, files, ref):
        nmb = list(range(0, 3 * len(files)))
        center = []
        err = []
        weights = []
        for run in self.runs60:
            for f in files:
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''SELECT pars FROM FitRes WHERE file = ? AND run = ?''', (f, run,))
                pars = ast.literal_eval(cur.fetchall()[0][0])
                con.close()
                center.append(pars['center'][0]-ref)
                err.append(pars['center'][1])
                weights.append(1 / (pars['center'][1] ** 2))
        mean = np.average(center, weights=weights)
        std = np.std(center)
        plt.figure(figsize=(1, 1))
        plt.errorbar(nmb, center, yerr=err, fmt='b.')
        plt.plot([0, nmb[-1]], [mean, mean], 'r-')
        plt.fill_between([0, nmb[-1]], mean - std, mean + std, alpha=0.2, linewidth=0, color='r')
        plt.title('Isotope shift 56Ni')
        plt.ylabel('MHz')
        print(mean)
        print(std)
        plt.show()
        print(center)
        print(err)


working_dir = 'D:\\Daten\\IKP\\Nickel-Auswertung\\Auswertung'
db = 'Nickel_BECOLA_60Ni-56Ni.sqlite'
line_vars = ['58_0','58_1','58_2']
runs60 = ['AsymVoigt0', 'AsymVoigt1', 'AsymVoigt2']
runs56 = ['AsymVoigt56_0', 'AsymVoigt56_1', 'AsymVoigt56_2', 'AsymVoigt56_All']
frequ_60ni = 850344183
reference_groups = [(6192,6191), (6208, 6207), (6243, 6242), (6254, 6253), (6259, 6253)]
calibration_groups = [((6192, 6208), (6202, 6203, 6204)), ((6208, 6243), (6211, 6213, 6214)),
                      ((6243, 6254), (6238, 6239, 6240)), ((6254, 6259), (6251, 6252))]
niAna = NiAnalysis(working_dir, db, line_vars, runs60, runs56, frequ_60ni, reference_groups, calibration_groups)
niAna.reset()
niAna.prep()
niAna.assign_cal()
print('-----------------ready-----------------------------')
files56 = niAna.get_files('56Ni')
for run in runs60:
    print('--------------------run', run, '---------------------')
    niAna.fit_all(files56, run)

files60 = niAna.get_files('60Ni')
center60, sigma60 = niAna.center_ref(files60)
print('Reference center is', center60, '+/-', sigma60)
print(files56)
files56.sort()
print(files56)
niAna.calc_iso(files56, center60)
#niAna.plot_results(files56)
