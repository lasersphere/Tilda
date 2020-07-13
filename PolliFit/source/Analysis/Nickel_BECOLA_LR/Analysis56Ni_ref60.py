import sqlite3
import numpy as np
import Physics
import os
import ast
from Measurement.XMLImporter import XMLImporter
import BatchFit


class NiAnalysis:

    def __init__(self, working_dir, db56, line_vars, runs, frequ_60ni, ref_groups, cal_groups):
        self.working_dir = working_dir
        self.db = os.path.join(self.working_dir, db56)
        self.data_path = os.path.join(self.working_dir, 'data')
        self.lineVar = line_vars
        self.runs = runs
        self.frequ_60ni = frequ_60ni
        self.ref_groups = ref_groups
        self.cal_groups = cal_groups

    def reset(self):
        # reset calibration
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Files SET accVolt = ?''', (29850,))
        con.commit()
        con.close()

    def prep(self):
        files = []
        for tup in self.ref_groups:
            files.append('BECOLA_' + str(tup[0]) + '.xml')
        print('Fitting files', files)
        for run in self.runs:
            self.fit_all(files, run)
        # Calibrate on absolute transition frequency of 60Ni
        self.calibrate_all()
        print('-------------------- Calibration done')
        for run in self.runs:
            self.fit_all(files, run)
        self.assign_cal()
        print('-------------------- Assignement done')
        for run in self.runs:
            for tup in self.ref_groups:
                file = 'BECOLA_' + str(tup[1]) + '.xml'
                print('Start fit')
                self.fit_all([file],run)
            for tup in self.cal_groups:
                for f in tup[1:]:
                    file = 'BECOLA_' + str(f) + '.xml'
                    print('Fitting file', f)
                    self.fit_all([file],run)

    def fit_all(self, files, run):
        for f in files:
            print('Fitting File', f)
            for tup in self.cal_groups:
                for element in tup[1:]:
                    if 'BECOLA_' + str(element) + '.xml' == f:
                        file60 = 'BECOLA_' + str(tup[0]) + '.xml'
                        con = sqlite3.connect(self.db)
                        cur = con.cursor()
                        cur.execute('''SELECT pars from FitRes WHERE file = ? AND run = ?''', (file60, run))
                        fitpars = ast.literal_eval(cur.fetchall()[0][0])
                        asy = fitpars['asy'][0]
                        cur.execute('''SELECT shape FROM Lines WHERE refRun = ?''', (run,))
                        setpars = ast.literal_eval(cur.fetchall()[0][0])
                        setpars['asy'] = asy
                        print('The Asymmetry Factor of file', file60 , ' and run', run, 'is', asy)
                        cur.execute('''UPDATE Lines SET shape = ?''', (str(setpars),))
                        cur.execute('''SELECT fixShape FROM Lines WHERE refRun = ?''', (run,))
                        fixShape = ast.literal_eval(cur.fetchall()[0][0])
                        fixShape['asy'] = True
                        cur.execute('''UPDATE Lines SET fixShape = ?''',(str(fixShape),))
                        con.commit()
                        con.close()
            for tup in self.ref_groups:
                if 'BECOLA_' + str(tup[1]) + '.xml' == f:
                    file60 = 'BECOLA_' + str(tup[0]) + '.xml'
                    con = sqlite3.connect(self.db)
                    cur = con.cursor()
                    cur.execute('''SELECT pars from FitRes WHERE file = ? AND run = ?''', (file60, run))
                    pars = ast.literal_eval(cur.fetchall()[0][0])
                    asy = pars['asy'][0]
                    cur.execute('''SELECT shape FROM Lines WHERE refRun = ?''', (run,))
                    setpars = ast.literal_eval(cur.fetchall()[0][0])
                    setpars['asy'] = asy
                    cur.execute('''UPDATE Lines SET shape = ? WHERE refRun = ?''', (str(setpars), run,))
                    cur.execute('''SELECT fixShape FROM Lines WHERE refRun = ?''', (run,))
                    fixShape = ast.literal_eval(cur.fetchall()[0][0])
                    fixShape['asy'] = True
                    cur.execute('''UPDATE Lines SET fixShape = ? WHERE refRun = ?''',(str(fixShape), run,))
                    con.commit()
                    con.close()
                    print('The Asymmetry Factor of file', file60, ' and run', run, 'is', asy)

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
            cur.execute('''SELECT frequency FROM Lines WHERE refRun = ? ''',(run,))
            frequ = cur.fetchall()[0][0]
            v = Physics.relVelocity(Physics.qe * center_voltage, mass * Physics.u)
            v = -v
            print('relDoppler:', Physics.relDoppler(laserFrequ, v))
            centerFrequ = Physics.relDoppler(laserFrequ, v) - frequ
            print('Dopplershifted Frequ:', centerFrequ)
            center = centerFrequ - 500
            self.adj_center(center, iso)

            # Fit
            print('Run to fit with:',run)
            BatchFit.batchFit(np.array([f]), self.db, run)

            # Reset FixShape
            cur.execute('''SELECT fixShape FROM Lines WHERE refRun = ?''', (run,))
            fixShape = ast.literal_eval(cur.fetchall()[0][0])
            fixShape['asy'] = [0,20]
            cur.execute('''UPDATE Lines SET fixShape = ? WHERE refRun = ?''', ( str(fixShape), run,))
            con.commit()
            con.close()

    def adj_offset(self, offset, run):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT shape FROM Lines WHERE refRun LIKE ? ''', (run,))
        shape_dict = ast.literal_eval(cur.fetchall()[0][0])
        shape_dict['offset'] = offset
        cur.execute('''UPDATE Lines SET shape = ? ''', (str(shape_dict),))
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
            for run in self.runs:
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

    def get_files(self, isotope):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT file FROM Files WHERE type LIKE ? ''', (isotope,))
        files = cur.fetchall()
        con.close()
        return [f[0] for f in files]

    def assign_cal(self):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        for tup in self.ref_groups:
            file60 = 'BECOLA_' + str(tup[0]) + '.xml'
            file58 = 'BECOLA_' + str(tup[1]) + '.xml'
            print(file60)
            cur.execute('''SELECT accVolt FROM Files WHERE file = ?''', (file60,))
            accVolt = cur.fetchall()[0][0]
            cur.execute('''UPDATE Files SET accVolt = ? WHERE file = ?''', (accVolt, file58,))
        for tup in self.cal_groups:
            file60 = 'BECOLA_' + str(tup[0]) + '.xml'
            cur.execute('''SELECT accVolt FROM Files WHERE file = ?''', (file60,))
            accVolt = cur.fetchall()[0][0]
            for file in tup[1:]:
                file56 = 'BECOLA_' + str(file) + '.xml'
                cur.execute('''UPDATE Files SET accVolt = ? WHERE file = ?''', (accVolt, file56,))
        con.commit()
        con.close()

working_dir = 'D:\\Daten\\IKP\\Nickel-Auswertung\\Auswertung'
db = 'Nickel_BECOLA_60Ni.sqlite'
line_vars = ['58_0','58_1','58_2']
runs = ['AsymVoigt0', 'AsymVoigt1', 'AsymVoigt2']
frequ_60ni = 850344183
reference_groups = [(6192, 6191), (6208, 6207), (6243, 6242), (6254, 6253)]
calibration_groups = [(6192, 6202, 6203, 6204), (6207,6211, 6213, 6214), (6242, 6238, 6239, 6240), (6253, 6251, 6252)]
niAna = NiAnalysis(working_dir, db, line_vars, runs, frequ_60ni, reference_groups, calibration_groups)
niAna.reset()
niAna.prep()