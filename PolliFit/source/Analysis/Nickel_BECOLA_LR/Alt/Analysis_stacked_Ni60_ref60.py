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

    def __init__(self, working_dir, db60, line_vars, runsRef, runs60, frequ_60ni, ref_groups, cal_groups):
        self.working_dir = working_dir
        self.db = os.path.join(self.working_dir, db60)
        self.data_path = os.path.join(self.working_dir, 'data')
        self.lineVar = line_vars
        self.runs60 = runs60
        self.runsRef = runsRef
        self.frequ_60ni = frequ_60ni
        self.laserFreq60 = 851224124.8007469
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

    def reset_fixShape(self):
        # Reset FixShape
        for run in self.runsRef:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT fixShape FROM Lines WHERE refRun = ?''', (run,))
            fixShape = ast.literal_eval(cur.fetchall()[0][0])
            fixShape['asy'] = [0, 20]
            cur.execute('''UPDATE Lines SET fixShape = ? WHERE refRun = ?''', (str(fixShape), run,))
            con.commit()
            con.close()

    def prep(self):
        # calibration of all files and fit of all reference files

        # get reference files
        files = []
        for tup in self.ref_groups:
            files.append('BECOLA_' + str(tup[0]) + '.xml')
        print('Fitting files', files)

        # fit reference files
        for run in self.runsRef:
            self.fit_all(files, run)

        self.plot_asy(files)

        self.assign_asy()

        # fit reference files
        for run in self.runs60:
            self.fit_all(files, run)

        # plot uncalibrated center
        center = []
        uncert = []
        for f in files:
            file_center = []
            weights = []
            for run in self.runsRef:
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
        for run in self.runsRef:
            self.fit_all(files, run)

        # plot calibrated center
        center = []
        uncert = []
        for f in files:
            file_center = []
            weights = []
            for run in self.runsRef:
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
        plt.plot([6192, 6259],[mean, mean], 'r')
        plt.plot([6192, 6259], [self.frequ_60ni, self.frequ_60ni], 'g')
        plt.title('Center calibrated')
        plt.show()
        print('calibrated centers:', center)

        #Assign calibration
        #self.assign_cal()

        files60 = self.get_files('60Ni')
        self.stack_files(files60)

    def plot_asy(self, files):
        for run in self.runsRef:
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

    def assign_asy(self):
        for i,run in enumerate(self.runs60):
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT shape FROM Lines WHERE refRun = ?''', (run,))
            pars56 = ast.literal_eval(cur.fetchall()[0][0])
            cur.execute('''SELECT shape FROM Lines WHERE refRun = ?''', (self.runsRef[i],))
            pars60 = ast.literal_eval(cur.fetchall()[0][0])
            print(pars56)
            pars56['asy'] = pars60['asy']
            print(pars56)
            cur.execute('''UPDATE Lines SET shape = ? WHERE refRun = ?''', (str(pars56), run,))
            cur.execute('''SELECT fixShape FROM Lines WHERE refRun = ?''', (run,))
            fix = ast.literal_eval(cur.fetchall()[0][0])
            fix['asy'] = True
            cur.execute('''UPDATE Lines Set fixShape = ? WHERE refRun = ?''', (str(fix), run,))
            con.commit()
            con.close()

    def fit_all(self, files, run):
        for f in files:
            #for tup in self.ref_groups:
                #if 'BECOLA_' + str(tup[1]) + '.xml' == f:
                    #fileRef = 'BECOLA_' + str(tup[0]) + '.xml'
                    #con = sqlite3.connect(self.db)
                    #cur = con.cursor()
                    #cur.execute('''SELECT pars from FitRes WHERE file = ? AND run = ?''', (fileRef, run))
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
                    #print('The Asymmetry Factor of file', fileRef, ' and run', run, 'is', asy)

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
            for run in self.runsRef:
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
            print('Absolute frequency befor calibration:', abs_frequ)
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
        print('differential doppler shift:', diffDopp60)

        # Calculate calibration
        delta_u = delta_frequ / diffDopp60
        print('Delta voltage:', delta_u)
        return accVolt + delta_u

    def assign_cal(self):
        # Assign calibration to 60Ni files
        for tup in self.cal_groups:
            print('Tuple to assigne to :', tup[1])
            for file in tup[1]:
                print('File:', file)
                file60 = 'BECOLA_' + str(file) + '.xml'
                fileRef0 = 'BECOLA_' + str(tup[0][0]) + '.xml'
                fileRef1 = 'BECOLA_' + str(tup[0][1]) + '.xml'

                # Query calibrated voltage from 60Ni
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''SELECT accVolt FROM files WHERE file LIKE ?''', (fileRef0,))
                cal_volt = cur.fetchall()[0][0]
                cur.execute('''SELECT accVolt FROM files WHERE file LIKE ?''', (fileRef1,))
                cal_volt = (cal_volt + cur.fetchall()[0][0]) / 2
                con.close()

                # Update 60Ni voltage to calibration
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''UPDATE Files SET accVolt = ? WHERE file LIKE ?''', (cal_volt, file60,))
                con.commit()
                con.close()

                print('Voltage updated for file', file60)

    def ana_60(self):
        for run in self.runs60:
            self.fit_stacked(run, sym= True)

    def stack_files(self, files):
        scalers = [0, 1, 2]

        # prepare arrays, one list for each scaler
        bin = 0.996947492252
        voltage = [[], [], []]
        sumcts = [[], [], []]
        sumbg = [[], [], []]
        scans = [[], [], []]
        err = [[], [], []]

        # iterate through scalers
        for s in scalers:
            print('scaler ', s)

            # find time gates
            t0, t_width = self.find_timegates(files, 0, s)
            t_min = (t0 - 2 * t_width) / 100
            t_max = (t0 + 2 * t_width) / 100
            print(t_min, t_max)

            # iterate through files and sum up
            volcts = [] # Will be a list of tuples: (DAC, cts, scans, bg)
            for f in files:
                # spectrum only in the specified time gate
                spec = XMLImporter(path=self.working_dir + '\\data\\' + str(f),
                                   softw_gates=[[-35, 15, t_min, t_max], [-35, 15, t_min, t_max],
                                                [-35, 15, t_min, t_max]])
                # spectrum of background
                bg = XMLImporter(path=self.working_dir + '\\data\\' + str(f),
                                 softw_gates=[[-35, 15, 0.5, 4], [-35, 15, 0.5, 4],
                                              [-35, 15, 0.5, 4]])
                #normalization of background to number of bins
                print('no normalization:', bg.cts[0][s])
                print(t_width)
                norm_factor = 4 * t_width / 100 / 3.5
                print('Normalization factor:', norm_factor)
                for i, c in enumerate(bg.cts[0][s]):
                    bg.cts[0][s][i] = c * norm_factor
                print('After normalization:', bg.cts[0][s])
                # plot uncalibrated spectrum
                plt.plot(spec.x[0], spec.cts[0][s])
                plt.plot(bg.x[0], bg.cts[0][s])

                # use calibration
                for j, x in enumerate(spec.x[0]):
                    con = sqlite3.connect(self.db)
                    cur = con.cursor()
                    cur.execute('''SELECT accVolt from Files WHERE file  = ?''', (f,))
                    accV = cur.fetchall()[0][0]
                    con.close()
                    offset = (accV - 29850)
                    volcts.append((x - offset, spec.cts[0][s][j], spec.nrScans[0], bg.cts[0][s][j]))

            plt.title('Uncalibrated, Scaler ' + str(s))
            plt.show()

            # create binned voltage list
            v = np.arange(-60, 30, bin)
            sumc = np.zeros(len(v)) # for summed counts
            sumb = np.zeros(len(v)) # for summed background
            sc = np.zeros(len(v))   # for summed scans

            # iterate through collected voltage and counts (all files included)
            for tup in volcts:
                for j, item in enumerate(v):    # find correct bin and add to counts, scans, background
                    if item - bin / 2 < tup[0] <= item + bin / 2:
                        sumc[j] += tup[1]
                        sc[j] += tup[2]
                        sumb[j] += tup[3]
            # find indices with zero background and remove from lists
            zInd = np.where(sumb == 0)
            sumc = np.delete(sumc, zInd)
            sumb = np.delete(sumb, zInd)
            v = np.delete(v, zInd)
            sc = np.delete(sc, zInd)

            zInd = np.where(sc < 100)
            sumc = np.delete(sumc, zInd)
            sumb = np.delete(sumb, zInd)
            v = np.delete(v, zInd)
            sc = np.delete(sc, zInd)

            # plot summed and calibrated counts and background
            plt.plot(v, sumc, 'b.')
            plt.plot(v, sumb, 'r.')
            plt.plot(v, sc, 'y.')
            plt.title('Calibrated and summed, Sclaer' + str(s))
            plt.show()

            # calculate statistic uncertainty
            unc = []
            for cts in sumc:
                unc.append(np.sqrt(cts))

            # normalize
            sumc_norm = []
            for i, cts in enumerate(sumc):
                sumc_norm.append(int((cts) / sumb[i] * np.mean(sumb)))

            plt.plot(v, sumc_norm, 'b.')
            plt.title('Calibrated, summed and normalized. Scaler' + str(s))
            plt.show()

            # assign normalized
            voltage[s] = v
            sumcts[s] = sumc_norm
            sumbg[s] = sumb
            scans[s] = sc
            err[s] = unc

        # prepare scaler array for xml-file
        scaler_array = []
        for s in scalers:
            timestep = 0
            for i, c in enumerate(sumcts[s]):
                scaler_array.append((s, i, timestep, c))
                timestep += 1

        # Create dictionary for xml export
        file_creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header_dict = {'type': 'trs',
                        'isotope': '60Ni',
                        'isotopeStartTime': file_creation_time,
                        'accVolt': 29850,
                        'laserFreq': Physics.wavenumber(self.laserFreq60) / 2,
                        'nOfTracks': 1,
                        'version': 99.0}
        track0_dict_header = {'trigger': {},  # Need a trigger dict!
                                'activePmtList': [0, 1, 2],  # Must be in form [0,1,2]
                                'colDirTrue': True,
                                'dacStartRegister18Bit': 0,
                                'dacStartVoltage': voltage[0][0],
                                'dacStepSize18Bit': None,  # old format xml importer checks whether val or None
                                'dacStepsizeVoltage': bin,
                                'dacStopRegister18Bit': len(voltage[0]) - 1,  # not real but should do the trick
                                'dacStopVoltage': voltage[0][-1],
                                'invertScan': False,
                                'nOfBins': len(voltage[0]),
                                'nOfCompletedSteps': float(len(voltage[0])),
                                'nOfScans': 1,
                                'nOfSteps': len(voltage[0]),
                                'postAccOffsetVolt': 0,
                                'postAccOffsetVoltControl': 0,
                                'softwGates': [[-40, -10, 0, timestep], [-40, -10, 0, timestep], [-40, -10, 0, timestep]],
                                #'softwGates': [[-252, -42, 0, 0.4], [-252, -42, 0, 0.4], [-252, -42, 0, 0.4]],
                                # For each Scaler: [DAC_Start_Volt, DAC_Stop_Volt, scaler_delay, softw_Gate_width]
                                'workingTime': [file_creation_time, file_creation_time],
                                'waitAfterReset1us': 0,  # looks like I need those for the importer
                                'waitForKepco1us': 0  # looks like I need this too
                                }
        data = '['
        for i, s in enumerate(scaler_array):
            data = data + str(scaler_array[i]) + ' '
        data = data[:len(data) - 1]
        data = data + ']'
        track0_dict_data = {
            'scalerArray_explanation': 'continously acquired data. List of Lists, each list represents the counts of '
                                       'one scaler as listed in activePmtList.Dimensions are: (len(activePmtList), '
                                       'nOfSteps), datatype: np.int32',
            'scalerArray': data}

        track0_vol_proj = {'voltage_projection': np.array(sumcts),
                           'voltage_projection_explanation': 'voltage_projection of the time resolved data. List of '
                                                             'Lists, each list represents the counts of one scaler as '
                                                             'listed in activePmtList.Dimensions are: '
                                                             '(len(activePmtList), nOfSteps), datatype: np.int32'}

        dictionary = {'header': header_dict,
                      'tracks': {'track0': {'header': track0_dict_header,
                                            'data': track0_dict_data,
                                            'projections': track0_vol_proj
                                            },
                                 }
                      }

        root = ET.Element('BecolaData')

        xmlWriteDict(root, dictionary)
        xml = ET.ElementTree(root)
        xml.write(self.working_dir + '\\data\\BECOLA_Stacked60.xml')

        # Add to database
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Files (file, filePath, date, type) VALUES (?, ?, ?, ?)''',
                    ('BECOLA_Stacked60.xml', 'data\BECOLA_Stacked60.xml', file_creation_time, '60Ni_sum'))
        con.commit()
        cur.execute(
            '''UPDATE Files SET offset = ?, accVolt = ?,  laserFreq = ?, laserFreq_d = ?, colDirTrue = ?, 
            voltDivRatio = ?, lineMult = ?, lineOffset = ?, errDateInS = ? WHERE file = ? ''',
            ('[0]', 29850, self.laserFreq60, 0, True, str({'accVolt': 1.0, 'offset': 1.0}), 1, 0,
             1, 'BECOLA_Stacked60.xml'))
        con.commit()
        con.close()

        stacked = XMLImporter(path=self.working_dir + '\\data\\' + 'BECOLA_Stacked60.xml')

    def fit_stacked(self, run, sym=True):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        print(run)
        cur.execute('''SELECT fixShape from Lines WHERE refRun = ? ''', (run,))
        shape = cur.fetchall()
        shape_dict = ast.literal_eval(shape[0][0])
        con.close()
        if sym:
            shape_dict['asy'] = True
        else:
            shape_dict['asy'] = [0,30]
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Lines SET fixShape = ? WHERE refRun = ?''', (str(shape_dict), run))
        con.commit()
        con.close()


        BatchFit.batchFit(['BECOLA_Stacked60.xml'], self.db, run, x_as_voltage=True,
                          save_file_as='.png')

        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT pars From FitRes WHERE run = ?''', (run,))
        paras = cur.fetchall()
        con.close()
        para_dict = ast.literal_eval(paras[0][0])
        print('Al =', para_dict['Al'], '\nAu =', para_dict['Au'], '\nBl =', para_dict['Bl'], '\nBu =', para_dict['Bu'])
        al = para_dict['Al'][0]
        au = para_dict['Au'][0]
        #print('A relation =', str(au / al))

    def find_timegates(self, file_list, track, scaler):
        # returns the center of the timegate and sigma
        # param:    file_list: list of files to sum up
        #           track: which track is used
        #           scaler: which scaler is used

        sum_t_proj = np.zeros(1024) # prepare the list of counts for each time step

        # iterate through files and sum up the counts for each time step
        for file in file_list:
            sum_t_proj = list(map(add, sum_t_proj, self.t_proj(file, track, scaler)))

        time = list(range(0, len(sum_t_proj)))  # list of time steps (for fitting and plotting)

        # Fit a gaussian function to the time projection
        a, sigma, center, offset = self.fit_time(time, sum_t_proj)
        print(a, sigma, center, offset)

        # Plot counts and fit
        plt.plot(time, sum_t_proj, 'b.')
        plt.title('Scaler' + str(scaler))
        plt.plot(time, self.lorentz(time, a, sigma, center, offset), 'r-')
        plt.axvline(center - 2 * sigma, color='y')
        plt.axvline(center + 2 * sigma, color='y')
        plt.show()
        return center, sigma

    def t_proj(self, file, track, scaler):
        # returns the time projection
        # param:    file: xml-file to get the data of
        #           track: which track is used
        #           scaler: which pmt is used
        spec = XMLImporter(path=self.working_dir + '\\data\\' + file)
        return spec.t_proj[track][scaler]

    def fit_time(self, time, cts):
        # fits a gaussian to the time projection
        # param:    time: list of time steps
        #           cts: list of counts

        # guess Start parameters and set amplitude and sigma positive
        start_par = np.array([max(cts), 10, time[cts.index(max(cts))], (time[0]+time[-1]) / 2])
        param_bounds = ([0, 0, -np.inf, -np.inf], [np.inf, np.inf, np.inf, np.inf])

        # Fitting
        #a, sigma, center, offset = curve_fit(self.gauss, time, cts, start_par, bounds=param_bounds)[0]
        a, gamma, center, offset = curve_fit(self.lorentz, time, cts, start_par, bounds=param_bounds)[0]
        #return a, sigma, center, offset
        return a, gamma, center, offset

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
        for run in self.runsRef:
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

    def calc_iso(self, ref):
        nmb = list(range(0, 3))
        center = []
        err = []
        weights = []
        for run in self.runs60:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT pars FROM FitRes WHERE run = ? AND file = ?''', (run,'BECOLA_Stacked60.xml',))
            pars = ast.literal_eval(cur.fetchall()[0][0])
            con.close()
            print(run)
            print(pars['center'][0])
            center.append(pars['center'][0]-ref)
            err.append(pars['center'][1])
            weights.append(1 / (pars['center'][1] ** 2))
        mean = np.average(center, weights=weights)
        std = np.std(center)
        #plt.figure(figsize=(1, 1))
        print(nmb)
        print(center)
        plt.errorbar(nmb, center, yerr=err, fmt='b.')
        plt.plot([0, nmb[-1]], [mean, mean], 'r-')
        plt.fill_between([0, nmb[-1]], mean - std, mean + std, alpha=0.2, linewidth=0, color='r')
        plt.title('Isotope shift 56Ni')
        plt.ylabel('MHz')
        print('Isotope shift:')
        print(mean)
        print(std)
        plt.show()

def get_asy_fac(list, dir, datab):
    asy = []
    weights = []
    for f in list:
        file = 'BECOLA_' + str(f) + '.xml'
        print(file)
        con = sqlite3.connect(os.path.join(dir, datab))
        cur = con.cursor()
        cur.execute('''SELECT pars FROM FitRes WHERE file = ?''', (file,))
        pars = ast.literal_eval(cur.fetchall()[0][0])
        print(pars)
        asy.append(pars['asy'][0])
        weights.append(1 / (pars['asy'][1] ** 2))
        con.close()
    mean_asy = np.average(asy, weights=weights)
    print('Mean asymmetry factor:', mean_asy)
    return mean_asy

def set_asy(asy, dir, datab):
    con = sqlite3.connect(os.path.join(dir, datab))
    cur = con.cursor()
    cur.execute('''SELECT lineVar FROM Lines''')
    lines = cur.fetchall()
    cur.execute('''SELECT shape FROM Lines''')
    shapes = cur.fetchall()
    cur.execute('''SELECT fixShape FROM Lines''')
    fix_shapes = cur.fetchall()
    con.close()
    print(lines, shapes)
    for i, line in enumerate(lines):
        shape = ast.literal_eval(shapes[i][0])
        shape['asy'] = asy
        fix_shape = ast.literal_eval(fix_shapes[i][0])
        fix_shape['asy'] = True
        con = sqlite3.connect(os.path.join(dir, datab))
        cur = con.cursor()
        cur.execute('''UPDATE Lines SET shape = ? WHERE lineVar = ?''', (str(shape), line[0],))
        cur.execute('''UPDATE Lines SET fixShape = ? WHERE lineVar = ?''', (str(fix_shape), line[0]))
        con.commit()
        con.close()



#working_dir = 'D:\\Daten\\IKP\\Nickel-Auswertung\\Auswertung'
working_dir = 'C:\\Users\\Laura Renth\\ownCloud\\User\\Laura\\Nickelauswertung'
db = 'Nickel_BECOLA_60Ni-60Ni-stacked.sqlite'
line_vars = ['58_0','58_1','58_2']
runsRef = ['AsymVoigt0', 'AsymVoigt1', 'AsymVoigt2']
runs60 = ['AsymVoigt56_0', 'AsymVoigt56_1', 'AsymVoigt56_2']
#runs60 = ['sidePVoigt55_0', 'sidePVoigt55_1', 'sidePVoigt55_2', 'sidePVoigt55_All']
frequ_60ni = 850344183
reference_groups = [(6192,6191), (6208, 6207), (6243, 6242), (6254, 6253), (6259, 6253)]
calibration_groups = [((6192, 6192), (6192), (6208, 6208), (6208)), ((6243, 6243), (6243)), ((6254, 6254), (6254)),
                      ((6259, 6259), (6259))]
niAna = NiAnalysis(working_dir, db, line_vars, runsRef, runs60, frequ_60ni, reference_groups, calibration_groups)
niAna.reset()
niAna.prep()
#asy = get_asy_fac([6192, 6208, 6243, 6254, 6259], working_dir, db)
#set_asy(asy, working_dir, db)
niAna.ana_60()
filesRef = niAna.get_files('60Ni')
centerRef, sigmaRef = niAna.center_ref(filesRef)
print('Reference center is', centerRef, '+/-', sigmaRef)
file60 = niAna.get_files('60Ni_sum')
con = sqlite3.connect(niAna.db)
cur = con.cursor()
cur.execute('''SELECT pars FROM FitRes WHERE file = ?''', (file60[0],))
pars = cur.fetchall()
con.close()
center60 = []
weigths60  = []
for res in pars:
    center_pars = ast.literal_eval(res[0])
    center60.append(center_pars['center'][0])
    weigths60.append(1 / (center_pars['center'][1] ** 2))
mean_center60 = np.average(center60, weights=weigths60)
print(mean_center60, '+/-', np.std(center60))
niAna.calc_iso(centerRef)