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

    def __init__(self, working_dir, db56, line_vars, runs60, runs55, frequ_60ni, ref_groups, cal_groups):
        self.working_dir = working_dir
        self.db = os.path.join(self.working_dir, db56)
        self.data_path = os.path.join(self.working_dir, 'data')
        self.lineVar = line_vars
        self.runs55 = runs55
        self.runs60 = runs60
        self.frequ_60ni = frequ_60ni
        self.laserFreq55 = 851264686.7203143
        self.ref_groups = ref_groups
        self.cal_groups = cal_groups

    def reset(self):
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
        plt.errorbar([6363, 6396, 6419, 6463, 6466, 6502], center, yerr=uncert)
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
        plt.errorbar([6363, 6396, 6419, 6463, 6466, 6502], center, yerr=uncert)
        plt.plot([6363, 6502], [mean, mean], 'r')
        plt.plot([6363, 6502], [self.frequ_60ni, self.frequ_60ni], 'g')
        plt.title('Center calibrated')
        plt.show()
        print('calibrated centers:', center)

        #Assign calibration
        self.assign_cal()

        files55 = self.get_files('55Ni')
        self.stack_files(files55)

    def fit_all(self, files, run):
        for f in files:
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
                mass = 55
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

            # Reset FixShape
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
        # Assign calibration to 55Ni files
        for tup in self.cal_groups:
            print('Tuple to assigne to :', tup[1])
            for file in tup[1]:
                print('File:', file)
                file55 = 'BECOLA_' + str(file) + '.xml'
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

                # Update 55Ni voltage to calibration
                con = sqlite3.connect(self.db)
                cur = con.cursor()
                cur.execute('''UPDATE Files SET accVolt = ? WHERE file LIKE ?''', (cal_volt, file55,))
                con.commit()
                con.close()

                print('Voltage updated for file', file55)

    def ana_55(self):
        for run in self.runs55:
            self.fit_stacked(run, sym=False)

    def stack_files(self, files):
        scalers = [0, 1, 2]

        # prepare arrays, one list for each scaler
        bin = 3
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
            t_min = (t0 - t_width) / 100
            t_max = (t0 + t_width) / 100
            # iterate through files and sum up
            volcts = [] # Will be a list of tuples: (DAC, cts, scans, bg)
            for f in files:
                # spectrum only in the specified time gate
                spec = XMLImporter(path=self.working_dir + '\\data\\' + str(f),
                                   softw_gates=[[-350, 0, t_min, t_max], [-350, 0, t_min, t_max],
                                                [-350, 0, t_min, t_max]])
                # spectrum of background
                bg = XMLImporter(path=self.working_dir + '\\data\\' + str(f),
                                 softw_gates=[[-350, 0, 0.5, 4], [-350, 0, 0.5, 4],
                                              [-350, 0, 0.5, 4]])
                # normalization of background to number of bins

                norm_factor = 2 * t_width / 100 / 3.5
                for i, c in enumerate(bg.cts[0][s]):
                    bg.cts[0][s][i] = c * norm_factor
                # plot uncalibrated spectrum
                plt.plot(spec.x[0], spec.cts[0][s])
                plt.plot(bg.x[0], bg.cts[0][s]-100)

                # use calibration
                for j, x in enumerate(spec.x[0]):
                    con = sqlite3.connect(self.db)
                    cur = con.cursor()
                    cur.execute('''SELECT accVolt from Files WHERE file  = ?''', (f,))
                    accV = cur.fetchall()[0][0]
                    con.close()
                    offset = accV - 29850
                    volcts.append((x - offset, spec.cts[0][s][j], spec.nrScans[0], bg.cts[0][s][j]))

            plt.title('Uncalibrated, Scaler ' + str(s))
            plt.show()

            # create binned voltage list
            v = np.arange(-267, -30, bin)
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

            zInd = np.where(sc < 1500)
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
                sumc_norm.append(int(cts / sumb[i] * np.mean(sumb)))

            plt.plot(v, sumc_norm, 'b-')
            plt.title('Calibrated, summed and normalized. Scaler' + str(s))
            plt.show()

            # assign normalized
            voltage[s] = v
            sumcts[s] = sumc_norm
            sumbg[s] = sumb
            scans[s] = sc
            err[s] = unc

        print('Voltage:', voltage)
        print('SUMCTs', sumcts)
        # prepare scaler array for xml-file
        scaler_array = []
        for s in scalers:
            timestep = 0
            for i, c in enumerate(sumcts[s]):
                scaler_array.append((s, int((voltage[s][i]-voltage[s][0]) / bin), int((voltage[s][i]-voltage[s][0]) / bin), c))
                timestep += int((voltage[s][i]-voltage[s][0]) / bin)
        print('ScalerArray', scaler_array)

        # Create dictionary for xml export
        file_creation_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        header_dict = {'type': 'trs',
                        'isotope': '55Ni',
                        'isotopeStartTime': file_creation_time,
                        'accVolt': 29850,
                        'laserFreq': Physics.wavenumber(self.laserFreq55) / 2,
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
                                'softwGates': [[-252, -42, 0, timestep], [-252, -42, 0, timestep], [-252, -42, 0, timestep]],
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
        xml.write(self.working_dir + '\\data\\BECOLA_Stacked55.xml')

        # Add to database
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''INSERT OR IGNORE INTO Files (file, filePath, date, type) VALUES (?, ?, ?, ?)''',
                    ('BECOLA_Stacked55.xml', 'data\BECOLA_Stacked55.xml', file_creation_time, '55Ni' + '_sum'))
        con.commit()
        cur.execute(
            '''UPDATE Files SET offset = ?, accVolt = ?,  laserFreq = ?, laserFreq_d = ?, colDirTrue = ?, 
            voltDivRatio = ?, lineMult = ?, lineOffset = ?, errDateInS = ? WHERE file = ? ''',
            ('[0]', 29850, self.laserFreq55, 0, True, str({'accVolt': 1.0, 'offset': 1.0}), 1, 0,
             1, 'BECOLA_Stacked55.xml'))
        con.commit()
        con.close()

        stacked = XMLImporter(path=self.working_dir + '\\data\\' + 'BECOLA_Stacked55.xml')
        print(stacked.x[0])

    def fit_stacked(self, run, sym=True):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT fixShape from Lines WHERE refRun = ? ''', (run,))
        shape = cur.fetchall()
        shape_dict = ast.literal_eval(shape[0][0])
        con.close()
        if sym:
            shape_dict['asy'] = True
        else:
            shape_dict['asy'] = False
        print(shape_dict['asy'])
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''UPDATE Lines SET fixShape = ? WHERE refRun = ?''', (str(shape_dict), run))
        con.commit()
        con.close()


        BatchFit.batchFit(['BECOLA_Stacked55.xml'], self.db, run, x_as_voltage=True,
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
        print('A relation =', str(au / al))

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

        # Plot counts and fit
        plt.plot(time, sum_t_proj, 'b.')
        plt.title('Scaler' + str(scaler))
        plt.plot(time, self.gauss(time, a, sigma, center, offset), 'r-')
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
        a, sigma, center, offset = curve_fit(self.gauss, time, cts, start_par, bounds=param_bounds)[0]
        return a, sigma, center, offset

    def gauss(self, t, a, s , t0, o):
        # prams:    t: time
        #           a: cts
        #           s: sigma
        #           t0: mid of time
        #           o: offset
        return o + a / np.sqrt(2 * np.pi * s ** 2) * np.exp(-1 / 2 * np.square((t - t0) / s))

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


working_dir = 'D:\\Daten\\IKP\\Nickel-Auswertung\\Auswertung'
db = 'Nickel_BECOLA_60Ni-55Ni.sqlite'
line_vars = ['58_0','58_1','58_2']
runs60 = ['AsymVoigt0', 'AsymVoigt1', 'AsymVoigt2']
runs55 = ['AsymVoigt55_0', 'AsymVoigt55_1', 'AsymVoigt55_2', 'AsymVoigt55_All']
#runs55 = ['sidePVoigt55_0', 'sidePVoigt55_1', 'sidePVoigt55_2', 'sidePVoigt55_All']
frequ_60ni = 850344183
reference_groups = [(6363, 6362), (6396, 6395), (6419, 6417), (6463, 6462), (6466, 6467), (6502, 6501)]
calibration_groups = [((6363, 6396), (6369, 6373, 6370, 6375, 6376, 6377, 6378, 6380, 6382, 6383, 6384, 6387, 6391,
                                      6392, 6393)),
                      ((6396, 6419), (6399, 6400, 6401, 6402, 6404, 6405, 6406, 6408, 6410, 6411, 6412)),
                      ((6419, 6463), (6428, 6429, 6430, 6431, 6432, 6433, 6434, 6436, 6438, 6440, 6441, 6444, 6445,
                                      6447, 6448)),
                      ((6466, 6502), (6468, 6470, 6471, 6472, 6473, 6478, 6479, 6480, 6493))]
niAna = NiAnalysis(working_dir, db, line_vars, runs60, runs55, frequ_60ni, reference_groups, calibration_groups)
niAna.reset()
niAna.prep()
niAna.ana_55()
files55 = niAna.get_files('60Ni')
center55, sigma55 = niAna.center_ref(files55)
print('Reference center is', center55, '+/-', sigma55)

# TODO find start parameters