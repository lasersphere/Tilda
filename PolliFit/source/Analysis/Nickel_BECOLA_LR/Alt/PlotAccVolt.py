import sqlite3
import os
from datetime import datetime
import matplotlib.dates as dates
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit


class calibVolt():

    def __init__(self, working_dir, db):
        self.working_dir = working_dir
        self.db = os.path.join(working_dir, db)
        self.files_60 = self.get_files('60Ni')
        self.times = self.get_time(self.files_60)
        self.volts = self.get_volts(self.files_60)
        new_times = [self.times[0]]
        new_volts = [self.volts[0]]
        print(self.times)
        for i, time in enumerate(self.times):
            for j, t in enumerate(new_times):
                if t == time:
                    break
                elif time > t:
                    new_times.insert(j, time)
                    new_volts.insert(j, self.volts[i])
                    break
                elif time < t:
                    new_times.insert(j+1, time)
                    new_volts.insert(j+1, self.volts[i])
                    break
        self.times = new_times
        self.volts = new_volts
        self.files_55 = self.get_files('55Ni')
        self.times_55 = self.get_time(self.files_55)
        self.volts_55 = self.get_volts(self.files_55)
        self.file_t_v_55 = []
        for i, f in enumerate(self.files_55):
            self.file_t_v_55.append((f, self.times_55[i], self.volts_55[i]))
        self.plot()
        self.calib()
        #self.volts_55 = []
        #for f in self.file_t_v_55:
            #self.volts_55.append(f[2])
        #self.plot()
        plt.show()


    def get_files(self, iso):
        con = sqlite3.connect(self.db)
        cur = con.cursor()
        cur.execute('''SELECT file FROM Files WHERE type = ?''', (iso,))
        pars = cur.fetchall()
        con.close()
        return [f[0] for f in pars]

    def get_time(self, files):
        times = []
        for f in files:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT date FROM Files WHERE file = ?''', (f,))
            pars = cur.fetchall()[0][0]
            con.close()
            time = datetime.strptime(pars, "%Y-%m-%d %H:%M:%S")
            times.append(time)
        return dates.date2num(times)

    def get_volts(self, files):
        volts = []
        for f in files:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''SELECT accVolt FROM Files WHERE file = ?''', (f,))
            pars = cur.fetchall()[0][0]
            con.close()
            volts.append(pars)
        return volts

    def plot(self):
        fig, ax = plt.subplots()
        plt.plot_date(self.times, self.volts, fmt='-b')
        ax.xaxis.set_major_formatter(dates.DateFormatter('%Y-%m-%d %H:%M:%S'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        #plt.plot_date(self.times_55, self.volts_55, fmt='r.')

    def lin_func(self, x, y0, m):
        return y0 + m * x

    def calib(self):
        print(self.file_t_v_55)
        for i, file in enumerate(self.file_t_v_55):
            if file[1] <= self.times[4]:
                print(file[1])
                y0, m = curve_fit(self.lin_func, self.times[-2:], self.volts[-2:])[0]
                self.file_t_v_55[i] = (self.file_t_v_55[i][0], self.file_t_v_55[i][1], y0 + m * file[1])
                self.file_t_v_55[i] = (file[0], file[1], y0 + m * file[1])
                plt.plot_date(file[1], self.file_t_v_55[i][2], fmt='y.')
            elif file[1] <= self.times[3]:
                y0, m = curve_fit(self.lin_func, self.times[-3:-1], self.volts[-3:-1])[0]
                self.file_t_v_55[i] = (self.file_t_v_55[i][0], self.file_t_v_55[i][1], y0 + m * file[1])
                self.file_t_v_55[i] = (file[0], file[1], y0 + m * file[1])
                plt.plot_date(file[1], self.file_t_v_55[i][2], fmt='y.')
            elif file[1] <= self.times[2]:
                y0, m = curve_fit(self.lin_func, self.times[-4:-2], self.volts[-4:-2])[0]
                self.file_t_v_55[i] = (self.file_t_v_55[i][0], self.file_t_v_55[i][1], y0 + m * file[1])
                self.file_t_v_55[i] = (file[0], file[1], y0 + m * file[1])
                plt.plot_date(file[1], self.file_t_v_55[i][2], fmt='y.')
            elif file[1] <= self.times[1]:
                y0, m = curve_fit(self.lin_func, self.times[-5:-3], self.volts[-5:-3])[0]
                self.file_t_v_55[i] = (self.file_t_v_55[i][0], self.file_t_v_55[i][1], y0 + m * file[1])
                plt.plot_date(file[1], file[2], fmt='r.')
                self.file_t_v_55[i] = (file[0], file[1], y0 + m * file[1])
                plt.plot_date(file[1], self.file_t_v_55[i][2], fmt='y.')
            elif file[1] <= self.times[0]:
                y0, m = curve_fit(self.lin_func, self.times[0:2], self.volts[0:2])[0]
                self.file_t_v_55[i] = (self.file_t_v_55[i][0], self.file_t_v_55[i][1], y0 + m * file[1])
                self.file_t_v_55[i] = (file[0], file[1], y0 + m * file[1])
                plt.plot_date(file[1], self.file_t_v_55[i][2], fmt='y.')
        for f in self.file_t_v_55:
            con = sqlite3.connect(self.db)
            cur = con.cursor()
            cur.execute('''UPDATE Files SET accVolt = ? WHERE file = ?''', (f[2], f[0]))
            con.commit()
            con.close()

working_dir = 'D:\\Daten\\IKP\\Nickel-Auswertung\\Auswertung'
db = 'Nickel_BECOLA_60Ni-55Ni.sqlite'
calibVolt(working_dir, db)
