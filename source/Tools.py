'''
Created on 21.05.2014

@author: hammen

The Tools module contains simple helper methods like createDB() to create a new empty database with the
right structure, isoPlot() to plot the spectrum of an isotope and centerplot() to plot acceleration
voltages depending on laser frequencies.

'''
import os
import sqlite3
import sys
import traceback

import matplotlib.pyplot as plt
import numpy as np

import MPLPlotter as plot
import Measurement.MeasLoad as Meas
import Physics as Physics
from DBIsotope import DBIsotope
from Spectra.FullSpec import FullSpec


def isoPlot(db, iso_name, isovar = '', linevar = '', as_freq=True, laserfreq=None,
            col=None, saving=False, show=True, isom_name=None, prec=10000):
    '''plot isotope iso'''
    iso = DBIsotope(db, iso_name, isovar, linevar)
    
    spec = FullSpec(iso)
    
    print(spec.getPars())
    if as_freq:
        plot.plot(spec.toPlot(spec.getPars(), prec=prec))
    else:
        plot.plot(spec.toPlotE(laserfreq, col, spec.getPars()))
        plot.get_current_axes().set_xlabel('Energy [eV]')
    plt.gca().get_lines()[-1].set_label(iso_name)
    plt.legend()
    if isom_name:
        isoPlot(db, isom_name, isovar, linevar, as_freq, laserfreq, col, saving, show)
    else:
        if saving:
            pathParts = str(db).split('/')
            path = ''
            for i in range(0,len(pathParts)-1,1):
                path += pathParts[i] + '/'
            path += 'simulations/'
            plot.save(path + iso_name + '.png')
        if show:
            plot.show()
        else:
            plot.clear()


def centerPlot(db, isoL, linevar = '', width = 1e6):
    '''Plot kinetic energy / eV, under which isotopes in isoL are on resonace depending on laser frequency up to width MHz away'''
    isos = [DBIsotope(db, iso, '', linevar) for iso in isoL]
    
    res = 100
    fx = np.linspace(isos[0].freq - width, isos[0].freq + width, res)
    wnx = Physics.wavenumber(fx)
    y = np.zeros((len(isos), len(fx)))
    for i, iso in enumerate(isos):
        for j, x in enumerate(fx):
            v = Physics.invRelDoppler(x, iso.freq + iso.center)
            y[i][j] = (iso.mass * Physics.u * v**2)/2 / Physics.qe
    
    fig = plt.figure(1, (8, 8))
    fig.patch.set_facecolor('white')
    
    for i, val in enumerate(y):
        plt.plot(wnx, val, '-', label=isoL[i])

    plt.legend()
    plt.xlabel("Laser wavenumber / cm^-1")
    plt.ylabel("Ion energy on resonance / eV")
    plt.axvline(Physics.wavenumber(isos[0].freq), 0, 20000, color = 'k')
    plt.gca().get_xaxis().get_major_formatter().set_useOffset(False)
    plt.show()
    

def crawl(db, crawl = '.', rec = True):
    '''Crawl the path and add all measurement files to the database, recursively if requested'''
    projectPath, dbname = os.path.split(db)
    print("Crawling", projectPath)
    oldPath = os.getcwd()
    
    os.chdir(projectPath)
    _insertFolder(crawl, rec, dbname)
    os.chdir(oldPath)
    
    print("Done")
    

def _insertFolder(path, rec, db):
    (p, d, f) = next(os.walk(path))
        
    if rec:
        for _d in d:
            _insertFolder(os.path.join(p, _d), rec, db)
    
    for _f in f:
        _insertFile(os.path.join(p, _f), db)


def _insertFile(f, db, x_as_voltage=True):
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT (1) FROM Files WHERE file = ?''', (os.path.basename(f),))
    if len(cur.fetchall()) != 0:
        print('Skipped', f, ': already in db.')
        return
    
    if not Meas.check(os.path.splitext(f)[1]):
        print('Skipped', f, ': not importable.')
        return
    
    try:
        cur.execute('''INSERT INTO Files (file, filePath) VALUES (?, ?)''', (os.path.basename(f), f))
        con.commit()
        spec = Meas.load(f, db, True, x_as_voltage)
        spec.export(db)  
    except:
        print("Error working on file", f, ":", sys.exc_info()[1])
        traceback.print_tb(sys.exc_info()[2])
    con.close() 


def _insertIso(db, iso, mass, mass_d, I, center, Al, Bl, Au, Bu, fixedArat,
               fixedBrat, intScale, fixedInt):
    con = sqlite3.connect(db)
    cur = con.cursor()

    cur.execute(
        ''' INSERT INTO Isotopes (iso, mass, mass_d, I, center,
    Al, Bl, Au, Bu, fixedArat,
    fixedBrat, intScale, fixedInt, relInt, m) VALUES (?, ?, ?, ?, ?,  ?, ?, ?, ?, ?,  ?, ?, ?, NULL, NULL)''',
        (iso, mass, mass_d, I, center, Al, Bl, Au, Bu, fixedArat, fixedBrat, intScale, fixedInt)
    )
    con.commit()
    con.close()


def fileList(db, type):
    '''Return a list of files with type'''
    con = sqlite3.connect(db)
    cur = con.cursor()
    
    cur.execute('''SELECT file FROM Files WHERE type = ? ORDER BY date''', (type,))
    files = cur.fetchall()
    
    return [f[0] for f in files]


def createDB(db):
    '''Initialize a new database. Does not alter existing tables.'''
    print('Initializing database:', db)
    con = sqlite3.connect(db)
    
    #Isotopes
    con.execute('''CREATE TABLE IF NOT EXISTS Isotopes (
    iso TEXT PRIMARY KEY  NOT NULL,
    mass FLOAT,
    mass_d FLOAT,
    I FLOAT,
    center FLOAT,
    Al FLOAT DEFAULT 0,
    Bl FLOAT DEFAULT 0,
    Au FLOAT DEFAULT 0,
    Bu FLOAT DEFAULT 0,
    fixedArat BOOL DEFAULT 0,
    fixedBrat BOOL DEFAULT 0,
    intScale DOUBLE DEFAULT 1,
    fixedInt BOOL DEFAULT 0,
    relInt TEXT,
    m TEXT
    )''')
    
    #Lines
    con.execute('''CREATE TABLE IF NOT EXISTS Lines (
    lineVar TEXT PRIMARY KEY  NOT NULL ,
    reference TEXT,
    refRun TEXT,
    frequency FLOAT,
    Jl FLOAT,
    Ju FLOAT,
    shape TEXT,
    fixShape TEXT,
    charge INT,
    FOREIGN KEY (reference) REFERENCES Isotopes (iso),
    FOREIGN KEY (refRun) REFERENCES Runs (run)
    )''')
    
    #Files
    con.execute('''CREATE TABLE IF NOT EXISTS Files (
    file TEXT PRIMARY KEY NOT NULL,
    filePath TEXT UNIQUE NOT NULL,
    date DATE,
    type TEXT,
    line TEXT,
    offset FLOAT,
    accVolt FLOAT,
    laserFreq FLOAT,
    colDirTrue BOOL,
    voltDivRatio TEXT,
    lineMult FLOAT,
    lineOffset FLOAT,
    FOREIGN KEY (type) REFERENCES Isotopes (iso),
    FOREIGN KEY (line) REFERENCES Lines (line)
    )''')
    
    #Runs
    con.execute('''CREATE TABLE IF NOT EXISTS Runs (
    run TEXT PRIMARY KEY NOT NULL,
    lineVar TEXT DEFAULT "",
    isoVar TEXT DEFAULT "",
    scaler TEXT,
    track TEXT,
    softwGates TEXT
    )''')
    try:
        con.execute('''INSERT OR IGNORE INTO Runs VALUES ("Run0", "", "", "[0]", "-1", "")''')
    except Exception as e:
        con.execute('''INSERT OR IGNORE INTO Runs VALUES ("Run0", "", "", "[0]", "-1")''')  # for older db versions

    
    #Fit results
    con.execute('''CREATE TABLE IF NOT EXISTS FitRes (
    file TEXT NOT NULL,
    iso TEXT NOT NULL,
    run TEXT NOT NULL,
    rChi FLOAT,
    pars TEXT,
    PRIMARY KEY (file, iso, run),
    FOREIGN KEY (file) REFERENCES Files (file),
    FOREIGN KEY (run) REFERENCES Runs (run)
    )''')
    
    #Combined results (averaged from fit)
    con.execute('''CREATE TABLE IF NOT EXISTS Combined (
    iso TEXT NOT NULL,
    parname TEXT,
    run TEXT,
    config TEXT DEFAULT "[]",
    final BOOL DEFAULT 0,
    rChi FLOAT,
    val FLOAT,
    statErr FLOAT,
    statErrForm TEXT DEFAULT err,
    systErr FLOAT,
    systErrForm TEXT DEFAULT 0,
    PRIMARY KEY (iso, parname, run),
    FOREIGN KEY (run) REFERENCES Runs (run)
    )''')

    con.close()


def extract_from_combined(runs_list, db, isotopes=None, par='shift', print_extracted=False):
    """
    will extract the results stored in Combined for the given parameter ('shift', 'center' etc.)
    :param runs_list: list, of strings with the name of the runs that should be extracted
    :param isotopes: list, of strings, with the isotopes that should be extracted
    :param par: str, parameter name, that should be extracted
    :return: dict, {'run_name': {'iso_name_1': (run, val, statErr, rChi), ...}}
    """
    result_dict = {}
    if runs_list == -1:
        # select all runs!
        for iso in isotopes:
            connection = sqlite3.connect(db)
            cursor = connection.cursor()
            cursor.execute(
                '''SELECT run, val, statErr, systErr, rChi FROM Combined WHERE iso = ? AND parname = ? ''',
                (iso, par))
            data = cursor.fetchall()
            connection.close()
            if len(data):
                if not result_dict.get(data[0][0], False):
                    result_dict[data[0][0]] = {}
                result_dict[data[0][0]][iso] = list(data[0][i] for i in range(1, 5))
    else:
        for selected_run in runs_list:
            result_dict[selected_run] = {}
            for iso in isotopes:
                connection = sqlite3.connect(db)
                cursor = connection.cursor()
                cursor.execute(
                    '''SELECT val, statErr, systErr, rChi FROM Combined WHERE iso = ? AND run = ? AND parname = ? ''',
                    (iso, selected_run, par))
                data = cursor.fetchall()
                connection.close()
                if len(data):
                    result_dict[selected_run][iso] = data[0]
    if print_extracted:
        for sel_run, run_results_dicts in sorted(result_dict.items()):
            print('--- \t%s\t%s\t ---' % (sel_run, par))
            print('run\tiso\t%s result\tstatErr\tsystErr\trChi' % par)
            for isot, vals in sorted(run_results_dicts.items()):
                print('%s\t%.5f\t%.5f\t%.5f\t%.5f' % (isot, vals[0], vals[1], vals[2], vals[3]))
    return result_dict