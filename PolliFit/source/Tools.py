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
            col=None, saving=False, show=True, isom_name=None, prec=10000, clear=False):
    '''plot isotope iso'''
    iso = DBIsotope(db, iso_name, isovar, linevar)
    
    spec = FullSpec(iso)
    
    print(spec.getPars())
    if as_freq:
        plot.plot(spec.toPlot(spec.getPars(), prec=prec))
        center_str = '%.1f MHz' % iso.center
        center_color = plt.gca().get_lines()[-1].get_color()
        plt.axvline(x=iso.center, color=center_color, linestyle='--', label='%s center: %s' % (iso_name, center_str))

    else:
        plot.plot(spec.toPlotE(laserfreq, col, spec.getPars()))
        plot.get_current_axes().set_xlabel('Energy [eV]')
        # convert center frequency to energy
        freq_center = iso.center + iso.freq
        vel_center = Physics.invRelDoppler(laserfreq, freq_center)  # velocity
        energ_center = (iso.mass * Physics.u * vel_center ** 2) / 2 / Physics.qe
        center_str = '%.1f eV' % energ_center
        center_color = plt.gca().get_lines()[-1].get_color()
        plt.axvline(x=energ_center, color=center_color, linestyle='--', label='%s center: %s' % (iso_name, center_str))

    plt.gca().get_lines()[-2].set_label(iso_name)
    plt.gcf().set_facecolor('w')
    plt.legend()
    if isom_name:
        isoPlot(db, isom_name, isovar, linevar, as_freq, laserfreq, col, saving, show)
    else:
        if saving:
            db_dir = os.path.dirname(db)
            path = os.path.join(db_dir, 'simulations')
            if not os.path.isdir(path):
                os.mkdir(path)
            file_path = os.path.join(path, iso_name + '.png')
            plot.save(file_path)
        if show:
            plot.show()
        if clear:
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
    m TEXT,
    midTof FLOAT,
    fixedAl BOOL DEFAULT 0,
    fixedBl BOOL DEFAULT 0,
    fixedAu BOOL DEFAULT 0,
    fixedBu BOOL DEFAULT 0
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
    softwGates TEXT,
    softwGateWidth FLOAT,
    softwGateDelayList TEXT
    )''')
    # try:
    #     con.execute('''INSERT OR IGNORE INTO Runs VALUES ("Run0", "", "", "[0]", "-1", "")''')
    # except Exception as e:
    #     con.execute('''INSERT OR IGNORE INTO Runs VALUES ("Run0", "", "", "[0]", "-1")''')  # for older db versions

    
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
    add_missing_columns(db)


def add_missing_columns(db):
    """ this will add missing columns to an already existing databases.
     For adding new columns add them to cols """
    cols = {
        'Isotopes': [
            (0, 'iso', 'TEXT', 1, '""', 1),
            (1, 'mass', 'FLOAT', 0, '0', 0),
            (2, 'mass_d', 'FLOAT', 0, '0', 0),
            (3, 'I', 'FLOAT', 0, '0', 0),
            (4, 'center', 'FLOAT', 0, '0', 0),
            (5, 'Al', 'FLOAT', 0, '0', 0),
            (6, 'Bl', 'FLOAT', 0, '0', 0),
            (7, 'Au', 'FLOAT', 0, '0', 0),
            (8, 'Bu', 'FLOAT', 0, '0', 0),
            (9, 'fixedArat', 'BOOL', 0, '0', 0),
            (10, 'fixedBrat', 'BOOL', 0, '0', 0),
            (11, 'intScale', 'DOUBLE', 0, '1', 0),
            (12, 'fixedInt', 'BOOL', 0, '0', 0),
            (13, 'relInt', 'TEXT', 0, '[]', 0),
            (14, 'm', 'TEXT', 0, '""', 0),
            (15, 'midTof', 'FLOAT', 0, '0', 0),
            (16, 'fixedAl', 'BOOL', 0, '0', 0),
            (17, 'fixedBl', 'BOOL', 0, '0', 0),
            (18, 'fixedAu', 'BOOL', 0, '0', 0),
            (19, 'fixedBu', 'BOOL', 0, '0', 0),
        ],
        'Runs': [
            (0, 'run', 'TEXT', 1, '""', 1),
            (1, 'lineVar', 'TEXT', 0, '""', 0),
            (2, 'isoVar', 'TEXT', 0, '""', 0),
            (3, 'scaler', 'TEXT', 0, '[]', 0),
            (4, 'track', 'TEXT', 0, '-1', 0),
            (5, 'softwGates', 'TEXT', 0, '[]', 0),
            (6, 'softwGateWidth', 'FLOAT', 0, '0', 0),
            (7, 'softwGateDelayList', 'TEXT', 0, '[]', 0),
        ]
    }
    for table_name, target_cols in cols.items():
        con = sqlite3.connect(db)
        cur = con.cursor()
        cur.execute(''' PRAGMA TABLE_INFO('%s')''' % table_name)
        exist_cols = cur.fetchall()
        # for each in exist_cols:
        #     print(each, ',')
        cols_name_flat = [each[1] for each in exist_cols]
        print('flat cols of %s : %s ' % (table_name, cols_name_flat))
        for each in target_cols:
            if each[1] not in cols_name_flat:
                print('column %s in table %s was not yet in db, adding now.' % (each[1], table_name))
                cur.execute(''' ALTER TABLE '%s' ADD COLUMN '%s' '%s' DEFAULT '%s' '''
                            % (table_name, each[1], each[2], each[4]))
        con.commit()
        con.close()


def extract_from_combined(runs_list, db, isotopes=None, par='shift', print_extracted=False):
    """
    will extract the results stored in Combined for the given parameter ('shift', 'center' etc.)
    :param runs_list: list, of strings with the name of the runs that should be extracted
    :param isotopes: list, of strings, with the isotopes that should be extracted
    :param par: str, parameter name, that should be extracted
    :return: dict, {'run_name': {'iso_name_1': (run, val, statErr, systErr, rChi), ...}}
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
        for sel_run, run_results_dicts in sorted(result_dict.items()):
            print('--- \t%s\t%s\t ---' % (sel_run, par))
            print('iso\t%s result(statErr)[systErr]\trChi' % par)
            for isot, vals in sorted(run_results_dicts.items()):
                print('%s\t%.3f(%.0f)[%.0f]\t%.2f' % (isot, vals[0], vals[1] * 1000, vals[2] * 1000, vals[3]))
    return result_dict


def extract_from_fitRes(runs_list, db, isotopes=None):
    """
    extract the fit results for an isotope from the database.
    :param runs_list: list of strings with the run names, -1 for all runs
    :param db: str, database path
    :param isotopes: list of str, isotope names of interes
    :return: dict, {run: {filename: (iso, runNum, rChi, fitres_dict), ...}}
    """
    unknown_run_number = -1
    ret_dict = {}
    ret_unsorted_data = []
    if runs_list == -1:
        # select all runs!
        for iso in isotopes:
            connection = sqlite3.connect(db)
            cursor = connection.cursor()
            cursor.execute('''SELECT file, iso, run, rChi, pars FROM FitRes WHERE iso = ? ORDER BY file''', (iso,))
            data = cursor.fetchall()
            connection.close()
            if len(data):
                ret_unsorted_data += data
    else:
        for run in runs_list:
            for iso in isotopes:
                connection = sqlite3.connect(db)
                cursor = connection.cursor()
                cursor.execute(
                    '''SELECT file, iso, run, rChi, pars FROM FitRes WHERE iso = ? AND run = ?  ORDER BY file''',
                    (iso, run))
                data = cursor.fetchall()
                connection.close()
                if len(data):
                    ret_unsorted_data += data
    for each in ret_unsorted_data:
        file_name, iso, run, r_chi_sq, pars_dict = each
        file_name_split = file_name.split('.')[0]
        if 'Run' in file_name_split:
            run_num = int(file_name_split[file_name_split.index('Run') + 3:file_name_split.index('Run') + 6])
            run_str = 'Run%03d_' % run_num
        else:
            run_num = unknown_run_number
            run_str = 'unknownRun%03d_' % abs(run_num)
            unknown_run_number -= 1
        pars_dict = eval(pars_dict)
        if not ret_dict.get(run, False):
            ret_dict[run] = {}  # was not existing yet
        ret_dict[run][file_name] = (iso, run_num, r_chi_sq, pars_dict)
    return ret_dict


def extract_file_as_ascii(db, file, sc, tr, x_in_freq=False, line_var='', save_to='', softw_gates=None):
    file_path = os.path.join(os.path.dirname(db), file)
    if save_to == '':  # automatic determination and store in Ascii_files relative to db
        save_to = os.path.join(os.path.dirname(db), 'Ascii_files', os.path.split(file)[1].split('.')[0] + '.txt')
    meas = Meas.load(file_path, db, raw=not x_in_freq, softw_gates=softw_gates)
    # i want an arith spec for each pmt here:
    arith_spec = [meas.getArithSpec([abs(each)], tr) for each in sc]
    if x_in_freq:
        iso = DBIsotope(db, meas.type, lineVar=line_var)
        if iso is not None:
            for i, e in enumerate(arith_spec[0][0]):  # transfer to frequency

                v = Physics.relVelocity(Physics.qe * e, iso.mass * Physics.u)
                v = -v if meas.col else v

                f = Physics.relDoppler(meas.laserFreq, v) - iso.freq
                arith_spec[0][0][i] = f

    if not os.path.exists(os.path.dirname(save_to)):
        os.mkdir(os.path.dirname(save_to))
    header = create_header_list(meas, sc, tr)
    x_name = 'f /MHz' if x_in_freq else 'dac_volts'
    with open(save_to, 'w') as f:
        for each in header:
            f.write(each + '\n')
        col_name = '%s\t' % x_name
        for each in sc:
            col_name += 'scaler_%s\t' % each
        f.write(col_name + '\n')
        for dac_i, dac_volts in enumerate(arith_spec[0][0]):
            if x_in_freq:
                line = '%.3f\t' % dac_volts
            else:
                line = '%.7f\t' % dac_volts
            for sc_i, each in enumerate(sc):
                line += '%.3f\t' % arith_spec[sc_i][1][dac_i]
            f.write(line + '\n')
    f.close()
    return save_to


def create_header_list(meas, sc, tr):
    from Measurement.XMLImporter import XMLImporter as Xml
    header = []
    header += 'date: %s' % meas.date,
    kepco = meas.type == 'Kepco'  # Kepco and MCP file
    tilda_file = isinstance(meas, Xml)
    if tilda_file:
        kepco = meas.seq_type == 'kepco'
        header += 'track working time: %s' % meas.working_time,
        header += 'isotope: %s' % meas.type,

    if kepco:
        pass
    else:
        header += 'laser frequency: %s' % meas.laserFreq,
        header += 'collinear: %s' % meas.col,
        header += 'scaler: %s' % str(sc),
        header += 'tracks: %s' % tr,
        if tilda_file:
            header += 'software gates [v_min, v_max, t_min, t_max]: %s' % meas.softw_gates,
    header += 'number of scans: %s' % meas.nrScans,
    if tilda_file:
        ret = meas.get_scaler_step_and_bin_num(tr)
        n_of_scalers_tr, n_of_steps_tr, n_of_bins_tr = zip(*ret)
        header += 'number of steps: %s' % str(n_of_steps_tr),
    else:
        header += 'number of steps: %s' % meas.nrSteps,
    if meas.offset_by_dev == {}:
        header += 'offset voltage: %s' % meas.offset,
    else:
        header += 'offset voltage: %s' % meas.offset_by_dev,
    if not kepco:
        header += 'acceleration voltage: %.5f' % meas.accVolt,
    header.append('###### end of header + 1 line column names ######')
    return header


if __name__ == '__main__':
    workdir = 'R:\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'
    db = os.path.join(workdir, 'Ni_workspace.sqlite')
    # save_to = os.path.join(workdir, 'Ascii_files', 'test.txt')
    # # files = ['Ni_April2016_mcp\\58Ni_no_protonTrigger_Run210.mcp',
    # #          'Ni_April2016_mcp\\59Ni_no_protonTrigger_Run113.mcp',
    # #          'Ni_April2016_mcp\\60Ni_no_protonTrigger_Run096.mcp',
    # #          'Ni_April2016_mcp\\61Ni_no_protonTrigger_Run159.mcp',
    # #          'Ni_April2016_mcp\\62Ni_no_protonTrigger_Run145.mcp',
    # #          'Ni_April2016_mcp\\63Ni_no_protonTrigger_Run169.mcp',
    # #          'Ni_April2016_mcp\\64Ni_no_protonTrigger_Run174.mcp',
    # #          'Ni_April2016_mcp\\65Ni_no_protonTrigger_Run181.mcp',
    # #          'Ni_April2016_mcp\\66Ni_no_protonTrigger_Run102.mcp',
    # #          'Ni_April2016_mcp\\67Ni_no_protonTrigger_3Tracks_Run191.mcp',
    # #          'Ni_April2016_mcp\\68Ni_no_protonTrigger_Run135.mcp',
    # #          'Ni_April2016_mcp\\70Ni_protonTrigger_Run248_sum_252_254_259_265.xml'
    # #          ]
    # files = ['Ni_April2016_mcp\\58Ni_no_protonTrigger_Run210.mcp',
    #          'Ni_April2016_mcp\\70Ni_protonTrigger_Run248_sum_252_254_259_265.xml',
    #          'Ni_April2016_mcp\\67Ni_no_protonTrigger_3Tracks_Run191.mcp'
    #          ]
    # for file in files:
    #     extract_file_as_ascii(db, file, [4, 5, 6, 7],
    #                           -1, line_var='tisa_60_asym_wide', x_in_freq=False)
    add_missing_columns(db)
