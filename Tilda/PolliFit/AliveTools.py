"""
Created on

@author: kkoenig


"""
import ast
import sqlite3
import datetime

from Tilda.PolliFit import Physics


def get_list_all_from_db(dbpath):
    """
    this will connect to the database and read all times, numbers and isotopes. Returns a list with all filenames and a
    list with the filenames of only HV measurements. Both list are sorted by time
    :param dbpath: str, path to .slite db
    :return: listAll(time, number, isotope, filename), listHV(time, number, isotope, filename)
    """
    con = sqlite3.connect(dbpath)
    cur = con.cursor()
    cur.execute('SELECT file, date, type FROM Files')
    _list = cur.fetchall()
    list_all = []

    for file in _list:
        chosen_files = file[0]
        i = chosen_files.find('_run') + 4
        f = chosen_files[i:].find('.') + i
        name_number = int(chosen_files[i:f])
        date_time = file[1]
        list_all = list_all + [{'date': date_time, 'number': name_number, 'type': file[2], 'filename': file[0]}]
        if any(v is None for k, v in list_all[-1].items()):
            print('Incomplete file information in database for file {}'.format(file))
            list_all = list_all[:-1]
    list_all.sort(key=lambda x: datetime.datetime.strptime(x['date'], '%Y-%m-%d %H:%M:%S'))
    return list_all


def find_ref_files(file, all_files, ref_str):
    """
    this will search for the previous and the next reference measurement for a given HV measurement file.
    :param file: HV measurement file.
    :param all_files: A list of all files.
    :param ref_str: The reference str.
    :return: ref_files.
    """
    i = 0
    j = 0
    ref_files = []
    for element in all_files:
        if element == file:
            i = 1

        if ref_str in element['type']:
            if i == 0:
                ref_files = [element]
            if i == 1:
                if j == 0:
                    ref_files = ref_files + [element]
                    j = 1
    return ref_files


def get_laserFreq_from_db(dbpath, chosenFiles):
    """
    this will connect to the database and read the laserFrequency for the chosenFiles
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :return: float, laserfrequency
    """
    if len(chosenFiles) > 0:
        file = chosenFiles  # [0]
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        cur.execute('''SELECT laserFreq, laserFreq_d FROM Files WHERE file = ?''', (file,))
        laserfreq = cur.fetchall()
        if laserfreq:
            laserfrequency = laserfreq[0][0]
            laserfrequency_d = laserfreq[0][1]
            return [laserfrequency, laserfrequency_d]
        else:
            return 0.0


def get_isotopeType_from_db(dbpath, chosenFiles):
    """
    this will connect to the database and read the isotopeType for the chosenFiles
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :return: str, isotopeType
    """
    if len(chosenFiles) > 0:
        file = chosenFiles  # [0]
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        cur.execute('''SELECT type FROM Files WHERE file = ?''', (file,))
        isoType = cur.fetchall()
        if isoType:
            isotopeType = isoType[0][0]
            return isotopeType
        else:
            return ''


def get_transitionFreq_from_db(dbpath, chosenFiles, run):
    """
    this will connect to the database and read the transitionFrequency for the chosenFiles
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :param run: str, run
    :return: float, transitionFrequency
    """
    if len(chosenFiles) > 0:
        type = get_isotopeType_from_db(dbpath, chosenFiles)
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        cur.execute('''SELECT frequency FROM Lines WHERE reference = ? AND refRun =?''', (type, run))
        freq = cur.fetchall()
        if freq:
            transitionFreq = freq[0][0]
            return transitionFreq
        else:
            raise Exception('error, transition frequency not found in Lines reference:%s, refRun: %s' % (type, run))
    else:
        return 0.0


def get_voltDivRatio_from_db(dbpath, chosenFiles):
    """
    this will connect to the database and read the voltage divider ratios for the chosenFiles
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :return: float, [offsetRatio,accVoltRatio]
    """
    if len(chosenFiles) > 0:
        file = chosenFiles  # [0]
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        cur.execute('''SELECT voltDivRatio FROM Files WHERE file = ?''', (file,))
        ratio = cur.fetchall()
        ratioStr = ratio[0][0]
        ratioDic = ast.literal_eval(ratioStr)
        offsetRatio = ratioDic["offset"]
        accVoltRatio = ratioDic["accVolt"]
        divRatio = [offsetRatio, accVoltRatio]
        return divRatio
    else:
        return 0.0


def get_accVolt_from_db(dbpath, chosenFiles):
    """
    this will connect to the database and read the accVolt for the chosenFiles
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :return: float, accVoltage
    """
    if len(chosenFiles) > 0:
        accVoltage = 0.0
        accVoltRatio = get_voltDivRatio_from_db(dbpath, chosenFiles)[1]
        file = chosenFiles  # [0]
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        cur.execute('''SELECT accVolt FROM Files WHERE file = ?''', (file,))
        accVolt = cur.fetchall()
        if accVolt:
            accVoltage = accVolt[0][0] * accVoltRatio
        return accVoltage
    else:
        return 0.0


def get_nameNumber_and_time(chosenFiles):
    """
    this will read the nameNumber and time for the chosenFiles
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :return: int, nameNumber
    """
    # War für altes Datenaufnahmesystem geschrieben. Nummer auslesen ist angepasst.
    # Zeit nicht, da automatischer Vergleich eh noch nicht implementiert
    if len(chosenFiles) > 0:
        fileType = chosenFiles[-4:]

        if fileType == '.dat':
            ind = chosenFiles.find('_')
            nameNumberString = chosenFiles[ind + 10:]
            ind = nameNumberString.find('_')
            nameNumber = int(nameNumberString[:ind])
            date = chosenFiles[:10]
            date = date.replace('-', '.')
            time = chosenFiles[11:19]
            time = time.replace('-', ':')
            dateTime = date + ' ' + time
        elif fileType == '.xml':
            ind = chosenFiles.find('.xml')
            nameNumber = int(chosenFiles[ind - 3:ind])
            dateTime = str(0)
        numberAndTime = [nameNumber, dateTime]

        return numberAndTime
    else:
        return 0.0


def get_offsetVolt_from_db(dbpath, chosenFiles):
    """
    this will connect to the database and read the offsetVolt for the chosenFiles
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :return: float, offsetVoltage
    """
    if len(chosenFiles) > 0:
        offsetVoltage = 0.0
        offsetVoltRatio = get_voltDivRatio_from_db(dbpath, chosenFiles)[0]
        file = chosenFiles  # [0]
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        cur.execute('''SELECT offset FROM Files WHERE file = ?''', (file,))
        offsetVolt = cur.fetchall()
        if offsetVolt == [('[0]',)]:
            offsetVolt = [(0.0,)]
        if offsetVolt:
            offsetVoltage = ast.literal_eval(offsetVolt[0][0])[0] * offsetVoltRatio
        return offsetVoltage
    else:
        return 0.00


def get_real_offsetVolt_from_db(dbpath, chosenFiles, measOffset, gain, voltDivRatio):
    """
    this will connect to the database and read the offsetVolt for the chosenFiles. Afterwards, the real offset is
    calculated with the divOffset, gain and divRatio.
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :return: float, offsetVoltage
    """
    if len(chosenFiles) > 0:
        file = chosenFiles  # [0]
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        cur.execute('''SELECT offset FROM Files WHERE file = ?''', (file,))
        offsetVolt = cur.fetchall()
        if offsetVolt == [('[0]',)]:
            offsetVolt = [(0.0,)]
        if offsetVolt:
            print("File: ", file)
            print("meas: ", ast.literal_eval(offsetVolt[0][0])[0])
            print("measOffset: ", measOffset)
            print("gain: ", gain)
            print("voltDivRatio: ", voltDivRatio)
            offsetVoltage = (ast.literal_eval(offsetVolt[0][0])[0] - measOffset) * gain * voltDivRatio
        return offsetVoltage
    else:
        return 0.00


def get_mass_from_db(dbpath, chosenFiles):
    """
    this will connect to the database and read the ion mass for the chosenFiles
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :param run: str, run
    :return: float, mass in m_u
    """
    if len(chosenFiles) > 0:
        ionMass = 0.0
        ionMass_d = 0.0
        type = get_isotopeType_from_db(dbpath, chosenFiles)
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        cur.execute('''SELECT mass FROM Isotopes WHERE iso = ? ''', (type,))
        mass = cur.fetchall()
        if mass:
            ionMass = mass[0][0]
        cur.execute('''SELECT mass_d FROM Isotopes WHERE iso = ? ''', (type,))
        mass_d = cur.fetchall()
        if mass_d:
            ionMass_d = mass_d[0][0]
        IonMass = [ionMass, ionMass_d]
        return IonMass
    else:
        return [0, 0]


def get_center_from_db(dbpath, chosenFiles, run):
    """
    this will connect to the database and read the center and stat. error of the fit for the chosenFiles
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :return: float, [center, stat. error]
    """
    if len(chosenFiles) > 0:
        file = chosenFiles  # [0]
        con = sqlite3.connect(dbpath)
        cur = con.cursor()
        cur.execute('''SELECT pars FROM FitRes WHERE file = ? AND run = ?''', (file, run))
        val = cur.fetchall()
        valStr = val[0][0]
        valDic = ast.literal_eval(valStr)
        center = valDic["center"]
        return center
    else:
        return 0.0


def transformFreqToVolt(dbpath, chosenFiles, run, center):
    """
    this will calculate the applied HV between ion source and optical detection region minus the Kepco voltage from the Doppler shift
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :param run: str, name of run
    :return: float, volt_Laser
    """
    if len(chosenFiles) > 0:
        speedOfLight = Physics.c
        electronCharge = Physics.qe
        atomicMassUnit = Physics.u
        fL = get_laserFreq_from_db(dbpath, chosenFiles)

        mass = get_mass_from_db(dbpath, chosenFiles)[0] * atomicMassUnit
        accVolt = get_accVolt_from_db(dbpath, chosenFiles)
        offsetVolt = get_offsetVolt_from_db(dbpath, chosenFiles)
        f0 = get_transitionFreq_from_db(dbpath, chosenFiles, run)

        # Calculation of Kepco Voltage
        v = Physics.invRelDoppler(fL[0], f0 + center)
        voltTotal = mass * speedOfLight ** 2 * ((1 - (v / speedOfLight) ** 2) ** (-1 / 2) - 1) / electronCharge
        voltKepco = voltTotal - abs(accVolt) - abs(offsetVolt)
        # Calculation of total voltage
        v_Laser = Physics.invRelDoppler(fL[0], f0)
        voltTotal_Laser = mass * speedOfLight ** 2 * (
                (1 - (v_Laser / speedOfLight) ** 2) ** (-1 / 2) - 1) / electronCharge
        volt_Laser = voltTotal_Laser - voltKepco

        return volt_Laser
    else:
        return 0.0


def calculateVoltage(dbpath, chosenFiles, run):
    """
    this will calculate the applied HV between ion source and optical detection region minus the Kepco voltage from the Doppler shift
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :param run: str, name of run
    :return: float, volt_Laser
    """
    center = get_center_from_db(dbpath, chosenFiles, run)
    laserFreq_d = get_laserFreq_from_db(dbpath, chosenFiles)
    # Not implemented yet. But should work by only adding it to next 3 lines

    volt_Laser = transformFreqToVolt(dbpath, chosenFiles, run, center[0])
    volt_Laser_max = transformFreqToVolt(dbpath, chosenFiles, run, center[0] - center[1])
    volt_Laser_min = transformFreqToVolt(dbpath, chosenFiles, run, center[0] + center[1])
    voltages = [volt_Laser, volt_Laser_max, volt_Laser_min]

    return voltages


def changeListFormat(_list):
    list1 = []
    list2 = []

    for data in _list:
        if len(data) == 2:
            list1 = list1 + [data[0]]
            list2 = list2 + [data[1]]
        if len(data) == 1:
            list1 = list1 + [data[0]]
            list2 = list2 + []
    plotdata = [list1, list2]

    return plotdata
