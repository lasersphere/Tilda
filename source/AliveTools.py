"""
Created on

@author: kkoenig


"""
import ast
import logging
import os
import sqlite3
from copy import deepcopy
import Physics

import numpy as np





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
        cur.execute('''SELECT laserFreq FROM Files WHERE file = ?''', (file,))
        laserfreq = cur.fetchall()
        if laserfreq:
            laserfrequency = laserfreq[0][0]
            return laserfrequency
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
    if len(chosenFiles)>0:
        type = get_isotopeType_from_db(dbpath,chosenFiles)
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


def get_nameNumber(chosenFiles):
    """
    this will read the nameNumber for the chosenFiles
    :param dbpath: str, path to .slite db
    :param chosenFiles: str, name of files
    :return: int, nameNumber
    """
    if len(chosenFiles) > 0:

        ind=chosenFiles.find('_')
        nameNumberString=chosenFiles[ind+10:]
        ind=nameNumberString.find('_')
        nameNumber=int(nameNumberString[:ind])

        return nameNumber
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
        if offsetVolt:
            offsetVoltage = offsetVolt[0][0] * offsetVoltRatio
        return offsetVoltage
    else:
        return 0.0


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


def get_center_from_db(dbpath, chosenFiles):
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
        cur.execute('''SELECT pars FROM FitRes WHERE file = ?''', (file,))
        val = cur.fetchall()
        valStr = val[0][0]
        valDic = ast.literal_eval(valStr)
        center = valDic["center"]
        return center
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
    if len(chosenFiles) > 0:

        speedOfLight = Physics.c
        electronCharge = Physics.qe
        atomicMassUnit = Physics.u
        fL = get_laserFreq_from_db(dbpath, chosenFiles)

        center = get_center_from_db(dbpath, chosenFiles)
        mass = get_mass_from_db(dbpath, chosenFiles)[0] * atomicMassUnit
        accVolt = get_accVolt_from_db(dbpath, chosenFiles)
        offsetVolt = get_offsetVolt_from_db(dbpath, chosenFiles)
        f0 = get_transitionFreq_from_db(dbpath, chosenFiles, run)

        # Calculation of Kepco Voltage
        v = Physics.invRelDoppler(fL, f0 + center[0])
        voltTotal = mass * speedOfLight ** 2 * ((1 - (v / speedOfLight) ** 2) ** (-1 / 2) - 1) / electronCharge
        voltKepco = voltTotal - abs(accVolt) - abs(offsetVolt)
        # Calculation of total voltage
        v_Laser = Physics.invRelDoppler(fL, f0)
        voltTotal_Laser = mass * speedOfLight ** 2 * (
        (1 - (v_Laser / speedOfLight) ** 2) ** (-1 / 2) - 1) / electronCharge
        volt_Laser = voltTotal_Laser - voltKepco


        return volt_Laser
    else:
        return 0.0
