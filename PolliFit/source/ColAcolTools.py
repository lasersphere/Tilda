"""
Module to convert center freq to volts


"""

import os
import ast
from TildaTools import select_from_db
import Physics
import MPLPlotter as plot
import numpy
import Analyzer


def absolute_frequency(db, fit1, run1, fit2, run2):
    """
    Calculates the absolute transition frequency from a collinear and an anti-collinear measurement.
    Returns a list [absFreq, absFreqError], where absFreqError is the combined gaussian error of laserFreq1_d,
    laserFreq2_d, fit1_d and fit2_d.
    """

    #Fit1 processing
    file1 = select_from_db(db, 'type, laserFreq, colDirTrue, laserFreq_d', 'Files', [['file'], [fit1]])
    #print(fit1)
    isotope1 = select_from_db(db, 'mass, mass_d', 'Isotopes', [['iso'], [file1[0][0]]])
    fitRes1 = select_from_db(db, 'run, pars', 'FitRes', [['file', 'run'], [fit1, run1]])
    lineVar1 = select_from_db(db, 'lineVar', 'Runs', [['run'], [run1]])
    f_run1 = select_from_db(db, 'frequency', 'Lines', [['lineVar'], [lineVar1[0][0]]])

    fitPars1 = ast.literal_eval("".join(fitRes1[0][1]))
    center1 = float(fitPars1['center'][0])
    center1_d = float(fitPars1['center'][1])
    refFreq1 = float(f_run1[0][0])
    relFreq1 = center1 + refFreq1
    relFreq1_d = center1_d
    isoMass1 = float(isotope1[0][0])
    isoMass1_d = float(isotope1[0][1])
    laserFreq1 = float(file1[0][1])
    colDir1 = int(file1[0][2])
    laserFreq1_d = float(file1[0][3])

    velCenter1 = abs(Physics.invRelDoppler(laserFreq1, relFreq1))  # Velocity
    velCenter1_d = ((2*Physics.c*relFreq1/((1+(relFreq1/laserFreq1)**2)*laserFreq1**2))-(2*Physics.c*relFreq1*(-1+(relFreq1/laserFreq1)**2)/((1+(relFreq1/laserFreq1)**2)**2 * laserFreq1**2)))*relFreq1_d
    #energCenter1 = (isoMass1 * Physics.u * velCenter1 ** 2) / 2 / Physics.qe
    energCenter1 = Physics.relEnergy(velCenter1, isoMass1*Physics.u) / Physics.qe
    #energCenter1_d = ((isoMass1 * Physics.u * velCenter1 / Physics.qe * velCenter1_d)**2 + (velCenter1 ** 2 / 2 / Physics.qe * isoMass1_d * Physics.u)**2)**0.5
    energCenter1_d = ((isoMass1 * Physics.u * velCenter1 * velCenter1_d / (
    1 - (velCenter1 / Physics.c) ** 2) ** 1.5 /Physics.qe) ** 2 + (isoMass1_d * Physics.u * Physics.c ** 2 * (
    1 / (1 - (velCenter1 / Physics.c) ** 2) ** 0.5 - 1) / Physics.qe) ** 2) ** 0.5

    #print([energCenter1, energCenter1_d])

    #Fit2 processing

    file2 = select_from_db(db, 'type, laserFreq, colDirTrue, laserFreq_d', 'Files', [['file'], [fit2]])
    #print(fit2)
    isotope2 = select_from_db(db, 'mass, mass_d', 'Isotopes', [['iso'], [file2[0][0]]])
    fitRes2 = select_from_db(db, 'run, pars', 'FitRes', [['file', 'run'], [fit2, run2]])
    lineVar2 = select_from_db(db, 'lineVar', 'Runs', [['run'], [run2]])
    f_run2 = select_from_db(db, 'frequency', 'Lines', [['lineVar'], [lineVar2[0][0]]])

    fitPars2 = ast.literal_eval("".join(fitRes2[0][1]))
    center2 = float(fitPars2['center'][0])
    center2_d = float(fitPars2['center'][1])
    refFreq2 = float(f_run2[0][0])
    relFreq2 = center2 + refFreq2
    relFreq2_d = center2_d
    isoMass2 = float(isotope2[0][0])
    isoMass2_d = float(isotope2[0][1])
    laserFreq2 = float(file2[0][1])
    colDir2 = int(file2[0][2])
    laserFreq2_d = float(file2[0][3])

    velCenter2 = abs(Physics.invRelDoppler(laserFreq2, relFreq2)) #Veolocity
    velCenter2_d = ((2*Physics.c*relFreq2/((1+(relFreq2/laserFreq2)**2)*laserFreq2**2))-(2*Physics.c*relFreq2*(-1+(relFreq2/laserFreq2)**2)/((1+(relFreq2/laserFreq2)**2)**2 * laserFreq2**2)))*relFreq2_d
    #energCenter2 = (isoMass2 * Physics.u * velCenter2 ** 2) / 2 / Physics.qe
    energCenter2 = Physics.relEnergy(velCenter2, isoMass2*Physics.u) / Physics.qe
    #energCenter2_d = ((isoMass2 * Physics.u * velCenter2 / Physics.qe * velCenter2_d) ** 2 + (velCenter2 ** 2 / 2 / Physics.qe * isoMass2_d * Physics.u) ** 2) ** 0.5
    energCenter2_d = ((isoMass2 * Physics.u * velCenter2 * velCenter2_d / (
    1 - (velCenter2 / Physics.c) ** 2) ** 1.5 / Physics.qe) ** 2 + (isoMass2_d * Physics.u * Physics.c ** 2 * (
    1 / (1 - (velCenter2 / Physics.c) ** 2) ** 0.5 - 1) / Physics.qe) ** 2) ** 0.5

    #print([energCenter2, energCenter1_d])

    # Voltage Difference

    voltDif = (energCenter1 - energCenter2)
    # print(voltDif)
    voltDif_d = (energCenter1_d ** 2 + energCenter2_d ** 2) ** 0.5
    diffDoppler = Physics.diffDoppler(refFreq1, (energCenter1+energCenter2)/2, isoMass1, real=True)
    f_corr = voltDif*diffDoppler
    f_corr_d = diffDoppler*voltDif_d
    # print([f_corr, f_corr_d])


    # Photon-Recoil Correction (see Krieger et al. 2017 (ApplPhysB))
    h = 6.626070150 * 10 ** -34  # CODATA17
    f_recoil = ((h * (laserFreq1 * 10**6) ** 2) / (2 * isoMass1 * Physics.u * Physics.c ** 2))*10**-6

    # Absolute Frequency

    if (colDir1 == 0 and colDir2 == 0) or (colDir1 == 1 and colDir2 == 1):
        print("Error in absolute Frequency measurement. A collinear and an anti-collinear measurement file expected.")
        return
    elif colDir1 == 0: #Means fit1 is the acol file
        #print('Fit1 acol')
        absFreq = ((laserFreq1 + f_corr) * laserFreq2) ** 0.5 - f_recoil
        error1 = laserFreq2 * laserFreq1_d / (2 * (laserFreq2 * (laserFreq1 + f_corr)) ** 0.5)
        error2 = (laserFreq1 + f_corr) * laserFreq2_d / (2 * (laserFreq2 * (laserFreq1 + f_corr)) ** 0.5)
        error3 = laserFreq2 * f_corr_d / (2 * (laserFreq2 * (laserFreq1 + f_corr)) ** 0.5)
    else: #Means fit2 is the acol file
        #print('Fit2 acol')
        absFreq = ((laserFreq2 - f_corr) * laserFreq1) ** 0.5 - f_recoil
        error1 = laserFreq1 * laserFreq2_d / (2 * (laserFreq1 * (laserFreq2 - f_corr)) ** 0.5)
        error2 = (laserFreq2 - f_corr) * laserFreq1_d / (2 * (laserFreq1 * (laserFreq2 - f_corr)) ** 0.5)
        error3 = laserFreq1 * f_corr_d / (2 * (laserFreq1 * (laserFreq2 - f_corr)) ** 0.5)


    #Stat. Error Calc
    #Consisting of a gaussian error of laserFreq1_d, laserFreq2_d and f_corr_d

    absFreq_d = (error1 ** 2 + error2 ** 2 + error3 ** 2) ** 0.5

    print(str(absFreq) + ' +- ' + str(absFreq_d))
    return [absFreq, absFreq_d, laserFreq1, laserFreq1_d, laserFreq2, laserFreq2_d, voltDif]


def files_to_csv(db, measList, pathOut):

    mL = []
    fL = []
    fL_d = []
    i=1
    print('Absolute transition frequency results: ')
    file = open(pathOut, 'w')
    file.write('MeasNr, AbsFreq, AbsFreq_d, LaserFreq1, LaserFreq1_d, LaserFreq2, LaserFreq2_d, U1 - U2\n')
    for pair in measList:
        print('Pair0: ' + str(pair[0]) + 'Pair1: ' + str(pair[1]) + ' Pair2: ' + str(pair[2]) + ' Pair3: ' + str(pair[3]))
        result = absolute_frequency(db, pair[0], pair[1], pair[2], pair[3])
        file.write(str(i) + ', ' + str(result[0]) + ', ' + str(result[1]) + ', ' + str(result[2]) + ', '
                   + str(result[3]) + ', ' + str(result[4]) + ', ' + str(result[5]) + ', ' + str(result[6]) + '\n')
        mL.append(i)
        fL.append(result[0])
        fL_d.append(result[1])
        i=i+1

    # Calculate weighted avg + error and std of the avg
    wAvg, wError, chi = Analyzer.weightedAverage(fL, fL_d)
    avgError = numpy.std(fL) / (len(fL))**0.5
    avg = numpy.mean(fL)
    file.write('\n')
    file.write('Avg: ' + str(avg) + '\n')
    file.write('Weighted Avg: ' + str(wAvg) + '\n')
    file.write('Weighted Avg Error: ' + str(wError) + '\n')
    file.write('Error of Mean: ' + str(avgError) + '\n')
    file.close()
    print('Done')
    plot.colAcolPlot(mL, fL, fL_d)
    plot.show()

