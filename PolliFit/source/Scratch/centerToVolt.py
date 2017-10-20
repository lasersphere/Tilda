"""
Module to convert center freq to volts


"""

import os
import ast
from TildaTools import select_from_db
import Physics
import numpy as np
import MPLPlotter as mpl

workdir = 'C:\\Users\\pimgram\\IKP ownCloud\\User\\Phillip\\_Barium\\Auswertungen\\'

datafolder = os.path.join(workdir, 'sums')

db = os.path.join(workdir, 'Barium_Data_D1.sqlite')



def absolute_frequency(db, fit1, fit2):
    """
    Calculates the absolute transition frequency from a collinear and an anti-collinear measurement.
    Returns a list [absFreq, absFreqError], where absFreqError is the combined gaussian error of laserFreq1_d,
    laserFreq2_d, fit1_d and fit2_d.
    """

    #Fit1 processing
    file1 = select_from_db(db, 'type, laserFreq, colDirTrue, laserFreq_d', 'Files', [['file'], [fit1]])
    isotope1 = select_from_db(db, 'mass', 'Isotopes', [['iso'], [file1[0][0]]])
    fitRes1 = select_from_db(db, 'run, pars', 'FitRes', [['file'], [fit1]])
    run1 = select_from_db(db, 'frequency', 'Lines', [['refRun'], [fitRes1[0][0]]])

    fitPars1 = ast.literal_eval("".join(fitRes1[0][1]))
    center1 = float(fitPars1['center'][0])
    center1_d = float(fitPars1['center'][1])
    refFreq1 = float(run1[0][0])
    relFreq1 = center1 + refFreq1
    relFreq1_d = center1_d
    isoMass1 = float(isotope1[0][0])
    laserFreq1 = float(file1[0][1])
    colDir1 = int(file1[0][2])
    laserFreq1_d = file1[0][3]

    velCenter1 = abs(Physics.invRelDoppler(laserFreq1, relFreq1)) #Veolocity
    velCenter1_d = ((2*Physics.c*relFreq1/((1+(relFreq1/laserFreq1)**2)*laserFreq1**2))-(2*Physics.c*relFreq1*(-1+(relFreq1/laserFreq1)**2)/((1+(relFreq1/laserFreq1)**2)**2 * laserFreq1**2)))*relFreq1_d
    energCenter1 = (isoMass1 * Physics.u * velCenter1 ** 2) / 2 / Physics.qe
    energCenter1_d = isoMass1 * Physics.u * velCenter1 / Physics.qe * velCenter1_d
    #print([energCenter1, energCenter1_d])

    #Fit2 processing

    file2 = select_from_db(db, 'type, laserFreq, colDirTrue, laserFreq_d', 'Files', [['file'], [fit2]])
    isotope2 = select_from_db(db, 'mass', 'Isotopes', [['iso'], [file2[0][0]]])
    fitRes2 = select_from_db(db, 'run, pars', 'FitRes', [['file'], [fit2]])
    run2 = select_from_db(db, 'frequency', 'Lines', [['refRun'], [fitRes2[0][0]]])

    fitPars2 = ast.literal_eval("".join(fitRes2[0][1]))
    center2 = float(fitPars2['center'][0])
    center2_d = float(fitPars2['center'][1])
    refFreq2 = float(run2[0][0])
    relFreq2 = center2 + refFreq2
    relFreq2_d = center2_d
    isoMass2 = float(isotope2[0][0])
    laserFreq2 = float(file2[0][1])
    colDir2 = int(file2[0][2])
    laserFreq2_d = file2[0][3]

    velCenter2 = abs(Physics.invRelDoppler(laserFreq2, relFreq2)) #Veolocity
    velCenter2_d = ((2*Physics.c*relFreq2/((1+(relFreq2/laserFreq2)**2)*laserFreq2**2))-(2*Physics.c*relFreq2*(-1+(relFreq2/laserFreq2)**2)/((1+(relFreq2/laserFreq2)**2)**2 * laserFreq2**2)))*relFreq2_d
    energCenter2 = (isoMass2 * Physics.u * velCenter2 ** 2) / 2 / Physics.qe
    energCenter2_d = isoMass1 * Physics.u * velCenter2 / Physics.qe * velCenter2_d

    #Voltage Difference

    voltDif = abs(energCenter1 - energCenter2)
    voltDif_d = (energCenter1_d ** 2 + energCenter2_d ** 2) ** 0.5
    diffDoppler = Physics.diffDoppler(refFreq1, energCenter1, isoMass1)
    f_corr = voltDif*diffDoppler
    f_corr_d = diffDoppler*voltDif_d
    #print([f_corr, f_corr_d])

    #Absolute Frequency

    if (colDir1 == 0 and colDir2 == 0) or (colDir1 == 1 and colDir2 == 1):
        print("Error is absolute Frequency measurement. A collinear and an anti-collinear measurement file expected.")
        return
    elif colDir1 == 0: #Means fit1 is the acol file
        absFreq = ((laserFreq1 + f_corr) * laserFreq2) ** 0.5
        error1 = laserFreq2 * laserFreq1_d / (2 * (laserFreq2 * (laserFreq1 + f_corr)) ** 0.5)
        error2 = (laserFreq1 + f_corr) * laserFreq2_d / (2 * (laserFreq2 * (laserFreq1 + f_corr)) ** 0.5)
        error3 = laserFreq2 * f_corr_d / (2 * (laserFreq2 * (laserFreq1 + f_corr)) ** 0.5)
    else: #Means fit2 is the acol file
        absFreq = ((laserFreq2 + f_corr) * laserFreq1) ** 0.5
        error1 = laserFreq1 * laserFreq2_d / (2 * (laserFreq1 * (laserFreq2 + f_corr)) ** 0.5)
        error2 = (laserFreq2 + f_corr) * laserFreq1_d / (2 * (laserFreq1 * (laserFreq2 + f_corr)) ** 0.5)
        error3 = laserFreq1 * f_corr_d / (2 * (laserFreq1 * (laserFreq2 + f_corr)) ** 0.5)


    #Stat. Error Calc
    #Consisting of a gaussian error of laserFreq1_d, laserFreq2_d and f_corr_d

    print(error1)
    print(error2)
    print(error3)
    absFreq_d = (error1 ** 2 + error2 ** 2 + error3 ** 2) ** 0.5


    return [absFreq, absFreq_d]


def files_to_csv(db, measList, pathOut):

    file = open(pathOut, 'w')
    for pair in measList:
        result = absolute_frequency(db, pair[0], pair[1])
        file.write(str(result[0]) + '; ' + str(result[1]))

    file.close()



#print(absolute_frequency(db, '138Ba_acol_cs_run128.xml', '138Ba_col_cs_run127.xml'))
#p = os.path.join(workdir, 'test.txt')

#print(files_to_csv(db, [['138Ba_acol_cs_run128.xml', '138Ba_col_cs_run127.xml']], p))

#file = open('test\\test.txt', 'w')
#file.write('hello')
#file.close()

#file = open(os.path.join(workdir, 'test\\test.txt'), 'w')
#file.write('hello')
#file.close()



#print(os.path.splitext('asd.txt'))

#dict = {'asf': 'asd'}
#b = dict.get('abc', {})

#print(b)
#print(bool(b))
#c = {}
#a = [[], [], [], []]
#a[1] = a[1] + [1, 2, 3] + c.get('aaa', [])

#print(a)

#a[2] = a[2] + [2, 3, 4]

#print(a)
#print(bool(a))
#print(np.mean(a[1]))

#mpl.colAcolPlot([1, 2, 3],[123.5, 124.6, 122.2], [0.2, 1, 0.9])
#mpl.show()

print(303870538.531*2)
print(0.237944336317*2)