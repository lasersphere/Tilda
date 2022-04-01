import numpy
import Physics
import os
import Analyzer

path = "C:\\Users\\pimgram\\IKP ownCloud\\Projekte\\KOALA\\C4+\\Online-Auswertung\\2022-03-17\\60"
file = "Voigt_bg_Ref_Scaler.txt"
ref_freq = 1319748572.5
ref_freq_d = 2.0

measurements = numpy.loadtxt(os.path.join(path, file), delimiter=',', skiprows=1)
savefile = open(os.path.join(path, file.split('.')[0]+'_refAnalyzed.txt'), 'w')
savefile.write('Meas. Nr, AbsFreq, AbsFreq_d\n')
results =[]
results_d = []
for meas in measurements:
    meas_nr = meas[0]
    meas_ref_laser_freq = meas[3]
    meas_ref_laser_freq_d = meas[4]
    meas_x_laser_freq = meas[5]
    meas_x_laser_freq_d = meas[6]
    meas_delta_Uref_Ux = meas[7]
    meas_delta_Uref_Ux_d = meas[8]
    meas_iso_m = meas[9]
    meas_iso_m_d = meas[10]

    meas_v_ref = abs(Physics.invRelDoppler(meas_ref_laser_freq, ref_freq))
    meas_v_ref_d = (4*Physics.c * meas_ref_laser_freq * ref_freq/(meas_ref_laser_freq**2 + ref_freq**2)**2)*(meas_ref_laser_freq_d**2 * ref_freq_d**2 + ref_freq**2 * meas_ref_laser_freq_d)**0.5
    meas_Ekin_ref = Physics.relEnergy(meas_v_ref, meas_iso_m*Physics.u) / Physics.qe
    meas_Ekin_ref_d = ((meas_v_ref_d**2 * (meas_iso_m*Physics.u)**2 * meas_v_ref**2 / (Physics.qe**2 * (1 - meas_v_ref**2/Physics.c**2)**3)) + (Physics.c**4 * (meas_iso_m_d*Physics.u)**2 / Physics.qe**2 * (-1 + (1/(1- meas_v_ref**2/Physics.c**2)**0.5))**2))**0.5

    Ekin_x = meas_Ekin_ref - meas_delta_Uref_Ux
    Ekin_x_d = (meas_Ekin_ref_d**2 + meas_delta_Uref_Ux_d**2)**0.5

    m = meas_iso_m*Physics.u
    m_d = meas_iso_m_d*Physics.u
    e = Ekin_x*Physics.qe
    e_d = Ekin_x_d*Physics.qe
    v_x = Physics.relVelocity(e, m)
    v_x_d = (Physics.c**10*m**2*(e_d**2*m**2 + e**2*m_d**2) / (e*(Physics.c**2*m + e)**4 * (e+2*Physics.c**2*m)))**0.5
    freq_x_lab_frame = Physics.relDoppler(meas_x_laser_freq, v_x)
    fL = meas_x_laser_freq
    fL_d = meas_x_laser_freq_d
    freq_x_lab_frame_d = ((fL_d**2 * (1 - v_x**2/Physics.c**2) / (1 - v_x/Physics.c)**2) + (v_x_d**2 * (fL*(1 - v_x**2/Physics.c**2)**0.5/(Physics.c*(1-v_x/Physics.c)**2) - fL*v_x /(Physics.c**2 * (1- v_x/Physics.c)*(1 - v_x**2/Physics.c**2)**0.5))**2))**0.5
    savefile.write('{}, {}, {}\n'.format(meas_nr, freq_x_lab_frame, freq_x_lab_frame_d))
    results.append(freq_x_lab_frame)
    results_d.append(freq_x_lab_frame_d)
    print(freq_x_lab_frame, freq_x_lab_frame_d)

wAvg, wError, chi = Analyzer.weightedAverage(results, results_d)
avgError = numpy.std(results) / (len(results))**0.5
avg = numpy.mean(results)

savefile.write('\n')
savefile.write('Mean: {}\n'.format(avg))
savefile.write('Weighted Mean: {}\n'.format(wAvg))
savefile.write('Weighted Mean Error: {}\n'.format(wError))
savefile.write('Error of Mean: {}'.format(avgError))

savefile.close()