import simulation_beam_overlap_tools as tools
import Physics
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
startTime = datetime.now()

''' Settings '''
rest_frame_transition_frequency = 1319748572  # in MHz
a_ik = 5.669E7  # Einstein-Coeff. in 1/s
relative_laser_scan_range = (-500, 500)  # in MHz
laser_scan_range_step_size = 1  # in MHz

ions_mean_velocity = 822214
ion_mean_velocity_sigma = 16
sample_number = 5

file_name = 'ions_linearVxVy0.5sigma_0.5x_1y_n300k.txt'

# tools.create_ion_file(file_name, 300000, 0, 0.5, 0, 1, ions_mean_velocity, ion_mean_velocity_sigma)

# Generate Space Grid
x_coord_matrix, y_coord_matrix, x_space, y_space = tools.generate_coord_matrix(20, 20, 0.1)

# Generate Laser Intensity and Vector Matrix in Grid
acol_laser_int_matrix = tools.generate_laser_int_matrix_gaussian(x_coord_matrix, y_coord_matrix, 1, 0.6, 0, 0.5, 0.5)  # Matrix for laser intensity (relativ to saturation parameter S = I/I_0
acol_laser_vector_matrix = tools.generate_laser_vector_matrix(x_coord_matrix, y_coord_matrix, [0, np.tan(0.000385), -1])  # Matrix for laser pointing vecotr ( (0, 0, -1) = acol, Standard)

col_laser_int_matrix = tools.generate_laser_int_matrix_gaussian(x_coord_matrix, y_coord_matrix, 1, 0, 0, 0.5, 0.5)  # Matrix for laser intensity (relativ to saturation parameter S = I/I_0
col_laser_vector_matrix = tools.generate_laser_vector_matrix(x_coord_matrix, y_coord_matrix, [0, 0, 1])  # Matrix for laser pointing vecotr ( (0, 0, -1) = acol, Standard)

# Read Ion File and get lists of:
# - position of Ion in Grid
# - Ion Velocity Value
# - Ion velocity vector
# - laser intensities at ion position (from laser matrices above)
# - laser vectors at ion position (from laser matrices above)
ion_pos_x_indices, ion_pos_y_indices, ion_pos_in_grid_x, ion_pos_in_grid_y, ion_vel, ion_vel_vectors, acol_int_at_ion_pos, acol_vector_at_ion_pos, col_int_at_ion_pos, col_vector_at_ion_pos = tools.readIonList(file_name, x_space, y_space, acol_laser_int_matrix, acol_laser_vector_matrix, col_laser_int_matrix, col_laser_vector_matrix)

# Generate Ion Number Matrix
ion_number_matrix = np.zeros((len(y_space), len(x_space)))
np.add.at(ion_number_matrix, tuple(np.vstack((ion_pos_y_indices, ion_pos_x_indices))), 1)

# Calculate Ion-Laser-Angles
acol_ion_laser_angle = tools.calculate_ion_laser_angle(ion_vel_vectors, acol_vector_at_ion_pos)
col_ion_laser_angle = tools.calculate_ion_laser_angle(ion_vel_vectors, col_vector_at_ion_pos)

# calculate effective Ion-Velocity (cos(alpha)*v0)
acol_ion_vel_eff = tools.calculate_ion_vel_eff_matrix(acol_ion_laser_angle, ion_vel)
col_ion_vel_eff = tools.calculate_ion_vel_eff_matrix(col_ion_laser_angle, ion_vel)

# calculate ion resonance frequency in laboratory frame
acol_ion_res_freq_lab_frame = tools.calculate_ion_res_freq_lab_frame_matrix(acol_ion_vel_eff, rest_frame_transition_frequency)
col_ion_res_freq_lab_frame = tools.calculate_ion_res_freq_lab_frame_matrix(col_ion_vel_eff, rest_frame_transition_frequency)

# generate laser frequency scan array
acol_laser_scan_range = tools.generate_laser_scan_range(acol_ion_res_freq_lab_frame, relative_laser_scan_range, laser_scan_range_step_size)
col_laser_scan_range = tools.generate_laser_scan_range(col_ion_res_freq_lab_frame, relative_laser_scan_range, laser_scan_range_step_size)

# calculate scattering rate tupple for each laser frequency
acol_scatter_results = tools.calculate_scattering_rate(a_ik, acol_int_at_ion_pos, acol_ion_res_freq_lab_frame, acol_laser_scan_range)/len(acol_ion_res_freq_lab_frame)
col_scatter_results = tools.calculate_scattering_rate(a_ik, col_int_at_ion_pos, col_ion_res_freq_lab_frame, col_laser_scan_range)/len(col_ion_res_freq_lab_frame)

# Start Fitting
acol_popt, acol_pcov = curve_fit(tools.Voigt, acol_laser_scan_range, acol_scatter_results, p0=[acol_scatter_results.max(), acol_laser_scan_range.mean(), 8, 80, 0])

acol_perr = np.sqrt(np.diag(acol_pcov))
# amp = popt[0]
# amp_d = perr[0]
acol_x0 = acol_popt[1]
acol_x0_d = acol_perr[1]
# gam = popt[2]
# gam_d = perr[2]
# sig_d = perr[3]
# offset_d = perr[4]
print("Center: {} MHz, Uncert: {} MHz".format(acol_x0, acol_x0_d))

col_popt, col_pcov = curve_fit(tools.Voigt, col_laser_scan_range, col_scatter_results, p0=[col_scatter_results.max(), col_laser_scan_range.mean(), 8, 80, 0])

col_perr = np.sqrt(np.diag(col_pcov))
# amp = popt[0]
# amp_d = perr[0]
col_x0 = col_popt[1]
col_x0_d = col_perr[1]
# gam = popt[2]
# gam_d = perr[2]
# sig_d = perr[3]
# offset_d = perr[4]
print("Center: {} MHz, Uncert: {} MHz".format(col_x0, col_x0_d))
col_acol_rest_frame = (col_x0*acol_x0)**0.5
col_acol_rest_frame_d = ((0.5/(col_x0*acol_x0)**0.5*col_x0*acol_x0_d)**2 + (0.5/(col_x0*acol_x0)**0.5*acol_x0*col_x0_d)**2)**0.5
dif = rest_frame_transition_frequency - col_acol_rest_frame
dif_d = col_acol_rest_frame_d
print("Col-Acol-rest-frame: {} +- {} MHz".format(col_acol_rest_frame, col_acol_rest_frame_d))
print("Real Rest-Frame: {} MHz".format(rest_frame_transition_frequency))
print("Dif: {}".format(dif))


# Generate Ion Effective Mean Velocity Matrix (for Plotting)
acol_ion_mean_vel_matrix = np.zeros((len(y_space), len(x_space)))
np.add.at(acol_ion_mean_vel_matrix, tuple(np.vstack((ion_pos_y_indices, ion_pos_x_indices))), acol_ion_vel_eff)
acol_ion_mean_vel_matrix[ion_pos_y_indices, ion_pos_x_indices] = acol_ion_mean_vel_matrix[ion_pos_y_indices, ion_pos_x_indices] / ion_number_matrix[ion_pos_y_indices, ion_pos_x_indices]

col_ion_mean_vel_matrix = np.zeros((len(y_space), len(x_space)))
np.add.at(col_ion_mean_vel_matrix, tuple(np.vstack((ion_pos_y_indices, ion_pos_x_indices))), col_ion_vel_eff)
col_ion_mean_vel_matrix[ion_pos_y_indices, ion_pos_x_indices] = col_ion_mean_vel_matrix[ion_pos_y_indices, ion_pos_x_indices] / ion_number_matrix[ion_pos_y_indices, ion_pos_x_indices]


# Plotting
acol_fit_vals = tools.Voigt(acol_laser_scan_range, *acol_popt)
acol_residuals = (acol_fit_vals - acol_scatter_results)
# tools.plotFit(acol_laser_scan_range, acol_scatter_results, acol_fit_vals, acol_residuals)
col_fit_vals = tools.Voigt(col_laser_scan_range, *col_popt)
col_residuals = (col_fit_vals - col_scatter_results)
# tools.plotFit(col_laser_scan_range, col_scatter_results, col_fit_vals, col_residuals)

acol_level = np.linspace(np.min(acol_ion_vel_eff), np.max(acol_ion_vel_eff), num=15)
col_level = np.linspace(np.min(col_ion_vel_eff), np.max(col_ion_vel_eff), num=15)

spectra_max = np.max((acol_scatter_results, col_scatter_results))*1.05

tools.plotBeamProfils(x_coord_matrix, y_coord_matrix, acol_laser_int_matrix, col_laser_int_matrix, ion_number_matrix,
                      acol_ion_mean_vel_matrix, col_ion_mean_vel_matrix, acol_laser_scan_range, acol_scatter_results,
                      acol_fit_vals, acol_residuals, col_laser_scan_range, col_scatter_results, col_fit_vals, col_residuals,
                      dif, dif_d, acol_level, col_level, spectra_max=spectra_max)



# print(ion_number_matrix)



#print(ion_mean_vel_matrix)
#np.divide.at(ion_mean_vel_matrix, tuple(np.vstack((ion_pos_y_indices, ion_pos_x_indices))), ion_number_matrix[ion_pos_y_indices, ion_pos_x_indices])
# ion_mean_vel_matrix[ion_pos_y_indices, ion_pos_x_indices] = ion_mean_vel_matrix[ion_pos_y_indices, ion_pos_x_indices] / ion_number_matrix[ion_pos_y_indices, ion_pos_x_indices]
# print(ion_number_matrix[ion_pos_y_indices, ion_pos_x_indices])
#print(ion_mean_vel_matrix)
# plt.contourf(x_coord_matrix, y_coord_matrix, ion_number_matrix)
# plt.show()




# ion_mean_vel_matrix[ion_pos_y_indices, ion_pos_x_indices] += ion_vel
# print(ion_mean_vel_matrix)
#ion_mean_vel_matrix[ion_pos_y_indices, ion_pos_x_indices] = ion_mean_vel_matrix[ion_pos_y_indices, ion_pos_x_indices] / ion_number_matrix[ion_pos_y_indices, ion_pos_x_indices]
# print(ion_mean_vel_matrix)

# ion_n_matrix = tools.generate_ion_n_matrix(x_coord_matrix, y_coord_matrix, 1, 0, 0, 1, 1)  # Matrix for ion number
#
# ion_vel_matrix = tools.generate_ion_velocity_matrix(x_coord_matrix, y_coord_matrix, {'name': 'Gauss', 'v0': ions_mean_velocity, 'sigma': ion_mean_velocity_sigma}, sample_number)# {'name': 'Static', 'v0': ions_mean_velocity})#{'name': 'Gauss', 'v0': 822214, 'sigma': 1000})  # Matrix for ion velocity distr. (skalar)
# ion_vel_vector_matrix = tools.generate_ion_vel_vector_matrix(x_coord_matrix, y_coord_matrix)  # Matrix for ion velocity vector
# print(ion_vel_matrix)
# calculate Ion-Laser Angle
# acol_ion_laser_angle_matrix = tools.calculate_ion_laser_angle_matrix(ion_vel_vector_matrix, acol_laser_vector_matrix)
# col_ion_laser_angle_matrix = tools.calculate_ion_laser_angle_matrix(ion_vel_vector_matrix, col_laser_vector_matrix)
#
# # calculate effective Ion-Velocity (cos(alpha)*v0)
# acol_ion_vel_eff_matrix = tools.calculate_ion_vel_eff_matrix(acol_ion_laser_angle_matrix, ion_vel_matrix)
# col_ion_vel_eff_matrix = tools.calculate_ion_vel_eff_matrix(col_ion_laser_angle_matrix, ion_vel_matrix)
#
# # calculate ion resonance frequency in laboratory frame
# acol_ion_res_freq_lab_frame_matrix = tools.calculate_ion_res_freq_lab_frame_matrix(acol_ion_vel_eff_matrix, rest_frame_transition_frequency)
# col_ion_res_freq_lab_frame_matrix = tools.calculate_ion_res_freq_lab_frame_matrix(col_ion_vel_eff_matrix, rest_frame_transition_frequency)
#
# # generate laser frequency scan tupple
#
# acol_laser_scan_range = tools.generate_laser_scan_range(rest_frame_transition_frequency, ions_mean_velocity, relative_laser_scan_range, laser_scan_range_step_size, colDir=False)
# col_laser_scan_range = tools.generate_laser_scan_range(rest_frame_transition_frequency, ions_mean_velocity, relative_laser_scan_range, laser_scan_range_step_size, colDir=True)
#
#
# # calculate scattering rate tupple for each laser frequency
# acol_scatter_results = tools.calculate_scattering_rate(a_ik, acol_laser_int_matrix, ion_n_matrix, acol_ion_res_freq_lab_frame_matrix, acol_laser_scan_range)/sample_number
# col_scatter_results = tools.calculate_scattering_rate(a_ik, col_laser_int_matrix, ion_n_matrix, col_ion_res_freq_lab_frame_matrix, col_laser_scan_range)/sample_number
#
# # Start Fitting
# acol_popt, acol_pcov = curve_fit(tools.Voigt, acol_laser_scan_range, acol_scatter_results, p0=[acol_scatter_results.max(), acol_laser_scan_range.mean(), 8, 80, 0])
#
# acol_perr = np.sqrt(np.diag(acol_pcov))
# # amp = popt[0]
# # amp_d = perr[0]
# acol_x0 = acol_popt[1]
# acol_x0_d = acol_perr[1]
# # gam = popt[2]
# # gam_d = perr[2]
# # sig_d = perr[3]
# # offset_d = perr[4]
# print("Center: {} MHz, Uncert: {} MHz".format(acol_x0, acol_x0_d))
#
# col_popt, col_pcov = curve_fit(tools.Voigt, col_laser_scan_range, col_scatter_results, p0=[col_scatter_results.max(), col_laser_scan_range.mean(), 8, 80, 0])
#
# col_perr = np.sqrt(np.diag(col_pcov))
# # amp = popt[0]
# # amp_d = perr[0]
# col_x0 = col_popt[1]
# col_x0_d = col_perr[1]
# # gam = popt[2]
# # gam_d = perr[2]
# # sig_d = perr[3]
# # offset_d = perr[4]
# print("Center: {} MHz, Uncert: {} MHz".format(col_x0, col_x0_d))
# col_acol_rest_frame = (col_x0*acol_x0)**0.5
# col_acol_rest_frame_d = ((0.5/(col_x0*acol_x0)**0.5*col_x0*acol_x0_d)**2 + (0.5/(col_x0*acol_x0)**0.5*acol_x0*col_x0_d)**2)**0.5
# dif = rest_frame_transition_frequency - col_acol_rest_frame
# dif_d = col_acol_rest_frame_d
# print("Col-Acol-rest-frame: {} +- {} MHz".format(col_acol_rest_frame, col_acol_rest_frame_d))
# print("Real Rest-Frame: {} MHz".format(rest_frame_transition_frequency))
# print("Dif: {}".format(dif))
#
# # Plotting
# acol_fit_vals = tools.Voigt(acol_laser_scan_range, *acol_popt)
# acol_residuals = (acol_fit_vals - acol_scatter_results)
# # tools.plotFit(acol_laser_scan_range, acol_scatter_results, acol_fit_vals, acol_residuals)
# col_fit_vals = tools.Voigt(col_laser_scan_range, *col_popt)
# col_residuals = (col_fit_vals - col_scatter_results)
# # tools.plotFit(col_laser_scan_range, col_scatter_results, col_fit_vals, col_residuals)
#
# tools.plotBeamProfils(x_coord_matrix, y_coord_matrix, acol_laser_int_matrix, col_laser_int_matrix, ion_n_matrix, np.mean(acol_ion_vel_eff_matrix, axis=2), np.mean(col_ion_vel_eff_matrix, axis=2), acol_laser_scan_range, acol_scatter_results, acol_fit_vals, acol_residuals, col_laser_scan_range, col_scatter_results, col_fit_vals, col_residuals, dif, dif_d)

# plt.contourf(x_coord_matrix, y_coord_matrix, np.mean(acol_ion_vel_eff_matrix, axis=2))
# plt.show()


# # print(ion_res_freq_lab_frame_matrix)
# #print(laser_scan_range)
# # print("----")
# #print(scatter_results)
# # plotting/fitting etc.
# #print(scatter_results)
print("Execution Time: {} s".format(datetime.now() - startTime))
# plt.plot(laser_scan_range, scatter_results)
# plt.show()
# print(ion_vel_eff_matrix[0][0])

# print(coord_matrix)
# print('---')
# print(laser_int_matrix)
#
# plt.contourf(x_coord_matrix, y_coord_matrix, laser_int_matrix)
# plt.show()
# plt.contourf(ion_n_matrix)
# plt.show()

# print(np.dot((2, 3), (1, 3)))
#
# ion_vel_matrix = np.zeros(shape=(3, 3,))
# ion_vel_matrix[:,:] += 2
# print(ion_vel_matrix)
#
# print(np.sum(ion_vel_vector_matrix, axis=-1))

# t1 = np.full((2, 2, 3), 2)
# t2 = np.full((5,), 5)
#
# print(t1)
# print(t1.shape)
# print(t2)
# print(t2.shape)
#
# t3 = np.expand_dims(t1, axis=3)
# print(t3)
# print(t3.shape)
# t4 = np.expand_dims(t2, axis=(0, 1, 2))
# print(t4)
# print(t4.shape)
#
# t5= t3-t4
# print(t5[0][0])
#
# t1 = np.array([[[1, 2, 3], [-1, -2, -3]], [[3, 2, 1], [-1, -2, -3]]])
# print(np.sum(t1, axis=(0, 1)))

