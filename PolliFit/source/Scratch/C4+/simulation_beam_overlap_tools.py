import numpy as np
from copy import deepcopy
from matplotlib import pyplot as plt
import Physics


def generate_coord_matrix(width, height, resolution):
    '''

    :param width: Matrix x-axis total width in mm
    :param height: Matrix y-axsis total height in mm
    :param resolution: Matrix step resolution in mm
    '''


    no_of_steps_width = int((width / resolution) + 1) # including 0
    no_of_steps_height = int((height / resolution) + 1)

    # create coord matrix
    x_coord_matrix = np.zeros((no_of_steps_height, no_of_steps_width))  # erstes Argument: Reihen
    y_coord_matrix = np.zeros((no_of_steps_height, no_of_steps_width))

    #create coord-system
    x_space = np.linspace(width/-2, width/2, num=no_of_steps_width)
    y_space = np.linspace(height/2, height/-2, num=no_of_steps_height)

    for r, rows in enumerate(x_coord_matrix):
        for c, col in enumerate(rows):
            x_coord_matrix[r][c] = x_space[c]
            y_coord_matrix[r][c] = y_space[r]

    return x_coord_matrix, y_coord_matrix, x_space, y_space

def generate_laser_int_matrix_gaussian(x_coord_matrix, y_coord_matrix, amp, x0, y0, sigma_x, sigma_y):
    # for y, row in enumerate(coord_system_matrix):
    #     for x, coord in enumerate(row):
    #         laser_int_matrix[y][x] = gaussian_func_2D(coord[0], coord[1], amp, x0, y0, sigma_x, sigma_y)

    return gaussian_func_2D(x_coord_matrix, y_coord_matrix, amp, x0, y0, sigma_x, sigma_y)

def generate_laser_vector_matrix(x_coord_matrix, y_coord_matrix, vector):
    laser_vector_matrix = np.zeros(shape=(x_coord_matrix.shape[0], x_coord_matrix.shape[1], 3))
    laser_vector_matrix[:,:, 0] += vector[0]
    laser_vector_matrix[:,:, 1] += vector[1]
    laser_vector_matrix[:,:, 2] += vector[2]
    # for y, row in enumerate(coord_system_matrix):
    #     for x, coord in enumerate(row):
    #         laser_vector_matrix[y][x] = (0, 0, -1)

    return laser_vector_matrix


def generate_ion_n_matrix(x_coord_matrix, y_coord_matrix, amp, x0, y0, sigma_x, sigma_y):
    # for y, row in enumerate(coord_system_matrix):
    #     for x, coord in enumerate(row):
    #         ion_n_matrix[y][x] = gaussian_func_2D(coord[0], coord[1], amp, x0, y0, sigma_x, sigma_y)

    return gaussian_func_2D(x_coord_matrix, y_coord_matrix, amp, x0, y0, sigma_x, sigma_y)

# def fill_ion_vel_matrix(ion_vel_matrix, coord_system_matrix, distr):
#     for y, row in enumerate(coord_system_matrix):
#         for x, coord in enumerate(row):
#             ion_vel_matrix[y][x] = distr
#
#     return ion_vel_matrix

def generate_ion_vel_vector_matrix(x_coord_matrix, y_coord_matrix):
    ion_vel_vector_matrix = np.zeros(shape=(x_coord_matrix.shape[0], x_coord_matrix.shape[1], 3))
    ion_vel_vector_matrix[:,:, 2] += 1

    return ion_vel_vector_matrix

def generate_ion_velocity_matrix(x_coord_matrix, y_coord_matrix, ion_vel_distr_dict, no_of_samples):
    if ion_vel_distr_dict['name'] == 'Static':
        ion_vel_matrix = np.zeros(shape=(x_coord_matrix.shape[0], x_coord_matrix.shape[1], no_of_samples))
        ion_vel_matrix[:,:,:] += ion_vel_distr_dict['v0']
    elif ion_vel_distr_dict['name'] == 'Gauss':
        ion_vel_matrix = np.random.normal(ion_vel_distr_dict['v0'], ion_vel_distr_dict['sigma'], size=(x_coord_matrix.shape[0], x_coord_matrix.shape[1], no_of_samples))
    else:
        print("Wrong ion_vel_distr_dict! Empty return!")
        ion_vel_matrix = []

    return ion_vel_matrix

    # for y, row in enumerate(ion_vel_distr_matrix):
    #     for x, entry in enumerate(row):
    #         if entry['name'] == 'Static':
    #                 val = np.array([entry['v0']])
    #         elif entry['name'] == 'Gauss':
    #             v0 = entry['v0']
    #             sigma = entry['sigma']
    #             val = np.random.normal(v0, sigma, no_of_samples)
    #         else:
    #             val = np.array(0)
    #         ion_vel_matrix[y][x] = val
    #
    # return ion_vel_matrix


def calculate_ion_laser_angle(ion_vel_vector_matrix, laser_vector_matrix):
    # for y, row in enumerate(ion_vel_vector_matrix):
    #     for x, ion_vel_vector in enumerate(row):
    #         laser_vector = laser_vector_matrix[y][x]
    #         angle_rad, angle_deg = vector_angle(ion_vel_vector, laser_vector)
    #         ion_laser_angle_matrix[y][x] = angle_rad

    return angle(ion_vel_vector_matrix, laser_vector_matrix, axis=1)

def calculate_ion_vel_eff_matrix(ion_laser_angle_matrix, ion_vel_matrix):
    # for y, row in enumerate(ion_vel_matrix):
    #     for x, velocities in enumerate(row):
    #         angle = ion_laser_angle_matrix[y][x]
    #         ion_vel_eff_matrix[y][x] = ion_vel_matrix[y][x] * np.cos(angle)
    # print(ion_vel_matrix.shape)
    # print(ion_laser_angle_matrix.shape)
    return ion_vel_matrix * np.cos(ion_laser_angle_matrix)

def calculate_ion_res_freq_lab_frame_matrix(ion_vel_eff_matrix, rest_frame_frequency):
    # for y, row in enumerate(ion_vel_eff_matrix):
    #     for x, velocities in enumerate(row):
    #         buffer_list = []
    #         for vel in velocities:
    #             buffer_list.append(rel_Doppler_to_lab_frame(rest_frame_frequency, vel))
    #         ion_res_freq_lab_frame_matrix[y][x] = buffer_list

    return rel_Doppler_to_lab_frame(rest_frame_frequency, ion_vel_eff_matrix)

def generate_laser_scan_range(doppler_shiftet_frequencies, scan_range, step_size):
    mean_freq = np.mean(doppler_shiftet_frequencies)

    return np.linspace(mean_freq+scan_range[0], mean_freq+scan_range[1], int((scan_range[1]-scan_range[0])/step_size + 1))

def calculate_scattering_rate(a_ik, laser_intensity, ion_lab_frame_resonance_frequency, laser_scan_frequencies):
    ion_scatter_rate = []

    # print("Laser_intensity_matrix.shape: {}".format(laser_intensity_matrix.shape))
    # print("ion_number_matrix.shape: {}".format(ion_number_matrix.shape))
    # print("ion_lab_frame_resonance_frequency_matrix.shape: {}".format(ion_lab_frame_resonance_frequency_matrix.shape))
    # print("laser_scan_frequencies.shape: {}".format(laser_scan_frequencies.shape))

    delta_matrix = np.expand_dims(ion_lab_frame_resonance_frequency, axis=1) - np.expand_dims(laser_scan_frequencies, axis=(0))
    #print(delta_matrix)
    #print("delta_matrix.shape: {}".format(delta_matrix.shape))

    rate_matrix = scatter_rate(a_ik, np.expand_dims(laser_intensity, axis=(1)), delta_matrix)
    # print(rate_matrix)
    #print("rate_matrix.shape: {}".format(rate_matrix.shape))
    # print(np.sum(rate_matrix, axis=(0, 1, 2)))
    return np.sum(rate_matrix, axis=0)
    # for i, laser_freq in enumerate(laser_scan_frequencies):
    #     print("Calculating laser freq " + str(i+1) + " / " + str(len(laser_scan_frequencies)))
    #     scattering_rate = []
    #     for y, row in enumerate(ion_lab_frame_resonance_frequency_matrix):
    #         for x, resonance_freqs in enumerate(row):
    #             for freqs in resonance_freqs:
    #                 delta = freqs - laser_freq
    #                 rate = ion_number_matrix[y][x]*scatter_rate(a_ik, laser_intensity_matrix[y][x], delta)
    #                 # print(delta, rate, ion_number_matrix[y][x], laser_intensity_matrix[y][x])
    #                 scattering_rate.append(rate)
    #
    #     ion_scatter_rate.append(np.sum(scattering_rate))


''' Help functions '''
def gaussian_func_1D(x, amp, x0, sigma_x):
    return amp*np.exp(-(x-x0)**2/(2*sigma_x**2))

def gaussian_func_2D(x, y, amp, x0, y0, sigma_x, sigma_y):
    return amp*np.exp(- (((x-x0)**2/(2*sigma_x**2)) + ((y -y0)**2/(2*sigma_y**2))))

def angle(x, y, axis=-1):
    """
    :param x: The first vectors (arb. units).
    :param y: The second vectors ([x]).
    :param axis: The axis along which the vector components are aligned.
    :returns: The angle between two vectors x and y (rad).
    :raises ValueError: The shapes of x and y must be compatible.
    """
    x, y = np.asarray(x), np.asarray(y)
    return np.arccos(np.sum(x * y, axis=axis) / np.sqrt(np.sum(x ** 2, axis=axis) * np.sum(y ** 2, axis=axis)))


def rel_Doppler_to_lab_frame(rest_frame_freq, velocity):
    beta = velocity / Physics.c
    gamma = (1 - beta**2)**(-0.5)

    return rest_frame_freq*gamma*(1 + beta)

def scatter_rate(a_ik, S, delta_freq):
    Gamma = a_ik*10**-6/(2*np.pi)
    return Gamma/2 * S / (1 + (2*delta_freq / Gamma)**2 + S)

def Voigt(x, amp, x0, gam, sig, offset):
    return amp*Physics.voigt(x-x0, sig, gam)/Physics.voigt(0, sig, gam) + offset

def plotFit(x, y_sim, y_fit, y_res):
    fontsize_ticks = 12
    fig = plt.figure(1, (8, 8))
    fig.patch.set_facecolor('white')
    ax1 = plt.axes([0.15, 0.35, 0.8, 0.50])
    data_line = plt.plot(x, y_sim, label='Simulation Data')
    main_plot = plt.plot(x, y_fit, 'g-', label='Fit Data')
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    ax2 = plt.axes([0.15, 0.1, 0.8, 0.2], sharex=ax1)
    plt.plot(x, y_res)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    ax2.locator_params(axis='y', nbins=5)

    plt.ylabel('relative residuals / a.u.', fontsize=fontsize_ticks)
    ax1.set_ylabel('cts / a.u.', fontsize=fontsize_ticks)

    plt.xlabel('frequency / MHz', fontsize=fontsize_ticks, labelpad=fontsize_ticks/2)

    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    lines = [data_line[0], main_plot[0]]
    labels = [each.get_label() for each in lines]
    fig.legend(lines, labels, loc='upper center', ncol=2,
               bbox_to_anchor=(0.15, 0.8, 0.8, 0.2), mode='expand',
               fontsize=fontsize_ticks+2, numpoints=1)
    plt.show()

def plotMatrices(x_coord_matrix, y_coord_matrix, value_matrix):
    plt.contourf(x_coord_matrix, y_coord_matrix, value_matrix)
    plt.show()

def plotBeamProfils(x_coord_matrix, y_coord_matrix, acol_profil, col_profil, ion_profil, acol_eff_vel, col_eff_vel,
                    acol_x, acol_y_sim, acol_y_fit, acol_y_res, col_x, col_y_sim, col_y_fit, col_y_res, dif, dif_d,
                    acol_levels, col_levels, spectra_max=None):
    # fig = plt.figure(1, (10, 4), tight_layout=True)
    fig = plt.figure(figsize=(12, 12), constrained_layout=True)
    # fig.suptitle('Test Results')

    subfigs = fig.subfigures(nrows=3, ncols=1)

    subfigs[0].suptitle('Intensity Positions')
    subplt = subfigs[0].subplots(nrows=1, ncols=3)
    # plt.subplot(1, 3, 1)
    subplt[0].contourf(x_coord_matrix, y_coord_matrix, acol_profil)
    subplt[0].axhline()
    subplt[0].axvline()
    subplt[0].set_title("Anticollinear Laser")
    subplt[0].set_xlabel('Position / mm')
    subplt[0].set_ylabel('Position / mm')

    # plt.subplot(1, 3, 2)
    subplt[1].contourf(x_coord_matrix, y_coord_matrix, col_profil)
    subplt[1].axhline()
    subplt[1].axvline()
    subplt[1].set_title("Collinear Laser")
    subplt[1].set_xlabel('Position / mm')
    subplt[1].set_ylabel('Position / mm')

    # plt.subplot(1, 3, 3)
    subplt[2].contourf(x_coord_matrix, y_coord_matrix, ion_profil)
    subplt[2].axhline()
    subplt[2].axvline()
    subplt[2].set_title("Ion beam")
    subplt[2].set_xlabel('Position / mm')
    subplt[2].set_ylabel('Position / mm')

    subfigs[1].suptitle('Effective Ion Velocities')
    subplt = subfigs[1].subplots(nrows=1, ncols=2)
    subplt[0].contourf(x_coord_matrix, y_coord_matrix, acol_eff_vel, levels=acol_levels)
    subplt[0].axhline()
    subplt[0].axvline()
    subplt[0].set_title("Anticollinear Laser")
    subplt[0].set_xlabel('Position / mm')
    subplt[0].set_ylabel('Position / mm')

    subplt[1].contourf(x_coord_matrix, y_coord_matrix, col_eff_vel, levels=col_levels)
    subplt[1].axhline()
    subplt[1].axvline()
    subplt[1].set_title("Collinear Laser")
    subplt[1].set_xlabel('Position / mm')
    subplt[1].set_ylabel('Position / mm')

    subfigs[2].suptitle('Results')
    subplt = subfigs[2].subplots(nrows=1, ncols=3)
    subplt[0].set_title("Anticollinear Laser")
    subplt[0].plot(acol_x, acol_y_sim, label='Simulation Data')
    subplt[0].plot(acol_x, acol_y_fit, 'g-', label='Fit Data')
    subplt[0].set_ylim(None, spectra_max)
    subplt[0].set_xlabel('Freq. / MHz')
    subplt[0].set_ylabel('Cts / a.u.')

    subplt[1].set_title("Collinear Laser")
    subplt[1].plot(col_x, col_y_sim, label='Simulation Data')
    subplt[1].plot(col_x, col_y_fit, 'g-', label='Fit Data')
    subplt[1].set_ylim(None, spectra_max)
    subplt[1].set_xlabel('Freq. / MHz')
    subplt[1].set_ylabel('Cts / a.u.')

    subplt[2].set_title("Difference to real value")
    subplt[2].errorbar(1, dif, yerr=dif_d, label='Simulation Data', marker='o')
    subplt[2].axhline()
    subplt[2].set_ylabel('Dif. / MHz')

    plt.show()


def create_ion_file(file_name, no_of_ions, x0, sigma_x, y0, sigma_y, v0, sigma_v):
    file = open(file_name, 'w')
    file.write("IonNo, Pos X, Pos Y, Velocity Value, Velocity Vector X, Velocity Vector Y, Velocity Vector Z")

    for i in range(no_of_ions):
        no = i + 1
        pos_x = np.random.normal(x0, sigma_x, 1)[0]
        pos_y = np.random.normal(y0, sigma_y, 1)[0]
        vel_skalar = np.random.normal(v0, sigma_v) #+pos_x*8+pos_y*8, sigma_v)
        vel_vector_x = 0
        vel_vector_y = 0
        vel_vector_z = 1

        file.write("\n{}, {}, {}, {}, {}, {}, {}".format(no, pos_x, pos_y, vel_skalar, vel_vector_x, vel_vector_y, vel_vector_z))

    file.close()

def readIonList(file, x_axis, y_axis, acol_int_matrix, acol_vector_matrix, col_int_matrix, col_vector_matrix):
    ionNo, ion_pos_x, ion_pos_y, ion_vel, ion_vec_x, ion_vec_y, ion_vec_z = np.loadtxt(file, delimiter=',', skiprows=1, unpack=True)

    ion_pos_x_indices = np.argmin(np.absolute(np.expand_dims(x_axis, axis=0) - np.expand_dims(ion_pos_x, axis=1)), axis=1)
    ion_pos_y_indices = np.argmin(np.absolute(np.expand_dims(y_axis, axis=0) - np.expand_dims(ion_pos_y, axis=1)), axis=1)

    ion_pos_in_grid_x = x_axis[ion_pos_x_indices]
    ion_pos_in_grid_y = y_axis[ion_pos_y_indices]

    ion_vel = ion_vel
    ion_vel_vectors = np.vstack((ion_vec_x, ion_vec_y, ion_vec_z)).T

    acol_int_at_ion_pos = acol_int_matrix[ion_pos_y_indices, ion_pos_x_indices]
    acol_vector_at_ion_pos = acol_vector_matrix[ion_pos_y_indices, ion_pos_x_indices]

    col_int_at_ion_pos = col_int_matrix[ion_pos_y_indices, ion_pos_x_indices]
    col_vector_at_ion_pos = col_vector_matrix[ion_pos_y_indices, ion_pos_x_indices]


    return ion_pos_x_indices, ion_pos_y_indices, ion_pos_in_grid_x, ion_pos_in_grid_y, ion_vel, ion_vel_vectors, acol_int_at_ion_pos, acol_vector_at_ion_pos, col_int_at_ion_pos, col_vector_at_ion_pos

    # print(acol_int_matrix[ion_pos_y_indices[1]][ion_pos_x_indices[1]])
    # print(acol_int_matrix[ion_pos_y_indices, ion_pos_x_indices])

    # print(np.expand_dims(ion_pos_x, axis=1))

    # print(np.expand_dims(x_axis, axis=0) - np.expand_dims(ion_pos_x, axis=1)))


    # ion_eff_velocities = []
    # ion_pos_indices_x = []
    # ion_pos_indices_y = []
    #
    #
    # for row in data:
    #     ion_x = row[1]
    #     ion_y = row[2]
    #     ion_vel = row[3]
    #     ion_vec_x = row[4]
    #     ion_vec_y = row[5]
    #     ion_vec_z = row[6]
    #
    #     # Find ion position in Grid
    #     pos_index_x = np.argmin(x_axis - ion_x)
    #     pos_index_y = np.argmin(y_axis - ion_y)
    #
    #     # Calculate Ion-Laser-Angle
    #     ion_laser_angle = 0





# base_matrix = [['Sarah', 0, (-1, 0, 2)], [23, (4, 2 ,1), 'agq']]
# base_matrix = np.zeros((4, 2), dtype='object')  # erstes Argument: Reihen
# base_matrix = np.array(base_matrix)
# base_matrix[2][0] = (1, -1, 0)
# base_matrix[1][1] = 2.454
# print(type(base_matrix[0][0]))

# t1 = np.zeros((3,3,))
# print(t1)
# t2 = np.insert(t1[1], 3, 100, axis=0)
# print(t2)
# print(t1[1])
#
#
# t3 = [[[], [2], [3, 2]], [[2, 3], [], [1]]]
# t4 = np.array(t3)
#
# print(t4)
