"""
Created on 

@author: simkaufm

Module Description: Droplet Model calculations etc.
fro details see here:
Nuclear Physics A410 (1983), W. D. Myers
Z. Phys. A322 (1985), D. Berdichevsky

"""
import os

import numpy as np
import TildaTools as TiTs
from matplotlib import pyplot as plt

workdir = 'E:\\Workspace\\OwnCloud\\Projekte\\COLLAPS\\Nickel\\Measurement_and_Analysis_Simon\\Ni_workspace'

datafolder = os.path.join(workdir, 'Ni_April2016_mcp')

db = os.path.join(workdir, 'Ni_workspace.sqlite')

# radii from Achim Schwenk via e-Mail:
# Die Rechnungen basieren auf der IM-SRG mit NN+3N Wechselwirkungen.

ni_theorie_radii = {
    # '56': 3.7923,
    # '57': 3.8122,
    '58': 3.83758,
    '59': 3.853517,
    '60': 3.875522,
    '61': 3.88883,
    '62': 3.909071,
    '63': 3.920739,
    '64': 3.940103,
    '65': 3.9507136,
    '69': 3.9964276,
    '70': 4.007,
    # '71': 4.01232,
    # '72': 4.021409,
    # '73': 4.0256229,
    # '74': 4.033381,
    # '75': 4.0364976,
    # '76': 4.04307,
    # '77': 4.045115
}


def droplet_model(Z, A):
    N = A - Z
    I = (N - Z) / A

    b2 = 0
    b4 = 0
    A2 = np.sqrt(5 / (4 * np.pi)) * b2
    A4 = np.sqrt(5 / (4 * np.pi)) * b4
    # ratio of surface area of a deformed nucleus to that of a sphere with equal volume :
    Bs = 1 + (2 / 5.) * A2 ** 2 - (4 / 105.) * A2 ** 3 - \
         (66 / 175.) * A2 ** 4 - (4 / 35.) * A2 ** 2 * A4 + A4 ** 2
    # analogous term for Coulomb energy:
    Bc = 1 - (1 / 5.) * A2 ** 2 - (4 / 105.) * A2 ** 3 + \
         (51 / 245.) * A2 ** 4 - (6 / 35.) * A2 ** 2 * A4 - (5 / 27.) * A4 ** 2
    # associated with variations in Coulomb potential over surface when a nucleus is deformed:
    Bv = 1 - (1 / 5.) * A2 ** 2 - (2 / 105.) * A2 ** 3 - \
         (253 / 1225.) * A2 ** 4 - (4 / 105.) * A2 ** 4 * A4 + (4 / 9.) * A4 ** 2

    b = 0.99  # nuclear diffuseness
    r0 = 1.145  # nuclear radius constant
    J = 29.5  # symmetry energy coefficient
    Q = 45.  # effective surface stiffness
    K = 240.  # compressibility coefficient
    L = -5.  # density symmetry coefficient
    a2 = 18.  # surface energy coefficient
    e = 1.199985  # electron charge in unit of MeV*fm
    c1 = (3 / 5.) * (e ** 2 / r0)  # Coulomb energy coefficient
    delta_avg = (I + (3 / 16.) * (c1 / Q) * Z * A ** (-2 / 3.) * Bv) / (
        1 + (9 / 4.) * (J / Q) * A ** (-1 / 3.) * Bs)  # neutron proton asymmetry
    epsilon_avg = (-2 * a2 * A ** (-1 / 3.) * Bs + L * delta_avg ** 2 + c1 * Z ** 2 * A ** (
        -4 / 3.) * Bc) / K  # deviation of density
    R = r0 * A ** (1 / 3.) * (1 + epsilon_avg)
    t = (2 / 3.) * R * (I - delta_avg) / Bs
    Rn = R + Z * t / A
    Rz = R - N * t / A
    c_prime = (1 / 2.) * (9 / (2. * K) + 1 / (4. * J)) * (Z * e ** 2 / Rz)
    # contribution from the size of the uniform distribution and its shape:
    r2_u = (3 / 5.) * Rz ** 2 * (1 + A2 ** 2 + (10 / 21.) * A2 ** 3 - (27 / 35.) * A2 ** 4 +
                                 (10 / 7.) * A2 ** 2 * A4 + (5 / 9.) * A4 ** 2)
    # contribution from the redistribution, and its shape dependence:
    r2_r = (12 / 175.) * c_prime * (Rz ** 2) * (1 + (14 / 5.) * A2 ** 2 +
                                                (28 / 15.) * A2 ** 3 - (29 / 5.) * A2 ** 4 +
                                                (116 / 15.) * A2 ** 2 * A4 + (70 / 26.) * A4 ** 2)
    r2_d = 3 * b ** 2  # diffuseness
    r2 = r2_u + r2_r + r2_d
    dm = {'r2': r2}
    return dm


print(droplet_model(68, 40))
ni_60_r_fricke = 3.806 ** 2
print('A\tN\tdelta_r_sq_exp\tdelta_r_sq_exp+r_sq_fricke\tdrop_let_vol\tdiff\t')
ni_radii_diff_even = []
ni_radii_diff_odd = []
ni_ch_radii = []
for N in range(30, 43):
    if N != 41:  # Ni 69 not measured
        Z = 28
        A = N + Z
        dm = droplet_model(Z, A)['r2']
        ret = TiTs.select_from_db(db, 'val, statErr, systErr', 'Combined',
                                  [['run', 'iso', 'parname'], ['wide_gate_asym', '%s_Ni' % A, 'delta_r_square']])
        if ret is not None:
            r_sq_exp, d_stat_r_sq_exp, d_sysr_sq_exp = ret[0]
            d_total = np.sqrt(d_sysr_sq_exp ** 2 + d_stat_r_sq_exp ** 2)
            ni_ch_radii.append((N, A, r_sq_exp, d_stat_r_sq_exp, d_sysr_sq_exp, d_total))

            r_sq_exp_no_plus = r_sq_exp
            r_sq_exp += ni_60_r_fricke
            r_theo = ni_theorie_radii.get('%d' % A, 0)
            print(('%s\t%s\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (
                A, N, r_sq_exp_no_plus, np.sqrt(r_sq_exp), r_theo, np.sqrt(r_sq_exp) - r_theo if r_theo != 0 else 0,
                r_sq_exp, dm, r_sq_exp - dm)).replace('.', ','))
            if N % 2 != 0:
                ni_radii_diff_odd.append(r_sq_exp - dm)
            else:
                ni_radii_diff_even.append(r_sq_exp - dm)

Ni_even_N = [30, 32, 34, 36, 38, 40, 42]

Ni_odd_N = [31, 33, 35, 37, 39]

Cu_even_N = [30, 32, 34, 36, 38, 40, 42, 44, 46]

Cu_odd_N = [29, 31, 33, 35, 37, 39, 41, 43, 45]

Zn_even_N = [32, 34, 36, 38, 40, 42, 44, 46, 48, 50]

Zn_odd_N = [33, 35, 37, 39, 41, 43, 45, 47, 49]

# data from Liang
Cu_even = [
    0.206643253,
    0.301169991,
    0.334780703,
    0.30894223,
    0.253206822,
    0.208196884,
    0.211592706,
    0.164122547,
    0.025554588

]

Cu_odd = [0.101105939,
          0.239358975,
          0.278000377,
          0.279447406,
          0.25621173,
          0.186882382,
          0.158112081,
          0.151606215,
          0.06511391

          ]

Zn_even = [0.425826394,
           0.47860827,
           0.461365704,
           0.428143274,
           0.401473243,
           0.385709975,
           0.327948835,
           0.2176181,
           0.110085629,
           -0.0714386
           ]

Zn_odd = [
    0.467700991,
    0.413889172,
    0.406244405,
    0.40683206,
    0.418745842,
    0.354559438,
    0.219699688,
    0.148322543,
    0.020571592
]
font_size = 20

plt.plot(Ni_even_N, ni_radii_diff_even, 'ro', label='Ni N even', linestyle='-')
plt.plot(Ni_odd_N, ni_radii_diff_odd, 'r^', label='Ni N odd', linestyle='--')

plt.plot(Zn_even_N, Zn_even, 'bo', label='Zn N even', linestyle='-')
plt.plot(Zn_odd_N, Zn_odd, 'b^', label='Zn N odd', linestyle='--')

plt.plot(Cu_even_N, Cu_even, 'ko', label='Cu N even', linestyle='-')
plt.plot(Cu_odd_N, Cu_odd, 'k^', label='Cu N odd', linestyle='--')

plt.gcf().set_facecolor('w')
plt.axvline(40, color='r', linestyle='-.', linewidth=1)
plt.margins(0.1)
plt.legend(fontsize=font_size)
plt.ylabel(r'$<r^2>-<r^2>_{volume} / fm^2$', fontsize=20)
plt.xlabel('N', fontsize=font_size)
plt.tick_params(labelsize=font_size)
plt.show()
plt.clf()

ni_ch_radii_x = [each[1] for each in ni_ch_radii]
ni_ch_radii_y = [each[2] for each in ni_ch_radii]
ni_ch_radii_y_err = [each[-1] for each in ni_ch_radii]

ni_ch_radii_th_x = [58, 59, 60, 61, 62, 63, 64, 65, 69, 70]
ni_ch_radii_th_y = [ni_theorie_radii['%d' % each] ** 2 - ni_theorie_radii['60'] ** 2 for each in ni_ch_radii_th_x]

plt.errorbar(ni_ch_radii_x, ni_ch_radii_y, ni_ch_radii_y_err, linestyle='None', fmt='ro',
             label='exp. data')
plt.plot(ni_ch_radii_th_x, ni_ch_radii_th_y, 'b^', linestyle='None', label='IM-SRG, NN+3N')
ax = plt.gca()
ax.set_ylabel(r'$\delta$ < r' + r'$^2$ > (fm $^2$) ', fontsize=font_size)
ax.set_xlabel('A', fontsize=font_size)
plt.xticks(fontsize=font_size)
plt.yticks(fontsize=font_size)
plt.margins(0.1)
plt.gcf().set_facecolor('w')
plt.legend(loc=2, fontsize=font_size)
plt.show()
