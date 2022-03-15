"""
Created on 23.03.2014

@author: hammen

A collection of physical constants and formulas
"""

import math

import numpy as np
from scipy import special
from scipy.stats import cauchy, norm

# _d marks uncertainty
c = 299792458  # speed of light
u = 1.660538921e-27  # atomic mass unit
u_d = 7.3e-35  # delta u
pi = 3.14159265
me = 9.10938291e-31  # electron mass
me_d = 4e-38
me_u = 5.4857990946e-4  # electron mass in u
me_u_d = 2.2e-13
qe = 1.602176565e-19  # electron charge
qe_d = 3.5e-27
h = 6.626070040e-34  # planck constant
hbar = h / (2 * np.pi)
LEMNISCATE = 2.6220575543


def relVelocity(e, m):
    """ Return the relativistic velocity of a body with kinetic energy e/J and mass m/kg """
    mcs = m * c * c
    return c * np.sqrt(1 - (mcs / (e + mcs)) ** 2)


def relEnergy(v, m):
    """ Return the relativistic energy of a body moving with velocity v/m/s and mass m/kg """
    mcs = m * c * c
    gamma = 1 / np.sqrt(1 - (v / c) ** 2)
    return mcs * (gamma - 1)


def addEnergyToFrequencyPoint(freq, energy, iso, laser_freq, col):
    """ Returns the frequency /MHz shifted by an energy /eV of the isotope. laser_Freq /MHZ, col /bool """
    try:
        velocity = invRelDoppler(laser_freq, freq + iso.freq)
        total_e = relEnergy(velocity, iso.mass * u)

        center_asym_joule = energy * qe

        v = relVelocity(total_e + center_asym_joule, iso.mass * u)
        v = -v if col else v

        shifted_f = relDoppler(laser_freq, v) - iso.freq
    except Exception as e:
        print('error while shifting frequency: %s ' % e)
        shifted_f = freq
    return shifted_f


def wavenumber(frequency):
    """ Returns the wavenumber /cm-1 at a given frequency /MHz """
    return frequency / c * 1e4


def freqFromWavenumber(wavenumber):
    """ Returns the frequency /MHz at a given wavenumber /cm-1 """
    return wavenumber * c * 1e-4


def freqFromWavelength(wavelength):
    """ Returns the frequency /MHz at a given wavelength /nm """
    return c / (wavelength * 10 ** -9) * 10 ** -6


def wavelenFromFreq(frequency):
    """ Returns the wavelength /nm at a given frequency /MHz """
    return c / (frequency * 10 ** 6) * 10 ** 9


def diffDoppler(nu_0, volt, m, charge=1, real=False):  # TODO Simplify / delete old?
    """ returns the differential doppler Factor [MHZ/V] """
    if real:  # Real calc. see Diss. K. König
        f_L = nu_0 * (1 + volt * qe * charge / (m * u * c ** 2)
                      * (1 - np.sqrt(1 + 2 * m * u * c ** 2 / (volt * qe * charge))))
        du = m * u * c ** 2 * (f_L ** 2 - nu_0 ** 2) / (2 * nu_0 * qe * charge * f_L ** 2)
        dv = -1 / du
    else:  # old approx.
        dv = nu_0 * qe * charge / np.sqrt(2 * qe * charge * volt * m * u * c ** 2)

    return dv

# def diffDoppler(nu_0, volt, m):
#     """returns the differential doppler Factor [MHZ/V]"""
#     return (nu_0/(m*u*c**2))*(qe+((qe*(m*u*c**2 + qe*volt))/((qe*volt*(2*m*u*c**2+qe*volt))**0.5)))


def diffDoppler2(nu_0, volt, m):  # TODO: ?
    f_L = nu_0 * (1 + volt * qe / (m * u * c ** 2) * (1 - (1 + 2 * m * u * c ** 2 / (volt * qe))**0.5))
    du = m * u * c**2 * (f_L**2 - nu_0**2) / (2*nu_0*qe*f_L**2)
    return -1/du


def relDoppler(laserFreq, v):
    """ Return the doppler shifted frequency of a frame moving with velocity v """
    # return laserFreq * np.sqrt((c + v) / (c - v))
    return laserFreq * np.sqrt(1 - (v / c) ** 2) / (1 - (v / c))


def invRelDoppler(laserFreq, dopplerFreq):
    """ Return the velocity, under which laserFreq is seen as dopplerFreq.
    Direction information gets lost in inverse function """
    # rs = (laserFreq/dopplerFreq)**2 """not right!?"""
    # rs = (dopplerFreq/laserFreq)**2
    # return c*(rs - 1)/(rs + 1)
    return c * (laserFreq ** 2 - dopplerFreq ** 2) / (laserFreq ** 2 + dopplerFreq ** 2)


def volt_to_rel_freq(volt, charge, mass, f_laser, f_0, col):
    """
    :param volt: The total acceleration voltage (V).
    :param charge: The charge of the particle (e).
    :param mass: The mass of the particle (u).
    :param f_laser: The laser frequency (arb. units).
    :param f_0: The reference frequency in the rest frame of the particle.
    :param col: Whether the lasers are aligned in collinear or anticollinear geometry.
    :returns: The doppler shifted frequency of the laser in the rest frame of a particle
     accelerated with 'volt' relative to a reference frequency 'f_0'.
    """
    pm = -1 if col else 1
    v = pm * relVelocity(qe * charge * volt, mass * u)
    return relDoppler(f_laser, v) - f_0


def rel_freq_to_volt(rel_freq, charge, mass, f_laser, f_0, col):
    """
    :param rel_freq: The relative frequency in the rest frame of a particle (MHz).
    :param charge: The charge of the particle (e).
    :param mass: The mass of the particle (u).
    :param f_laser: The laser frequency (arb. units).
    :param f_0: The reference frequency in the rest frame of the particle.
    :param col: Whether the lasers are aligned in collinear or anticollinear geometry.
    :returns: The total acceleration voltages which shift the laser frequency to 'rel_freq',
     the frequencies relative to 'f_0'.
    """
    pm = 1 if col else -1
    v = pm * invRelDoppler(f_laser, rel_freq + f_0)
    return relEnergy(v, mass * u) / (charge * qe)


def voigt(x, sig, gam):
    """ Voigt profile, unnormalized, using the Faddeeva function """
    return special.wofz((x + 1j * gam) / (sig * np.sqrt(2))).real / (sig * np.sqrt(2 * np.pi))


def fanoVoigt(x, sig, gam, dispersive):
    """ Fano Voigt profile, unnormalized, using the Faddeeva function and their imaginary part """
    return special.wofz((x + 1j * gam) / (sig * np.sqrt(2))).real / (sig * np.sqrt(2 * np.pi)) \
        + dispersive * special.wofz((x + 1j * gam) / (sig * np.sqrt(2))).imag / (sig * np.sqrt(2 * np.pi))


def source_energy_pdf(x, x0, sigma, xi, collinear=True):
    """ PDF of rest frame frequencies after acceleration of thermally and normally distributed kinetic energies. """
    pm = 1. if collinear else -1.
    x = np.asarray(x)
    sig = (sigma / (2. * xi)) ** 2
    _norm = np.exp(-0.5 * sig) / (sigma * np.sqrt(2. * np.pi))
    mu = -pm * (x - x0) / (2. * xi) - sig
    nonzero = mu.astype(bool)
    mu = mu[nonzero]
    b_arg = mu ** 2 / (4. * sig)
    main = np.full(x.shape, np.sqrt(LEMNISCATE * np.sqrt(sig / np.pi)))
    main_nonzero = np.empty_like(x[nonzero], dtype=float)
    mask = mu < 0.

    main_nonzero[mask] = np.sqrt(-0.5 * mu[mask] / np.pi) * np.exp(-mu[mask]) \
        * np.exp(-b_arg[mask]) * special.kv(0.25, b_arg[mask])
    main_nonzero[~mask] = 0.5 * np.sqrt(mu[~mask] * np.pi) * np.exp(-mu[~mask]) \
        * (special.ive(0.25, b_arg[~mask]) + special.ive(-0.25, b_arg[~mask]))
    main[nonzero] = main_nonzero
    return main * _norm


def thermalLorentz(x, loc, gam, xi, colDirTrue, order):
    """ Lineshape developed in Kretzschmar et al., Appl. Phys. B, 79, 623 (2004) """
    lw = 2 * gam  # linewidth FWHM
    col = 1 if colDirTrue else -1

    if order < 0:
        z0 = np.sqrt((-col * (x - loc) - 1j * lw / 2) / (2 * xi))
        return np.imag(np.exp(z0 ** 2) * (1 - special.erf(z0)) / z0) / (2 * np.sqrt(np.pi) * xi)

    denominator = ((x - loc) ** 2 + lw ** 2 / 4)
    sum_order = 0
    if order > 0:
        sum_order = (col * 2 * xi * (x - loc) - 3 * xi ** 2) / denominator
    if order > 1:
        sum_order += (4 * 3 * xi ** 2 * (x - loc) ** 2 - col * 4 * 15 * xi ** 3 * (x - loc) + 105 * xi ** 4) / (
                    denominator ** 2)
    if order > 2:
        sum_order += (col * 8 * 15 * xi ** 3 * (x - loc) ** 3 - 12 * 105 * xi ** 4 * (
                    x - loc) ** 2 + col * 6 * 945 * xi ** 5 * (x - loc) - 10395 * xi ** 6) / (denominator ** 3)
    if order > 3:
        sum_order += (16 * 105 * xi ** 4 * (x - loc) ** 4 - col * 32 * 945 * xi ** 5 * (
                    x - loc) ** 3 + 24 * 10395 * xi ** 6 * (x - loc) ** 2
                      - col * 8 * 135135 * xi ** 7 * (x - loc) + 2027025 * xi ** 8) / (denominator ** 4)
    return lorentz(x, loc, gam) * (1 + sum_order)


def lorentz(x, loc, gam):
    """ Lorentzian profile """
    return cauchy.pdf(x, loc, gam)


def lorentzQI(x, loc, loc2, gam):
    """ Quantum interference of lorentzian profile """
    cross = gam / np.pi / ((x - loc + 1j * gam) * (x - loc2 - 1j * gam))
    return 2 * cross.real


def gaussian(x, mu, sig, amp):
    """ gaussian function """
    # return amp / (sig * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)
    return amp * norm.pdf(x, mu, sig)


def gaussian_offset(x, mu, sig, amp, off):
    """ same as gaussian but adds offset with slope = 0 """
    return gaussian(x, mu, sig, amp) + off


def asymmetric_gaussian(x, mu, sig, amp, off, skew):
    """ normal distribution exponentially modified """
    return amp * skew / 2 * np.exp(
        skew / 2 * (2 * mu + skew * (sig ** 2) - 2 * x)) * special.erfc(
        (mu + skew * (sig ** 2) - x) / (np.sqrt(2) * sig)) + off


def transit(x, t):
    """
    Transit broadening function. Demtroeder Laserspectroscopy (german) Eq. (3.58).
    Uses same method as numpy.sinc(x) to calculate x = 0.
    :param x: frequency
    :param t: transit time
    :returns: The transit broadened peak profile.
    """
    x = np.asanyarray(x)
    y = 2 * np.pi * np.where(x == 0, 1.0e-20, x) * 1e6  # from frequency to angular frequency
    return (np.sin(0.5 * t * y)) ** 2 / y ** 2


def HFCoeff(I, J, F, old=True):
    """ Return the tuple of hyperfine coefficients for A and B-factor for a given quantum state """
    if old:
        C = 0. if I < 0.5 or J < 0.5 else (F * (F + 1) - I * (I + 1) - J * (J + 1))
        coA = 0.5 * C

        coB = 0. if I < 1 or J < 1 else (0.75 * C * (C + 1) - J * (J + 1) * I * (I + 1)) \
            / (2 * I * (2 * I - 1) * J * (2 * J - 1))

        return coA, coB

    # Functions using the new version should be able to handle an arbitrary amount of coefficients.
    # First three orders are taken from https://journals.aps.org/pra/abstract/10.1103/PhysRevA.103.032826.
    if I < 0.5 or J < 0.5:
        return tuple()

    # Magnetic dipole, the corresponding hyperfine constant is A = mu / (IJ) * <T_1>.
    K = F * (F + 1) - I * (I + 1) - J * (J + 1)
    coA = 0.5 * K
    if I < 1 or J < 1:
        return coA,

    # Electric quadrupole, the corresponding hyperfine constant is B = 2eQ * <T_2>.
    coB = (0.75 * K * (K + 1) - J * (J + 1) * I * (I + 1)) / (2 * I * (2 * I - 1) * J * (2 * J - 1))
    if I < 1.5 or J < 1.5:
        return coA, coB

    # Magnetic octupole, the corresponding hyperfine constant is C = -Omega * <T_3>.
    coC = K ** 3 + 4 * K ** 2 + 0.8 * K * (-3 * I * (I + 1) * J * (J + 1) + I * (I + 1) + J * (J + 1) + 3) \
        - 4 * I * (I + 1) * J * (J + 1)
    coC /= I * (I - 1) * (2 * I - 1) * J * (J - 1) * (2 * J - 1)
    coC *= 1.25
    if I < 2 or J < 2:
        return coA, coB, coC
    return coA, coB, coC  # Highest implemented order.


def HFTrans(I, Jl, Ju, old=True):
    """
    Calculate all allowed hyperfine transitions and their hyperfine coefficients.
    Returns (Fl, Fu, coAl, coBl, coAu, coBu)
    """
    # print('calculating the hyperfine transitions and hyperfine coeffients')
    if old:
        return [(Fl, Fu) + HFCoeff(I, Jl, Fl) + HFCoeff(I, Ju, Fu)
                for Fl in np.arange(abs(I - Jl), (I + Jl + 0.5))
                for Fu in np.arange(abs(I - Ju), (I + Ju + 0.5))
                if abs(Fl - Fu) == 1 or (Fl - Fu == 0 and Fl != 0 and Fu != 0)]
    return [[(Fl, Fu), HFCoeff(I, Jl, Fl, old=False), HFCoeff(I, Ju, Fu, old=False)]
            for Fl in np.arange(abs(I - Jl), (I + Jl + 0.5))
            for Fu in np.arange(abs(I - Ju), (I + Ju + 0.5))
            if abs(Fl - Fu) == 1 or (Fl - Fu == 0 and Fl != 0 and Fu != 0)]


def HFShift(hyper_l, hyper_u, coeff_l, coeff_u):
    """
    :param hyper_l: The hyperfine structure constants of the lower state (Al, Bl, Cl, ...).
    :param hyper_u: The hyperfine structure constants of the upper state (Au, Bu, Cu, ...).
    :param coeff_l: The coefficients of the lower state to be multiplied by the constants (coAl, coBl, coCl, ...).
    :param coeff_u: The coefficients of the lower state to be multiplied by the constants (coAu, coBu, coCu, ...).
    :returns: The hyperfine structure shift of an optical transition.
    """
    return sum(const * coeff for const, coeff in zip(hyper_u, coeff_u)) \
        - sum(const * coeff for const, coeff in zip(hyper_l, coeff_l))


def HFLineSplit(Al, Bl, Au, Bu, transitions):
    """ Calculate line splittings from (Au, Bu, Al, Bl) and list of transitions (see calcHFTrans) """
    return [Au * coAu + Bu * coBu - Al * coAl - Bl * coBl
            for x, y, coAl, coBl, coAu, coBu in transitions]


def HFInt(I, Jl, Ju, transitions, old=True):
    """ Calculate relative line intensities """
    if old:
        # print('Calculate relative line intensities for I, Jl, Ju, transitions ', I, Jl, Ju, transitions)
        res = [(2 * Fu + 1) * (2 * Fl + 1) * (sixJ(Jl, Fl, I, Fu, Ju, 1) ** 2) for Fl, Fu, *r in transitions]
        # print('result is: %s' % [round(each, 3) for each in res])
        return res
    return [np.around((2 * Fu + 1) * (2 * Fl + 1) * (sixJ(Jl, Fl, I, Fu, Ju, 1) ** 2), decimals=9)
            for (Fl, Fu), *r in transitions]


def sixJ(j1, j2, j3, J1, J2, J3):
    """ 6-J symbol used for Racah coefficients """
    # print('6-J symbol used for Racah coefficients, j1, j2, j3, J1, J2, J3: ', j1, j2, j3, J1, J2, J3)
    ret = 0
    for i in range(int(round(max(max(j1 + j2 + j3, j1 + J2 + J3), max(J1 + j2 + J3, J1 + J2 + j3)))),
                   int(round(min(min(j1 + j2 + J1 + J2, j2 + j3 + J2 + J3), j3 + j1 + J3 + J1) + 1))):
        ret = (ret + pow(-1, i) * math.factorial(i + 1)
               / math.factorial(round(i - j1 - j2 - j3))
               / math.factorial(round(i - j1 - J2 - J3))
               / math.factorial(round(i - J1 - j2 - J3))
               / math.factorial(round(i - J1 - J2 - j3))
               / math.factorial(round(j1 + j2 + J1 + J2 - i))
               / math.factorial(round(j2 + j3 + J2 + J3 - i))
               / math.factorial(round(j3 + j1 + J3 + J1 - i)))

    return math.sqrt(deltaJ(j1, j2, j3) * deltaJ(j1, J2, J3) * deltaJ(J1, j2, J3) * deltaJ(J1, J2, j3)) * ret


def threeJ(j1, m1, j2, m2, j3, m3):
    """ 3-J symbol used for Racah coefficients """
    # print('3-J symbol used for Racah coefficients')
    ret = 0
    for i in range(round(max(max(0, j2 - j3 - m1), m2 + j1 - j3)),
                   round(min(min(j1 + j2 - j3, j1 - m1), j2 + m2) + 1)):
        ret = (ret + pow(-1., i) / math.factorial(i) / math.factorial(round(j3 - j2 + i + m1)) / math.factorial(
            round(j3 - j1 + i - m2))
               / math.factorial(round(j1 + j2 - j3 - i)) / math.factorial(round(j1 - i - m1)) / math.factorial(
                    round(j2 - i + m2)))

    return (pow(-1., round(j1 - j2 - m3)) * math.sqrt(
        deltaJ(j1, j2, j3) * math.factorial(round(j1 + m1)) * math.factorial(round(j1 - m1))
        * math.factorial(round(j2 + m2)) * math.factorial(round(j2 - m2)) * math.factorial(
            round(j3 + m3)) * math.factorial(round(j3 - m3))) * ret)


def deltaJ(j1, j2, j3):    
    """ Delta-symbol used for Racah coefficients """
    # print('Delta-symbol used for Racah coefficients, j1, j2, j3: ', j1, j2, j3)
    return math.factorial(round(j1 + j2 - j3)) * math.factorial(round(j1 - j2 + j3)) * math.factorial(
        round(-j1 + j2 + j3)) / math.factorial(round(j1 + j2 + j3 + 1))


"""
|  not in use  |
|  erroneous   |
v    rework    v
"""

def shiftFreqToVoltage(m, nuOff, deltaNu, nuL):
    """ Returns the Voltage for a given frequency shift. ([m]=u, All frequencies need to be given in the same unit!) """
    return m * u * c ** 2 / (2 * qe) * ((nuOff + deltaNu) / nuL) ** 2


def dopplerAngle(nu, v, angle):
    """ Returns the frequency at a given angle (in rad) and velocity """
    return nu * np.sqrt(1 - v ** 2 / c ** 2) / (1 - v / c * np.cos(angle))


def getLineStrength(k_s, eps_L, F_i, F_apos, F_f, J, Japos, I):
    """ Returns f(\vec(kappa_s), \vec(epsilon_L), F_i, F') """
    Fi = F_i
    Fapos = F_apos
    epsL = eps_L
    ks = k_s
    epsS = ks
    Ff = F_f
    Cs = 0
    while Ff > 0:
        mf = Ff
        while mf >= -Ff:
            mi = Fi
            while mi >= -Fi:
                Cs = Cs + np.abs(C_if(Fapos, Fi, mi, Ff, mf, epsS, epsL, J, Japos, I)) ** 2
                mi = mi - 1
            mf = mf - 1
        Ff = Ff - 1
    return 3 / (2 * g_T(F_i)) * Cs


""" helper functions """


def g_T(F_i):
    gT = 0
    while F_i > 0:
        gT = gT + 2 * F_i + 1
        F_i = F_i - 1
    print('gT: ' + str(gT))
    return gT


def C_if(Fapos, Fi, Ff, epsS, epsL, J, Japos, I):
    mapos = -Fapos
    ret = 0
    while mapos <= Fapos:
        mi = -Fi
        while mi <= Fi:
            mf = -Ff
            while mf <= Ff:
                ret = ret + A(epsS, Fapos, mapos, Ff, mf, J, Japos, I) * A(epsL, Fapos, mapos, Fi, mi, J, Japos, I)
                mf += 1
            mi += 1
        mapos += 1
    # print('Calculating C_i->f, result: ' + str(ret))
    return ret


def A(eps, Fapos, mapos, F, m, J, Japos, I):
    ret = 0
    q = int(mapos - m)
    A = np.sqrt(2 * Japos + 1) / np.sqrt(2 * Fapos + 1) * CGK(F, m, 1, q, Fapos, mapos) * sqrtF(F, Fapos, J, Japos, I)
    if A != 0 and q in [1, 2, 3]:
        ret = eps[q - 1] * A
        # print('calculating A, result: '  + str(ret))
    return ret


def CGK(j1, m1, j2, m2, J, M):
    """ returns Clebsch Gordan Coefficient for <j1 m1 ; j2 m2 | J M> """
    if M != m1 + m2 or j1 + j2 < J or j1 - j2 > J or j2 - j1 > J:
        return 0
    n = 0
    sumn = 0
    while (j1 + j2 - J - n) >= 0 and (j1 - m1 - n) >= 0 and (
            j2 + m2 - n) >= 0 and J - j2 + m1 + n >= 0 and J - j1 - m2 + n >= 0:
        sumn = sumn + ((-1) ** n * np.sqrt(
            math.factorial(j1 + m2) * math.factorial(j1 - m1) * math.factorial(j2 + m2) * math.factorial(
                j2 - m2) * math.factorial(J + M) * math.factorial(J - M)) /
                       (math.factorial(n) * math.factorial(j1 + j2 - J - n) * math.factorial(
                           j1 - m1 - n) * math.factorial(j2 + m2 - n) * math.factorial(
                           J - j2 + m1 + n) * math.factorial(J - j1 - m2 + n)))
        n += 1
    # print(str(j1 + j2 - J) + '  ' + str(j1 + j2 - J) + '  ' + str(j1 - j2 + J)
    #       + '  ' + str(J + j2 - j1) + '  ' + str(j1 + j2 + J + 1))
    ret = np.sqrt((2 * J + 1) * math.factorial(j1 + j2 - J) * math.factorial(j1 - j2 + J) * math.factorial(
        J + j2 - j1) / math.factorial(j1 + j2 + J + 1)) * sumn
    # print('Calculating CGK for' + str(j1) + ',' + str(m1) + ',' + str(j2) + ',' + str(m2) 
    #       + ',' + str(J) + ',' + str(M) + ', result: ' + str(c))
    return ret


def sqrtF(F, Fapos, Japos, J, I):
    p = (F + I + 1 + Japos)
    # print('SqrtF of: ' +str(F) + str(Fapos) + str(Japos) + str(J) + str(I))
    ret = (-1) ** p * np.sqrt(2 * F + 1) * np.sqrt(2 * Fapos + 1) * sixJ(Japos, J, 1, F, Fapos, I)
    # if ret != 0:
    #     print('Calculating sqrtF, result: ' + str(ret))
    return ret
