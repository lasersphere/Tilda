'''
Created on 23.03.2014

@author: hammen

A collection of physical constants and formulas
'''

import math

import numpy as np
from scipy import special
from scipy.stats import cauchy

# _d marks uncertainty
c = 299792458;    #speed of light
u = 1.660538921e-27;    #atomic mass unit
u_d = 7.3e-35;    #delta u
pi = 3.14159265;
me = 9.10938291e-31;    #electron mass
me_d = 4e-38;
me_u = 5.4857990946e-4;    #electron mass in u
me_u_d = 2.2e-13;
qe = 1.602176565e-19;    #electron charge
qe_d = 3.5e-27;


def relVelocity(e, m):
    '''Return the relativistic velocity of a body with kinetic energy e/J and mass m/kg'''
    mcs = m*c*c
    return c * math.sqrt(1 - (mcs / (e + mcs))**2)


def relEnergy(v, m):
    """ return the relativistic energy of a body moving with velocity v/m/s and mass m/kg """
    mcs = m * c * c
    gamma = 1 / np.sqrt(1 - (v / c) ** 2)
    return mcs * (gamma - 1)


def addEnergyToFrequencyPoint(freq, energy, iso, laser_freq, col):
    """ Returns the frequency /MHz shifted by an energy /eV of the isotop. laser_Freq / MHZ, col / bool """
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
    '''Returns the wavenumber/1/cm at a given frequency/MHz'''
    return 10**4* frequency / c

def freqFromWavenumber(wavenumber):
    '''Returns the frequency/MHz at a given wavenumber/1/cm'''
    return wavenumber * c / 10**4

def freqFromWavelength(wavelength):
    ''' Returns the frequency/MHz at a given wavelength/nm '''
    return c / (wavelength * 10 ** -9) * 10 ** -6

def wavelenFromFreq(frequency):
    ''' Returns the wavelength in nm at a given frequency/MHz '''
    return c / (frequency * 10 ** 6) * 10 ** 9

def diffDoppler(nu_0, volt, m, charge=1, real=False):
    '''returns the differential doppler Factor [MHZ/V]'''
    if real:
        f_L = nu_0 * (1 + volt * qe * charge / (m * u * c ** 2) * (1 - (1 + 2 * m * u * c ** 2 / (volt * qe * charge)) ** 0.5))
        du = m * u * c ** 2 * (f_L ** 2 - nu_0 ** 2) / (2 * nu_0 * qe * charge * f_L ** 2) # Real calc. see Diss. K. König
        dv = -1/du
    else:
        dv = nu_0 * qe * charge / np.sqrt(2 * qe * charge * volt * m * u * c ** 2) # old approx.

    return dv

# def diffDoppler(nu_0, volt, m):
#     '''returns the differential doppler Factor [MHZ/V]'''
#     return (nu_0/(m*u*c**2))*(qe+((qe*(m*u*c**2 + qe*volt))/((qe*volt*(2*m*u*c**2+qe*volt))**0.5)))

def diffDoppler2(nu_0, volt, m):
    f_L = nu_0 * (1 + volt * qe / (m * u * c ** 2) * (1 - (1 + 2 * m * u * c ** 2 / (volt * qe))**0.5))
    du = m * u * c**2 * (f_L**2 - nu_0**2) / (2*nu_0*qe*f_L**2)
    return -1/du

def relDoppler(laserFreq, v):
    '''Return the doppler shifted frequency of a frame moving with velocity v'''
    return laserFreq * math.sqrt((c + v) / (c - v))

def invRelDoppler(laserFreq, dopplerFreq):
    '''Return the velocity, under which laserFreq is seen as dopplerFreq'''
    #rs = (laserFreq/dopplerFreq)**2 '''not right!?'''
    rs = (dopplerFreq/laserFreq)**2
    return c*(rs - 1)/(rs + 1)

def voigt(x, sig, gam):
    '''Voigt profile, unnormalized, using the Faddeeva function'''
    return special.wofz((x + 1j * gam)/(sig * math.sqrt(2))).real / (sig * math.sqrt(2 * math.pi))

def fanoVoigt(x, sig, gam, dispersive):
    '''Fano Voigt profile, unnormalized, using the Faddeeva function and their imaginary part'''
    return special.wofz((x + 1j * gam)/(sig * math.sqrt(2))).real / (sig * math.sqrt(2 * math.pi)) +\
           dispersive * special.wofz((x + 1j * gam)/(sig * math.sqrt(2))).imag / (sig * math.sqrt(2 * math.pi))

def thermalLorentz(x, loc, gam, xi, colDirTrue, order):
    '''Lineshape developed in Kretzschmar et al., Appl. Phys. B, 79, 623 (2004)'''
    lw = gam*2  # linewidth FWHM
    if colDirTrue:
        col = 1
    else:
        col = -1

    if order < 0:
        z0 = np.sqrt((-col*(x - loc) - 1j*lw/2)/(2*xi))
        return np.imag(np.exp(z0**2)*(1 - special.erf(z0))/z0)/(2*np.sqrt(np.pi)*xi)

    denominator = ((x - loc)**2 + lw**2/4)
    sum_order = 0
    if order > 0:
        sum_order = (col*xi*(x - loc)-3*xi**2)/denominator
    if order > 1:
        sum_order += (4*3*xi**2*(x - loc)**2 - col*4*15*xi**3*(x - loc) + 105*xi**4)/(denominator**2)
    if order > 2:
        sum_order += (col*8*15*xi**3*(x - loc)**3 - 12*105*xi**4*(x - loc)**2 + col*6*945*xi**5*(x - loc) - 10395*xi**6)\
            / (denominator**3)
    if order > 3:
        sum_order += (16*105*xi**4*(x - loc)**4 - col*32*945*xi**5*(x - loc)**3 + 24*10395*xi**6*(x - loc)**2
                      - col*8*135135*xi**7*(x - loc) + 2027025*xi**8)/(denominator**4)
    return lorentz(x, loc, gam, own=True)*(1 + sum_order)

def lorentz(x, loc, gam, own=False):
    '''Lorentzian profile '''
    if own:
        lw = gam*2
        return (2/(math.pi*lw))*((lw**2/4)/((x-loc)**2+(lw**2/4)))
    else:
        return cauchy.pdf(x, loc, gam)


def lorentzQI (x, loc, loc2, gam):
    '''Quantum interference of lorentzian profile '''
    cross = gam / math.pi / ((x-loc + 1j*gam)*(x-loc2 - 1j*gam))

    return 2 * cross.real

def gaussian(x, mu, sig, amp):
    """
    gaussian function
    :param x:
    :param mu:
    :param sig:
    :return:
    """
    return amp/(sig * np.sqrt(2 * math.pi)) * np.exp(-0.5 * ((x - mu) / sig) ** 2)


def gaussian_offset(x, mu, sig, amp, off):
    """
    same as gaussian but adds offset with slope = 0
    :param x:
    :param mu:
    :param sig:
    :param amp:
    :param off:
    :return:
    """
    return gaussian(x, mu, sig, amp) + off


def asymmetric_gaussian(x, mu, sig, amp, off, skew):
    """
    normal distribution exponentially modified
    :param x:
    :param mu:
    :param sig:
    :param amp:
    :param off:
    :param skew:
    :return:
    """
    return amp * skew / 2 * np.exp(
        skew / 2 * (2 * mu + skew * (sig ** 2) - 2 * x)) * special.erfc(
        (mu + skew * (sig ** 2) - x) / (np.sqrt(2) * sig)) + off


def transit(x, t):
    """
    transit broadening function. Demtröder Laserspectroscopy (german) Eq. (3.58).
    Uses same method as numpy.sinc(x) to calculate x = 0.
    :param x: frequency
    :param t: transit time
    :return:
    """

    x = np.asanyarray(x)
    y = 2*np.pi*np.where(x == 0, 1.0e-20, x)*10**6 # from frequency to angular frequency
    return (np.sin(0.5*t*y))**2/y**2


def HFCoeff(I, J, F):
    '''Return the tuple of hyperfine coefficients for A and B-factor for a given quantum state'''
    #print('Return the tuple of hyperfine coefficients for A and B-factor for I = ', I, ' J = ', J, ' F = ', F)
    C = 0.0 if I == 0 else (F*(F+1) - I*(I+1) - J*(J+1))
    coA = 0.5 * C
    
    #catch case of low spins
    coB = 0.0 if I < 0.9 or J < 0.9 else (0.75 * C*(C+1) - J*(J+1)*I*(I+1)) / (2*I*(2*I-1)*J*(2*J-1))
                                                 
    return (coA, coB)

def HFTrans(I, Jl, Ju):
    '''Calculate all allowed hyperfine transitions and their hyperfine coefficients. Returns (Fl, Fu, coAl, coBl, coAu, coBu)''' 
    #print('calculating the hyperfine transitions and hyperfine coeffients')
    return [(Fl, Fu) + HFCoeff(I, Jl, Fl) + HFCoeff(I, Ju, Fu)
                        for Fl in np.arange(abs(I - Jl), (I + Jl + 0.5))
                        for Fu in np.arange(abs(I - Ju), (I + Ju + 0.5)) if abs(Fl - Fu) == 1 or (Fl - Fu == 0 and Fl != 0 and Fu != 0)]

def HFLineSplit(Al, Bl, Au, Bu, transitions):
    '''Calculate line splittings from (Au, Bu, Al, Bl) and list of transitions (see calcHFTrans)'''
    return [Au * coAu + Bu * coBu - Al * coAl - Bl * coBl
                    for x, y, coAl, coBl, coAu, coBu in transitions]

def HFInt(I, Jl, Ju, transitions):
    '''Calculate relative line intensities'''
    print('Calculate relative line intensities for I, Jl, Ju, transitions ',I, Jl, Ju, transitions)
    res = [(2*Fu+1)*(2*Fl+1)*(sixJ(Jl, Fl, I, Fu, Ju, 1)**2) for Fl, Fu, *r in transitions]
    print('result is: %s' % [round(each, 3) for each in res])
    return res

def sixJ(j1, j2, j3, J1, J2, J3):
    '''6-J symbol used for Racah coefficients'''
    #print('6-J symbol used for Racah coefficients, j1, j2, j3, J1, J2, J3: ', j1, j2, j3, J1, J2, J3)
    ret = 0
    for i in range(int(round(max(max(j1+j2+j3,j1+J2+J3),max(J1+j2+J3,J1+J2+j3)))),
                   int(round(min(min(j1+j2+J1+J2,j2+j3+J2+J3),j3+j1+J3+J1) + 1))):
        ret= (ret + pow(-1,i) * math.factorial(i+1.)
            /math.factorial(round(i-j1-j2-j3))
            /math.factorial(round(i-j1-J2-J3))
            /math.factorial(round(i-J1-j2-J3))
            /math.factorial(round(i-J1-J2-j3))
            /math.factorial(round(j1+j2+J1+J2-i))
            /math.factorial(round(j2+j3+J2+J3-i))
            /math.factorial(round(j3+j1+J3+J1-i)) )
        
    return math.sqrt(deltaJ(j1,j2,j3)*deltaJ(j1,J2,J3)*deltaJ(J1,j2,J3)*deltaJ(J1,J2,j3))*ret
        
def threeJ(j1, m1, j2, m2, j3, m3):
    '''3-J symbol used for Racah coefficients'''
    #print('3-J symbol used for Racah coefficients')
    ret=0;
    for i in range(round(max(max(0.,j2-j3-m1), m2+j1-j3)),
                    round(min(min(j1+j2-j3,j1-m1),j2+m2) + 1)):

        ret = (ret+pow(-1.,i)/math.factorial(i) / math.factorial(round(j3-j2+i+m1)) / math.factorial(round(j3-j1+i-m2))
            /math.factorial(round(j1+j2-j3-i)) / math.factorial(round(j1-i-m1)) / math.factorial(round(j2-i+m2)) )
    
    return (pow(-1.,round(j1-j2-m3)) * math.sqrt(deltaJ(round(j1,j2,j3)) * math.factorial(round(j1+m1)) * math.factorial(round(j1-m1))
        *math.factorial(round(j2+m2))*math.factorial(round(j2-m2))*math.factorial(round(j3+m3))*math.factorial(round(j3-m3)))*ret)
    
def deltaJ(j1, j2, j3):    
    '''Delta-symbol used for Racah coefficients'''
    #print('Delta-symbol used for Racah coefficients, j1, j2, j3: ', j1, j2, j3)
    return math.factorial(round(j1+j2-j3))*math.factorial(round(j1-j2+j3))*math.factorial(round(-j1+j2+j3))/math.factorial(round(j1+j2+j3+1))

def shiftFreqToVoltage(m,nuOff,deltaNu,nuL):
    '''Returns the Voltage for a given frequency shift. ([m]=u, All frequencies need to be given in the same unit!)'''
    return m*u*c**2/(2*qe)*((nuOff+deltaNu)/nuL)**2

def dopplerAngle(nu, v, angle):
    '''Returns the frequency at a given angle (in rad) and velocity'''
    return nu*np.sqrt(1-v**2/c**2)/(1-v/c*np.cos(angle))

def getLineStrength(k_s, eps_L, F_i , F_apos , F_f, J, Japos, I):
    '''Returns f(\vec(kappa_s), \vec(epsilon_L), F_i, F') '''
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
            while mi >=-Fi:
                Cs = Cs + np.abs(C_if(Fapos, Fi, mi, Ff, mf, epsS, epsL, J, Japos, I))**2
                mi = mi-1
            mf = mf -1
        Ff = Ff -1
    return 3/(2*g_T(F_i)) * Cs

'''helpfunctions'''
def g_T(F_i):
    gT = 0
    while F_i > 0:
        gT = gT + 2*F_i + 1
        F_i = F_i - 1
    print('gT: ' + str(gT))
    return gT

def C_if(Fapos, Fi, Ff, epsS, epsL, J, Japos, I):
    mapos = -Fapos
    c = 0
    while mapos <= Fapos:
        mi = -Fi
        while mi <= Fi:
            mf = -Ff
            while mf <= Ff:
                c = c + A(epsS, Fapos, mapos, Ff, mf, J, Japos, I) * A(epsL, Fapos, mapos, Fi, mi, J, Japos, I)  
                mf +=1
            mi += 1
        mapos += 1
    #print('Calculating C_i->f, result: ' + str(c))
    return c

def A(eps, Fapos, mapos, F, m, J, Japos, I):
    A = 0
    c = 0
    q = int(mapos - m)
    A = np.sqrt(2*Japos+1)/np.sqrt(2*Fapos+1)* CGK(F, m, 1, q, Fapos, mapos) * sqrtF(F, Fapos, J, Japos, I)
    if A != 0 and q in [1,2,3]:
        c = eps[q-1] * A
        # print('calculating A, result: '  + str(c))
    return c

def CGK(j1, m1, j2, m2, J, M):
    '''returns Clebsch Gordan Coefficient for <j1 m1 ; j2 m2 | J M>'''
    if M != m1+m2 or j1+j2 < J or j1 - j2 > J or j2 - j1 > J:
        return 0
    n = 0
    sumn = 0
    while (j1 + j2 - J - n) >= 0 and (j1 - m1 - n) >=0 and (j2+m2-n)>=0 and J - j2 + m1 + n >=0 and J - j1 - m2 + n >=0:
        sumn = sumn + ((-1)**n * np.sqrt(math.factorial(j1 + m2) * math.factorial(j1 - m1) * math.factorial(j2 + m2) * math.factorial(j2 - m2) * math.factorial(J + M) * math.factorial(J - M)) /
                         ( math.factorial(n) * math.factorial(j1 + j2 - J - n) * math.factorial(j1 - m1 - n) * math.factorial(j2 + m2 - n) * math.factorial(J - j2 + m1 + n) * math.factorial(J - j1 - m2 + n) ) )
        n += 1
    #print(str(j1 + j2 - J) + '  ' +  str(j1 + j2 - J) + '  ' +  str(j1 - j2 + J) + '  ' +  str(J + j2 - j1) + '  ' +  str(j1 + j2 + J + 1))
    c = np.sqrt((2 * J + 1) * math.factorial(j1 + j2 - J) * math.factorial(j1 - j2 + J) * math.factorial(J + j2 - j1) / math.factorial(j1 + j2 + J + 1)) * sumn 
    #print('Calculating CGK for' + str(j1) + ',' + str(m1) + ','  + str(j2) + ',' + str(m2) + ',' + str(J) + ',' + str(M) +', result: ' + str(c))
    return c

def sqrtF(F, Fapos, Japos, J, I):
    p = (F + I + 1 + Japos)
    #print('SqrtF of: ' +str(F) + str(Fapos) + str(Japos) + str(J) + str(I))
    c = (-1)**p * np.sqrt(2 * F + 1) * np.sqrt(2*Fapos + 1) * sixJ(Japos, J, 1, F, Fapos, I)
#     if c != 0:
#         print('Calculating sqrtF, result: ' + str(c) )
    return c