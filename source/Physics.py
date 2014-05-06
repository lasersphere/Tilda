'''
Created on 23.03.2014

@author: hammen
'''

import math
import numpy as np
from scipy import special

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

def relDoppler(laserFreq, v):
    '''Return the doppler shifted frequency of a frame moving with velocity v'''
    return laserFreq * math.sqrt((c + v) / (c - v))

def clasEnergy():
    pass


def voigt(x, sig, gam):
    '''Voigt profile, unnormalized'''
    return special.wofz((x + 1j * gam)/(sig * math.sqrt(2))).real / (sig * math.sqrt(2 * math.pi))

def HFCoeff(I, J, F):    
    '''Return the tuple of hyperfine coefficients for A and B-factor for a given quantum state'''
    C = 0.0 if I == 0 else (F*(F+1) - I*(I+1) - J*(J+1))
    coA = 0.5 * C
    
    #catch case of low spins
    coB = 0.0 if I < 0.9 or J < 0.9 else (0.75 * C*(C+1) - J*(J+1)*I*(I+1)) / (2*I*(2*I-1)*J*(2*J-1))
                                                 
    return (coA, coB)

def HFTrans(I, Jl, Ju):
    '''Calculate all allowed hyperfine transitions and their hyperfine coefficients. Returns (Fl, Fu, coAl, coBl, coAu, coBu)''' 
    return [(Fl, Fu) + HFCoeff(I, Jl, Fl) + HFCoeff(I, Ju, Fu)
                        for Fl in np.arange(abs(I - Jl), (I + Jl + 0.5))
                        for Fu in np.arange(abs(I - Ju), (I + Ju + 0.5)) if abs(Fl - Fu) == 1 or (Fl - Fu == 0 and Fl != 0 and Fu != 0)]

def HFLineSplit(Al, Bl, Au, Bu, transitions):
    '''Calculate line splittings from (Au, Bu, Al, Bl) and list of transitions (see calcHFTrans)'''
    return [Au * coAu + Bu * coBu - Al * coAl - Bl * coBl
                    for x, y, coAl, coBl, coAu, coBu in transitions]

def HFInt(I, Jl, Ju, transitions):
    '''Calculate relative line intensities'''
    return [(2*Fu+1)*(2*Fl+1)*(sixJ(Jl, Fl, I, Fu, Ju, 1)**2) for Fl, Fu, *r in transitions]

def sixJ(j1, j2, j3, J1, J2, J3):
    '''6-J symbol used for Racah coefficients'''
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
    ret=0;
    for i in range(round(max(max(0.,j2-j3-m1), m2+j1-j3)),
                    round(min(min(j1+j2-j3,j1-m1),j2+m2) + 1)):

        ret = (ret+pow(-1.,i)/math.factorial(i) / math.factorial(round(j3-j2+i+m1)) / math.factorial(round(j3-j1+i-m2))
            /math.factorial(round(j1+j2-j3-i)) / math.factorial(round(j1-i-m1)) / math.factorial(round(j2-i+m2)) )
    
    return (pow(-1.,round(j1-j2-m3)) * math.sqrt(deltaJ(round(j1,j2,j3)) * math.factorial(round(j1+m1)) * math.factorial(round(j1-m1))
        *math.factorial(round(j2+m2))*math.factorial(round(j2-m2))*math.factorial(round(j3+m3))*math.factorial(round(j3-m3)))*ret)
    
def deltaJ(j1, j2, j3):    
    '''Delta-symbol used for Racah coefficients'''
    return math.factorial(round(j1+j2-j3))*math.factorial(round(j1-j2+j3))*math.factorial(round(-j1+j2+j3))/math.factorial(round(j1+j2+j3+1))