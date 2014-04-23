'''
Created on 23.03.2014

@author: hammen
'''

import math

# _d marks uncertainty
c = 299792458;    #speed of light
u_kg = 1.660538921e-27;    #atomic mass unit
u_kg_d = 7.3e-35;    #delta u
pi = 3.14159265;
me = 9.10938291e-31;    #electron mass
me_d = 4e-38;
me_u = 5.4857990946e-4;    #electron mass in u
me_u_d = 2.2e-13;
qe = 1.602176565e-19;    #electron charge
qe_d = 3.5e-27;


'''returns the relativistic velocity of a body of mass m and kinetic energy e'''
def relVelocity(e, m):
    mc = m*c*c
    return c * math.sqrt(1 - pow(mc / (e + mc), 2))

'''returns the doppler shifted frequency of a frame moving with velocity v'''
def relDoppler(laserFreq, v):
    return laserFreq * math.sqrt((c + v) / (c - v))

def clasEnergy():
    pass


'''
Returns the tuple of total angular momentum F andhyperfine coefficients for A and B-factor
for a given quantum state
'''    
def hypCoeff(I, J, F):
    C = (F*(F+1) - I*(I+1) - J*(J+1))
    coA = 0.5 * C
    
    #catch case of low spins
    if I < 0.9 or J < 0.9:
        coB = 0
    else:
        coB = (0.75 * C*(C+1) - J*(J+1)*I*(I+1)) / (2*I*(2*I-1)*J*(2*J-1))
                                                 
    return (coA, coB)

'''
calculate all allowed hyperfine transitions and their hyperfine coefficients. Returns (Fu, Fl, coAu, coBu, coAl, coBl)
''' 
def calcHFTrans(I, Ju, Jl):
    return [(Fu, Fl) + hypCoeff(I, Ju, Fu) + hypCoeff(I, Jl, Fl)
                        for Fu in range(abs(I - Ju), (I + Ju))
                        for Fl in range(abs(I - Jl), (I + Jl)) if abs(Fl - Fu) == 1 or (Fl - Fu == 0 and Fl != 0 and Fu != 0)]

'''
calculate line positions from (Au, Bu, Al, Bl) and list of transitions (see calcHFTrans)
'''
def calcHFLinePos(Au, Bu, Al, Bl, transitions):
    return [Au * coAu + Bu * coBu - Al * coAl - Bl * coBl
                    for (coAu, coBu, coAl, coBl) in transitions[2:]]

'''6-J symbol used for Racah coefficients'''
def sixJ(j1, j2, j3, J1, J2, J3):
    ret = 0
    for i in range(max(max(j1+j2+j3,j1+J2+J3),max(J1+j2+J3,J1+J2+j3)),
                   min(min(j1+j2+J1+J2,j2+j3+J2+J3),j3+j1+J3+J1) + 1):
        ret= (ret + pow(-1,i) * math.factorial(i+1.)
            /math.factorial(i-j1-j2-j3)
            /math.factorial(i-j1-J2-J3)
            /math.factorial(i-J1-j2-J3)
            /math.factorial(i-J1-J2-j3)
            /math.factorial(j1+j2+J1+J2-i)
            /math.factorial(j2+j3+J2+J3-i)
            /math.factorial(j3+j1+J3+J1-i) )
        
    return math.sqrt(deltaJ(j1,j2,j3)*deltaJ(j1,J2,J3)*deltaJ(J1,j2,J3)*deltaJ(J1,J2,j3))*ret

'''3-J symbol used for Racah coefficients'''        
def threeJ(j1, m1, j2, m2, j3, m3):
    ret=0;
    for i in range(max(max(0.,j2-j3-m1), m2+j1-j3),
                    min(min(j1+j2-j3,j1-m1),j2+m2) + 1):

        ret = (ret+pow(-1.,i)/math.factorial(i) / math.factorial(j3-j2+i+m1) / math.factorial(j3-j1+i-m2)
            /math.factorial(j1+j2-j3-i) / math.factorial(j1-i-m1) / math.factorial(j2-i+m2) )
    
    return (pow(-1.,j1-j2-m3) * math.sqrt(deltaJ(j1,j2,j3) * math.factorial(j1+m1) * math.factorial(j1-m1)
        *math.factorial(j2+m2)*math.factorial(j2-m2)*math.factorial(j3+m3)*math.factorial(j3-m3))*ret)
    
'''delta-symbol used for Racah coefficients'''    
def deltaJ(j1, j2, j3):
    return math.factorial(j1+j2-j3)*math.factorial(j1-j2+j3)*math.factorial(-j1+j2+j3)/math.factorial(j1+j2+j3+1)