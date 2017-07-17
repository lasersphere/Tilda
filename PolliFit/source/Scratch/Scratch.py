'''
Created on 31.03.2014

@author: gorges
'''

import BatchFit
import Tools
import Analyzer
import numpy as np
import Physics
import copy

kBE = 8.61733*10**-5 # eV/K
kBJ = 1.38064852*10**-23 #J/K
T = 1275
U = 10000
qe = 1.602176565e-19    #electron charge

c = 299792458    #speed of light
u = 1.660538921e-27    #atomic mass unit
for i in [40,48]:
    m = i*u

    v = np.sqrt(2*qe*U/m)

    nu0 = 755222765.9
    gamma = 1/np.sqrt(1-(v/c)**2)
    R = 1/2*np.sqrt(kBE*T/U)


    w = nu0 * gamma * np.sqrt(2*kBJ*T/(m*c**2))

    xeta = R*w/2

    print(xeta)

# print(Physics.freqFromWavenumber(16303.13364))
# bestdiff = 10**12
# values = [-0.9816, 0.785, -1.096, 0.653, -1.3469, -1.463, -0.573]
# valueErrs = [0.0049, 0.010, 0.45, 0.007, 0.005, 0.015, 0.015]
# gl = 0
# gs = 0
# bestglgs = [gl, gs]
# bestvals = []
# print(Analyzer.weightedAverage([-0.9,-0.915,-0.997,-1.044],
#                                [0.02339,0.00875,0.0085,0.00881]))
#
# for i in range(0, 400, 1):
#     for j in range(-100,100, 1):
#         gl = j/100
#         gs = -i/100
#         a = gs/2
#         b = 3/5*(gl*3-gs/2)
#         c = 2*gl+gs/2
#         d = 7/9*(gl*5-gs/2)
#         e = 5*gl+gs/2
#         f = 3*gl+gs/2
#         g1 = gl - (gs - gl) / 5
#         g2 = gl + (gs - gl) / 11
#         g = 7/2*( g1 + g2 - (g1 - g2)*128/224)
#         comparison = [a, b, c, d, e, f, g]
#         diff = 0
#         for l, m in enumerate(values):
#             diff = diff+((m-comparison[l])/(valueErrs[l]))**2
#         if diff<bestdiff:
#             bestdiff = diff
#             bestglgs = [gl,gs]
#             bestvals = comparison
#
# print(bestglgs)
# print(bestvals)