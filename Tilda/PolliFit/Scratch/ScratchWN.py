'''
Created on 31.03.2014

@author: noertershaeuser
'''
# import os
# import sqlite3
# import math
# import Tools
# import Tilda.PolliFit.TildaTools

#db = 'C:\\Users\\wnoerter\\PycharmProjects\\PolliFit\\test\\Project\\CdIon.sqlite'
# Tools.createDB(db)
# Tools.add_missing_columns(db)
#print(TildaTools.select_from_db(db, 'mass', 'Isotopes', [['iso'], ['106_Cd']]))

#from Tilda.PolliFit import MPLPlotter as plot
#import numpy as np

#------------------------------

# import numpy as np
# # x = np.asarray([[1,3,6,1,4,6,8,9,2,2],
# #                 [2,7,6,1,5,6,3,9,5,7],
# #                 [1,3,4,7,4,8,5,1,8,2],
# #                 [1,1,1,1,1,1,1,1,1,1]]).T
# # y = np.asarray([1,2,3,4,5,6,7,8,9,10]).T
#
#
# x = np.asarray([[1,2,2,3,1],
#                 [1,1,2,1,3],
#                 [1,1,1,1,1]]).T
# y = np.asarray([6,7,9,8,10]).T
#
#
# a,b,c = np.linalg.pinv((x.T).dot(x)).dot(x.T.dot(y))
# print(a,b,c)
#------------------------------
import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *
import scipy.odr
import random

# Initiate some data, giving some randomness using random.random().
#x = np.array([[1, 2, 2, 3, 1, 5],[1, 1, 2, 1, 3, 4]])
#y = np.array([6.1,6.9,9.05,8.1,9.99,15.84])
    #[i**2 + random.random() for i in x])
x = np.array([[1.05, 2, 3.1, 4, 5.01], [1, 1.99, 3, 3.98, 5]])
y = np.array([6, 9, 12, 15, 18])


# x_err = np.array([[0.1,0.2,0.15,0.3,0.1,0.05],[0.01,0.1,0.1,0.05,0.2,0.1]])
x_err = 0.1
y_err = 0.1
# y_err = np.array([3, 0.5, 1, 2, 0.3, 4.])

# Define a function (quadratic in our case) to fit the data with.
def mlr(p, x):
     print (p[0],p[1],p[2], x[0],x[1])
     print (p[0] * x[0] + p[1] * x[1] + p[2])
     return p[0] * x[0] + p[1] * x[1] + p[2]

# Create a model for fitting.
# mlr = Model(multilinear)

# Create a RealData object using our initiated data from above.
data = RealData(x, y, sx=x_err, sy=y_err)
linmod = scipy.odr.Model(mlr)

# Set up ODR with the model and data.
odr = ODR(data, linmod, beta0=[1. , 2., 3.])

# Run the regression.
out = odr.run()

# Use the in-built pprint method to give us results.
out.pprint()
'''Beta: [ 1.01781493  0.48498006]
Beta Std Error: [ 0.00390799  0.03660941]
Beta Covariance: [[ 0.00241322 -0.01420883]
 [-0.01420883  0.21177597]]
Residual Variance: 0.00632861634898189
Inverse Condition #: 0.4195196193536024
Reason(s) for Halting:
  Sum of squares convergence'''

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d as m3d
data = np.concatenate((x[0][:, np.newaxis],
                       x[1][:, np.newaxis],
                       y[:, np.newaxis]),
                      axis=1)
linepts = odr.output.beta[0] * x[0] + odr.output.beta[1] * x[1] + odr.output.beta[2]
#linepts = vv[0] * np.mgrid[0:7:2j][:, np.newaxis]
# linepts += datamean

ax = m3d.Axes3D(plt.figure())
ax.scatter3D(*data.T)
#ax.plot3D(*linepts.T)
plt.show()

# x_fit = np.linspace(0, 5, 1000)
# y_fit = mlr(out.beta, x_fit)
#
# plt.errorbar(x, y, xerr=x_err, yerr=y_err, linestyle='None', marker='x')
# plt.plot(x_fit, y_fit)
#
# plt.show()
# ------------------------------
#ORIGINAL
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.odr import *
#
# import random
#
# # Initiate some data, giving some randomness using random.random().
# x = np.array([0, 1, 2, 3, 4, 5])
# y = np.array([i**2 + random.random() for i in x])
#
# x_err = np.array([random.random() for i in x])
# y_err = np.array([random.random() for i in x])
#
# # Define a function (quadratic in our case) to fit the data with.
# def quad_func(p, x):
#      m, c = p
#      return m*x**2 + c
#
# # Create a model for fitting.
# quad_model = Model(quad_func)
#
# # Create a RealData object using our initiated data from above.
# data = RealData(x, y, sx=x_err, sy=y_err)
#
# # Set up ODR with the model and data.
# odr = ODR(data, quad_model, beta0=[0., 1.])
#
# # Run the regression.
# out = odr.run()
#
# # Use the in-built pprint method to give us results.
# out.pprint()
# '''Beta: [ 1.01781493  0.48498006]
# Beta Std Error: [ 0.00390799  0.03660941]
# Beta Covariance: [[ 0.00241322 -0.01420883]
#  [-0.01420883  0.21177597]]
# Residual Variance: 0.00632861634898189
# Inverse Condition #: 0.4195196193536024
# Reason(s) for Halting:
#   Sum of squares convergence'''
#
# x_fit = np.linspace(x[0], x[-1], 1000)
# y_fit = quad_func(out.beta, x_fit)
#
# plt.errorbar(x, y, xerr=x_err, yerr=y_err, linestyle='None', marker='x')
# plt.plot(x_fit, y_fit)
#
# plt.show()
#---------------------------------


# import numpy as np
# import statsmodels.api as sm

# y = [1,2,3,4,3,4,5,4,5,5,4,5,4,5,4,5,6,5,4,5,4,3,4]
#
# x = [
#      [4,2,3,4,5,4,5,6,7,4,8,9,8,8,6,6,5,5,5,5,5,5,5],
#      [4,1,2,3,4,5,6,7,5,8,7,8,7,8,7,8,7,7,7,7,7,6,5],
#      [4,1,2,5,6,7,8,9,7,8,7,8,7,7,7,7,7,7,6,6,4,4,4]
#     ]

# y = [6,7,9,8,10]
#
# x = [
#      [1,1,2,1,3],
#      [1,2,2,3,1]
# ]
#
# def reg_m(y, x):
#     ones = np.ones(len(x[0]))
#      X = sm.add_constant(np.column_stack((x[0], ones)))
#     for ele in x[1:]:
#         X = sm.add_constant(np.column_stack((ele, X)))
#    results = sm.OLS(y, X).fit()
#     return results
#
# print (reg_m(y, x).summary())

#-----------------------------------
#from Tilda.PolliFit.DBIsotope import DBIsotope
#from Tilda.PolliFit.SPFitter import SPFitter
#from Tilda.PolliFit.Spectra.FullSpec import FullSpec
#from datetime import datetime
#import Tilda.PolliFit.BatchFit
#import Tilda.PolliFit.Analyzer
#import Tilda.PolliFit.Tools
#import Tilda.PolliFit.Physics

#path = 'C:/Users/wnoerters/Synology/CloudStation/Databases'
#db = os.path.join(path, 'Ni_Test.sqlite')
#Tools.centerPlot(db, ['64_Ni', '65_Ni','67_Ni'])


# import scipy.optimize as optimize
# import numpy as np
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt


#A = np.array([(1,2,1,5,10), (1,1,2,3,-1), (14,17,21,40,27)])

#def func(data, p0, px, py):
#    return data[:,0]*px + data[:,1]*py + p0
#def func(data, a, b, c):
#    return data[:,0]*a+data[:,1]*b + c

#guess = (3,7,4)
#params, pcov = optimize.curve_fit(func, A[:,:2], A[:,2], guess)
#print(params)

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#ax.scatter((1,2,1,5,10), (1,1,2,3,-1), (14,17,21,40,27))
#plt.show()


# x = np.mgrid[-2:5:120j]
# y = np.mgrid[1:9:120j]
# z = np.mgrid[-5:3:120j]
#
# data = np.concatenate((x[:, np.newaxis],
#                        y[:, np.newaxis],
#                        z[:, np.newaxis]),
#                       axis=1)
#
# # Perturb with some Gaussian noise
# data += np.random.normal(size=data.shape) * 0.4
#
#
# # Calculate the mean of the points, i.e. the 'center' of the cloud
# datamean = data.mean(axis=0)
#
# # Do an SVD on the mean-centered data.
# uu, dd, vv = np.linalg.svd(data - datamean)
#
# # Now vv[0] contains the first principal component, i.e. the direction
# # vector of the 'best fit' line in the least squares sense.
#
# # Now generate some points along this best fit line, for plotting.
#
# # I use -7, 7 since the spread of the data is roughly 14
# # and we want it to have mean 0 (like the points we did
# # the svd on). Also, it's a straight line, so we only need 2 points.
# linepts = vv[0] * np.mgrid[-7:7:2j][:, np.newaxis]
#
# # shift by the mean to get the line in the right place
# linepts += datamean
#
# # Verify that everything looks right.
#
# import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d as m3d
#
# ax = m3d.Axes3D(plt.figure())
# ax.scatter3D(*data.T)
# ax.plot3D(*linepts.T)
# plt.show()