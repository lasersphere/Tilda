

'''
Created on 04.03.2018 - 18.03.2018

@author: nörtershäuser
'''

import numpy
import os
import numpy as np
from KingFitter import KingFitter
from Scratch.KingFitter2Lines import KingFitter2Lines


analysis_folder = os.path.dirname(__file__)
#db = 'C:/owncloud/Projekte/COLLAPS/Sn/KingPlot_Wilfried/Sn_paper.sqlite'
db = 'U:/owncloud/Projekte/COLLAPS/Sn/KingPlot_Wilfried/Sn_paper.sqlite'

'''
 Run#   Transition, Analyzed by
 Run0 = SP, Christian Gorges
 Run1 = PP, Liss
 '''


'''performing a King fit analysis'''

# king = KingFitter(db, showing=True, litvals=litvals)
# # run = -1
# run='Run1'
# king.kingFit(alpha=849, findBestAlpha=True, run=run)
# king.kingFit(findBestAlpha=False, run=run)
# king.calcChargeRadii(isotopes=isoL, run=run)




'''Der KingFitter berücksichtigt x- und y- Fehler nach dem Algorithmus in York_AmJPhys_72_367(2004)'''

''' perform a king fit based on the SP (Run0, results from Christian) and the PP transition (Run1, Results from Liss). 
here we take only the even isotopes into account
the isotope 118 from the PP transition is excluded since it disagrees with the value from Anselment et al. and shows 
a considerable deviation in both, the King plot of the two transitions with respect to each other and in the king-plot 
wrt the charge radii from muonic atoms and electron scattering
Please note that the y axis should be used for the transition to which the projection should be made.'''
#ACHTUNG! Referenz-Isotop darf nicht in der Liste stehen!

''' King Fit des SP Übergangs mit Korrelationselimination '''
# litvals = {'112_Sn':[-0.748025649,.0077],
#             '114_Sn':[-0.601624554,.0077],
#            '116_Sn':[ -0.464108311,.0077],
#            '118_Sn':[-0.327818629,.0077],
#             '120_Sn':[-0.202198458,.0080],
#             '122_Sn':[-0.093007073,.0077]}#Fricke charge radii
#
# isotopes = {'112_Sn': [0],
#             '114_Sn': [0],
#             '116_Sn': [0],
#             '118_Sn': [0],
#             '120_Sn': [0],
#             '122_Sn': [0],
#             '126_Sn': [0],
#             '128_Sn': [0],
#             '130_Sn': [0],
#             '132_Sn': [0],
#             '134_Sn': [0]}
#
# king = KingFitter(db, litvals, showing=True, ref_run='Run0')
# king.kingFit(alpha=830,findBestAlpha=True, find_slope_with_statistical_error=False, run='Run0')
# king.calcChargeRadii(run='Run0', incl_projected=False)

'''PP INDIVIDUAL 
=================='''

# ''' King Fit des PP Übergangs mit Korrelationselimination, 118 nicht berücksichtigt '''

# litvals = {'112_Sn':[-0.748025649,.0077],
#             '114_Sn':[-0.601624554,.0077],
#            '116_Sn':[ -0.464108311,.0077],
#            # '118_Sn':[-0.327818629,.0077],
#             '120_Sn':[-0.202198458,.0080],
#             '122_Sn':[-0.093007073,.0077]}#Fricke charge radii
#
# isotopes = {'108_Sn': [0],
#             '110_Sn': [0],
#            '112_Sn': [0],
#            '114_Sn': [0],
#            '116_Sn': [0],
# #           '118_Sn': [0],
#            '120_Sn': [0],
#            '122_Sn': [0],
#            '126_Sn': [0],
#            '128_Sn': [0],
#            '130_Sn': [0],
#            '132_Sn': [0],
#            '134_Sn': [0]}
#
# king = KingFitter(db, litvals, showing=True, ref_run='Run1')
# king.kingFit(alpha=830,findBestAlpha=True, find_slope_with_statistical_error=False, run='Run1')
# king.calcChargeRadii(run='Run1', incl_projected=False)

'''SP and PP with Projection PP --> SP  
========================================'''
# litvals = {'112_Sn':[-0.748025649,.0077],
#             '114_Sn':[-0.601624554,.0077],
#            '116_Sn':[ -0.464108311,.0077],
# #           '118_Sn':[-0.327818629,.0077],
#             '120_Sn':[-0.202198458,.0080],
#             '122_Sn':[-0.093007073,.0077]}#Fricke charge radii
#
#
# '''King Fit of both lines '''
# isotopes = {'108_Sn': [0],
#             '110_Sn': [0],
#             '112_Sn': [0],
#             '114_Sn': [0],
#             '116_Sn': [0],
#             # '118_Sn': [0],
#             '120_Sn': [0],
#             '122_Sn': [0],
#             '126_Sn': [0],
#             '128_Sn': [0],
#             '130_Sn': [0],
#             '132_Sn': [0],
#             '134_Sn': [0]}

# king = KingFitter2Lines(db, isotopelist=isotopes, showing=True, plot_mhz=False, ref_run_y='Run0', ref_run_x='Run1')
# king.kingFit2Lines(alpha=-1, run_y='Run0', find_slope_with_statistical_error=False, run_x='Run1', findBestAlpha=True)
#
# ''' now we can also project the isotope shift in the atomic line onto the ionic line.
# They appear in the database as "projected IS" with the run identification of the line on which
# it is projected (here"Run0") '''
# king.calcProjectedIS(run_kp='KingPlotRun1Run0',run_y='Run0',run_x='Run1')

# king2 = KingFitter(db, litvals, showing=True, ref_run='Run0', incl_projected=False)
# king2.kingFit(alpha=830,findBestAlpha=True, find_slope_with_statistical_error=False, run='Run0')
# king2.calcChargeRadii(run='Run0', incl_projected=True)
#


'''Data of Liss for SP (Run2)  
========================================'''

''' King Fit des SP Übergangs mit Korrelationselimination '''
litvals = {'112_Sn':[-0.748025649,.0077],
            '114_Sn':[-0.601624554,.0077],
           '116_Sn':[ -0.464108311,.0077],
           #'118_Sn':[-0.327818629,.0077],
            '120_Sn':[-0.202198458,.0080],
            '122_Sn':[-0.093007073,.0077]}#Fricke charge radii

isotopes = {'112_Sn': [0],
            '114_Sn': [0],
            '116_Sn': [0],
            #'118_Sn': [0],
            '120_Sn': [0],
            '122_Sn': [0],
            '126_Sn': [0],
            '128_Sn': [0],
            '130_Sn': [0],
            '132_Sn': [0],
            '134_Sn': [0]}

king = KingFitter(db, litvals, showing=True, ref_run='Run2')
king.kingFit(alpha=830,findBestAlpha=True, find_slope_with_statistical_error=False, run='Run2')
king.calcChargeRadii(run='Run2', incl_projected=False)