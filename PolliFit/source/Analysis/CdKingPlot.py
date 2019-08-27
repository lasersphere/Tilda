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
db = os.path.normpath(os.path.join(analysis_folder,
                                        os.path.pardir, os.path.pardir,
                                        'test\\Project\\CdCombined.sqlite'))
#                                        'test\\Project\\CdAtom.sqlite'))
#                                        'test\\Project\\CdIon.sqlite'))

# db = 'C:\\Workspace\\PolliFit\\test\\Project\\CdIon.sqlite'

'''performing a King fit analysis with values and uncertainties taken from Michael's first manuscript version of the charge radii paper'''
# litvals = {'106_Cd': [-0.6983,.0174],
#            '108_Cd': [-0.5124,.0174],
#            '110_Cd': [-0.3333,.0174],
#            '111_Cd': [-0.2889,.0169],
#            '112_Cd': [-0.1592,.0173],
#            '113_Cd': [-0.1160,.0164],
#            '116_Cd': [+0.1349,.0174]}

'''performing a King fit analysis with lambda values, and uncertainties taken from Michael's first manuscript version of the charge radii paper'''
litvals = {'106_Cd': [-0.6712,.0174],
           '108_Cd': [-0.4989,.0174],
           '110_Cd': [-0.3245,.0174],
           '111_Cd': [-0.2811,.0169],
           '112_Cd': [-0.1550,.0173],
           '113_Cd': [-0.1129,.0164],
           '116_Cd': [+0.1314,.0174]}

# king = KingFitter(db, litvals, showing=True, ref_run='Run0')
# king.kingFit(alpha=1000,findBestAlpha=True, run='Run0')
# king.calcChargeRadii(run='Run0', incl_projected=False)

'''performing a King fit analysis with literature values and uncertainties from Michaels worksheet IS-Merged.xlsx'''
'''this is only the statistical uncertainty and can be used to disentangle statistical and systematical uncertainties'''
# litvals = {'106_Cd': [-0.68934,0.00292],
#            '108_Cd': [-0.51241,0.00225],
#            '110_Cd': [-0.33333,0.00160],
#            '111_Cd': [-0.28884,0.00160],
#            '112_Cd': [-0.15924,0.00101],
#            '113_Cd': [-0.11601,0.00160],
#            '116_Cd': [+0.13490,0.00102]}

'''the same with lambda'''
# litvals = {'106_Cd': [-0.6712,0.00292],
#            '108_Cd': [-0.4989,0.00225],
#            '110_Cd': [-0.3245,0.00160],
#            '111_Cd': [-0.2811,0.00160],
#            '112_Cd': [-0.1550,0.00101],
#            '113_Cd': [-0.1129,0.00160],
#            '116_Cd': [+0.1314,0.00102]}


# king = KingFitter(db, litvals, showing=True, ref_run='Run0')
# king.kingFit(alpha=1040,findBestAlpha=True, run='Run0')
# king.calcChargeRadii(run='Run0', incl_projected=False)

'''Der KingFitter berücksichtigt x- und y- Fehler nach dem Algorithmus in York_AmJPhys_72_367(2004)
 Example for performing a King fit of 2 lines when only
 the values of one line are included in the database
 to run this, the db CdAtom must be assigned to db.
 Please note that the y axis should be used for the transition to which the projection should be made.
 Ionendaten als x-Achse (only stable ones)'''
isotopeShiftList = {'106_Cd': [2.991119, 1E-3*np.sqrt(np.square(2.2)+np.square(6.2))],
           '108_Cd': [2.193991, 1E-3*np.sqrt(np.square(2.2)+np.square(4.6))],
           '110_Cd': [1.432204, 1E-3*np.sqrt(np.square(2.3)+np.square(3.0))],
           '111_Cd': [1.314298, 1E-3*np.sqrt(np.square(2.2)+np.square(2.3))],
           '112_Cd': [ .674618, 1E-3*np.sqrt(np.square(2.2)+np.square(1.5))],
           '113_Cd': [ .555244, 1E-3*np.sqrt(np.square(2.3)+np.square(0.8))],
           '116_Cd': [-.526466, 1E-3*np.sqrt(np.square(2.2)+np.square(1.5))]}

# king = KingFitter2Lines(db, isotopeList=isotopeShiftList, showing=True, ref_run_x='Run1')
# king.kingFit2Lines(alpha=-1, run_y='Run0', find_slope_with_statistical_error=False, findBestAlpha=True)
''' this esymple shows how to perform a king fit based on 2 lines that are in th same database. In the
 example of CdCombined.sqlite (attach riht database to db above) "Run0" is the ionic data
 and "Run1" the atomic data.
 with the isotopeList one can exclude some isotopes from the KingPlot. Here, the isotopes 117-120
 and 122 were excluded since the IS in the ionic transition was hampered most probably by changing
 pressure in ISCOOL. These isotopes therefore also showed a deviation from the regression  line
 when they were included
 Please note that the y axis should be used for the transition to which the projection should be made.'''
isotopes = {'106_Cd': [0],
            '107_Cd': [0],
            '108_Cd': [0],
            '109_Cd': [0],
            '110_Cd': [0],
            '111_Cd': [0],
            '112_Cd': [0],
            '113_Cd': [0],
            '115_Cd': [0],
            '116_Cd': [0],
#            '117_Cd': [0],
#            '118_Cd': [0],
#            '119_Cd': [0],
#            '120_Cd': [0],
            '121_Cd': [0],
#            '122_Cd': [0],
            '123_Cd': [0],
            '124_Cd': [0],
            '126_Cd': [0]}
# king = KingFitter2Lines(db, isotopelist=isotopes, showing=True, plot_mhz=False, ref_run_y='Run0', ref_run_x='Run2')
# king.kingFit2Lines(alpha=-1, run_y='Run0', find_slope_with_statistical_error=True, run_x='Run2', findBestAlpha=True)

''' now we can also project the isotope shift in the atomic line onto the ionic line.
They appear in the database as "projected IS" with the run identification of the line on which
it is projected (here"Run0") '''
# king.calcProjectedIS(run_kp='KingPlotRun2Run0',run_y='Run0',run_x='Run2')

'''Example for calculating the charge radii based on a database
where isotope shifts (marked "shifts") and projected isotope shifts
(marked "projected IS") are in the database.
if you want to run it, the database CdCombined must be assigned to db
 Run#   Charge State, Analyzed by
 Run0 = Ion, Michael
 Run1 = Atom, Nadja
 Run2 = Atom, Deyan 
 '''
king = KingFitter(db, litvals, showing=True, ref_run='Run0', incl_projected=False)
king.kingFit(alpha=0,findBestAlpha=False, find_slope_with_statistical_error=False, run='Run0')
king.calcChargeRadii(run='Run0', incl_projected=False)