'''
Created on 31.03.2014

@author: gorges
'''
import os
from KingFitter import KingFitter

# db = 'C:\\Workspace\\PolliFit\\test\\Project\\Fe.sqlite'
analysis_folder = os.path.dirname(__file__)
db = os.path.normpath(os.path.join(analysis_folder,
                                        os.path.pardir, os.path.pardir,
                                        'test\\Project\\Fe.sqlite'))
#'''performing a King fit analysis'''
litvals = {'57_Fe':[0.124,.028],
            '58_Fe':[0.283,.028],
           '54_Fe':[-0.313,.026]}

king = KingFitter(db, litvals,showing=True)
king.kingFit(alpha=0,findBestAlpha=True)
king.calcChargeRadii()


