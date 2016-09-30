'''
Created on 31.03.2014

@author: gorges
'''

from KingFitter import KingFitter

db = 'V:/User/Christian/databases/Fe.sqlite'

'''performing a King fit analysis'''
litvals = {'57_Fe':[0.124,.028],
            '58_Fe':[0.283,.028],
           '54_Fe':[-0.313,.026]}

king = KingFitter(db, litvals,alpha=0,findBestAlpha=False,showing=True)
king.calcChargeRadii()