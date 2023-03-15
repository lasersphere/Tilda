"""
Created on 

@author: simkaufm

Module Description:
"""
import numpy as np
from copy import copy

num_of_tracks = 3
num_of_steps = 5
num_of_pmts = [5, 4, 3]
# tracks = [[(list(np.random.randint(0, 100, 5))) for i in range(0, num_of_pmts[tr])] for tr in range(0, num_of_tracks)]
# pmts = [[str(i + tr) for i in range(0, 5 - tr)] for tr in range(0, num_of_tracks)]
# print(tracks)
# print(pmts)

tracks_fix = [[[0, 10, 38, 29, 99], [50, 67, 18, 67, 84], [35, 14, 85, 68, 94], [63, 52, 44, 11, 35],
               [31, 56, 55, 13, 90]],
              [[60, 15, 82, 87, 32], [31, 7, 54, 10, 11], [35, 6, 71, 37, 39], [47, 41, 0, 99, 31]],
              [[16, 84, 74, 19, 55], [42, 2, 50, 32, 92], [54, 72, 75, 82, 35]]]
pmts_fix = [['0', '1', '2', '3', '4'], ['0', '2', '3', '4'], ['0', '3', '4']]

pmts_flat = [item for sublist in pmts_fix for item in sublist]
pmts_ok = [pmt_name for pmt_name in pmts_fix[0] if pmts_flat.count(pmt_name) == num_of_tracks]
tracks_ok = [[pmt for pmt_ind, pmt in enumerate(tracks_fix[tr_ind]) if pmts_fix[tr_ind][pmt_ind] in pmts_ok]
             for tr_ind, tr in enumerate(tracks_fix)]

print(pmts_ok)
print(tracks_ok)
