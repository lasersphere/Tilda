"""
Created on 

@author: simkaufm

Module Description:  Module to play around with the sparse array datastructure of PANDA
"""

import pandas as pd
from scipy import sparse
from pandas._sparse import BlockIndex  # seems to be wrong, maybe newer python version and newer pandas version?

a = pd.SparseArray(1, index=range(1), kind='block',
                   sparse_index=BlockIndex(10, [8], [1]),
                   fill_value=0)
b = pd.SparseArray(1, index=range(1), kind='block',
                   sparse_index=BlockIndex(10, [8], [1]),
                   fill_value=0)

sparse.coo_matrix


print(a)
print(a + b)