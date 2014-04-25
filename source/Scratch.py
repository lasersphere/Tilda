'''
Created on 31.03.2014

@author: hammen
'''


import Physics
import numpy as np

I = 0
Jl = 2.5
Ju = 3.5

#a = np.float(2.3)
#print(a)
#print(type(round(a)))

trans = Physics.HFTrans(I, Jl, Ju)
 
print(trans)
split = Physics.HFLineSplit(500, 0, 2000, 0, trans)
print(split)


#Physics.sixJ(Jl, 0.5, I, 1.5, Ju, 1)

intens = Physics.HFInt(I, Jl, Ju, trans)
print(intens)

