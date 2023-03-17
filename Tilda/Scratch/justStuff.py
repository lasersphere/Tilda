"""

Created on '27.05.2015'

@author:'simkaufm'

"""

import numpy as np

x = np.arange(9.)

print(x)
print(np.where( x > 5 ))
x[np.where( x > 3.0 )]               # Note: result is 1D.

np.where(x < 5, x, -1)

for i in range(8):
    print(i)