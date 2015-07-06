"""

Created on '30.06.2015'

@author:'simkaufm'

"""

import Driver.Heinzinger.Heinzinger as hz
import Driver.Heinzinger.HeinzingerCfg as hzCfg
import Scratch.freeToUse as ft

# hz0 = hz.Heinzinger(hzCfg.comportHeinzinger0) #start Heinzinger 0
# hz1 = hz.Heinzinger(hzCfg.comportHeinzinger1) #start Heinzinger 1
bla = ft.printer()

print('type in q to quit')
eingabe = ''
while eingabe != 'q':
    eingabe = input('Enter Commandstring: ')
    try:
        print(eval(eingabe))
    except:
        print('Command ' + eingabe + ' not accepted')
print(eingabe)
