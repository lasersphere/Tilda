"""

Created on '30.06.2015'

@author:'simkaufm'

"""

import Driver.Heinzinger.Heinzinger as hz
import Driver.Heinzinger.HeinzingerCfg as hzCfg

heinz = hz.Heinzinger(hzCfg.comportHeinzinger0)
print(heinz.setVoltage(100), 'setVolt')
print(heinz.getProgrammedVolt(), 'prgoVOlt')
print(heinz.getVoltage(), 'measVolt ')

print(heinz.getCurrent())
print(heinz.setVoltage(0))
print(heinz.getVoltage())