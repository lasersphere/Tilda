"""
Created on 

@author: simkaufm

Module Description: On 25th & 26nd Nov. 2015 Tilda was run several times for testing Tilda itself,
but also for testing if the asymetry in the peaks could be caused by the scanning direction.

Therefore
    track0 is: -2 to -1.698804
    track1 is: -2 to -1.698804 (invert direction = TRUE)
    track3 is: -1.698804 to -2

Nevertheless, ion current was really low and also deviating of around 10%,
whis for sure will influence the results.
"""

import Tools


"""
databases:
"""

db = 'R:\Projekte\TRIGA\Measurements and Analysis_Simon' + \
     '\Tilda Offline Tests 15_11\BothDaysCombined\BothDaysCombined.sqlite'

'''crawl'''
Tools.crawl(db)

