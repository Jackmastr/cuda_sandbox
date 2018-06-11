import numpy as np 
import scipy.signal as sps
import mf_redux

print "3x3 window with 73x1 image"

mf_redux.MedianFilter(ws=3, n=1, m=73, timing=True)
mf_redux.MedianFilter(ws=3, n=1, m=73, timing=True)
mf_redux.MedianFilter(ws=3, n=1, m=73, timing=True)
mf_redux.MedianFilter(ws=3, n=1, m=73, timing=True)

print "\n3x3 window with 1x73 image"

mf_redux.MedianFilter(ws=3, n=73, m=1, timing=True)
mf_redux.MedianFilter(ws=3, n=73, m=1, timing=True)
mf_redux.MedianFilter(ws=3, n=73, m=1, timing=True)
mf_redux.MedianFilter(ws=3, n=73, m=1, timing=True)

print "\n5x5 window with 93x93 image"

mf_redux.MedianFilter(ws=3, n=93, timing=True)
mf_redux.MedianFilter(ws=3, n=93, timing=True)
mf_redux.MedianFilter(ws=3, n=93, timing=True)
mf_redux.MedianFilter(ws=3, n=93, timing=True)

# print "\n3x3 window with 2048x2048 image"

# mf_redux.MedianFilter(ws=3, n=2048, m=2048, timing=True)
# mf_redux.MedianFilter(ws=3, n=2048, m=2048, timing=True)
# mf_redux.MedianFilter(ws=3, n=2048, m=2048, timing=True)
# mf_redux.MedianFilter(ws=3, n=2048, m=2048, timing=True)



print "\n9x9 window with 997x997 image"

mf_redux.MedianFilter(ws=9, n=997, timing=True)
mf_redux.MedianFilter(ws=9, n=997, timing=True)
mf_redux.MedianFilter(ws=9, n=997, timing=True)
mf_redux.MedianFilter(ws=9, n=997, timing=True)


### JUNE 8 11:00 AM ###

# 3x3 window with 73x1 image
# THIS FUNCTION:  372.356445312 ms
# SCIPY MEDFILT 0.120448000729 ms
# THIS FUNCTION:  169.163070679 ms
# SCIPY MEDFILT 0.204640001059 ms
# THIS FUNCTION:  172.044006348 ms
# SCIPY MEDFILT 0.167999997735 ms
# THIS FUNCTION:  170.972640991 ms
# SCIPY MEDFILT 0.131807997823 ms

# 3x3 window with 1x73 image
# THIS FUNCTION:  173.245925903 ms
# SCIPY MEDFILT 0.129472002387 ms
# THIS FUNCTION:  185.689987183 ms
# SCIPY MEDFILT 0.227871999145 ms
# THIS FUNCTION:  173.246459961 ms
# SCIPY MEDFILT 0.153919994831 ms
# THIS FUNCTION:  173.778747559 ms
# SCIPY MEDFILT 0.145983994007 ms

# 5x5 window with 93x93 image
# THIS FUNCTION:  175.999847412 ms
# SCIPY MEDFILT 2.86240005493 ms
# THIS FUNCTION:  169.462524414 ms
# SCIPY MEDFILT 2.82899188995 ms
# THIS FUNCTION:  164.101409912 ms
# SCIPY MEDFILT 2.84774398804 ms
# THIS FUNCTION:  167.226364136 ms
# SCIPY MEDFILT 2.8751039505 ms

# 9x9 window with 997x997 image
# THIS FUNCTION:  204.197540283 ms
# SCIPY MEDFILT 1292.91540527 ms
# THIS FUNCTION:  181.769699097 ms
# SCIPY MEDFILT 1452.16601562 ms
# THIS FUNCTION:  197.205444336 ms
# SCIPY MEDFILT 1304.50415039 ms
# THIS FUNCTION:  182.313568115 ms
# SCIPY MEDFILT 1264.5078125 ms

### JUNE 11 10:00 AM ###

# 3x3 window with 73x1 image
# THIS FUNCTION:  169.807678223 ms
# SCIPY MEDFILT 0.0877119973302 ms
# THIS FUNCTION:  2.83980798721 ms
# SCIPY MEDFILT 0.0553920008242 ms
# THIS FUNCTION:  2.49667191505 ms
# SCIPY MEDFILT 0.0558080002666 ms
# THIS FUNCTION:  2.58236789703 ms
# SCIPY MEDFILT 0.0561279989779 ms

# 3x3 window with 1x73 image
# THIS FUNCTION:  2.60511994362 ms
# SCIPY MEDFILT 0.0596799999475 ms
# THIS FUNCTION:  2.46063995361 ms
# SCIPY MEDFILT 0.0552319996059 ms
# THIS FUNCTION:  2.47459197044 ms
# SCIPY MEDFILT 0.0545919984579 ms
# THIS FUNCTION:  2.46016001701 ms
# SCIPY MEDFILT 0.0549440011382 ms

# 5x5 window with 93x93 image
# THIS FUNCTION:  3.02246403694 ms
# SCIPY MEDFILT 1.57433605194 ms
# THIS FUNCTION:  4.63820791245 ms
# SCIPY MEDFILT 1.60131204128 ms
# THIS FUNCTION:  4.22070407867 ms
# SCIPY MEDFILT 1.54985594749 ms
# THIS FUNCTION:  3.95968008041 ms
# SCIPY MEDFILT 1.54771196842 ms

# 9x9 window with 997x997 image
# THIS FUNCTION:  34.1038703918 ms
# SCIPY MEDFILT 1359.34936523 ms
# THIS FUNCTION:  19.271232605 ms
# SCIPY MEDFILT 1277.9375 ms
# THIS FUNCTION:  13.9361276627 ms
# SCIPY MEDFILT 1330.84509277 ms
# THIS FUNCTION:  12.8464317322 ms
# SCIPY MEDFILT 1164.88293457 ms