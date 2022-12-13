import numpy as np

file = "../../data/copper_cavity_loop_antennas/AutoSave47.s1p"
header = 1312
footer = 1300
every = 40
comments = '!'
delimiter = '\t'

data =  np.loadtxt(open(file,'rt').readlines()[header::every], comments=comments,delimiter=delimiter)
np.savetxt(f"../../data/copper_cavity_loop_antennas/AutoSave47_every{every}.s1p",data, fmt='%f')
