# -*- coding: utf-8 -*-
"""
Created on Tue Aug 04 23:22:07 2015

@author: Ilia
"""

from PD_fractals1 import *

fname_prefix = 'combined'

b=1.8
P = 0.01
T = b
S = 0
R = 1
#payoff_mat = np.array([[P,T],[S,R]])
SetPayoffMat(P,T,S,R)

N1 = 100
M1 = 100

f=3

N2 = N1*f
M2 = M1*f

N=[N1,N2]
M=[M1,M2]

indices1 = make_indices(N1,M1)
indices2 = make_indices(N2,M2)
indices=[indices1,indices2]

starting_lattice1 = np.ones((N1,M1),dtype=np.int)
starting_lattice1[N1/2,M1/2] = 0

central_squares = np.arange(N2/2-1,N2/2+2,dtype=np.int).reshape((1,3))
starting_lattice2 = np.ones((N2,M2),dtype=np.int)
starting_lattice2[central_squares.transpose(),central_squares] = 0

starting_lattices=[starting_lattice1,starting_lattice2]

fname_suffix = 'center'

#starting_lattice = np.ones((N,M),dtype=np.int)
#starting_lattice[N/2,M-1] = 0
#starting_lattice[N/2,0] = 0
#fname_suffix = 'wave'

#starting_lattice = MakeRandLattice(N,M,0.9)
#fname_suffix='rand0.9'

num_neighbors = 4

disp_pop = False

if num_neighbors == 4:
    iteration_m = iterate_4_neighbors
else:
    iteration_m = iterate_8_neighbors

#L = SingleGeneration(starting_lattice,indices,iteration_m)
num_generations = 120


from matplotlib import animation
FFMpegWriter = animation.writers['ffmpeg']
writer = FFMpegWriter(fps=25)

fig = plt.figure()
ax1 = plt.subplot(1,2,1)
ax2 = plt.subplot(1,2,2);
ax=[ax1,ax2]

import copy
L = copy.copy(starting_lattices)
pyoffs=[None]*2
img_h = [None]*2
for i in range(len(L)):
    pyoffs[i] = CalcTotalPayoffs(L[i],indices[i],iteration_m)
    if not disp_pop:
        img_h[i] = ax[i].imshow(pyoffs[i],interpolation='none',clim=[0,b*(num_neighbors+1)*0.7])
        
    else:
        img_h[i] = plt.imshow(L[i],interpolation='none')

if not disp_pop:
    plt.colorbar(img_h[i],ax=ax)

fname = '%s_%d_%d_%d_%d_%s_b%.2f_%s.mp4' % \
        (fname_prefix,N[0],M[0],num_generations,num_neighbors,'pop' if disp_pop else '',b,fname_suffix)
with writer.saving(fig,fname,150):
    writer.grab_frame()
    for gen in range(num_generations):
        if gen%10 == 0:
            print( gen)
        L[0] = SingleGeneration(L[0],indices[0],iteration_m)
        for j in [None]*f:
            L[1] = SingleGeneration(L[1],indices[1],iteration_m)
        for i in range(len(L)):
            if not disp_pop:
                pyoffs[i] = CalcTotalPayoffs(L[i],indices[i],iteration_m)
                img_h[i].set_data(pyoffs[i])
            else:
                img_h[i].set_data(L[i])
        writer.grab_frame()

##################################################
