# -*- coding: utf-8 -*-
"""
Created on Mon Aug 03 16:11:00 2015

Based on 'Evolutionary games and spatial chaos by Martin et al.'

@author: Ilia
"""

import numpy as np

from matplotlib import pyplot as plt

def SetPayoffMat(P_,T_,S_,R_):
    global payoff_mat
    payoff_mat = np.array([[P_,T_],[S_,R_]])

def make_indices(N,M):
    indices = {}
    
    ind1 = list(range(1,M))
    ind1.append(0)
    
    indices['left'] = np.array(ind1)
    
    ind1 = [M-1]
    ind1.extend(range(0,M-1))
    
    indices['right'] = np.array(ind1)
    
    indices['hcenter'] = np.arange(M)
    
    ind1 = list(range(1,N))
    ind1.append(0)
    
    indices['up'] = np.array(ind1)
    
    ind1 = [N-1]
    ind1.extend(range(0,N-1))
    
    indices['down'] = np.array(ind1)
    
    indices['vcenter'] = np.arange(N)
    return indices

def reindex(lattice,ind1,ind2):
    return lattice[ind1.reshape((ind1.size,1)),ind2]

def CalcPayoffs(lattice,ind1,ind2):
    L = reindex(lattice,ind1,ind2)
    payoffs = payoff_mat[lattice,L]
    return payoffs

def iterate_8_neighbors():
    for vind in ['up','vcenter','down']:
        for hind in ['left','hcenter','right']:
            #if not (vind == 'vcenter' and hind=='hcenter'):
            if True:
                yield (vind,hind)

def iterate_4_neighbors():
    for vind in ['up','down']:
        yield (vind,'hcenter')
    for hind in ['left','right']:
        yield ('vcenter',hind)
    yield ('vcenter','hcenter')

def CalcTotalPayoffs(lattice,indices,neighbors_iter = iterate_8_neighbors):
    payoffs = np.zeros(lattice.shape)
    for vind,hind in neighbors_iter():
        #print vind,hind
        ind1 = indices[vind]
        ind2 = indices[hind]
        this_payoffs = CalcPayoffs(lattice,ind1,ind2)
        payoffs += this_payoffs
    return payoffs
    
#CalcTotalPayoffs(lattice,indices)

def SingleGeneration(lattice,indices,neighbors_iter = iterate_8_neighbors):
    N = lattice.shape[0]
    M = lattice.shape[1]
    payoffs = CalcTotalPayoffs(lattice,indices,neighbors_iter = neighbors_iter)
    #all_indices = [('vcenter','hcenter')]
    all_indices=[]
    all_indices.extend((vind,hind) for vind,hind in neighbors_iter())
    all_payoff_array = np.zeros(lattice.shape+(len(all_indices),))
    all_lattices_array = np.zeros(lattice.shape+(len(all_indices),),dtype=np.int)
    for z,(vind,hind) in enumerate(all_indices):
        ind1 = indices[vind]
        ind2 = indices[hind]
        all_payoff_array[:,:,z] =  reindex(payoffs,ind1,ind2)
        all_lattices_array[:,:,z] = reindex(lattice,ind1,ind2)
    #return all_payoff_array
    max_indices = np.argmax(all_payoff_array,axis=2)
    new_lattice = all_lattices_array[np.arange(N).reshape((N,1)),np.arange(M),max_indices]
    return new_lattice

from scipy import sparse

def MakeRandLattice(N,M,avg_density):
    return np.array(np.ceil(sparse.rand(N,M,avg_density).toarray()),dtype=np.int)

if __name__ == '__main__':
    #SingleGeneration(lattice,indices)
    
    b=1.9
    P = 0.01
    T = b
    S = 0
    R = 1
    payoff_mat = np.array([[P,T],[S,R]])
    
    
    N = 35
    M = 200
    
    indices = make_indices(N,M)
    
    
    starting_lattice = np.ones((N,M),dtype=np.int)
    starting_lattice[N/2,M/2] = 0
    
    #iteration_m = iterate_4_neighbors
    iteration_m = iterate_8_neighbors
    
    #L = SingleGeneration(starting_lattice,indices,iteration_m)
    
    L = starting_lattice
    
    num_generations = 200
    for gen in range(num_generations):
        if gen%10 == 0:
            print (gen)
        L = SingleGeneration(L,indices,iteration_m)
    
    plt.figure()
    m=plt.imshow(L,interpolation='none')
    
    pyoffs = CalcTotalPayoffs(L,indices,iteration_m)
    plt.figure()
    plt.imshow(pyoffs,interpolation='none',clim=[0,10])
    plt.colorbar()
    
    #####################################
    
    b=1.8
    P = 0.01
    T = b
    S = 0
    R = 1
    payoff_mat = np.array([[P,T],[S,R]])
    
    
    N = 200
    M = 200
    
    indices = make_indices(N,M)
    
    
    starting_lattice = np.ones((N,M),dtype=np.int)
    starting_lattice[N/2,M/2] = 0
    fname_suffix = 'center'
    
    #starting_lattice = np.ones((N,M),dtype=np.int)
    #starting_lattice[N/2,M-1] = 0
    #starting_lattice[N/2,0] = 0
    #fname_suffix = 'wave'
    
    #starting_lattice = MakeRandLattice(N,M,0.9)
    #fname_suffix='rand0.9'
    
    num_neighbors = 8
    
    disp_pop = False
    
    if num_neighbors == 4:
        iteration_m = iterate_4_neighbors
    else:
        iteration_m = iterate_8_neighbors
    
    #L = SingleGeneration(starting_lattice,indices,iteration_m)
    num_generations = 1000
    
    
    from matplotlib import animation
    FFMpegWriter = animation.writers['ffmpeg']
    writer = FFMpegWriter(fps=25)
    
    fig = plt.figure()
    
    L = starting_lattice
    pyoffs = CalcTotalPayoffs(L,indices,iteration_m)
    
    if not disp_pop:
        img_h = plt.imshow(pyoffs,interpolation='none',clim=[0,b*(num_neighbors+1)*0.7])
        plt.colorbar()
    else:
        img_h = plt.imshow(L,interpolation='none')
    
    with writer.saving(fig,'out2_%d_%d_%d_%d_%s_b%.2f_%s.mp4' % \
            (N,M,num_generations,num_neighbors,'pop' if disp_pop else '',b,fname_suffix)
            ,100):
        writer.grab_frame()
        for gen in range(num_generations):
            if gen%10 == 0:
                print( gen)
            L = SingleGeneration(L,indices,iteration_m)
            if not disp_pop:
                pyoffs = CalcTotalPayoffs(L,indices,iteration_m)
                img_h.set_data(pyoffs)
            else:
                img_h.set_data(L)
            writer.grab_frame()
    
    ##################################################
    
    
    
