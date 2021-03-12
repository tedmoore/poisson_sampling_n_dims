# Ted Moore
# ted@tedmooremusic.com
# www.tedmooremusic.com
# inspiration: https://www.youtube.com/watch?v=flQgnCUxHlw&list=LLF6FNfqNM4Tm7Z3Fu1wIUdg&index=2540

import math
import numpy as np
import time
import argparse
import csv

def generateSamples(n_dims,r,k=30,verbose=False):
    cell_side = r / math.sqrt(2)
    n_cells_per_dim = int(math.ceil(1.0 / cell_side))
    array_shape = np.ones(n_dims) * n_cells_per_dim
    array_shape = [int(v) for v in array_shape]

    grid = np.ndarray(shape=array_shape,buffer=np.ones(pow(n_cells_per_dim,n_dims)))
    grid *= -1

    saved_points = []
    saved_index = 0
    spawn_points = []

    spawn_points.append(np.ones(n_dims) * 0.5)

    candidatesChecked = 1
    while len(spawn_points) > 0:
        spawn_index = int(np.random.rand() * len(spawn_points))
        spawn_center = spawn_points[spawn_index]
        candidateAccepted = False
        for i in range(k):
            offset = randomPoint(n_dims,r)
            candidate = spawn_center + offset
            grid_pos = getGridPos(candidate,cell_side)
            if isValid(candidate,saved_points,grid,n_cells_per_dim,r,grid_pos,n_dims,verbose):
                saved_points.append(candidate)
                spawn_points.append(candidate)
                grid[tuple(grid_pos)] = saved_index
                saved_index += 1
                candidateAccepted = True
                break
        if verbose:
            print("candidates checked",candidatesChecked,"n saved points",len(saved_points))
        candidatesChecked +=1
        if not candidateAccepted:
            del spawn_points[spawn_index]
    
    return saved_points

def isValid(candidate,saved_points,grid,n_cells_per_dim,r,grid_pos,n_dims,verbose):
    if (candidate >= 0).all() and (candidate <= 1).all():
        total_neighbors = 5 ** n_dims
        if total_neighbors < len(saved_points):
            neighbor_offset = np.array([int(v) for v in np.ones(n_dims) * -2])
            while (neighbor_offset >= -2).all():
                if verbose:
                    print("neighbor offset",neighbor_offset)
                neighbor_indices = grid_pos + neighbor_offset
                if (neighbor_indices >= 0).all() and (neighbor_indices < n_cells_per_dim).all():
                    at_grid = grid[tuple(neighbor_indices)]
                    #print("at_grid",at_grid)
                    if at_grid != -1:
                        other_point = saved_points[int(at_grid)]
                        dist = getDistSquared(candidate,other_point)
                        if dist < r*r:
                            return False
                neighbor_offset = getNextNeighborOffset(neighbor_offset,n_dims)
            return True
        else:
            for other in saved_points:
                if getDistSquared(other,candidate) < r*r:
                    return False
            return True
    else:
        return False

def getNextNeighborOffset(no,n_dims):
    if (no == 2).all():
        return np.ones(n_dims) * -3
    
    i = n_dims - 1

    while i >= 0:
        val = no[i] + 1
        if val <= 2:
            no[i] = val
            return no
        else:
            # we have to carry
            no[i] = -2
            i -= 1


def getDistSquared(a,b):
    return np.sum((a - b) ** 2)

def getGridPos(candidate, cell_side):
    return [int(v) for v in candidate / cell_side]

def getNeighborOffsets(n_dims,verbose):
    neighbor_offsets = []
    for i in range(int(5**n_dims)):
        vec = []
        for j in range(n_dims):
            val = int(math.floor(i / (5**j)) % 5)
            vec.append(val)
        vec.reverse()
        vec = np.array(vec) - 2
        if verbose:
            print(vec)
        neighbor_offsets.append(vec)
    return neighbor_offsets
    
def randomPoint(n_dims,r):
    u = np.random.normal(0,1,n_dims)
    norm = np.sum(u**2)**(0.5)
    return (u / norm) * ((np.random.rand() * r) + r)

def makeFile(n_dims,r,k,verbose):
    startTime = time.process_time()
    samples = generateSamples(n_dims,r,k,verbose)
    samples = np.array(samples)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    filePath = f'generated_samples/poisson_sample_set_ndims={n_dims}_npoints={len(samples)}_r={r:.2f}_k={k}_{timestamp}.csv'
    with open(filePath,"w") as csvfile:
        writer = csv.writer(csvfile)
        for row in samples:
            writer.writerow(row)

    print("n_dims   ",n_dims)
    print("k        ",k)
    print("r        ",r)
    print("time     ",time.process_time() - startTime)
    print("n samples",len(samples))
    print("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-k",action='store',dest='k',type=int,default=20,help='Number of failures looking for a spot before giving up')
    parser.add_argument("-d",action='store',dest='n_dims',type=int,default=2,help='Number of dimensions')
    parser.add_argument("-r",action='store',dest='r',type=float,default=0.1,help='Minimum radius of closeness')
    parser.add_argument("-v",action='store_true',dest='verbose',default=False,help='Verbose flag')
    args = parser.parse_args()

    makeFile(args.n_dims,args.r,args.k,args.verbose)
