import os
import sys
import json
from source_distribution_tau import *
from plots import *
from petsc4py import PETSc
from datetime import date


if __name__ == "__main__":
    # Set-up parallelisation
    comm = PETSc.COMM_WORLD
    rank = comm.Get_rank()                  # Rank of process
    size = comm.Get_size()                  # Total numbers of processes

    ## CASE DEFINITION AND SET UP
    # Get casefile name for the argument in the script
    if len(sys.argv)==1 :
        print('ERROR : Casefile not specified !')
        sys.exit()
    else :
        casefile = str(sys.argv[1])
    
    # Read casefile
    with open(casefile, 'r') as f:
        casename = json.load(f)
    path = casename["path"] if "path" in list(casename.keys()) else os.path.dirname(casefile)
    
    # Check topography file
    topofile = casename['topofile'] if "topofile" in list(casename.keys()) else None
    if (topofile==None)|(os.path.isfile(path+topofile)==False):
        print('ERROR : Topography file not found !')
        sys.exit()
    
    if rank==0 :
        print("\nCasefile : "+casefile)
        print("Date : "+str(date.today()))
        plotTopo(path+topofile,field='hh',savefig=True)
    

    ## SOURCE DISTRIBUTION COMPUTATION
    comm.Barrier()         # all processes wait here
    compute_source_distribution(comm,casename)
    
    # Plot sources
    if (rank==0):
        list_pmax = casename['list_pmax']     # Max number of modes
        for pmax in list_pmax:
            sourcefile = "source_pmax"+str(pmax)+".npy"
            plotSource(path+sourcefile, savefig=True)
    
