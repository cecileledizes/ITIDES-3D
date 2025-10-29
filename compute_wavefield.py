import os
import sys
import json
from wavefield import *
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
    
    
    ## WAVEFIELD CALCULATION 
    comm.Barrier()
    wavefieldCalculation = casename['wavefieldCalculation'] if "wavefieldCalculation" in list(casename.keys()) else True
    list_pmax = [casename['list_pmax'][-1]]
    sourcefile = "source_pmax"+str(list_pmax[0])+".npy"
    if os.path.isfile(path+sourcefile)==False:
        print('ERROR : Source file not found !')
        sys.exit()
    
    compute_wavefield(comm, casename, list_pmax)
    
    # Plot wavefile 
    plotWavefield = casename['plotWavefield'] if "plotWavefield" in list(casename.keys()) else True
    nmodes = casename['nmodes']               # Number of modes (to reconstruct the field)
    H0, Gamma, L = np.abs(casename["H0"]), casename["Gamma"], casename["L"]
    N, omega, f = casename["N"], casename["omega"], casename["f"]
    mu = np.sqrt((N**2-omega**2)/(omega**2 - f**2))
    if (rank==0)&(plotWavefield==True):
        for pmax in list_pmax:
            sourcefile = "source_pmax"+str(pmax)+".npy"
            wavefile = "wavefile_pmax"+str(pmax)+"_nmodes"+str(casename['nmodes'])+".npy"
            
            figname = wavefile.split("_")[0]+"_"+wavefile.split("_")[1]+"_p"+str(nmodes)+"_"
            print('Read sourcefile and wavefile ...')
            wavefield = np.load(path+wavefile, allow_pickle=True).item()
            source = np.load(path+sourcefile, allow_pickle=True).item()
            print('... Done !')

            planeYPlot(path,source,wavefield,nmodes,field='w',cstY=0.0,figname=figname+"w_Y0.png")
            planeYPlot(path,source,wavefield,nmodes,field='u',cstY=0.0,figname=figname+"u_Y0.png")
            planeYPlot(path,source,wavefield,nmodes,field='v',cstY=0.0,figname=figname+"v_Y0.png")
            planeZPlot(path,wavefield,nmodes,field='w',cstZ=-1.0,figname=figname+"w_Z1.png")
            planeZPlot(path,wavefield,nmodes,field='u',cstZ=-1.0,figname=figname+"u_Z1.png")
            planeZPlot(path,wavefield,nmodes,field='v',cstZ=-1.0,figname=figname+"v_Z1.png")
            # planeZPlot(path,wavefile,nmodes,field='Jx',cstZ=-1.0,figname=figname+"Jx.png")
            # planeZPlot(path,wavefile,nmodes,field='Jy',cstZ=-1.0,figname=figname+"Jy.png")
            plotConversionRate(path,wavefield,nmodes,Cadim=(Gamma**2*L*mu*np.pi**(3/2)/8)/H0**3,figname=figname)

            planeZVectors(path,sourcefile,wavefield,nmodes,field='J',cstZ=-1.0,plot=True,savefig=True,scale=150,figname=figname)
            
            Rmax = casename['Xmax']
            Rmin = Rmax/3
            R = np.linspace(1.1*Rmin, 0.9*Rmax, 6)
            norm = colors.Normalize(1.1*Rmin, 0.9*Rmax, 6)
            fig, ax = plt.subplots(2,1)
            for r in R:
                c = cm.inferno(norm(np.round(r,2)))
                radialFlux(path,source,wavefield,nmodes,R=r,ax=ax,savefig=False,color=c,linestyle='solid', label="R = "+str(np.round(r,2)))
            ax[1].legend()
            ax[0].grid()
            ax[1].grid()

    plt.show()
