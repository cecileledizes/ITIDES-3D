import numpy as np
from scipy.special import hankel1
from scipy.signal import convolve2d
from mpi4py import MPI
import time

def compute_wavefield(comm, casename, list_pmax):
    """ 
    Inputs :
        comm = the MPI.world for parallel compution
        casename = dictionnary containing all the parameters for this case
            Global info
                "casename" : "testcase",  
                "path"    : "./Calculs/",
                "Nnodes"  : number of nodes,
                "Ncores"  : number of cores (=32*Nnodes),
            Physical parameters
                "N_type" : Brunt-Vaissala frequency type : 'constant', 'file'
                "N"      : Brunt-Vaissala frequency value (Hz) or filename
                "omega"  : Tidal frequency (Hz)
                "f"      : Coriolis frequency (Hz)
                "U0"     : Velocity of the tide (cm/s) in the x-direction (cm/s)
                "V0"     : Velocity of the tide in the y-direction (cm/s)
                "rho0"   : Reference density (kg/m3)
                "Ho"     : Ocean depth (negative value) (m)
            Topography
                "topofile" : topography file (npy format) ; contain 'x'(m),'y'(m),'h'(m),'dhx','dhy'
                "L"        : Standard length of the topo (for adim) (m)
                "Gamma"    : Standard height of the topo (m)
            Sources
                "sourceCalculation" : if True, the source dstribution is computed,
                "list_pmax"   : list of maximal mode used for source distribution
                "solver_name" : name of petsc4py solver used for system resolution (default 'gmres')
                "pc_name"     : name of preconditionner used for system resolution (default 'jacobi')
                "rtol", "atol", "divtol:, "maxit" :  convergence criteria (default = 1e-10,1e-15,100,1000)
            Wavefield
                "wavefieldCalculation" : if True, the wavefield is computed
                "nmodes" : wavefield reconstructed for modes between 1 and nmodes
                "Xmax", "Ymax"  : Domain used for reconstruction [-Xmax, Xmax]x[-Ymax,Ymax]
                "plotWavefield" : if True, the wavefield s plotted (modify directly the main if we want to plot other things)

    Outputs: 
        wavefile : contains a dictionnary with the following keys :
    """
    comm = comm.tompi4py()
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank==0:
        print("\nCOMPUTING WAVEFIELD")
    
    # Get parameters
    path = casename['path']
    nmodes = casename['nmodes']
    Xmax, Ymax = casename['Xmax'], casename["Ymax"]
    H0, Gamma, L = casename["H0"], casename["Gamma"], casename["L"]
    N, omega, f = casename["N"], casename["omega"], casename["f"]
    mu = np.sqrt((N**2-omega**2)/(omega**2 - f**2))      # Slope parameter mu
    beta = f/omega                                       # Frequency ratio beta

    # Define coordinates on which to calculate the convolution and on which to plot the field
    sourcefile = "source_pmax"+str(list_pmax[0])+".npy"
    solution = np.load(path+sourcefile, allow_pickle=True).item()
    XX, YY = solution['XX'], solution['YY']
    Delta = solution['Delta']
    Xadd, Yadd = np.arange(np.max(XX)+Delta,Xmax+np.max(XX)-Delta/2,Delta), np.arange(np.max(YY)+Delta,Ymax+np.max(YY)-Delta/2,Delta)
    Xconv, Yconv = np.append(-Xadd[::-1], XX, 0), np.append(-Yadd[::-1], YY, 0)
    Xconv, Yconv = np.append(Xconv,Xadd), np.append(Yconv,Yadd)
    Ymesh, Xmesh = np.meshgrid(Yconv, Xconv)
    dist = np.sqrt(Xmesh**2+Ymesh**2)
    
    nx, ny = len(Xconv)-len(XX)+1, len(Yconv)-len(YY)+1
    Xplot, Yplot = np.linspace(-Xmax, Xmax, nx), np.linspace(-Ymax, Ymax, ny)

    # Share modes on processes
    comm.Barrier()
    plist = np.arange(1,nmodes+1)
    allmodes = [(pmax, p) for pmax in list_pmax for p in plist]
    ptot = len(allmodes)
    pstart, pend = rank*(nmodes//size)+min(rank,nmodes%size)+1, (rank+1)*(nmodes//size)+min(rank+1,nmodes%size)+1
    ## Evaluate the modal field at points (Xplot, Yplot) : process 'rank' deals with mode from pstart to pend
    w = np.zeros((pend-pstart,nx,ny), dtype=np.complex64)
    dw_X = np.zeros((pend-pstart,nx,ny), dtype=np.complex64)
    dw_Y = np.zeros((pend-pstart,nx,ny), dtype=np.complex64)
    for pmax in list_pmax :
        tinit = time.time()
        if rank==0 :
            print("\nCalculating convolutions for Pmax = "+str(pmax))
           
        ## Load source distribution
        sourcefile = "source_pmax"+str(pmax)+".npy"
        solution = np.load(path+sourcefile, allow_pickle=True).item()
        SS_matrix = solution['S']
        HH = solution['HH']

        ## Compute convolutions
        ip = 0
        for p in range(pstart, pend):
            A_p = 1+2j/np.pi*(np.log(p*Delta/2) + np.euler_gamma - 3 + np.log(2) + np.pi/2)
            H0    = np.ma.masked_array(hankel1(0,p*dist), mask=(dist==0)).filled(A_p)
            dH0_X = np.ma.masked_array(-p*Xmesh/dist*hankel1(1,p*dist), mask=(dist==0)).filled(0.0)
            dH0_Y = np.ma.masked_array(-p*Ymesh/dist*hankel1(1,p*dist), mask=(dist==0)).filled(0.0)
            phi = np.sin(p*HH)*SS_matrix/2j/np.pi
            w[ip,:,:] = convolve2d(H0,phi,mode='valid')*Delta**2
            dw_X[ip,:,:] = convolve2d(dH0_X,phi,mode="valid")*Delta**2
            dw_Y[ip,:,:] = convolve2d(dH0_Y,phi,mode="valid")*Delta**2
            ip = ip+1
        
        comm.Barrier()
        tend = time.time()
        tinit2 = time.time()
        if rank==0:
            print("Done ! dt = "+str(tend-tinit))
            print("\nGathering data on rank=0 ...")
    
        ## Gather all results on process rank = 0
        # Compute sizes and offsets for Gatherv
        dist_mode = np.array([nmodes//size+min(r+1,nmodes%size)-min(r,nmodes%size) for r in range(size)])
        sizes_memory = nx*ny*dist_mode
        offsets = np.zeros(size)
        offsets[1:] = np.cumsum(sizes_memory)[:-1]
        # Prepare buffer for Gatherv
        w_all, dwx_all, dwy_all = None, None, None
        if rank==0:
            w_all = np.empty((nmodes,nx,ny), dtype=np.complex64)
            dwx_all = np.empty((nmodes,nx,ny), dtype=np.complex64)
            dwy_all = np.empty((nmodes,nx,ny), dtype=np.complex64)
        comm.Gatherv(w, recvbuf=[w_all, sizes_memory.tolist(), offsets.tolist(), MPI.COMPLEX], root=0)
        comm.Gatherv(dw_X, recvbuf=[dwx_all, sizes_memory.tolist(), offsets.tolist(), MPI.COMPLEX], root=0)
        comm.Gatherv(dw_Y, recvbuf=[dwy_all, sizes_memory.tolist(), offsets.tolist(), MPI.COMPLEX], root=0)
        tend = time.time()
        if rank==0:
            print("Done ! dt = "+str(tend-tinit))

        ## Save wavefield file
        if (rank == 0):
            modes = np.tile(np.arange(1,nmodes+1), (ny,nx,1)).transpose()       # size = (nm, nx, ny)

            data = dict([])
            data['X'], data['Y'] = Xplot, Yplot
            data['w'] = w_all                                      # Velocity adimensionned by Utide = np.sqrt(np.real(U0)**2+np.real(V0)**2)
            data['p'] = 1j/np.pi*mu**2/modes * w_all               # Pressure adimensionned by rho0*Utide*(omega**2-f**2)/omega*Ho
            data['u'] = mu/modes*(dwx_all+1j*beta*dwy_all)         # Velocity adimensionned by Utide
            data['v'] = mu/modes*(dwy_all-1j*beta*dwx_all)         # Velocity adimensionned by Utide
            # Jx = rho0 * Utide**2 * (omega**2-f**2)/omega * Ho**2/pi * int(<p u>)
            #    = rho0 * Utide**2 * (omega**2-f**2)/omega * Ho**2/pi * Sum Re(pi/4 (conj(p_n) u_n))
            #    = Sum Jn_x
            # Jn_x = rho0 * Utide**2 * (omega**2-f**2)/omega * Ho**2/pi * Re(pi/4 (conj(p_n) u_n))
            #      = rho0 * Utide**2 * (omega**2-f**2)/omega * Ho**2/pi * Re(pi/4 * (-1j)/pi/modes**2 * mu**3 * np.conj(w_all) * (dwx_all+1j*beta*dwy_all))
            #      = rho0 * Utide**2 * (omega**2-f**2)/omega * Ho**2 * Re((-1j)/4/pi/modes**2 * mu**3 * np.conj(w_all) * (dwx_all+1j*beta*dwy_all))
            data['Jx'] = np.real( -1j/4/np.pi/modes**2 * mu**3 * np.conjugate(w_all) * (dwx_all+1j*beta*dwy_all))   # Flux adim by rho0*Utide**2*(omega**2-f**2)/omega*Ho**2
            data['Jy'] = np.real( -1j/4/np.pi/modes**2 * mu**3 * np.conjugate(w_all) * (dwy_all-1j*beta*dwx_all))
            # Conversion rate 
            ix_p, iy_p = np.argmin(np.abs(Xplot-np.max(XX))), np.argmin(np.abs(Yplot-np.max(YY)))
            ix_n, iy_n = np.argmin(np.abs(Xplot-np.min(XX))), np.argmin(np.abs(Yplot-np.min(YY)))
            convRate  = np.sum(data['Jx'][:,ix_p,iy_n:iy_p+1],axis=1) - np.sum(data['Jx'][:,ix_n,iy_n:iy_p+1],axis=1)
            convRate += np.sum(data['Jy'][:,ix_n:ix_p+1,iy_p],axis=1) - np.sum(data['Jy'][:,ix_n:ix_p+1,iy_n],axis=1)
            # Clsy = rho0 * Utide**2 * mu * (omega**2-f**2)/omega * hmax**2 * np.pi**(3/2)/8 * L
            data['C'] = convRate*Delta*mu/np.pi      # C adim by rho0*Utide**2*(omega**2-f**2)/omega*Ho**3

            wavefile = "wavefile_pmax"+str(pmax)+"_nmodes"+str(nmodes)+'.npy'
            print("Save "+wavefile)
            np.save(path+wavefile,data,allow_pickle=True)
    return 0


