import numpy as np
import time
from petsc4py import PETSc
from scipy.special import hankel1
from sympy.integrals.quadrature import gauss_gen_laguerre
#from plots import plotSource
import logging

def compute_source_distribution(comm,casename) :
    """ 
    Inputs :
        comm = the MPI.world for parallel compution
        casename = dictionnary containing the parameters for this case
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
                "topofile" : topography file (npy format) ; contain 'x'(m),'y'(m),'h'(negative depth m),'dhx','dhy'
                "L"        : Standard length of the topo (for adim) (m)
                "Gamma"    : Standard height of the topo (m)
            Sources
                "sourceCalculation" : if True, the source distribution is computed,
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
        sourcefile : contains a dictionnary with the following keys :
            'N', 'f', 'omega' : stratification frequency, coriolis frequency and barotropic forcing frequency (Hz)
            'Ho' : ocean depth (m)
            'S': distribution of sources S(X,Y) along the topography (size nX*nY)
            'X', 'Y': dimensionless square coordinates (size Ntot)
            'XX', 'YY': dimensionless square coordinates (size nX / nY)
            'Delta': non-dimensional size of square cell
            'H' : dimensionless height of topography (size Ntot)
            'dH_X', 'dH_Y' : stretched derivatives of the topography (size Ntot)
            'HH' : dimensionless height of topography filled in matrix (size nX*nY)
    """
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Get parameters
    path = casename['path']
    topofile = casename['topofile']
    U0, V0 = casename["U0"], casename["V0"]
    Utide = max(np.sqrt(np.real(U0)**2+np.real(V0)**2),np.sqrt(np.imag(U0)**2+np.imag(V0)**2))
    N, omega, f = casename["N"], casename["omega"], casename["f"]
    mu = np.sqrt((N**2-omega**2)/(omega**2 - f**2))       # Slope parameter mu
    beta = f/omega                                        # Frequency ratio beta

    # Set Log file
    if rank==0:
        print("\nCOMPUTING SOURCE DISTRIBUTION")
        logging.basicConfig(filename=path+'sources.log', filemode='w', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        logger = logging.getLogger()
        logger.info("Casefile : "+casename['casename'])
        logger.info("")
        logger.info("COMPUTING SOURCE DISTRIBUTION")
        logger.info("")
    else : 
        logger = 'None'
    

    ### COMPUTING THE NON DIMENSIONNAL TOPOGRAPHY
    topo = np.load(path+topofile, allow_pickle=True).item()       # load the topography file
    Ho = np.abs(casename['H0']) if "H0" in list(casename.keys()) else np.abs(np.min(topo['h']))
    X, Y  = topo['x']*np.pi/mu/Ho, topo['y']*np.pi/mu/Ho   # X_m, Y_n
    nX,nY = len(np.unique(X)),len(np.unique(Y))
    Delta = topo['delta']*np.pi/mu/Ho
    H     = (np.pi/Ho)*(topo['h']+Ho)    # h[ind] = h(x[ind],y[ind]) into its non-dimensional form
    dH_X  = mu*topo['dh_x']              # dhx into its non-dimensional form
    dH_Y  = mu*topo['dh_y']              # dhy into its non-dimensional form
    Ntot  = len(H)

    ### SOLVE MATRIX PROBLEM
    comm.Barrier()

    # Set-up solver
    if rank==0:
        start_time = time.time()
        print("Setting solver")
        logger.info("Setting solver")
    solver_name = casename["solver_name"] if "solver_name" in list(casename.keys()) else 'gmres'
    pc_name     = casename["pc_name"] if "pc_name" in list(casename.keys()) else 'None'
    rtol    = casename["rtol"] if "rtol" in list(casename.keys()) else 1e-10
    atol    = casename["atol"] if "atol" in list(casename.keys()) else 1e-15
    divtol  = casename["divtol"] if "divtol" in list(casename.keys()) else 100.0
    max_it  = casename["max_it"] if "max_it" in list(casename.keys()) else 1000
    solv = PETSc.KSP()
    solv.create(comm=comm)
    solv.setType(solver_name)
    solv.setTolerances(rtol=rtol, atol=atol, divtol=divtol, max_it=max_it)
    pc = solv.getPC()
    if pc_name!="None":
        pc.setType(pc_name)
    solv.setPCSide(1)  # Right preconditionning
    
    def monitor(ksp, its, rnorm, rank=rank, logger=logger):
        if rank==0:
            logger.info("      its, rnorm = "+str((its,rnorm)))
    solv.setMonitor(monitor)
    if rank==0:
        print('   Solver : '+str(solv.getType()))
        print('   Pre-conditionner : '+str(pc.getType()))
        logger.info('   Solver : '+str(solv.getType()))
        logger.info('   Pre-conditionner : '+str(pc.getType()))
        logger.info('   Tolerance : rtol, atol, divtol, max_it = '+str(solv.getTolerances()))
        logger.info('')

    # Filling the M matrix in parallel   
    M = PETSc.Mat()
    M.create(comm=comm)
    M.setSizes([Ntot, Ntot])
    M.setType('dense')
    rstart, rend = M.getOwnershipRange()
    M.zeroEntries()
    M.assemblyBegin()
    M.assemblyEnd()
    p0 = 0
    tau = casename["tau"]
    ptau = int(np.round(tau/Delta,0))

    # For faster calculation of H0, H1, I0 and I1
    Xp = np.arange(0,np.max(X)-np.min(X)+Delta/2,Delta)
    Yp = np.arange(0,np.max(Y)-np.min(Y)+Delta/2,Delta)
    Ymesh, Xmesh = np.meshgrid(Yp,Xp)

    for pmax in casename["list_pmax"] :
        
        # Small p values from p0 such that : p Delta <= tau
        if ptau>=p0+1:
            newM = M.copy()
            if rank==0:
                start_time = time.time()
                print('Assembling matrix for p = '+str(p0+1)+' to '+str(min(ptau,pmax))+'...')
                logger.info('Assembling matrix for p = '+str(p0+1)+' to '+str(min(ptau,pmax))+'...')
            
            psmall = np.arange(p0+1,min(ptau,pmax)+1)
            A_p = 1+2j/np.pi*(np.log(psmall*Delta/2) + np.euler_gamma - 25/12 + np.log(2)/3 + np.pi/3)
            B = np.sum(1j/2/np.pi - psmall**2*Delta**2/24)

            Xtile, Ytile = np.tile(Xmesh, (len(psmall),1,1)), np.tile(Ymesh, (len(psmall),1,1))
            ptile = np.transpose(np.tile(psmall, (np.shape(Xmesh)[1], np.shape(Xmesh)[0], 1)))
            H0 = hankel1(0,ptile*np.sqrt(Xtile**2+Ytile**2))
            H1 = hankel1(1,ptile*np.sqrt(Xtile**2+Ytile**2))

            Htile = np.tile(H,(len(psmall),1))
            ptile2 = np.transpose(np.tile(psmall, (np.shape(H)[0], 1)))
            sinpH = np.sin(ptile2*Htile)
            cospH = np.cos(ptile2*Htile)
            for row in range(rstart, rend):
                alpha_x, alpha_y = dH_X[row]-1j*beta*dH_Y[row], dH_Y[row]+1j*beta*dH_X[row]
                
                # Diagonal
                oldVal = M.getValues(row,row)
                val = np.sum(sinpH[:,row]**2 * A_p)
                val += (dH_X[row]**2+dH_Y[row]**2) * B
                newM.setValues(row,row,oldVal+1/2j/np.pi*(val))

                # Extra-diagonal
                listcol = np.append(np.arange(0,row), np.arange(row+1,Ntot))
                for col in listcol:
                    oldVal = M.getValues(row,col)
                    dX, dY = (X[row]-X[col]), (Y[row]-Y[col])
                    dmn_ij = np.sqrt(dX**2+dY**2)
                    iX, iY = int(np.round(np.abs(dX)/Delta,0)), int(np.round(np.abs(dY)/Delta,0))
                    val = np.sum(H0[:,iX,iY] * sinpH[:,col] * sinpH[:,row])
                    val += (dX*alpha_x + dY*alpha_y)/dmn_ij * np.sum(H1[:,iX,iY] * sinpH[:,col] * cospH[:,row])
                    newM.setValues(row,col,oldVal+1/2j/np.pi*(val))
            newM.assemblyBegin() # Assembling the matrix makes it "useable".
            newM.assemblyEnd()
            M = newM.copy()

            if rank==0:
                end_time = time.time()
                print("... DONE ! dt = "+str(end_time-start_time)) 
                logger.info("... DONE ! dt = "+str(end_time-start_time))

        # Larger p values such as : p Delta > tau
        if pmax>=ptau+1:
            newM = M.copy()
            if rank==0:
                print('Assembling matrix for p = '+str(max(p0,ptau)+1)+' to '+str(pmax)+'...')
                logger.info('Assembling matrix for p = '+str(max(p0,ptau)+1)+' to '+str(pmax)+'...')
                start_time = time.time()

            plarge = np.arange(max(p0,ptau)+1,pmax+1)

            #Ymesh, Xmesh = np.meshgrid(np.unique(Y), np.unique(X))
            Xtile, Ytile = np.tile(Xmesh, (len(plarge),1,1)), np.tile(Ymesh, (len(plarge),1,1))
            ptile = np.transpose(np.tile(plarge, (np.shape(Xmesh)[1], np.shape(Xmesh)[0], 1)))

            ptile2 = np.transpose(np.tile(plarge, (np.shape(H)[0], 1)))
            Htile = np.tile(H,(len(plarge),1))
            exppH_P, exppH_N = np.exp(1j*ptile2*Htile), np.exp(-1j*ptile2*Htile)

            # Approximation for the slowly varying values I0 and I1
            nI = 6
            xl, wl = gauss_gen_laguerre(nI,-0.5,2*nI)
            xL = [float(T) for T in xl]
            wL = [float(T) for T in wl]
            xn, wn = gauss_gen_laguerre(nI,+0.5,2*nI)
            xN = [float(T) for T in xn]
            wN = [float(T) for T in wn]
            dtile = np.sqrt(Xtile**2+Ytile**2)
            dtile = np.ma.masked_array(dtile, dtile==0).filled(1.0)
            I0 = np.zeros_like(ptile, dtype='complex')
            for l in range(len(xL)):
                I0 += wL[l]*np.sqrt(1/(2+1j*xL[l]/ptile/dtile))
            I1 = np.zeros_like(ptile, dtype='complex')
            for l in range(len(xN)):
                I1 += wN[l]*np.sqrt((2*dtile+1j*xN[l]/ptile))
            
            for row in range(rstart, rend):
                alpha_x, alpha_y = dH_X[row]-1j*beta*dH_Y[row], dH_Y[row]+1j*beta*dH_X[row]

                # Diagonal : the cell 'row' is divided by n cells
                oldVal = M.getValues(row,row)
                val = 0.0+0.0j
                for p in plarge:
                    n = int(np.ceil(p*Delta/tau))
                    newpoints = -Delta/2 + Delta/n/2*np.arange(1,n+1,1)
                    ynew, xnew = np.meshgrid(newpoints, newpoints)
                    xnew = xnew.flatten()
                    ynew = ynew.flatten()

                    Hp = H[row] + dH_X[row]*xnew + dH_Y[row]*ynew
                    sinHp, cosHp = np.sin(p*Hp), np.cos(p*Hp)

                    A = 1+2j/np.pi*(np.log(p*Delta/n/2) + np.euler_gamma - 25/12 + np.log(2)/3 + np.pi/3)
                    B = 1j/2/np.pi - p**2*Delta**2/n**2/24
                    val += (A*np.sum(sinHp**2) + (dH_X[row]**2+dH_Y[row]**2)*B*len(Hp)) / n**4
                    for prow in range(len(sinHp)) :
                        dp = np.sqrt((xnew[prow]-xnew)**2+(ynew[prow]-ynew)**2)
                        dp[prow] = 1
                        lineL = hankel1(0,p*dp) * sinHp
                        lineL[prow] = 0.0
                        lineN = -((xnew[prow]-xnew)*alpha_x + (ynew[prow]-ynew)*alpha_y)/dp * hankel1(1,p*dp) * sinHp

                        val += (sinHp[prow]*np.sum(lineL) - cosHp[prow]*np.sum(lineN)) / n**4
                newM.setValues(row,row,oldVal+1/2j/np.pi*(val))

                # Extra-diagonal : approximation of H0 and H1 as a slowly varying value I0 and I1 times e^ipr
                listcol = np.append(np.arange(0,row), np.arange(row+1,Ntot))
                for col in listcol:
                    oldVal = M.getValues(row,col)
                    
                    dX, dY = (X[row]-X[col]), (Y[row]-Y[col])
                    dmn_ij = np.sqrt(dX**2+dY**2)
                    iX, iY = int(np.round(np.abs(dX)/Delta,0)), int(np.round(np.abs(dY)/Delta,0))
                    ax, ay = dX/dmn_ij, dY/dmn_ij
                    if (dH_X[row]==0)&(ax==0):
                        UmnXP = plarge*Delta/2
                        UmnXN = plarge*Delta/2
                    else:
                        UmnXP = np.sin(plarge*Delta/2*(dH_X[row]+ax))/(dH_X[row]+ax)
                        UmnXN = np.sin(plarge*Delta/2*(dH_X[row]-ax))/(dH_X[row]-ax)
                    if (dH_Y[row]==0)&(ay==0):
                        UmnYP = plarge*Delta/2
                        UmnYN = plarge*Delta/2
                    else:
                        UmnYP = np.sin(plarge*Delta/2*(dH_Y[row]+ay))/(dH_Y[row]+ay)
                        UmnYN = np.sin(plarge*Delta/2*(dH_Y[row]-ay))/(dH_Y[row]-ay)
                    
                    if (dH_X[col]==0)&(ax==0):
                        UijXP = plarge*Delta/2
                        UijXN = plarge*Delta/2
                    else:
                        UijXP = np.sin(plarge*Delta/2*(dH_X[col]-ax))/(dH_X[col]-ax)
                        UijXN = np.sin(plarge*Delta/2*(dH_X[col]+ax))/(dH_X[col]+ax)
                    if (dH_Y[col]==0)&(ay==0):
                        UijYP = plarge*Delta/2
                        UijYN = plarge*Delta/2
                    else:
                        UijYP = np.sin(plarge*Delta/2*(dH_Y[col]-ay))/(dH_Y[col]-ay)
                        UijYN = np.sin(plarge*Delta/2*(dH_Y[col]+ay))/(dH_Y[col]+ay)

                    Smnij = -2j*(exppH_P[:,row]*UmnXP*UmnYP-exppH_N[:,row]*UmnXN*UmnYN)
                    Sijmn = -2j*(exppH_P[:,col]*UijXP*UijYP-exppH_N[:,col]*UijXN*UijYN)
                    Cmnij = 2*(exppH_P[:,row]*UmnXP*UmnYP+exppH_N[:,row]*UmnXN*UmnYN)

                    L_plarge =  np.exp(1j*plarge*dmn_ij) * I0[:,iX,iY]/plarge**4/np.sqrt(plarge) * Smnij * Sijmn
                    val = 2*np.exp(-1j*np.pi/4)/np.pi/Delta**4/np.sqrt(dmn_ij) * np.sum(L_plarge)

                    N_plarge =  np.exp(1j*plarge*dmn_ij) * I1[:,iX,iY]/plarge**4/np.sqrt(plarge) * Cmnij * Sijmn
                    val += 2*np.exp(-1j*3*np.pi/4)/np.pi/Delta**4 * (dX*alpha_x+dY*alpha_y)/dmn_ij**2 * np.sum(N_plarge)
                    newM.setValues(row,col,oldVal+1/2j/np.pi*val)
            newM.assemblyBegin() # Assembling the matrix makes it "useable".
            newM.assemblyEnd()
            M = newM.copy()
            
            if rank==0:
                end_time = time.time()
                print("... DONE ! dt = "+str(end_time-start_time))
                logger.info("... DONE ! dt = "+str(end_time-start_time))
        

        # Filling right-hand side in parallel
        S, rhs = M.createVecs()
        indexes = np.arange(rstart,rend,1,dtype='int32')
        rhs.setValues(indexes, (U0*dH_X[rstart:rend] + V0*dH_Y[rstart:rend])/Delta**2/mu/Utide)
        rhs.assemblyBegin()
        rhs.assemblyEnd()
        '''
        # Initial guess : WTA solution
        S.setValues(indexes, -np.pi/np.sin(H[rstart:rend]) * rhs.getValues(indexes))
        S.assemblyBegin()
        S.assemblyEnd()
        solv.setInitialGuessNonzero(True)
        '''

        # Solving linear system
        solv.setOperators(M)
        stopcrit = max(atol,rhs.norm()*rtol)
        if rank==0:
            start_time = time.time()
            print("Solving linear system for pmax = "+str(pmax)+" ...")
            logger.info("Solving linear system for pmax = "+str(pmax)+" ...")
            logger.info('   Stop if rnorm = ||M*S - b|| < max(atol, rtol*||b||) = '+str(stopcrit))
        
        comm.Barrier()
        solv.solve(rhs, S)
        
        if rank==0:
            end_time = time.time()
            print("... DONE ! dt = "+str(end_time-start_time))
            logger.info("... DONE ! dt = "+str(end_time-start_time))
        
        
        # Reconstruct and save source file
        scatter, S_vec = PETSc.Scatter.toZero(S)
        scatter.scatter(S, S_vec, False, PETSc.Scatter.Mode.FORWARD)
        comm.barrier()
        if rank==0:
            sourcefile = "source_pmax"+str(pmax)+".npy"    # Source distribution
            
            XX, counts = np.unique(X, return_counts=True)
            YY, counts = np.unique(Y, return_counts=True)
            nX, nY = len(XX), len(YY)
            SS_matrix = np.zeros((nX,nY),dtype='complex')
            HH = np.zeros((nX,nY))
            dHH_X, dHH_Y = np.zeros((nX,nY)), np.zeros((nX,nY))
            for ind in range(Ntot):
                i = np.where(XX==X[ind])[0][0]
                j = np.where(YY==Y[ind])[0][0]
                SS_matrix[i,j] = S_vec[ind]
                HH[i,j] = H[ind]
                dHH_X[i,j] = dH_X[ind]
                dHH_Y[i,j] = dH_Y[ind]

            solution = dict([])
            solution['N'], solution['omega'], solution['cf'], solution['Ho'], solution['pmax'] = N, omega, f, Ho, pmax
            solution['U0'], solution['V0'] = U0, V0
            solution['X'], solution['Y'] = X, Y
            solution['XX'], solution['YY'] = XX, YY
            solution['S'] = SS_matrix
            solution['Delta'] = Delta
            solution['H'], solution['HH'] = H, HH
            solution['dH_X'], solution['dH_Y'] = dH_X, dH_Y
            np.save(path+sourcefile,solution,allow_pickle=True)
            print("Saving file "+sourcefile+"\n")
            logger.info("Saving file "+sourcefile+"\n")

        p0 = pmax
    
    return 0