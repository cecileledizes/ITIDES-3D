# ITIDES-3D

### Description
This program is BEM solver for internal tide generation by three-dimensional isolated seamounts. For now, the code only supports constant stratification, but the version for vertically-varying stratification is currently being validated.

### Citing this Work

Please cite the following reference in any published work, presentations, or derived projects that make use of this code or its results.

> **Le Dizes, C.**, Grisouard, N., Thual, O., & Mercier, M. J. (2025).  
> *Three-dimensional modelling of internal tide generation over isolated seamounts in a rotating ocean.*  
> **Journal of Fluid Mechanics, 1022**, A5.  
> [https://doi.org/10.1017/jfm.2025.10647](https://doi.org/10.1017/jfm.2025.10647)

This article provides a detailed description of the numerical method implemented in this source code, as well as validation cases and theoretical background.

### Getting started
#### Dependencies
This program is a parallel code in Python, using the following packages : numpy, scipy, os, sys, json, time, logging, petsc4py, mpi4py, matplolib.

#### Executing program
To run the program, the user 
```python main.py casename.json```


#### Program structure
The file **main.py** calls the two following functions : 
* *compute_source_distribution(comm,casefile)* from **source_distribution.py** :
This function reads the casefile containing informations on the frequencies, the maximum number of modes and the solver to use, as well as the topography file (with the data on the bathymetry). The function then computes the matrix M in parallel using the dimensionless values of the 
topography and solves the boundary integral equation using a linear solver from the petsc4py package.
Once the source distribution is computed, it is saved in the sourcefile.
* *compute_wavefield(comm, casefile, list_pmax)*  from **wavefield.py** :
This function reads the casefile as well as the sourcefile, to compute the convolution gn*S and saves in the output file wavefile the dimensionless values
of the modal velocities -un,vn,wn-, the modal pressure -pn-, as well as the modal energy flux -Jn_x, Jn_y- and conversion rate Cn.

All the input and output files are detailled in the next section.


Several other python files are also contained in the project : 
* **topographyGeneration.py** : generates topography files for simple idealised topographies (gaussian, bumps, ...)
* **generateCasefile.py** : generates the casefile (in the json format) and the topography file
* **plots.py** : contains all the functions to plot the topography, the source distribution or the wavefield
* various test files ...


### Input files
#### Casefile
The casefile is a .json file containing the following dictionnary, with information on the frequencies omega and f, the stratification and the topography.
```
casefile = {
    "path"     : "dir",            # Directory in which the output files are saved
    "casename" : 'casename',       # Name of case

    # Physical parameters
    "N_type" : 'constant',         # Brunt-Vaissala frequency type : 'constant', 'file'
    "N"      : 0.002,              # Brunt-Vaissala frequency value (Hz) or filename
    "omega"  : 0.00014 ,           # Tidal frequency (Hz)
    "f"      : 0.00001,            # Coriolis frequency (Hz)
    "U0"     : 0.04,               # Velocity of the tide (cm/s) in the x-direction (cm/s)
    "V0"     : 0,                  # Velocity of the tide in the y-direction (cm/s) - complex for elliptic tide
    "rho0"   : 1e3,                # Reference density (kg/m3)
    "H0"     : -3000,              # Ocean depth (m)

    # Topography
    "topofile" : 'topofile.npy',   # Name of the npy file that describes the topography
    "L"        : 10000,            # Standard length of the topo (for adim) (m)
    "Gamma"    : 1000,             # Standard height of the topo (m)

    # Sources
    "sourceCalculation" : True,    # if True, the source distribution is computed
    "plotSources" : True,          # if True, the source distribution is plotted
    "list_pmax"   : [100,150],     # List of maximal mode used for source distribution
    "solver_name" :'gmres',        # Name of petsc4py solver used for system resolution (default 'gmres')
    "pc_name"     :'None',         # Name of preconditionner used for system resolution (default 'None')
    "rtol"        : 1e-8,          # Convergence criteria (default = 1e-10)
    "atol"        : 1e-15,         # Convergence criteria (default = 1e-15)
    "divtol"      : 100,           # Divergence criteria (default = 100)
    "max_it"      : 1500,          # Max number of iterations for system resolution (default = 1000)

    # Wavefield
    "wavefieldCalculation" : False, # if True, the wavefield is computed
    "plotWavefield" : False,        # if True, the wavefield is plotted
    "nmodes" : 20,                  # Wavefield reconstructed for modes between 1 and nmodes
    "Xmax"   : 6*Xmax,              # Domain used for reconstruction [-Xmax, Xmax]x[-Ymax,Ymax]
    "Ymax"   : 6*Xmax
}
```

#### Topography 
The topography file is a .npy file containing the discrete topography. It is a .npy file with the following dictionary. Topography files can be generated by the program topographyGeneration.py for simple seamounts.
```
topofile = {
    "delta" : float,           # Size of square cell (m)
    
    # Flattened topography
    "x"     : array (Ntot),    # Array of x-coordinate (m) (size Ntot = nX*nY = total number of cells)
    "y"     : array (Ntot),    # Array of y-coordinate (m)
    "h"     : array (Ntot),    # Array of topography height (m)
    "dh_x"  : array (Ntot),    # Array of topography x-slope
    "dh_y"  : array (Ntot),    # Array of topography y-slope
    
    # For rectangular topography
    "xx"    : array (nX),      # Array of x-coordinate (size nX = number of cells in the x-direction)
    "yy"    : array (nY),      # Array of y-coordinate (size nY = number of cells in the y-direction)
    "hh"    : array (nX,nY),   # Array of topography height
    "dhh_x" : array (nX,nY),   # Array of topography x-slope
    "dhh_y" : array (nX,nY),   # Array of topography y-slope
}
```

#### Stratification
For variable stratification - not implemented yet !


### Output files
#### Source distribution
```
sourcefile = { 
    "N"    : float,           # Brunt-Vaissala frequency value (Hz) or filename
    "omega": float,           # Tidal frequency (Hz)
    "f"    : float,           # Coriolis frequency (Hz)
    'Ho'   : float,           # Ocean depth (m) - negative value
    'Delta': float,           # Dimensionless size of square cell
    'X'    : array (Ntot),    # Dimensionless x-coordinates
    'Y'    : array (Ntot),    # Dimensionless y-coordinates
    'S'    : array (Ntot),    # Distribution of sources along the topography
    'H'    : array (Ntot),    # Dimensionless height of topography
    'dH_X' : array (Ntot),    # Dimensionless x-derivatives of the topography
    'dH_Y' : array (Ntot),    # Dimensionless y-derivatives of the topography
    'XX'   : array (nX),      # Dimensionless y-coordinates
    'YY'   : array (nY),      # Dimensionless y-coordinates
    'HH'   : array (nX,nY),   # Dimensionless height of topography
}
```


#### Wavefield
```
wavefield = {
    'X' : array (nXplot),                   # Dimensionless x-coordinates for which to plot the wavefield
    'Y' : array (nYplot),                   # Dimensionless y-coordinates for which to plot the wavefield
    'w' : array (nmodes, nXplot, nYplot),   # Velocity adimensionned by Utide = np.sqrt(np.real(U0)**2+np.real(V0)**2)
    'u' : array (nmodes, nXplot, nYplot),   # Velocity adimensionned by Utide
    'v' : array (nmodes, nXplot, nYplot),   # Velocity adimensionned by Utide
    'p' : array (nmodes, nXplot, nYplot),   # Pressure adimensionned by rho0*Utide*(omega**2-f**2)/omega*Ho
    'Jx': array (nmodes, nXplot, nYplot),   # Energy flux (x-component) adimensionned by rho0*Utide**2*(omega**2-f**2)/omega*Ho**2
    'Jy': array (nmodes, nXplot, nYplot),   # Energy flux (y-component) adimensionned by rho0*Utide**2*(omega**2-f**2)/omega*Ho**2
    'C' : array (nmodes),                   # Conversion rate adimensionned by rho0*Utide**2*(omega**2-f**2)/omega*Ho**3
}
```

## Test case
Not released yet !

## Author
* **Cécile Le Dizes**

And contributors :
* Matthieu Mercier, CNRS, IMFT
* Nicolas Grisouard, University of Toronto
* Olivier Thual, IMFT, CERFACS

## License

This project is under the license ``GNU General Public License `` - see [LICENSE](LICENSE) for more information

