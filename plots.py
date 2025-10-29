import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.ticker as tck
import plotly.colors as co
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import animation
from scipy.special import hankel1
from scipy.interpolate import RegularGridInterpolator
#import cmocean

#plt.rcParams['text.usetex'] = True

# Topography
def plotTopoPoints(topofile, field='h', savefig=True):
    topo = np.load(topofile, allow_pickle=True).item()
    x, y = topo['x'], topo['y']
    nX, nY = len(np.unique(x)), len(np.unique(y))
    h = topo[field]
    Ntot = len(x)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    norm = colors.Normalize(np.min(h), np.max(h))
    color_values = cm.viridis(norm(h.tolist()))
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title("Dimensional topography (m) : "+str(Ntot)+" cells")
    label=field
    if field=='h':
        ax.set_zlim(bottom=-3000,top=0)
        label += " (m)"
    ax.scatter(x/1000, y/1000,h,color=color_values,marker="s")
    #ax.bar3d(x/1000, y/1000, bottom, width, depth, h-bottom, color=color_values, shade=False)
    ax.set_zlabel(label)
    ax.set_box_aspect((nX,nY,max(nX,nY)))
    if savefig :
        figname = topofile.split(".npy")[0]+".png"
        plt.savefig(figname)
    return 0


def plotTopo(topofile, field='hh', savefig=True):
    topo = np.load(topofile, allow_pickle=True).item()
    x, y = topo['xx'], topo['yy']
    Y, X = np.meshgrid(y, x)
    nX, nY = len(x), len(y)
    h = topo[field]

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(X/1000, Y/1000, h, cmap='viridis')
    ax.set_xlabel('x (km)')
    ax.set_ylabel('y (km)')
    ax.set_title("Dimensional topography (m) : "+str(len(x))+"x"+str(len(y)))
    if field=='hh':
        ax.set_zlim(bottom=np.min(h),top=0)
        ax.set_zlabel('z (m)')
    ax.set_box_aspect((nX,nY,min(nX,nY)))
    if savefig :
        figname = topofile.split(".npy")[0]+".png"
        plt.savefig(figname)


def plotTopo3D(topofile, field='hh', savefig=True):
    topo = np.load(topofile, allow_pickle=True).item()
    x, y = topo['xx'], topo['yy']
    Y, X = np.meshgrid(y, x)
    h = topo[field]+3000
    hmax = np.max(h)

    xmid, ymid = (x[:-1]+x[1:])/2, (y[:-1]+y[1:])/2
    Ymid, Xmid = np.meshgrid(ymid,xmid)
    hmid = (h[:-1,:]+h[1:,:])/2
    hmid = (hmid[:,:-1]+hmid[:,1:])/2
    
    fig1 = go.Figure(data=[go.Surface(
            #contours = {"z": {"show": True, "start": -3000+hmax/10, "end": -3000+hmax, "size": hmax/5}},
            x=x/1000,
            y=y/1000,
            z=-3000+h,
            surfacecolor=np.log(h),
            cmin = 0.5,
            cmax = 7,
            colorscale="BuPu", #surfacecolor=np.ones((len(XX),len(YY))),
            colorbar_nticks=3, # colorbar ticks correspond to isosurface values
            opacity=1.0,
            showscale=False #colorbar_x=-0.07
        )])
    fig1.add_scatter3d(x=Xmid.flatten()/1000, y=Ymid.flatten()/1000, z=hmid.flatten()-3000+50, mode='markers', 
                marker=dict(size=1.5, color="red"))
    line_marker = dict(color="rgb(150,150,150)", width=2)
    for xx, yy, zz in zip(X, Y, h-3000):
        fig1.add_scatter3d(x=xx/1000, y=yy/1000, z=zz+10, mode='lines', line=line_marker, name='')
    
    X, Y = np.meshgrid(x,y)
    for xx, yy, zz in zip(X, Y, h-3000):
        fig1.add_scatter3d(x=xx/1000, y=yy/1000, z=zz+10, mode='lines', line=line_marker, name='')

    fig1.update_layout(scene = dict(
                    xaxis_title='x (km)',
                    xaxis = dict(
                         backgroundcolor="rgb(230,230,230)",
                         gridcolor="grey",
                         showbackground=True,
                         zerolinecolor="grey",
                         range=[-np.max(x)/1000-4,np.max(x)/1000+4]),
                    yaxis_title='y (km)',
                    yaxis = dict(
                        backgroundcolor="rgb(185,185,185)",
                        gridcolor="grey",
                        showbackground=True,
                        zerolinecolor="grey",
                        range=[-np.max(y)/1000-4,np.max(y)/1000+4]),
                    zaxis_title='z (m)',
                    zaxis = dict(
                        backgroundcolor="rgb(255,255,255)",
                        gridcolor="grey",
                        showbackground=False,
                        range=[-3000,0],
                        zerolinecolor="grey")))
    fig1.update_coloraxes(colorbar_len=0.7, colorbar_thickness=10,colorbar_xpad=5)
    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': '3Dimage',
            'height': 700,
            'width': 1000,
            'scale':10 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }
    fig1.show(config=config)


def plotTopoBox3D(topofile, field='hh', savefig=True):
    topo = np.load(topofile, allow_pickle=True).item()
    x, y = topo['xx'], topo['yy']
    h = topo[field]+3000
    hmax = np.max(h)

    Xplot = np.linspace(-np.max(x)/1000-5,np.max(x)/1000+5,5)
    Yplot = np.linspace(-np.max(y)/1000-5,np.max(y)/1000+5,5)
    Zplot = np.linspace(-3000,0,5)
    nX, nY, nZ = len(Xplot), len(Yplot), len(Zplot)
    Yd, Xd = np.meshgrid(Yplot,Xplot)
    
    fig1 = go.Figure(data=[go.Surface(
            #contours = {"z": {"show": True, "start": -3000+hmax/10, "end": -3000+hmax, "size": hmax/5}},
            x=x/1000,
            y=y/1000,
            z=-3000+h,
            #surfacecolor=np.log(h),
            #cmin = 0.5,
            #cmax = 7,
            colorscale="BuPu", #surfacecolor=np.ones((len(XX),len(YY))),
            colorbar_nticks=3, # colorbar ticks correspond to isosurface values
            opacity=1.0,
            showscale=False #colorbar_x=-0.07
        ),
        go.Surface(               # Plane Z = cstZ
            x=Xd,
            y=Yd,
            z=-np.zeros((len(Xd),len(Yd))),
            colorscale='gray',
            surfacecolor=np.ones((len(Xd),len(Yd))),
            opacity=0.5,
            cmin=-0.0015, #np.min(np.real(data['w'].flatten())),
            cmax=0.0015,
            showscale=False #colorbar_x=-0.07
        ),
        go.Surface(               # Plane Y=-cstY
            x=np.transpose(np.array([Xplot]*nZ)),
            y=np.min(Yplot)*np.ones((nX,nZ)),
            z=np.array([Zplot]*len(Yplot)),
            colorscale='gray',
            opacity=0.5,
            surfacecolor=np.ones((len(Xd),len(Yd))),
            cmin=-0.0015, #np.min(np.real(data['w'].flatten())),
            cmax=0.0015,
        ),
        go.Surface(               # Plane Y=+cstY
            x=np.transpose(np.array([Xplot]*nZ)),
            y=np.max(Yplot)*np.ones((nX,nZ)),
            z=np.array([Zplot]*len(Yplot)),
            colorscale='gray',
            opacity=0.5,
            surfacecolor=np.ones((len(Xd),len(Yd))),
            cmin=-0.0015, #np.min(np.real(data['w'].flatten())),
            cmax=0.0015,
        ),
        go.Surface(               # Plane X=-cstX
            x=np.min(Xplot)*np.ones((nY,nZ)),
            y=np.transpose(np.array([Yplot]*nZ)),
            z=np.array([Zplot]*len(Xplot)),
            colorscale='gray',
            opacity=0.5,
            surfacecolor=np.ones((len(Xd),len(Yd))),
            cmin=-0.0015, #np.min(np.real(data['w'].flatten())),
            cmax=0.0015,
        ),
        go.Surface(               # Plane X=+cstX
            x=np.max(Xplot)*np.ones((nY,nZ)),
            y=np.transpose(np.array([Yplot]*nZ)),
            z=np.array([Zplot]*len(Xplot)),
            colorscale='gray',
            opacity=0.5,
            surfacecolor=np.ones((len(Xd),len(Yd))),
            cmin=-0.0015, #np.min(np.real(data['w'].flatten())),
            cmax=0.0015,
        )])
    line_marker = dict(color="rgb(150,150,150)", width=2)
    fig1.add_scatter3d(x=Xplot, y=np.min(Yplot)*np.ones(nY), z=np.zeros(nY), mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=Xplot, y=np.min(Yplot)*np.ones(nY), z=-3000*np.ones(nY), mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=Xplot, y=np.max(Yplot)*np.ones(nY), z=np.zeros(nY), mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=Xplot, y=np.max(Yplot)*np.ones(nY), z=-3000*np.ones(nY), mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=np.min(Xplot)*np.ones(nY), y=Yplot, z=np.zeros(nY), mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=np.min(Xplot)*np.ones(nY), y=Yplot, z=-3000*np.ones(nY), mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=np.max(Xplot)*np.ones(nY), y=Yplot, z=np.zeros(nY), mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=np.max(Xplot)*np.ones(nY), y=Yplot, z=-3000*np.ones(nY), mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=np.min(Xplot)*np.ones(nY), y=np.min(Yplot)*np.ones(nY), z=Zplot, mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=np.min(Xplot)*np.ones(nY), y=np.max(Yplot)*np.ones(nY), z=Zplot, mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=np.max(Xplot)*np.ones(nY), y=np.min(Yplot)*np.ones(nY), z=Zplot, mode='lines', line=line_marker, name='')
    fig1.add_scatter3d(x=np.max(Xplot)*np.ones(nY), y=np.max(Yplot)*np.ones(nY), z=Zplot, mode='lines', line=line_marker, name='')
    fig1.update_layout(scene = dict(
                    xaxis_title='x (km)',
                    xaxis = dict(range=[-np.max(x)/1000-15,np.max(x)/1000+15]),
                    yaxis_title='y (km)',
                    yaxis = dict(range=[-np.max(y)/1000-15,np.max(y)/1000+15]),
                    zaxis_title='z (m)',
                    zaxis = dict(range=[-3000,0])))
    fig1.update_coloraxes(colorbar_len=0.7, colorbar_thickness=10,colorbar_xpad=5)
    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': '3Dimage',
            'height': 700,
            'width': 2000,
            'scale':10 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }
    fig1.show(config=config)



# Sources
def plotSource(sourcefile, savefig=True):
    # Plotting source definition in symetric log scale
    solution = np.load(sourcefile, allow_pickle=True).item()
    SS_matrix = solution['S']
    XX, YY = solution['XX'], solution['YY']
    yplot, xplot = np.meshgrid(YY, XX)

    #fig = plt.figure(figsize=plt.figaspect(0.365))
    fig, axes = plt.subplots(nrows=1,ncols=2,figsize=plt.figaspect(0.365),sharey=True)

    ax0 = axes[0]
    #ax0 = fig.add_subplot(1, 2, 1)
    pcm = ax0.pcolormesh(xplot,yplot,np.abs(SS_matrix),cmap='Reds',norm=colors.LogNorm(vmin=1e-3))
    #ax0.set_xlabel(r'$\frac{\pi x}{\mu H_0}$', fontsize=30)
    #ax0.set_ylabel(r'$ \frac{\pi y}{\mu H_0}$', fontsize=30, rotation=0)
    ax0.tick_params(axis='x', labelsize=20)
    ax0.tick_params(axis='y', labelsize=20)
    ax0.axis("equal")
    clb = fig.colorbar(pcm, ax=ax0)
    #clb.ax.set_xlabel(r'$\| S \|$', fontsize=30)
    clb.ax.tick_params(labelsize=20)
    
    ax1 = axes[1]
    #ax1 = fig.add_subplot(1, 2, 2)
    pcm = ax1.pcolormesh(xplot,yplot,np.angle(SS_matrix),cmap='twilight_shifted_r',vmin=-np.pi,vmax=np.pi)
    #ax1.set_xlabel(r'$\frac{\pi x}{\mu H_0}$', fontsize=30)
    #ax1.set_ylabel(r'$ \frac{\pi y}{\mu H_0}$', fontsize=30, rotation=0)
    ax1.tick_params(axis='x', labelsize=20)
    ax1.tick_params(axis='y', labelsize=20)
    ax1.axis("equal")
    clb = fig.colorbar(pcm, ax=ax1)
    #clb.ax.set_xlabel(r'$Arg (S)$', fontsize=30)
    clb.set_ticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    clb.set_ticklabels([r'$-\pi$',r'$-\frac{\pi}{2}$', r'$0$', r'$\frac{\pi}{2}$', r'$\pi$'])
    clb.ax.tick_params(labelsize=25)
    #fig.suptitle("Source distribution : norm (left) and angle (right)")
    #fig.tight_layout()
    if savefig :
        figname = sourcefile.split(".npy")[0]+".png"
        plt.savefig(figname, dpi=1000)


def plotSourceReIm(sourcefile, savefig=True):
    # Plotting source definition in symetric log scale
    solution = np.load(sourcefile, allow_pickle=True).item()
    SS_matrix = solution['S']
    XX, YY = solution['XX'], solution['YY']
    yplot, xplot = np.meshgrid(YY, XX)

    fig = plt.figure(figsize=plt.figaspect(0.37))
    ax0 = fig.add_subplot(1, 2, 1)
    lev_exp = np.arange(-2, np.ceil(np.log10(np.max(np.abs(np.real(SS_matrix))))+1))
    levs_pos = np.power(10, lev_exp)
    levs_neg = np.append(-levs_pos[::-1],[0],axis=0)
    levs = np.append(levs_neg,levs_pos,axis=0)
    pcm = ax0.contourf(xplot,yplot,np.real(SS_matrix), levs, cmap='RdBu_r', norm=colors.SymLogNorm(linthresh=1e-4,linscale=1,base=10))
    ax0.set_xlabel(r'$\frac{\pi x}{\mu H_0}$', fontsize=15)
    ax0.set_ylabel(r'$\frac{\pi y}{\mu H_0}$', fontsize=15, rotation=0)
    ax0.axis("equal")
    fig.colorbar(pcm, ax=ax0)
    
    ax1 = fig.add_subplot(1, 2, 2)
    lev_exp = np.arange(-2, np.ceil(np.log10(np.max(np.abs(np.imag(SS_matrix))))+1))
    levs_pos = np.power(10, lev_exp)
    levs_neg = np.append(-levs_pos[::-1],[0],axis=0)
    levs = np.append(levs_neg,levs_pos,axis=0)
    pcm = ax1.contourf(xplot,yplot,np.imag(SS_matrix), levs, cmap='RdBu_r', norm=colors.SymLogNorm(linthresh=1e-4,linscale=1,base=10))
    ax1.set_xlabel(r'$\frac{\pi x}{\mu H_0}$', fontsize=15)
    ax1.axis("equal")
    fig.colorbar(pcm, ax=ax1)
    fig.suptitle("Source distribution : Re (left) and Im (right)")
    #fig.tight_layout()
    if savefig :
        figname = sourcefile.split(".npy")[0]+".png"
        plt.savefig(figname)  #, dpi=600)


# Wavefield
def planeZPlot(path,modefile,wavefield,nm,field='w',cstZ=-1.0,figname="planeZplot.png",vmax=None):
    # Plot the chosen field at Z=cstZ 
    #   Inputs are:
    #       'path'          directory in which to save the files
    #       'wavefield'     output of compute_wavefield
    #       'nm'            number of modes used to compute the wavefield
    #       'field'         field to plot : choose between 'u','v','w','p','Jx','Jy'
    #       'cstZ'          Z value at which to plot a plane
    # Read wavefile
    #wavefield = np.load(path+wavefile, allow_pickle=True).item()
    w = wavefield[field]
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    nX, nY = len(Xplot), len(Yplot)

    # Mode an(z) definition
    modes = np.load(path+modefile, allow_pickle=True).item()       # load the topography file
    kn = modes['k'][:nm]
    Z = modes['Z']
    argZ = np.argmin(np.abs(Z-cstZ))
    if field=='w':
        an = modes['modes_Z'][argZ,:nm]
        title = field+r'$/U_0$ for $\frac{\pi z}{H_0} = $'+str(cstZ)+r' calculated with '+str(nm)+r' modes'
    elif (field=='u')|(field=='v')|(field=='p'): 
        an = 1/kn*modes['modesP_Z'][argZ,:nm]
        title = field+r'$/U_0$ for $\frac{\pi z}{H_0} = $'+str(cstZ)+r' calculated with '+str(nm)+r' modes'
    elif (field=='Jx')|(field=='Jy'): 
        an = np.ones(kn.shape)
        title = field+r'$/J_0$ calculated with '+str(nm)+r' modes'
    else :
        print("Error ! Field "+field+" not valid")

    # Plots for Z=cstZ
    Yd, Xd = np.meshgrid(Yplot,Xplot, copy=False)
    antile = np.tile(an,(nY,nX,1)).transpose()      # shape = (nm, nX, nY)
    wtile = w[:nm,:,:]                              # shape = (nm, nX, nY)
    print(np.max(wtile))
    values = np.real(np.sum(wtile*antile, axis=0))
    if vmax==None:
        vmax = np.max(np.abs(values))

    fig, ax = plt.subplots()
    cax = ax.pcolormesh(Xd,Yd,values,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='gouraud')
    clb = fig.colorbar(cax)
    clb.ax.set_xlabel(r'$\frac{'+field+r'}{U_0}$', fontsize=30)
    clb.ax.tick_params(labelsize=20)
    ax.set_xlim([np.min(Xd),np.max(Xd)])
    ax.set_ylim([np.min(Yd),np.max(Yd)])
    ax.set_xlabel(r'$ \frac{x}{H_0}$', fontsize=30)
    ax.set_ylabel(r'$ \frac{y}{H_0}$', fontsize=30, rotation=0)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.set_aspect("equal")
    #ax.set_title(title)
    #plt.tight_layout()
    #figname = wavefile.split("_")[0]+"_"+wavefile.split("_")[1]+"_p"+str(nm)+"_"+field+"_Z"+str(np.abs(cstZ))+".png"
    fig.savefig(path+figname)
    return 0


def planeYPlot(path,modefile,source,wavefield,nm,field='w',cstY=0.0,figname="planeYplot.png",vmax=None):
    # Plot the planes at Y=cstY 
    #   Inputs are:
    #       'path'          directory in which to save the files
    #       'source'        output of compute_source_distribution
    #       'wavefield'     output of compute_wavefield
    #       'nm'            number of modes used to compute the wavefield
    #       'field'         field to plot : choose between 'u','v','w','p'
    #       'cstY'          Y value at which to plot a plane
    #source = np.load(path+sourcefile, allow_pickle=True).item()
    XX, YY = source['XX'], source['YY']
    HH = source['HH']

    # Read wavefile
    w = wavefield[field]
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    nX = len(Xplot)

    # Mode an(z) definition
    modes = np.load(path+modefile, allow_pickle=True).item()       # load the topography file
    kn = modes['k']
    ech = 10
    Zplot = modes['Z'][::ech]
    nZ = len(Zplot)
    ktile = np.tile(kn[:nm], (nZ,1))
    if field=='w':
        an = modes['modes_Z'][::ech,:nm]  # shape = (nZ, nm)
    elif (field=='u')|(field=='v')|(field=='p'): 
        an = 1/ktile*modes['modesP_Z'][::ech,:nm]

    # Plots for Y=0
    Zd, Xd = np.meshgrid(Zplot,Xplot)
    antile = np.tile(an.transpose(),(nX,1,1))              # shape = (nX, nm, nZ)

    indexY = np.argmin(np.abs(Yplot-cstY))
    wtile = np.tile(w[:nm,:,indexY],(nZ,1,1)).transpose()  # shape = (nX, nm, nZ)
    values = np.real(np.sum(wtile*antile, axis=1))
    if vmax==None:
        vmax = np.max(np.abs(values))

    index_y = np.argmin(np.abs(cstY-YY))
    ztopo = np.linspace(-1,np.max(HH-1),1000)
    Ztopo, Xtopo = np.meshgrid(ztopo, XX)
    htopo = HH[:,index_y]-1
    Htopo = np.tile(htopo,(1000,1)).transpose()
    ZtopoMasked = np.ma.masked_array(Ztopo, mask=Ztopo>Htopo)

    fig, ax = plt.subplots()
    cax = ax.pcolormesh(Xd,Zd,values,cmap='RdBu_r',vmin=-vmax,vmax=vmax,shading='gouraud')
    clb = fig.colorbar(cax)
    clb.ax.set_xlabel(r'$\frac{'+field+r'}{U_0}$', fontsize=30)
    clb.ax.tick_params(labelsize=20)
    #ax.pcolormesh(Xtopo,Ztopo,ZtopoMasked,cmap='gray_r',vmin=-np.pi)
    ax.plot(XX,HH[:,index_y]-1,'k',linewidth=1)
    ax.set_ylim([-1, 0])
    ax.set_xlim([np.min(Xd),np.max(Xd)])
    ax.set_xlabel(r'$\frac{x}{H_0}$', fontsize=30)
    ax.set_ylabel(r'$\frac{z}{H_0}$', fontsize=30, rotation=0)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    #ax.set_title(field+r' for $\frac{\pi y}{\mu H_0} = $'+str(cstY)+r' calculated with '+str(nm)+r' modes')
    #ax.set_aspect('equal')
    #plt.tight_layout()
    #figname = wavefile.split("_")[0]+"_"+wavefile.split("_")[1]+"_p"+str(nm)+"_"+field+"_Y"+str(np.abs(cstY))+".png"
    fig.savefig(path+figname)
    return 0


def compConvRate(path,sourcefile,wavefield,nm,sigma,list_coord):
    # Read source file
    source = np.load(path+sourcefile, allow_pickle=True).item()
    N, omega, f, Ho = source['N'], source['omega'], source['cf'], source['Ho'] 
    mu = np.sqrt((N**2-omega**2)/(omega**2-f**2))
    delta = source['Delta']
    Hmax = np.max(source['H'])

    # Read wavefile
    #wavefield = np.load(path+wavefile, allow_pickle=True).item()
    Xplot, Yplot = wavefield['X'], wavefield['Y']

    plt.figure()
    p = np.arange(1,nm+1,1)
    for coord in list_coord :
        x, y = coord
        C = np.zeros(nm)
        # Conversion rate
        ix_p, iy_p = np.argmin(np.abs(Xplot-x)), np.argmin(np.abs(Yplot-y))
        ix_n, iy_n = np.argmin(np.abs(Xplot+x)), np.argmin(np.abs(Yplot+y))
        
        for ip in range(nm):

            convRate  = np.sum(wavefield['Jx'][ip,ix_p,iy_n:iy_p+1],axis=1) - np.sum(wavefield['Jx'][ip,ix_n,iy_n:iy_p+1],axis=1)
            convRate += np.sum(wavefield['Jy'][ip,ix_n:ix_p+1,iy_p],axis=1) - np.sum(wavefield['Jy'][ip,ix_n:ix_p+1,iy_n],axis=1)
            # Clsy = rho0 * Utide**2 * mu * (omega**2-f**2)/omega * hmax**2 * np.pi**(3/2)/8 * sigma
            C[ip] = np.real(convRate)*delta*mu*Ho/np.pi * mu**2 * np.pi**(1/2)/(Hmax**2/8*sigma)      # C adim by Clsy
            C[ip] = np.real(convRate)*delta*mu*Ho/np.pi / (np.pi**(3/2)/8*sigma)

        plt.plot(p,np.cumsum(C), '+', label="Coord = "+str((x,y)))
    plt.xlabel('Mode number N')
    plt.ylabel(r'$\sum_{n=1}^N C_n / C_{WTA}$')
    plt.legend()
    plt.tight_layout()
    return 0


def plotConversionRate(path,wavefield,nm,Cadim=1,figname=""):
    p = np.arange(1,nm+1,1)
    #wavefield = np.load(path+wavefile, allow_pickle=True).item()
    C = np.real(wavefield['C'])[:nm]/Cadim
    
    plt.figure()
    plt.grid(zorder=0)
    plt.bar(p,C,zorder=3)
    plt.xlabel('Mode number n')
    plt.ylabel('$C_n / C_{WTA}$')
    plt.yscale("log")
    #figname = wavefile.split("_")[0]+"_"+wavefile.split("_")[1]+"_p"+str(nm)+"_Cp.png"
    plt.tight_layout()
    plt.savefig(path+figname+"Cp.png")

    plt.figure()
    plt.grid(zorder=0)
    plt.bar(p,np.cumsum(C),zorder=3)
    plt.xlabel('Mode number N')
    plt.ylabel(r'$\sum_{n=1}^N C_n / C_{WTA}$')
    #figname = wavefile.split("_")[0]+"_"+wavefile.split("_")[1]+"_p"+str(nm)+"_Cpsum.png"
    plt.tight_layout()
    plt.savefig(path+figname+"Cpsum.png")
    return 0


def planeZVectors(path,sourcefile,wavefield,nm,field='J',cstZ=-1.0,plot=True,savefig=True,vmin=None,vmax=None,scale=10,figname=""):
    # Plot the norm and the direction of the energy flux vectors
    #   Inputs are:
    #       'path'          directory in which to save the files
    #       'sourcefile'    output file of compute_source_distribution
    #       'wavefield'     output of compute_wavefield
    #       'nm'            number of modes used to compute the wavefield
    #       'field'         field to plot : choose from 'Uh' and 'J'
    #   Output : 
    #       'N', 'omega', 'f'       Values of frequencies
    #       'angles'                Angles between the maximum vector at the left and right of the topo with (Oy)

    # Read source file
    source = np.load(path+sourcefile, allow_pickle=True).item()
    X, Y = source['XX'], source['YY']
    N, omega, f = source['N'], source['omega'], source['cf']

    # Read wavefile
    #wavefield = np.load(path+wavefile, allow_pickle=True).item()
    Xplot, Yplot = wavefield['X'][::2], wavefield['Y'][::2]
    nX, nY = len(Xplot), len(Yplot)
    if field=='J' :
        valx, valy = np.sum(np.real(wavefield['Jx'][:nm,::2,::2]), axis=0), np.sum(np.real(wavefield['Jy'][:nm,::2,::2]),axis=0)     # shape = (nXplot, nYplot)
        title = r'$\vec J$ calculated with '+str(nm)+r' modes'
        figname = figname+"normJ.png"
    elif field=='Uh':
        p = np.arange(1,nm+1)
        anp = np.tile(np.cos(p*cstZ),(nY,nX,1)).transpose()
        valx, valy = np.sum(wavefield['u'][:nm,::2,::2]*anp, axis=0), np.sum(wavefield['v'][:nm,::2,::2]*anp,axis=0)     # shape = (nXplot, nYplot)
        valx, valy = np.real(valx), np.real(valy)
        title = r'$\vec u_H$ for $\frac{\pi z}{H_0} = $'+str(cstZ)+r'calculated with '+str(nm)+r' modes'
        figname = figname+"normUh_Z"+str(np.abs(cstZ))+".png"
    else : 
        print('Error ! Wavefield '+field+' not supported !')
    Yd, Xd = np.meshgrid(Yplot,Xplot, copy=False)
    valnorm = np.sqrt(valx**2+valy**2)    # shape = (nXplot, nYplot)

    # Find direction of max norm
    maskP = (Xd<=np.max(X)+0.2)&(Yd<=np.max(Y))&(Yd>=np.min(Y))
    valmaxP = np.ma.masked_array(valnorm, mask=maskP)
    iP,jP = np.unravel_index(np.argmax(valmaxP),np.shape(valmaxP))
    maskM = (Xd>=np.min(X)-0.2)&(Yd<=np.max(Y))&(Yd>=np.min(Y))
    valmaxM = np.ma.masked_array(valnorm, mask=maskM)
    iM,jM = np.unravel_index(np.argmax(valmaxM),np.shape(valmaxM))
    angles = valy[iP,jP]/valx[iP,jP], valy[iM,jM]/valx[iM,jM]

    if plot==True:
        # Arrows
        listX = np.linspace(np.min(Xplot),np.max(Xplot),21)
        listY = np.linspace(np.min(Yplot),np.max(Yplot),21)
        ind_i = np.array([np.argmin(np.abs(Xplot-listX[i])) for i in range(len(listX)) for j in range(len(listY)) if (listX[i]!=0)|(listY[j]!=0)])
        ind_j = np.array([np.argmin(np.abs(Yplot-listY[j])) for i in range(len(listX)) for j in range(len(listY)) if (listX[i]!=0)|(listY[j]!=0)])
        # Plot
        fig, ax = plt.subplots()
        cax = ax.pcolormesh(Xd, Yd, valnorm, cmap='Reds',vmin=vmin,vmax=vmax, shading='gouraud')
        clb = fig.colorbar(cax, ax=ax)
        clb.ax.set_xlabel(r'$\frac{\|J\|}{J_0}$', fontsize=30)
        clb.ax.tick_params(labelsize=20)
        ax.quiver(Xd[ind_i,ind_j], Yd[ind_i,ind_j], scale*valx[ind_i,ind_j], scale*valy[ind_i,ind_j], angles='xy', scale=1, scale_units='xy',width=0.005)
        ax.set_xlim([np.min(Xd),np.max(Xd)])
        ax.set_ylim([np.min(Yd),np.max(Yd)])
        ax.set_xlabel(r'$ \frac{\pi x}{\mu H_0}$', fontsize=30)
        #ax.set_ylabel(r'$ \frac{\pi y}{\mu H_0}$', fontsize=30, rotation=0)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelleft=False) #labelsize=20)
        ax.set_aspect("equal")
        #ax.set_title(title)
        plt.tight_layout()
        if savefig:
            fig.savefig(path+figname, dpi=1200)

    return (N, omega, f, angles)


def planeZstream(path,sourcefile,wavefield,nm,field='J',cstZ=-1.0,plot=True,savefig=True,vmin=None,vmax=None,figname=""):
    # Plot the norm and the direction of the energy flux vectors
    #   Inputs are:
    #       'path'          directory in which to save the files
    #       'sourcefile'    output file of compute_source_distribution
    #       'wavefield'     output of compute_wavefield
    #       'nm'            number of modes used to compute the wavefield
    #       'field'         field to plot : choose from 'Uh' and 'J'
    #   Output : 
    #       'N', 'omega', 'f'       Values of frequencies
    #       'angles'                Angles between the maximum vector at the left and right of the topo with (Oy)

    # Read source file
    source = np.load(path+sourcefile, allow_pickle=True).item()
    X, Y = source['XX'], source['YY']
    N, omega, f = source['N'], source['omega'], source['cf']

    # Read wavefile
    #wavefield = np.load(path+wavefile, allow_pickle=True).item()
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    nX, nY = len(Xplot), len(Yplot)
    if field=='J' :
        valx, valy = np.sum(np.real(wavefield['Jx'][:nm,:,:]), axis=0), np.sum(np.real(wavefield['Jy'][:nm,:,:]),axis=0)     # shape = (nXplot, nYplot)
        title = r'$\vec J$ calculated with '+str(nm)+r' modes'
        figname = figname+"streamJ.png"
    elif field=='Uh':
        p = np.arange(1,nm+1)
        anp = np.tile(np.cos(p*cstZ),(nY,nX,1)).transpose()
        valx, valy = np.sum(wavefield['u'][:nm,:,:]*anp, axis=0), np.sum(wavefield['v'][:nm,:,:]*anp,axis=0)     # shape = (nXplot, nYplot)
        valx, valy = np.real(valx), np.real(valy)
        title = r'$\vec u_H$ for $\frac{\pi z}{H_0} = $'+str(cstZ)+r'calculated with '+str(nm)+r' modes'
        figname = figname+"streamUh_Z"+str(np.abs(cstZ))+".png"
    else : 
        print('Error ! Wavefield '+field+' not supported !')
    Yd, Xd = np.meshgrid(Yplot,Xplot, copy=False)
    valnorm = np.sqrt(valx**2+valy**2)    # shape = (nXplot, nYplot)

    # Find direction of max norm
    maskP = (Xd<=np.max(X)+0.2)&(Yd<=np.max(Y))&(Yd>=np.min(Y))
    valmaxP = np.ma.masked_array(valnorm, mask=maskP)
    iP,jP = np.unravel_index(np.argmax(valmaxP),np.shape(valmaxP))
    maskM = (Xd>=np.min(X)-0.2)&(Yd<=np.max(Y))&(Yd>=np.min(Y))
    valmaxM = np.ma.masked_array(valnorm, mask=maskM)
    iM,jM = np.unravel_index(np.argmax(valmaxM),np.shape(valmaxM))
    angles = valy[iP,jP]/valx[iP,jP], valy[iM,jM]/valx[iM,jM]

    if plot==True:
        fig, ax = plt.subplots()
        strm = ax.streamplot(Xplot, Yplot, valx.transpose(), valy.transpose(), color=valnorm.transpose(), linewidth=2, cmap='Reds', density=1)
        fig.colorbar(strm.lines)
        ax.set_xlim([np.min(Xd),np.max(Xd)])
        ax.set_ylim([np.min(Yd),np.max(Yd)])
        ax.set_xlabel(r'$ \frac{\pi x}{\mu H_0}$', fontsize=20)
        ax.set_ylabel(r'$ \frac{\pi y}{\mu H_0}$', fontsize=20, rotation=0)
        ax.set_aspect("equal")
        #ax.set_title(title)
        plt.tight_layout()
        if savefig:
            fig.savefig(path+figname)

    return (N, omega, f, angles)


def radialFlux(path,source,wavefield,nm,R,ax=None,savefig=False,color='0.0',linestyle='solid', label=None):
    # Plot the value of J.er for theta between -pi and pi and R given
    #   Inputs are :
    #       'path'          directory in which to save the files
    #       'sourcefile'    output file of compute_source_distribution
    #       'wavefield'     output of compute_wavefield
    #       'nm'            number of modes used to compute the wavefield
    #       'R'             radius used for the computation of J.er
    #   Output : 
    #       'N', 'omega', 'f'       Values of frequencies
    #       'angleMax'              Angle for which J.er is max
 
    # Read source file
    #source = np.load(path+sourcefile, allow_pickle=True).item()
    N, omega, f = source['N'], source['omega'], source['cf']
    #pmax = (wavefile.split("_")[1]).split("pmax")[0]

    # Read wavefile
    #wavefield = np.load(path+wavefile, allow_pickle=True).item()
    Jx, Jy = np.sum(np.real(wavefield['Jx'][:nm,:,:]),axis=0), np.sum(np.real(wavefield['Jy'][:nm,:,:]),axis=0)            # shape = (nXplot, nYplot)
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    theta = np.linspace(-np.pi, -np.pi/4, 100, endpoint=False)
    theta = np.append(theta,np.linspace(-np.pi/4,np.pi/4,10000,endpoint=False))
    theta = np.append(theta,np.linspace(np.pi/4,np.pi,100,endpoint=True))
    interpPoints = np.array([[R*np.cos(theta[i]), R*np.sin(theta[i])] for i in range(len(theta))])
    rgiJx = RegularGridInterpolator((Xplot, Yplot), Jx)
    rgiJy = RegularGridInterpolator((Xplot, Yplot), Jy) 
    Jx_theta = rgiJx(interpPoints)
    Jy_theta = rgiJy(interpPoints)
    Jr = Jx_theta*np.cos(theta)+Jy_theta*np.sin(theta)
    Jtheta = -Jx_theta*np.sin(theta)+Jy_theta*np.cos(theta)
    
    if (type(ax)!=np.ndarray):
        fig, ax = plt.subplots(2,1)
        color = '0.0'
    
    # Radial plot
    ax[0].plot(theta, Jr*R, c=color, linestyle=linestyle, label=label) #'f/$\omega$ = '+str(np.round(f/omega,2))+'; R = '+str(R)
    imax = np.argmax(np.ma.masked_array(Jr,mask=(theta>=np.pi/2)|(theta<=-np.pi/2)))
    angleMax = theta[imax]
    Jmax = Jr[imax]
    ax[0].vlines(angleMax,0,Jmax*R,color=color,linestyle='dashed')
    ax[0].set_ylim(bottom=0.0)
    ax[0].set_ylabel(r'$r (\vec J . \vec e_r)$', fontsize=15)

    # Orthoradial plot
    ax[1].plot(theta, Jtheta*R**2, c=color, linestyle=linestyle, label=label) #'f/$\omega$ = '+str(np.round(f/omega,2))+'; R = '+str(R)
    ax[1].set_xlabel(r'$\theta$ (rad)', fontsize=15)
    ax[1].set_ylabel(r'$r^2 (\vec J . \vec e_{\theta})$', fontsize=15)

    if savefig: 
        ax[0].grid()
        ax[1].grid()
        #figname = wavefile.split("_")[0]+"_pmax"+pmax+"_p"+str(nm)+"_Jr_R"+str(R)+".png"
        fig.tight_layout()
        fig.savefig(path+"Jr_R"+str(R)+".png")
        #plt.close(fig)
    return (N,omega,f,angleMax,Jmax)


def directionFlux(path,sourcefile,wavefield,nm,ax=None,savefig=False,figname=""):
    # Plot the angle between the flux vector for Y=0 and the y axis (for diff values of X)
    #   Inputs are :
    #       'path'          directory in which to save the files
    #       'sourcefile'    output file of compute_source_distribution
    #       'wavefile'      output file of compute_wavefield
    #       'nm'            number of modes used to compute the wavefield
 
    # Read source file
    source = np.load(path+sourcefile, allow_pickle=True).item()
    N, omega, f = source['N'], source['omega'], source['cf']

    # Read wavefile
    #wavefield = np.load(path+wavefile, allow_pickle=True).item()
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    jY0 = np.argmin(np.abs(Yplot))
    Jx, Jy = np.sum(np.real(wavefield['Jx'][:nm,:,jY0]),axis=0), np.sum(np.real(wavefield['Jy'][:nm,:,jY0]),axis=0)      # shape = (nXplot)
    
    norm = colors.Normalize(-1, 1)
    c = cm.viridis(norm(np.round(f/omega,2)))
    if ax==None:
        fig, ax = plt.subplots()
        ax.set_xlabel(r'$\frac{\pi x}{\mu H_0}$', fontsize=15)
        ax.set_ylabel(r'$\theta $ (rad)', fontsize=15)
        ax.grid()
        c='0.0'
    #val = np.array([np.arctan(Jy[i]/Jx[i]) if Xplot[i]>=0 else -np.pi+np.arctan(Jy[i]/Jx[i]) for i in range(len(Xplot))])
    ax.plot(Xplot, np.arctan(Jy/Jx), c=c, label=r'$f/\omega = $'+str(np.round(f/omega,2)))

    if savefig: 
        figname = figname+"dirJ_Y0.png"
        fig.tight_layout()
        fig.savefig(path+figname)
        #plt.close(fig)



# 3D plots
def isopycnes3D(path, source, wavefield, nm, Xmax, Ymax, Utide=None):
    # Plot in 3D the planes at Z=cstZ, Y=cstY, X=cstX as well as the topography 
    #   Inputs are:
    #       'path'          directory in which to save the files
    #       'sourcefile'    output file of compute_source_distribution
    #       'wavefile'      output file of compute_wavefield
    #       'nm'            number of modes used to compute the wavefield
    #       'field'         field to plot : choose between 'u','v','w','p'
    #       'cstZ'          Z value at which to plot a plane
    #       'cstY'          same with Y
    p = np.arange(1,nm+1,1)

    # Read source file
    XX, YY = source['XX'], source['YY']
    HH, H0 = source['HH'], source['Ho']
    hmax = np.max(HH)
    if Utide==None:
        U0, V0 = source["U0"], source["V0"]
        Utide = max(np.sqrt(np.real(U0)**2+np.real(V0)**2),np.sqrt(np.imag(U0)**2+np.imag(V0)**2))
    N, omega, f = source["N"], source["omega"], source["cf"]

    # Read wavefile
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    ix_p, ix_n = np.argmin(np.abs(Xplot-Xmax)), np.argmin(np.abs(Xplot+Xmax))
    iy_p, iy_n = np.argmin(np.abs(Yplot-Ymax)), np.argmin(np.abs(Yplot+Ymax))
    Xplot, Yplot = Xplot[ix_n:ix_p:2], Yplot[iy_n:iy_p:2]
    delta = Xplot[1]-Xplot[0]
    Zplot = np.arange(-np.pi,0,2*delta)
    nX, nY, nZ = len(Xplot), len(Yplot), len(Zplot)
    w = wavefield['w'][:,ix_n:ix_p:2,iy_n:iy_p:2]
    
    # Compute full wavefield
    ptile = np.tile(p,(nZ,nY,nX,1)).transpose()        # shape = (nm, nX, nY, nZ)
    Ztile = np.tile(Zplot,(nm,nX,nY,1))                # shape = (nm, nX, nY, nZ)
    wtile = np.tile(w[:nm,:,:].transpose(),(nZ,1,1,1)).transpose()  # shape = (nm, nX, nY, nZ)
    w = np.sum(wtile*np.sin(ptile*(Ztile+np.pi)), axis=0)         # shape = (nX, nY, nZ)
    #print(np.max(val))

    # Plane Y=0
    indexY = np.argmin(np.abs(Yplot))
    color_Y0 = np.real(w[:,indexY,:])

    # Plot
    # Plot
    Y, X, Z = np.meshgrid(Yplot, Xplot, Zplot)
    #iso1, iso2 = omega*H0/np.pi/Utide*(-1.0), omega*H0/np.pi/Utide*(-2.0)
    fig1 = go.Figure([go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=Z.flatten()-np.pi*Utide/omega/H0*np.real(1j*w).flatten(),
            colorscale='gray_r',
            isomin=-np.pi/4,
            isomax=-np.pi/4,
            showscale=False,
            surface_count = 1,
            opacity=0.5,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ),go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=Z.flatten()-np.pi*Utide/omega/H0*np.real(1j*w).flatten(),
            colorscale='gray_r',
            isomin=-3*np.pi/4,
            isomax=-3*np.pi/4,
            showscale=False,
            surface_count = 1,
            opacity=0.5,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ),go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=Z.flatten()-np.pi*Utide/omega/H0*np.real(1j*w).flatten(),
            colorscale='gray_r',
            isomin=-2*np.pi/4,
            isomax=-2*np.pi/4,
            showscale=False,
            surface_count = 1,
            opacity=0.5,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ),
        go.Surface(
            contours = {
                "z": {"show": True, "start": hmax/10-np.pi, "end": hmax-np.pi, "size": hmax/5}
            },
            x=XX,
            y=YY,
            z=HH-np.pi,
            colorscale="gray_r", #surfacecolor=np.ones((len(XX),len(YY))),
            colorbar_nticks=3, # colorbar ticks correspond to isosurface values
            opacity=1.0,
            showscale=False #colorbar_x=-0.07
        ),
        go.Surface(               # Plane Y=0
            x=np.transpose(np.array([Xplot]*nZ)),
            y=np.zeros((nX,nZ)),
            z=np.array([Zplot]*len(Yplot)),
            colorscale='RdBu_r',
            surfacecolor=color_Y0,
            showscale=False,
            opacity = 1.0
        )])
    
    fig1.update_layout(showlegend=False, 
                scene = dict(
                    aspectratio=dict(x=1,y=1,z=0.6),
                    xaxis = dict(visible=False),
                    yaxis = dict(visible=False),
                    zaxis =dict(visible=False)))
    #fig1.update_coloraxes(colorbar_len=0.7, colorbar_thickness=10,colorbar_xpad=5)
    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': '3Dimage',
            'height': 1000,
            'width': 1300,
            'scale': 5 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }

    fig1.show(config=config)
    return 0


def planePlot3D(path,sourcefile,wavefile,nm,field='w',cstZ=-1.0,cstY=0.0):
    # Plot in 3D the planes at Z=cstZ, Y=cstY, X=cstX as well as the topography 
    #   Inputs are:
    #       'path'          directory in which to save the files
    #       'sourcefile'    output file of compute_source_distribution
    #       'wavefile'      output file of compute_wavefield
    #       'nm'            number of modes used to compute the wavefield
    #       'field'         field to plot : choose between 'u','v','w','p'
    #       'cstZ'          Z value at which to plot a plane
    #       'cstY'          same with Y
    p = np.arange(1,nm+1,1)

    # Read source file
    source = np.load(path+sourcefile, allow_pickle=True).item()
    XX, YY = source['XX'], source['YY']
    HH = source['HH']
    
    # Read wavefile
    wavefield = np.load(path+wavefile, allow_pickle=True).item()
    w = wavefield[field]
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    delta = Xplot[1]-Xplot[0]
    Zplot = np.arange(-np.pi,0,delta)
    nX, nY, nZ = len(Xplot), len(Yplot), len(Zplot)

    # Color for Y=cstY
    indexY = np.argmin(np.abs(Yplot-cstY))
    ptile = np.tile(p,(nZ,nX,1)).transpose()        # shape = (nm, nX, nZ)
    Ztile = np.tile(Zplot,(nm,nX,1))                # shape = (nm, nX, nZ)
    w_cstY = np.tile(w[:nm,:,indexY].transpose(),(nZ,1,1)).transpose()
    color_cstY = np.real(np.sum(w_cstY*np.sin(ptile*Ztile), axis=0))
    # Color for Z=cstZ
    ptile = np.tile(p,(nY,nX,1)).transpose()        # shape = (nm, nX, nY)
    w_cstZ = w[:nm,:,:]                             # shape = (nm, nX, nY)
    color_cstZ = np.real(np.sum(w_cstZ*np.sin(ptile*cstZ), axis=0))

    # Plot
    Yd, Xd = np.meshgrid(Yplot,Xplot)
    fig = go.Figure(data=[go.Surface(               # Plane Y=cstY
            x=np.transpose(np.array([Xplot]*nZ)),
            y=cstY*np.ones((nX,nZ)),
            z=np.array([Zplot]*len(Yplot)),
            colorscale='RdBu_r',
            surfacecolor=color_cstY,
        ),
        go.Surface(                                 # Plane Z = 0 (for boundary)
            x=Xplot,
            y=Yplot,
            z=np.zeros((len(Xplot),len(Yplot))),
            colorscale='RdBu_r',
            opacity=0,
            showscale=False #colorbar_x=-0.07
        ),
        go.Surface(                                 # Plane Z = cstZ
            x=Xd,
            y=Yd,
            z=cstZ*np.ones((len(Xd),len(Yd))),
            colorscale='RdBu_r',
            surfacecolor=color_cstZ,
            cmin=-0.0015, #np.min(np.real(data['w'].flatten())),
            cmax=0.0015,
            showscale=False #colorbar_x=-0.07
        ),
        go.Surface(                                # Topography
            contours = {
                "z": {"show": True, "start": 0.01-np.pi, "end": 0.05-np.pi, "size": 0.015}
            },
            x=XX,
            y=YY,
            z=HH-np.pi,
            colorscale="gray_r", #surfacecolor=np.ones((len(XX),len(YY))),
            colorbar_nticks=3, # colorbar ticks correspond to isosurface values
            opacity=0.5,
            showscale=False #colorbar_x=-0.07
        )])
    fig.show()


def isosurfacePlot(path,sourcefile,wavefile,nm,field,iso,Xmax,Ymax,cmax=None,opacity=1):
    # Plot in 3D the isosurface as well as the topography 
    #   Inputs are:
    #       'path'          directory in which to save the files
    #       'sourcefile'    output file of compute_source_distribution
    #       'wavefile'      output file of compute_wavefield
    #       'nm'            number of modes used to compute the wavefield
    #       'iso'           isovalue
    #       'field'         field to plot : choose between 'u','v','w','p'
    p = np.arange(1,nm+1,1)

    # Read source file
    source = np.load(path+sourcefile, allow_pickle=True).item()
    XX, YY = source['XX'], source['YY']
    HH = source['HH']
    hmax = np.max(HH)

    # Read wavefile
    wavefield = np.load(path+wavefile, allow_pickle=True).item()
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    ix_p, ix_n = np.argmin(np.abs(Xplot-Xmax)), np.argmin(np.abs(Xplot+Xmax))
    iy_p, iy_n = np.argmin(np.abs(Yplot-Ymax)), np.argmin(np.abs(Yplot+Ymax))
    Xplot, Yplot = Xplot[ix_n:ix_p:2], Yplot[iy_n:iy_p:2]
    delta = Xplot[1]-Xplot[0]
    Zplot = np.arange(-np.pi,0,2*delta)
    nX, nY, nZ = len(Xplot), len(Yplot), len(Zplot)
    w = wavefield[field][:,ix_n:ix_p:2,iy_n:iy_p:2]
    
    # Compute full wavefield
    ptile = np.tile(p,(nZ,nY,nX,1)).transpose()        # shape = (nm, nX, nY, nZ)
    Ztile = np.tile(Zplot,(nm,nX,nY,1))                # shape = (nm, nX, nY, nZ)
    wtile = np.tile(w[:nm,:,:].transpose(),(nZ,1,1,1)).transpose()
    val = np.sum(wtile*np.sin(ptile*(Ztile+np.pi)), axis=0)
    #print(np.max(val))
    if cmax==None:
        cmax = np.max(np.abs(val))

    # Plot
    Y, X, Z = np.meshgrid(Yplot, Xplot, Zplot)
    Yd, Xd = np.meshgrid(Yplot,Xplot)
    fig1 = go.Figure([go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=np.real(val).flatten(),
            colorscale='RdBu_r',
            cmin=-cmax,
            cmax=cmax,
            isomin=-np.abs(iso),
            isomax=np.abs(iso),
            showscale=False,
            surface_count = 2,
            opacity=opacity,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ),
        go.Surface(
            contours = {
                "z": {"show": True, "start": hmax/10-np.pi, "end": hmax-np.pi, "size": hmax/5}
            },
            x=XX,
            y=YY,
            z=HH-np.pi,
            colorscale="gray_r", #surfacecolor=np.ones((len(XX),len(YY))),
            colorbar_nticks=3, # colorbar ticks correspond to isosurface values
            opacity=1.0,
            showscale=False #colorbar_x=-0.07
        ),
        go.Surface(               # Plane Z = -1.0
            x=Xd,
            y=Yd,
            z=-1.0*np.ones((len(Xd),len(Yd))),
            colorscale='gray_r',
            surfacecolor=0.3*np.ones((len(Xd),len(Yd))),
            cmin=0, #np.min(np.real(data['w'].flatten())),
            cmax=1,
            showscale=False,
            opacity = 0.5
        ),
        go.Surface(               # Plane Y=0
            x=np.transpose(np.array([Xplot]*nZ)),
            y=np.zeros((nX,nZ)),
            z=np.array([Zplot]*len(Yplot)),
            colorscale='gray_r',
            surfacecolor=np.zeros((len(Xd),len(Yd))),
            cmin=0, #np.min(np.real(data['w'].flatten())),
            cmax=1,
            showscale=False,
            opacity = 0.3
        ),
        go.Scatter3d(
            x=np.max(Xplot)*np.ones_like(Zplot), y=np.zeros_like(Zplot), z=Zplot,
            mode="lines",
            line=dict(color='darkblue',width=3)
        ),
        go.Scatter3d(
            x=-np.max(Xplot)*np.ones_like(Zplot), y=np.zeros_like(Zplot), z=Zplot,
            mode="lines",
            line=dict(color='darkblue',width=3)
        ),
        go.Scatter3d(
            x=Xplot, y=np.zeros_like(Xplot), z=np.max(Zplot)*np.ones_like(Xplot),
            mode="lines",
            line=dict(color='darkblue',width=3)
        ),
        go.Scatter3d(
            x=Xplot, y=np.zeros_like(Xplot), z=-np.pi*np.ones_like(Xplot),
            mode="lines",
            line=dict(color='darkblue',width=3)
        ),
        go.Scatter3d(
            x=np.max(Xplot)*np.ones_like(Yplot), y=Yplot, z=-1.0*np.ones_like(Yplot),
            mode="lines",
            line=dict(color='goldenrod',width=3)
        ),
        go.Scatter3d(
            x=-np.max(Xplot)*np.ones_like(Yplot), y=Yplot, z=-1.0*np.ones_like(Yplot),
            mode="lines",
            line=dict(color='goldenrod',width=3)
        ),
        go.Scatter3d(
            x=Xplot, y=np.max(Yplot)*np.ones_like(Xplot), z=-1.0*np.ones_like(Xplot),
            mode="lines",
            line=dict(color='goldenrod',width=3)
        ),
        go.Scatter3d(
            x=Xplot, y=-np.max(Yplot)*np.ones_like(Xplot), z=-1.0*np.ones_like(Xplot),
            mode="lines",
            line=dict(color='goldenrod',width=3)
        ),
        go.Scatter3d(
            x=-Xmax*np.ones_like(Yplot), y=Yplot, z=-np.pi*np.ones_like(Yplot),
            mode="lines",
            line=dict(color='black',width=3)
        ),
        go.Scatter3d(
            x=Xplot, y=-Ymax*np.ones_like(Xplot), z=-np.pi*np.ones_like(Xplot),
            mode="lines",
            line=dict(color='black',width=3)
        ),
        go.Scatter3d(
            x=-Xmax*np.ones_like(Zplot), y=-Ymax*np.ones_like(Zplot), z=Zplot,
            mode="lines",
            line=dict(color='black',width=3)
        ),
        go.Scatter3d(
            x=-Xmax*np.ones_like(Zplot), y=Ymax*np.ones_like(Zplot), z=Zplot,
            mode="lines",
            line=dict(color='gray',width=1)
        ),
        go.Scatter3d(
            x=Xmax*np.ones_like(Zplot), y=Ymax*np.ones_like(Zplot), z=Zplot,
            mode="lines",
            line=dict(color='gray',width=1)
        )])
    """
        go.Scatter3d(
            x=XX, y=(-np.max(XX))*np.ones_like(XX), z=(-np.pi+0.02)*np.ones_like(XX),
            mode="lines",
            line=dict(color='black',width=3)
        ),
        go.Scatter3d(
            x=( np.max(XX))*np.ones_like(YY), y=YY, z=(-np.pi+0.02)*np.ones_like(YY),
            mode="lines",
            line=dict(color='black',width=3)
        ),
        go.Scatter3d(
            x=(-np.max(XX))*np.ones_like(YY), y=YY, z=(-np.pi+0.02)*np.ones_like(YY),
            mode="lines",
            line=dict(color='black',width=3)
        ),
        go.Scatter3d(
            x=XX, y=( np.max(YY))*np.ones_like(XX), z=(-np.pi+0.02)*np.ones_like(XX),
            mode="lines",
            line=dict(color='black',width=3)
        ),
    """
    fig1.update_layout(font=dict(size=14,color="black"),
                showlegend=False,
                scene = dict(
                    aspectmode='manual',
                    aspectratio=dict(x=1,y=1,z=0.7),
                    #xaxis_title='X',
                    xaxis = dict(
                        #backgroundcolor="rgb(255,255,255)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white",
                        range=[-Xmax-0.01,Xmax+0.01]),
                    #yaxis_title='Y',
                    yaxis = dict(
                        #backgroundcolor="rgb(255,255,255)", #"rgb(185,185,185)",
                        gridcolor="white",
                        showbackground=False,
                        zerolinecolor="white",
                        range=[-Ymax-0.01,Ymax+0.01]),
                    #zaxis_title='Z',
                    zaxis = dict(
                        backgroundcolor="rgb(230,230,230)",
                        gridcolor="white",
                        showbackground=True,
                        zerolinecolor="gray",
                        range=[-np.pi-0.01,0.01])))
    
    fig1.update_coloraxes(colorbar_len=0.7, colorbar_thickness=10,colorbar_xpad=5)
    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': '3Dimage',
            'height': 1000,
            'width': 1000,
            'scale':10 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }
    fig1.show(config=config)
    return 0


def isosurfacePlotNew(path,sourcefile,wavefile,nm,field,iso,Xmax,Ymax,cmax=None,opacity=1):
    # Plot in 3D the isosurface as well as the topography 
    #   Inputs are:
    #       'path'          directory in which to save the files
    #       'sourcefile'    output file of compute_source_distribution
    #       'wavefile'      output file of compute_wavefield
    #       'nm'            number of modes used to compute the wavefield
    #       'iso'           isovalue
    #       'field'         field to plot : choose between 'u','v','w','p'
    p = np.arange(1,nm+1,1)

    # Read source file
    source = np.load(path+sourcefile, allow_pickle=True).item()
    XX, YY = source['XX'], source['YY']
    HH = source['HH']
    hmax = np.max(HH)

    # Read wavefile
    wavefield = np.load(path+wavefile, allow_pickle=True).item()
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    ix_p, ix_n = np.argmin(np.abs(Xplot-Xmax)), np.argmin(np.abs(Xplot+Xmax))
    iy_p, iy_n = np.argmin(np.abs(Yplot-Ymax)), np.argmin(np.abs(Yplot+Ymax))
    Xplot, Yplot = Xplot[ix_n:ix_p:2], Yplot[iy_n:iy_p:2]
    delta = Xplot[1]-Xplot[0]
    Zplot = np.arange(-np.pi,0,2*delta)
    nX, nY, nZ = len(Xplot), len(Yplot), len(Zplot)
    w = wavefield[field][:,ix_n:ix_p:2,iy_n:iy_p:2]
    
    # Compute full wavefield
    ptile = np.tile(p,(nZ,nY,nX,1)).transpose()        # shape = (nm, nX, nY, nZ)
    Ztile = np.tile(Zplot,(nm,nX,nY,1))                # shape = (nm, nX, nY, nZ)
    wtile = np.tile(w[:nm,:,:].transpose(),(nZ,1,1,1)).transpose()
    val = np.sum(wtile*np.sin(ptile*(Ztile+np.pi)), axis=0)
    #print(np.max(val))
    if cmax==None:
        cmax = np.max(np.abs(val))

    # Plot
    Y, X, Z = np.meshgrid(Yplot, Xplot, Zplot)
    Yd, Xd = np.meshgrid(Yplot,Xplot)
    fig1 = go.Figure([go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=np.real(val).flatten(),
            colorscale='balance', #'RdBu_r',
            cmin=-cmax,
            cmax=cmax,
            isomin=-np.abs(iso),
            isomax=np.abs(iso),
            surface_count = 2,
            opacity=opacity,
            caps=dict(x_show=False, y_show=False, z_show=False)
        ),
        go.Surface(
            contours = {
                "z": {"show": True, "start": hmax/10-np.pi, "end": hmax-np.pi, "size": hmax/5}
            },
            x=XX,
            y=YY,
            z=HH-np.pi,
            colorscale="gray_r", #surfacecolor=np.ones((len(XX),len(YY))),
            colorbar_nticks=3, # colorbar ticks correspond to isosurface values
            opacity=1.0,
            showscale=False #colorbar_x=-0.07
        ),
        go.Surface(               # Plane Z = -1.0
            x=Xd,
            y=Yd,
            z=-1.0*np.ones((len(Xd),len(Yd))),
            colorscale='gray_r',
            surfacecolor=0.3*np.ones((len(Xd),len(Yd))),
            cmin=0, #np.min(np.real(data['w'].flatten())),
            cmax=1,
            showscale=False,
            opacity = 0.8
        ),
        go.Surface(               # Plane Y=0
            x=np.transpose(np.array([Xplot]*nZ)),
            y=np.zeros((nX,nZ)),
            z=np.array([Zplot]*len(Yplot)),
            colorscale='gray_r',
            surfacecolor=np.zeros((len(Xd),len(Yd))),
            cmin=0, #np.min(np.real(data['w'].flatten())),
            cmax=1,
            showscale=False,
            opacity = 0.5
        ),
        go.Scatter3d(
            x=np.max(Xplot)*np.ones_like(Zplot), y=np.zeros_like(Zplot), z=Zplot,
            mode="lines",
            line=dict(color='darkblue',width=3)
        ),
        go.Scatter3d(
            x=-np.max(Xplot)*np.ones_like(Zplot), y=np.zeros_like(Zplot), z=Zplot,
            mode="lines",
            line=dict(color='darkblue',width=3)
        ),
        go.Scatter3d(
            x=Xplot, y=np.zeros_like(Xplot), z=np.max(Zplot)*np.ones_like(Xplot),
            mode="lines",
            line=dict(color='darkblue',width=3)
        ),
        go.Scatter3d(
            x=Xplot, y=np.zeros_like(Xplot), z=-np.pi*np.ones_like(Xplot),
            mode="lines",
            line=dict(color='darkblue',width=3)
        ),
        go.Scatter3d(
            x=np.max(Xplot)*np.ones_like(Yplot), y=Yplot, z=-1.0*np.ones_like(Yplot),
            mode="lines",
            line=dict(color='goldenrod',width=3)
        ),
        go.Scatter3d(
            x=-np.max(Xplot)*np.ones_like(Yplot), y=Yplot, z=-1.0*np.ones_like(Yplot),
            mode="lines",
            line=dict(color='goldenrod',width=3)
        ),
        go.Scatter3d(
            x=Xplot, y=np.max(Yplot)*np.ones_like(Xplot), z=-1.0*np.ones_like(Xplot),
            mode="lines",
            line=dict(color='goldenrod',width=3)
        ),
        go.Scatter3d(
            x=Xplot, y=-np.max(Yplot)*np.ones_like(Xplot), z=-1.0*np.ones_like(Xplot),
            mode="lines",
            line=dict(color='goldenrod',width=3)
        ),
        go.Scatter3d(
            x=np.linspace(-np.max(Xplot),np.max(Xplot)+0.4,len(Xplot)), y=(-np.max(Yplot))*np.ones_like(Xplot), z=-np.pi*np.ones_like(Xplot),
            mode="lines",
            line=dict(color='gray',width=2)
        ),
        go.Scatter3d(
            x=(-np.max(Xplot))*np.ones_like(Yplot), y=np.linspace(-np.max(Yplot),np.max(Yplot)+0.4,len(Yplot)), z=-np.pi*np.ones_like(Yplot),
            mode="lines",
            line=dict(color='gray',width=2)
        ),
        go.Scatter3d(
            x=(-np.max(Xplot))*np.ones_like(Zplot), y=(-np.max(Yplot))*np.ones_like(Zplot), z=np.linspace(-np.pi,0.4,len(Zplot)),
            mode="lines",
            line=dict(color='gray',width=2)
        ),
        go.Scatter3d(
            x=XX, y=(-np.max(XX))*np.ones_like(XX), z=(-np.pi+0.02)*np.ones_like(XX),
            mode="lines",
            line=dict(color='black',width=3)
        ),
        go.Scatter3d(
            x=( np.max(XX))*np.ones_like(YY), y=YY, z=(-np.pi+0.02)*np.ones_like(YY),
            mode="lines",
            line=dict(color='black',width=3)
        ),
        go.Scatter3d(
            x=(-np.max(XX))*np.ones_like(YY), y=YY, z=(-np.pi+0.02)*np.ones_like(YY),
            mode="lines",
            line=dict(color='black',width=3)
        ),
        go.Scatter3d(
            x=XX, y=( np.max(YY))*np.ones_like(XX), z=(-np.pi+0.02)*np.ones_like(XX),
            mode="lines",
            line=dict(color='black',width=3)
        )])
    fig1.update_layout(showlegend=False,
        scene = dict(
            aspectmode='manual',
            aspectratio=dict(x=1,y=1,z=0.7),
            xaxis = dict(visible=False),
            yaxis = dict(visible=False),
            zaxis =dict(visible=False)
            ))
    config = {
        'toImageButtonOptions': {
            'format': 'png', # one of png, svg, jpeg, webp
            'filename': '3Dimage',
            'height': 1000,
            'width': 1000,
            'scale':10 # Multiply title/legend/axis/canvas sizes by this factor
        }
    }
    fig1.show(config=config)
    return 0


# Animations
def animationPlaneY0(path,sourcefile,wavefield,nm,t,field='w',vmin=None,vmax=None):
    # Read source file
    source = np.load(path+sourcefile, allow_pickle=True).item()
    omega = source['omega']
    XX, YY = source['XX'], source['YY']
    HH = source['HH']

    # Read wavefile
    w = wavefield[field][:nm,:,:]
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    delta = Xplot[1]-Xplot[0]
    Zplot = np.arange(-np.pi,0,delta)
    nX, nY, nZ = len(Xplot), len(Yplot), len(Zplot)
    
    # Mode an(z) definition
    if field=='w':
        an = lambda n, z : np.sin(n*(np.pi+z))
    else : 
        an = lambda n, z : np.cos(n*(np.pi+z))
    p = np.arange(1,nm+1,1)

    # Initial values
    Zd, Xd = np.meshgrid(Zplot,Xplot)
    ptile = np.tile(p,(nZ,nX,1)).transpose()        # shape = (nm, nX, nZ)
    Ztile = np.tile(Zplot,(nm,nX,1))                # shape = (nm, nX, nZ)
    indexY = np.argmin(np.abs(Yplot))
    wtile = np.tile(w[:nm,:,indexY].transpose(),(nZ,1,1)).transpose()
    values = np.sum(wtile*an(ptile, Ztile), axis=0)

    # Animation
    fig, ax = plt.subplots(figsize=(30,10))
    ax.set_aspect('equal')
    ax.set_ylim([-np.pi, 0])
    ax.set_xlim([np.min(Xd), np.max(Xd)])

    ims=[]
    for i in range(len(t)):
        val = np.real(values*np.exp(-1j*omega*t[i]))
        im = ax.imshow(val.transpose(),cmap='RdBu_r',origin='lower',extent=[np.min(Xd),np.max(Xd),-np.pi,0],vmin=vmin,vmax=vmax,animated=True)
        #ax.colorbar()
        #index_y = np.argmin(np.abs(YY))
        #ax.scatter(XX,HH[:,index_y]-np.pi,'k-')
        ims.append([im])
    
    ax.set_xlabel(r'$\frac{\pi x}{\mu H_0}$', fontsize=15)
    ax.set_ylabel(r'$\frac{\pi z}{H_0}$', fontsize=15, rotation=0)
    fig.colorbar(im)
    ani = animation.ArtistAnimation(fig,ims)
    figname = "animation_"+field+"_"+(sourcefile.split(".npy")[0]).split("_")[1]+"_p"+str(nm)+"_Y0.mp4"
    ani.save(path+figname, writer="ffmpeg")
    return 0


def animationPlaneZc(path,sourcefile,wavefield,field,nm,cstZ,nbframe,vmax=None):
    # Read source file
    source = np.load(path+sourcefile, allow_pickle=True).item()
    omega = source['omega']
    XX, YY = source['XX'], source['YY']
    HH = source['HH']

    # Read wavefile
    w = wavefield[field][:nm,:,:]
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    delta = Xplot[1]-Xplot[0]
    Zplot = np.arange(-np.pi,0,delta)
    nX, nY, nZ = len(Xplot), len(Yplot), len(Zplot)
    
    # Mode an(z) definition
    if field=='w':
        an = lambda n, z : np.sin(n*(np.pi+z))
    else : 
        an = lambda n, z : np.cos(n*(np.pi+z))
    p = np.arange(1,nm+1,1)

    # Initial values
    Yd, Xd = np.meshgrid(Yplot,Xplot)
    ptile = np.tile(p,(nY,nX,1)).transpose()        # shape = (nm, nX, nZ)
    Ztile = cstZ*np.ones((nm,nX,nY))                # shape = (nm, nX, nZ)
    indexY = np.argmin(np.abs(Yplot))
    values = np.sum(w*an(ptile, Ztile), axis=0)
    if vmax==None:
        vmax = max(np.max(np.real(values)),np.max(np.imag(values)))

    ## ANIMATION
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_ylim([np.min(Yd), np.max(Yd)])
    ax.set_xlim([np.min(Xd), np.max(Xd)])
    ax.set_xlabel(r'$\frac{\pi x}{\mu H_0}$', fontsize=15)
    ax.set_ylabel(r'$\frac{\pi y}{\mu H_0}$', fontsize=15, rotation=0)
    ti = ax.set_title('Non-dimensional '+field+r' for $\frac{\pi z}{H_0} = $'+str(cstZ)+r' and $\omega t$ = 0')
    val = np.real(values)
    cax = ax.pcolormesh(Xd, Yd, val, cmap='RdBu_r',vmin=-vmax,vmax=vmax, shading='gouraud')
    fig.colorbar(cax)

    def animate(num, cax, values, nbframe):
        t = 2*np.pi/nbframe*num
        val = np.real(values*np.exp(-1j*t))
        cax.set_array(val.ravel())
        ti.set_text("Non-dimensionnal "+field+r' for $\frac{\pi z}{H_0} = $'+str(cstZ)+r' and $\omega t =$'+str(2*num)+r'$\pi /$'+str(nbframe))
    
    anim = animation.FuncAnimation(fig, animate, fargs=(cax, values, nbframe), frames=nbframe, blit=False)
    figname = "animation_"+field+"_"+(sourcefile.split(".npy")[0]).split("_")[1]+"_p"+str(nm)+"_Zc.mp4"
    anim.save(path+figname, writer="ffmpeg")
    plt.close()

    return 0


def animationUHor(path,sourcefile,wavefield,nm,nbframe,cstZ,vmax=None):
    # Read source file
    source = np.load(path+sourcefile, allow_pickle=True).item()
    omega = source['omega']
    XX, YY = source['XX'], source['YY']
    HH = source['HH']

    # Read wavefile
    u, v = wavefield['u'][:nm,:,:], wavefield['v'][:nm,:,:]             # shape = (nm, nX, nY)
    Xplot, Yplot = wavefield['X'], wavefield['Y']
    nX, nY = len(Xplot), len(Yplot)

    Yd, Xd = np.meshgrid(Yplot,Xplot, copy=False)   # shape = (nX, nY)
    p = np.arange(1,nm+1,1)
    ptile = np.tile(p,(nY,nX,1)).transpose()        # shape = (nm, nX, nY)
    uvalues = np.sum(u*np.cos(ptile*(np.pi+cstZ)), axis=0)     # shape = (nX, nY)
    vvalues = np.sum(v*np.cos(ptile*(np.pi+cstZ)), axis=0)     # shape = (nX, nY)
    if vmax==None:
        vmax = max(np.max(np.sqrt(np.real(uvalues)**2+np.real(vvalues)**2)), np.max(np.sqrt(np.imag(uvalues)**2+np.imag(vvalues)**2)))

    ## ANIMATION
    fig, ax = plt.subplots(1, 1)
    ax.set_aspect('equal')
    ax.set_ylim([np.min(Yd), np.max(Yd)])
    ax.set_xlim([np.min(Xd), np.max(Xd)])
    ax.set_xlabel(r'$\frac{\pi x}{\mu H_0}$', fontsize=15)
    ax.set_ylabel(r'$\frac{\pi y}{\mu H_0}$', fontsize=15, rotation=0)
    ti = ax.set_title(r'$\| \vec u_H \| / U_0$ for $\frac{\pi z}{H_0} = $'+str(cstZ)+r' and $\omega t$ = 0')
    norm = np.sqrt(np.real(uvalues)**2 + np.real(vvalues)**2)
    cax = ax.pcolormesh(Xd, Yd, norm, cmap='Reds',vmin=0,vmax=vmax, shading='gouraud')
    fig.colorbar(cax)

    # Arrows and ellipse
    nArrows = 7
    R, theta = 5, np.linspace(0,2*np.pi,13)
    scale = 20
    ind_i = np.array([np.argmin(np.abs(Xplot-R*np.cos(theta[i]))) for i in range(len(theta))])
    ind_j = np.array([np.argmin(np.abs(Yplot-R*np.sin(theta[i]))) for i in range(len(theta))])
    t = np.linspace(0, 2*np.pi,100)
    for i in range(len(theta)):
        xi, yi = Xplot[ind_i[i]], Yplot[ind_j[i]]
        ax.plot(xi+scale*np.real(uvalues[ind_i[i],ind_j[i]]*np.exp(-1j*t)), yi+scale*np.real(vvalues[ind_i[i],ind_j[i]]*np.exp(-1j*t)), 'k-', linewidth=1)
    qr = ax.quiver(Xd[ind_i,ind_j],Yd[ind_i,ind_j],scale*np.real(uvalues[ind_i,ind_j]),scale*np.real(vvalues[ind_i,ind_j]), angles='xy', scale=1, scale_units='xy',width=0.005)

    def animate(num, qr, cax, uvalues, vvalues, ind_i, ind_j, nbframe):
        t = 2*np.pi/nbframe*num
        ut = np.real(uvalues*np.exp(-1j*t))
        vt = np.real(vvalues*np.exp(-1j*t))
        norm = np.sqrt(ut**2+vt**2)
        cax.set_array(norm.ravel())
        qr.set_UVC(scale*ut[ind_i,ind_j], scale*vt[ind_i,ind_j])
        ti.set_text(r"Non-dimensionnal $\| u_H \|$ for pi z / mu = "+str(cstZ)+r" and $\omega t =$"+str(2*num)+r"$\pi /$"+str(nbframe))
    
    anim = animation.FuncAnimation(fig, animate, fargs=(qr, cax, uvalues, vvalues, ind_i, ind_j, nbframe), frames=nbframe, blit=False)
    figname = "animation_uH_"+(sourcefile.split(".npy")[0]).split("_")[1]+"_p"+str(nm)+"_Y0.mp4"
    anim.save(path+figname, writer="ffmpeg")
    plt.close()

    return 0



# Others
def plotHankel():
    x = np.linspace(0.01, 10, 1000)
    h = hankel1(0,x)
    plt.figure()
    plt.grid()
    plt.plot(x,np.real(h),label = "$Re (H_0^{(1)}(x))$")
    plt.plot(x,np.imag(h),label = "$Im (H_0^{(1)}(x))$")
    plt.ylim(bottom=-2.0)
    plt.xlabel("x")
    plt.legend()

    h = hankel1(1,x)
    plt.figure()
    plt.grid()
    plt.plot(x,np.real(h),label = "$Re( H_1^{(1)}(x))$")
    plt.plot(x,np.imag(h),label = "$Im( H_1^{(1)}(x))$")
    #plt.plot(x,np.imag(h0),label = "Expansion at 0")
    plt.ylim(bottom=-2.0)
    plt.xlabel("x")
    plt.legend()

    # H0 : expansions
    #gamma = 0.5772156649
    #h0 = 1 + 2*1j/np.pi*(np.log(x/2)+gamma)

    # H1 : expansions
    #xinf = np.linspace(0.5,10, 1000)
    #hinf = np.sqrt(2/np.pi/xinf) * np.exp(1j*(xinf-3*np.pi/4))
    #x0 = 1.5 * 10**(np.linspace(-10, 0, 1000))
    #h0 = x0/2 - 1j/np.pi*2/x0
    #plt.plot(xinf,np.imag(hinf),label = "Expansion at infinity")
    #plt.plot(x0,np.imag(h0),label = "Expansion at 0")

def plotP0rho0():
    # Plot typical pressure, density and stratification profiles in the ocean
    N0, Nmax = 6*1e-4, 5.48*1e-3
    zc, sig = -400, 250
    z = np.linspace(-3000, 0, 1000)
    N = N0 + (Nmax-N0)*np.exp(-(z-zc)**2/sig**2)

    rho0 = 1030 * np.ones(len(z))
    P0 = 300*101325*np.ones(len(z))
    for i in range(len(z)-1):
        rho0[i+1] = rho0[i] - 1000/9.81*N[i]**2*(z[i+1]-z[i])
        P0[i+1] = P0[i] - 9.81*rho0[i]*(z[i+1]-z[i])

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True)
    #fig.suptitle('Horizontally stacked subplots')
    ax1.plot(P0/101325,z)
    ax1.set_ylabel("z")
    ax1.set_xlabel("Pressure $P_0(z)$ (atm)")
    ax1.grid()
    ax2.plot(rho0/1000,z)
    ax2.set_xlabel("Density ($g.cm^{-3}$)")
    ax2.grid()
    ax3.plot(N,z)
    ax3.set_xlabel("$N(z)$")
    ax3.grid()