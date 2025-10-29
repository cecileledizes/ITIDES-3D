import numpy as np
import matplotlib.pyplot as plt
from plots import plotTopoPoints, plotTopo, plotTopo3D, plotTopoBox3D
from scipy.signal import convolve2d
from scipy import interpolate
from scipy.optimize import fsolve


def generateGaussianTopography(path,topofile,nx,ny,hmax,sigX,sigY,H0,nsig=3,smooth=None):
    # Inputs : 
    #   'path'  directory in which to save topofile
    #   'topofile'      File with the topography description
    #   'nx', 'ny'      Number of points in the x and y directions
    #   'hmax'          Maximum height of topo
    #   'sigX', 'sigY'  X - Y standard deviation
    #   'H0'            Global height of the ocean
    xmax, ymax = nsig*sigX, nsig*sigY     # Maximum value of x and y
    xx, yy = np.linspace(-xmax,xmax,nx), np.linspace(-ymax,ymax,ny)
    delta = xx[1]-xx[0]
    print("dx = "+str(delta))
    print("dy = "+str(yy[1]-yy[0]))

    ymesh, xmesh = np.meshgrid(yy,xx)
    gauss = hmax*np.exp(-xmesh**2/2/sigX**2 - ymesh**2/2/sigY**2)
    dgauss_x = -hmax*xmesh/sigX**2*np.exp(-xmesh**2/2/sigX**2 - ymesh**2/2/sigY**2)
    dgauss_y = -hmax*ymesh/sigY**2*np.exp(-xmesh**2/2/sigX**2 - ymesh**2/2/sigY**2)
    
    xlim, ylim = xmax-sigX, ymax-sigY
    if smooth=='mean':
        # Moving average over npoints
        npoint = int(np.round(nx/2/nsig,0))
        fconv = np.ones((npoint,npoint))/npoint**2
        gauss_conv = np.where(np.abs(xmesh)>(xmax+xlim)/2,0.0,gauss)
        hmean = convolve2d(gauss_conv,fconv,mode='same')
        hh = np.where((hmean<gauss)&(np.abs(xmesh)>xlim),hmean,gauss)
        dhh_x, dhh_y = np.gradient(hh,xx[1]-xx[0],yy[1]-yy[0])
        dhh_x = np.where((hmean<gauss)&(np.abs(xmesh)>xlim),dhh_x,dgauss_x)
        dhh_y = np.where((hmean<gauss)&(np.abs(xmesh)>xlim),dhh_y,dgauss_y)
        '''
        elif smooth=='tanh':
            dec = 10
            f_x  = np.tanh(dec*(xmesh+xmax))-np.tanh(dec*(xmesh-xmax))-1
            df_x = dec*(-np.tanh(dec*(xmesh+xmax))**2 + np.tanh(dec*(xmesh-xmax))**2)
            f_y  = np.tanh(dec*(ymesh+ymax))-np.tanh(dec*(ymesh-ymax))+1
            df_y = dec*(np.tanh(dec*(ymesh+ymax))**2+ np.tanh(dec*(ymesh-ymax))**2)
            hh = gauss*f_x*f_y
            dhh_x = gauss*df_x*f_y+dgauss_x*f_x*f_y
            dhh_y = gauss*f_x*df_y+dgauss_y*f_x*f_y
        '''
    elif smooth=='exp':
            exp = 2*int(nsig+1)
            print(exp)
            # f_x  = np.where(xmesh<-(nsig-1)*sigX, np.exp(1 - 1/(1-(xmesh/sigX+(nsig-1))**exp)), 1.0)
            # #df_x = np.where(xmesh<-(nsig-1)*sigX, exp*xmax**exp*xmesh**(exp-1)/(xmax**exp-xmesh**exp)**2*np.exp(1 - 1/(1-(xmesh/sigX+(nsig-1))**exp)), 0.0)
            # f_x  = np.where(xmesh>+(nsig-1)*sigX, np.exp(1 - 1/(1-(xmesh/sigX-(nsig-1))**exp)), f_x)
            # #df_x = np.where(xmesh>+(nsig-1)*sigX, exp*xmax**exp*xx[i]**(exp-1)/(xmax**exp-xx[i]**exp)**2*np.exp(1 - 1/(1-(xmesh/sigX+(nsig-1))**exp)), df_x)
            # f_y  = np.where(ymesh<-(nsig-1)*sigY, np.exp(1 - 1/(1-(ymesh/sigY+(nsig-1))**exp)), 1.0)
            # f_y  = np.where(ymesh>+(nsig-1)*sigY, np.exp(1 - 1/(1-(ymesh/sigY-(nsig-1))**exp)), f_y)
            f_x = np.exp(1 - 1/(1-(xmesh/xmax)**exp))
            f_y = np.exp(1 - 1/(1-(ymesh/ymax)**exp))
            hh = gauss*f_x*f_y
            dhh_x, dhh_y = np.gradient(hh, delta, delta)
            #dhh_x = gauss*df_x*f_y+dgauss_x*f_x*f_y
            #dhh_y = gauss*f_x*df_y+dgauss_y*f_x*f_y
    elif smooth=='expaxi':
            exp = 2#2*int(nsig+1)
            r = np.sqrt(xmesh**2+ymesh**2)
            f_r = np.ma.masked_array(np.exp(1 - 1/(1-(r/xmax)**exp)),mask=r>=xmax).filled(0.0)
            hh = gauss*f_r+np.min(gauss)
            dhh_x, dhh_y = np.gradient(gauss*f_r, delta, delta)
    elif smooth=='cos':
        f_x  = np.where(xmesh<-xlim, 0.5*(1+np.cos(np.pi/(xmax-xlim)*(xmesh+xlim))), 1.0)
        f_x  = np.where(xmesh>+xlim, 0.5*(1+np.cos(np.pi/(xmax-xlim)*(xmesh-xlim))), f_x)
        df_x  = np.where(xmesh<-xlim, -0.5*np.pi/(xmax-xlim)*np.sin(np.pi/(xmax-xlim)*(xmesh+xlim)), 0.0)
        df_x  = np.where(xmesh>+xlim, -0.5*np.pi/(xmax-xlim)*np.sin(np.pi/(xmax-xlim)*(xmesh-xlim)), df_x)

        f_y  = np.where(ymesh<-ylim, 0.5*(1+np.cos(np.pi/(ymax-ylim)*(ymesh+ylim))), 1.0)
        f_y  = np.where(ymesh>+ylim, 0.5*(1+np.cos(np.pi/(ymax-ylim)*(ymesh-ylim))), f_y)
        df_y  = np.where(ymesh<-ylim, -0.5*np.pi/(ymax-ylim)*np.sin(np.pi/(ymax-ylim)*(ymesh+ylim)), 0.0)
        df_y  = np.where(xmesh>+xlim, -0.5*np.pi/(ymax-ylim)*np.sin(np.pi/(ymax-ylim)*(ymesh-ylim)), df_y)

        hh = gauss*f_x*f_y
        dhh_x = gauss*df_x*f_y+dgauss_x*f_x*f_y
        dhh_y = gauss*f_x*df_y+dgauss_y*f_x*f_y
    else : 
        smooth = None
        hh = gauss
        dhh_x = dgauss_x
        dhh_y = dgauss_y

    if smooth!=None:
        xmesh, ymesh = xmesh[1:-1, 1:-1], ymesh[1:-1, 1:-1]
        hh = hh[1:-1, 1:-1]
        dhh_x, dhh_y = dhh_x[1:-1, 1:-1], dhh_y[1:-1, 1:-1]

        ind_y0 = np.argmin(np.abs(yy))
        fig, ax = plt.subplots()
        ax.plot(xx, gauss[:,ind_y0],'k+-')
        ax.plot(xmesh[:,ind_y0], hh[:,ind_y0],'r+-')

        fig, ax = plt.subplots()
        ax.plot(xx, dgauss_x[:,ind_y0],'k+-')
        ax.plot(xmesh[:,ind_y0], dhh_x[:,ind_y0],'r+-')

        fig, ax = plt.subplots()
        ax.plot(xx, dgauss_x[:,0],'k+-')
        ax.plot(xmesh[:,0], dhh_x[:,0],'r+-')
    
        xx, yy = xx[1:-1], yy[1:-1]
    
    x, y, h = xmesh.flatten(), ymesh.flatten(), hh.flatten()
    dh_x, dh_y = dhh_x.flatten(), dhh_y.flatten()
    print("min(dh_x) = "+str(np.min(np.abs(np.ma.masked_array(dh_x,mask=dh_x==0)))))
    topo = dict([('delta',delta),('x',x),('y',y),('h',H0+h),('dh_x',dh_x),('dh_y',dh_y),('xx',xx),('yy',yy),('hh',H0+hh),('dhh_x',dhh_x),('dhh_y',dhh_y)])
    np.save(path+topofile,topo,allow_pickle=True)
    return 0

def generateRoundSmoothGaussianTopo(path,topofile,nx,ny,hmax,sigX,sigY,H0,nsig=3,test_plot=True):
    # Inputs : 
    #   'path'  directory in which to save topofile
    #   'topofile'      File with the topography description
    #   'nx', 'ny'      Number of points in the x and y directions
    #   'hmax'          Maximum height of topo
    #   'sigX', 'sigY'  X - Y standard deviation
    #   'H0'            Global height of the ocean
    xmax, ymax = nsig*sigX, nsig*sigY     # Maximum value of x and y
    xx, yy = np.linspace(-xmax,xmax,nx), np.linspace(-ymax,ymax,ny)
    delta = xx[1]-xx[0]
    rmax = max(xmax,ymax)

    nx, ny = len(xx), len(yy)
    ymesh, xmesh = np.meshgrid(yy,xx)
    gauss = hmax*np.exp(-xmesh**2/2/sigX**2 - ymesh**2/2/sigY**2)
    dgauss_x = -hmax*xmesh/sigX**2*np.exp(-xmesh**2/2/sigX**2 - ymesh**2/2/sigY**2)
    dgauss_y = -hmax*ymesh/sigY**2*np.exp(-xmesh**2/2/sigX**2 - ymesh**2/2/sigY**2)
    
    exp = 2*int(nsig+1)
    r = np.sqrt(xmesh**2+ymesh**2)
    f_r = np.ma.masked_array(np.exp(1 - 1/(1-(r/rmax)**exp)),mask=r>=rmax).filled(0.0)
    hh = gauss*f_r
    dhh_x, dhh_y = np.gradient(hh, delta, delta)
    dhh_x = np.nan_to_num(dhh_x)
    dhh_y = np.nan_to_num(dhh_y)

    maskf = (r>rmax).flatten()
    h = hh.flatten()[maskf==False]
    dh_x = dhh_x.flatten()[maskf==False]
    dh_y = dhh_y.flatten()[maskf==False]
    x, y = xmesh.flatten()[maskf==False], ymesh.flatten()[maskf==False]
    if test_plot==True:
        xmesh, ymesh = xmesh[1:-1, 1:-1], ymesh[1:-1, 1:-1]
        hh = hh[1:-1, 1:-1]
        dhh_x, dhh_y = dhh_x[1:-1, 1:-1], dhh_y[1:-1, 1:-1]

        ind_y0 = np.argmin(np.abs(yy))
        fig, ax = plt.subplots()
        ax.plot(xx, gauss[:,ind_y0],'k+-')
        ax.plot(xmesh[:,ind_y0], hh[:,ind_y0],'r+-')

        fig, ax = plt.subplots()
        ax.plot(xx, dgauss_x[:,ind_y0],'k+-')
        ax.plot(xmesh[:,ind_y0], dhh_x[:,ind_y0],'r+-')

        fig, ax = plt.subplots()
        ax.plot(xx, dgauss_x[:,0],'k+-')
        ax.plot(xmesh[:,0], dhh_x[:,0],'r+-')
    
        xx, yy = xx[1:-1], yy[1:-1]

    
    print("min(dh_x) = "+str(np.min(np.abs(np.ma.masked_array(dh_x,mask=dh_x==0)))))
    topo = dict([('delta',delta),('x',x),('y',y),('h',H0+h),('dh_x',dh_x),('dh_y',dh_y),('xx',xx),('yy',yy),('hh',H0+hh),('dhh_x',dhh_x),('dhh_y',dhh_y)])
    np.save(path+topofile,topo,allow_pickle=True)
    return 0


def generateRoundGaussianTopo(path,topofile,nx,ny,hmax,sigX,sigY,H0,nsig=3):
    # Inputs : 
    #   'path'  directory in which to save topofile
    #   'topofile'      File with the topography description
    #   'nx', 'ny'      Number of points in the x and y directions
    #   'hmax'          Maximum height of topo
    #   'sigX', 'sigY'  X - Y standard deviation
    #   'H0'            Global height of the ocean
    xmax, ymax = nsig*sigX, nsig*sigY     # Maximum value of x and y
    xx, yy = np.linspace(-xmax,xmax,nx), np.linspace(-ymax,ymax,ny)
    delta = xx[1]-xx[0]
    nx, ny = len(xx), len(yy)
    ymesh, xmesh = np.meshgrid(yy, xx)
    #rmesh = np.sqrt((xmesh**2+ymesh**2))
    #sig = max(sigX,sigY)
    #mask = rmesh>=nsig*sig
    hgauss = hmax*np.exp(-xmesh**2/2/sigX**2 - ymesh**2/2/sigY**2)
    hmin = min(hmax*np.exp(np.max(yy)**2/2/sigY**2), hmax*np.exp(-np.max(xx)**2/2/sigX**2))
    mask = hgauss<=hmin
    hh = np.ma.masked_array(hgauss, mask=mask)
    hh = H0+hh-np.min(hh)
    h = (hh.flatten()).compressed()
    hh = hh.filled(H0)

    dhh_x = np.ma.masked_array(-hmax*xmesh/sigX**2*np.exp(-xmesh**2/2/sigX**2 - ymesh**2/2/sigY**2), mask=mask)
    dhh_y = np.ma.masked_array(-hmax*ymesh/sigY**2*np.exp(-xmesh**2/2/sigX**2 - ymesh**2/2/sigY**2), mask=mask)
    dh_x, dh_y = (dhh_x.flatten()).compressed(), (dhh_y.flatten()).compressed()
    dhh_x, dhh_y = dhh_x.filled(0.0), dhh_y.filled(0.0)

    x, y = (np.ma.masked_array(xmesh, mask=mask).flatten()).compressed(), (np.ma.masked_array(ymesh, mask=mask).flatten()).compressed()

    print("min(dh_x) = "+str(np.min(np.abs(dh_x))))
    print("min(dh_x) = "+str(np.min(np.abs(np.ma.masked_array(dh_x,mask=dh_x==0)))))
    print("max(dh_x) = "+str(np.max(np.abs(dh_x))))
    topo = dict([('delta',delta),('x',x),('y',y),('h',h),('dh_x',dh_x),('dh_y',dh_y),('xx',xx),('yy',yy),('hh',hh),('dhh_x',dhh_x),('dhh_y',dhh_y)])
    np.save(path+topofile,topo,allow_pickle=True)
    return 0


def generateMaasTopo(path,topofile,nx,ny,mu,H0):
    # Inputs : 
    #   'path'  directory in which to save topofile
    #   'topofile'      File with the topography description
    #   'nx', 'ny'      Number of points in the x and y directions
    #   'hmax'          Maximum height of topo
    #   'sigX', 'sigY'  X - Y standard deviation
    #   'H0'            Global height of the ocean


    # Dimensionless value with slope 1 and H0 = -1
    D = 2
    t = lambda z: D/(-1-z)*np.sinh(z)-np.cosh(z)-1
    zmax = fsolve(t, -0.5)[0]
    zH = np.linspace(-1, zmax, 60000)
    rH = np.arccosh(D/(-1-zH)*np.sinh(zH)-np.cosh(zH))
    zH = np.array([zH[i] for i in range(len(rH)) if np.isnan(rH[i])==False])
    rH = np.array([rH[i] for i in range(len(rH)) if np.isnan(rH[i])==False])
    rH = np.append(rH, [0])
    zH = np.append(zH, [zmax])

    f = interpolate.interp1d(rH, zH, kind='cubic')

    xmax, ymax = 8, 8    # Maximum value of x and y
    xx, yy = np.linspace(-xmax,xmax,nx), np.linspace(-ymax,ymax,ny)
    delta = xx[1]-xx[0]

    ymesh, xmesh = np.meshgrid(xx, yy)
    rmesh = np.sqrt(xmesh**2+ymesh**2)
    hh = f(rmesh)
    dhh_x, dhh_y = np.gradient(hh, delta, axis=0), np.gradient(hh, delta, axis=1)
    x, y, h = xmesh.flatten(), ymesh.flatten(), hh.flatten()
    dh_x, dh_y = dhh_x.flatten(), dhh_y.flatten()
    print("min(dh_x) = "+str(np.min(np.abs(np.ma.masked_array(dh_x,mask=dh_x==0)))))
    print("eps = "+str(np.max(np.abs(dh_x))))

    # Dimensional values
    topo = dict([('delta',delta*mu*np.abs(H0)),('x',x*mu*np.abs(H0)),('y',y*mu*np.abs(H0)),
                 ('h',h*np.abs(H0)),('dh_x',dh_x/mu),('dh_y',dh_y/mu),
                 ('xx',xx*mu*np.abs(H0)),('yy',yy*mu*np.abs(H0)),
                 ('hh',hh*np.abs(H0)),('dhh_x',dhh_x/mu),('dhh_y',dhh_y/mu)])
    
    np.save(path+topofile,topo,allow_pickle=True)

    ind = np.argmax(dh_x)
    Lambda = np.max(h)*np.abs(H0)-H0
    L = x[ind]*mu*np.abs(H0)
    return Lambda, L, delta*np.pi, np.max(x)*np.pi


def generateRingTopography(path,topofile,nx,ny,hmax,R,sig,H0,nsig=3,smooth=True):
    # Inputs : 
    #   'path'  directory in which to save topofile
    #   'topofile'      File with the topography description
    #   'nx', 'ny'      Number of points in the x and y directions
    #   'hmax'          Maximum height of topo
    #   'sigX', 'sigY'  X - Y standard deviation
    #   'H0'            Global height of the ocean
    xmax, ymax = nsig*sig, nsig*sig     # Maximum value of x and y
    xx, yy = np.linspace(-xmax,xmax,nx), np.linspace(-ymax,ymax,ny)
    delta = xx[1]-xx[0]
    nx, ny = len(xx), len(yy)
    hh = np.zeros((nx,ny))
    dhh_x, dhh_y = np.zeros((nx,ny)), np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            r = np.sqrt(xx[i]**2+yy[j]**2)
            hh[i,j] = H0+hmax*np.exp(-(r-R)**2/2/sig**2)
            dhh_x[i,j] = -hmax/sig**2*xx[i]/r*(r-R)*np.exp(-(r-R)**2/2/sig**2)   #*(1+np.random.rand(1)/50)
            dhh_y[i,j] = -hmax/sig**2*yy[i]/r*(r-R)*np.exp(-(r-R)**2/2/sig**2)   #*(1+np.random.rand(1)/50)
    
    ymesh, xmesh = np.meshgrid(yy,xx)
    x, y, h = xmesh.flatten(), ymesh.flatten(), hh.flatten()
    dh_x, dh_y = dhh_x.flatten(), dhh_y.flatten()
    print("min(dh_x) = "+str(np.min(np.abs(np.ma.masked_array(dh_x,mask=dh_x==0)))))
    topo = dict([('delta',delta),('x',x),('y',y),('h',H0+h),('dh_x',dh_x),('dh_y',dh_y),('xx',xx),('yy',yy),('hh',H0+hh),('dhh_x',dhh_x),('dhh_y',dhh_y)])
    np.save(path+topofile,topo,allow_pickle=True)
    return 0


def generateArcTopography(path,topofile,nx,ny,hmax,R,sig,H0,nsig=3,theta_0=np.pi/2):
    # Inputs : 
    #   'path'  directory in which to save topofile
    #   'topofile'      File with the topography description
    #   'nx', 'ny'      Number of points in the x and y directions
    #   'hmax'          Maximum height of topo
    #   'sigX', 'sigY'  X - Y standard deviation
    #   'H0'            Global height of the ocean
    xmax, ymax = nsig*sig, nsig*sig     # Maximum value of x and y
    xx, yy = np.linspace(-xmax,xmax,nx), np.linspace(-ymax,ymax,ny)
    delta = xx[1]-xx[0]
    nx, ny = len(xx), len(yy)
    hh = np.zeros((nx,ny))
    dhh_x, dhh_y = np.zeros((nx,ny)), np.zeros((nx,ny))
    for i in range(nx):
        for j in range(ny):
            r = np.sqrt(xx[i]**2+yy[j]**2)
            theta = np.arccos(xx[i]/r)*np.copysign(1,yy[j])
            if (theta>=0):
                env = 1-np.tanh(3*(theta-theta_0))
            else :
                env = 1+np.tanh(3*(theta+theta_0))
            hh[i,j] = H0+hmax*np.exp(-(r-R)**2/2/sig**2)*env
    dhh_x = np.gradient(hh,axis=0)   #*(1+np.random.rand(1)/50)
    dhh_y = np.gradient(hh,axis=1)   #*(1+np.random.rand(1)/50)
    
    ymesh, xmesh = np.meshgrid(yy,xx)
    x, y, h = xmesh.flatten(), ymesh.flatten(), hh.flatten()
    dh_x, dh_y = dhh_x.flatten(), dhh_y.flatten()
    print("min(dh_x) = "+str(np.min(np.abs(np.ma.masked_array(dh_x,mask=dh_x==0)))))
    topo = dict([('delta',delta),('x',x),('y',y),('h',H0+h),('dh_x',dh_x),('dh_y',dh_y),('xx',xx),('yy',yy),('hh',H0+hh),('dhh_x',dhh_x),('dhh_y',dhh_y)])
    np.save(path+topofile,topo,allow_pickle=True)
    return 0


def generateBump(path,topofile,nx,ny,hmax,R,H0,nsig=1,exp=2):
# Inputs : 
    #   'path'  directory in which to save topofile
    #   'topofile'      File with the topography description
    #   'nx', 'ny'      Number of points in the x and y directions
    #   'hmax'          Maximum height of topo
    #   'sigX', 'sigY'  X - Y standard deviation
    #   'H0'            Global height of the ocean
    xmax, ymax = nsig*R, nsig*R     # Maximum value of x and y
    xx, yy = np.linspace(-xmax,xmax,nx), np.linspace(-ymax,ymax,ny)
    delta = xx[1]-xx[0]

    nx, ny = len(xx), len(yy)
    hh, dhh_x, dhh_y = np.zeros((nx,ny)), np.zeros((nx,ny)), np.zeros((nx,ny))
    x, y, h, dh_x, dh_y = [], [], [], [], []
    for i in range(nx):
        for j in range(ny):
            r = np.sqrt(xx[i]**2+yy[j]**2)
            if (r>=R):
                hh[i,j] = H0
                dhh_x[i,j] = 0 
                dhh_y[i,j] = 0
            else :
                hh[i,j] = H0+hmax*np.exp(1)*np.exp(-1/(1-(r/R)**exp))
                dhh_x[i,j] = -hmax*np.exp(1) * exp*xx[i]*r**(exp-2)/R**exp/(1-(r/R)**exp)**2 * np.exp(-1/(1-(r/R)**exp))
                dhh_y[i,j] = -hmax*np.exp(1) * exp*yy[j]*r**(exp-2)/R**exp/(1-(r/R)**exp)**2 * np.exp(-1/(1-(r/R)**exp))
                x += [xx[i]]
                y += [yy[j]]
                h += [hh[i,j]]
                dh_x += [dhh_x[i,j]]
                dh_y += [dhh_y[i,j]]
    dhh_x = np.nan_to_num(dhh_x)
    dhh_y = np.nan_to_num(dhh_y)
    x, y, h = np.array(x), np.array(y), np.array(h)
    dh_x, dh_y = np.array(dh_x), np.array(dh_y)

    topo = dict([('delta',delta),('x',x),('y',y),('h',h),('dh_x',dh_x),('dh_y',dh_y),('xx',xx),('yy',yy),('hh',hh),('dhh_x',dhh_x),('dhh_y',dhh_y)])
    np.save(path+topofile,topo,allow_pickle=True)
    return 0


def generateEllipsoid(path,topofile,Delta,hmax,Lx,Ly,H0):
        # Inputs : 
    #   'path'  directory in which to save topofile
    #   'topofile'      File with the topography description
    #   'Delta'         Cell size
    #   'hmax'          Maximum height of topo
    #   'H0'            Global height of the ocean

    #xpos, ypos = np.arange(0,Lx,Delta), np.arange(0,Ly,Delta)
    #xx, yy = np.append(-xpos[1:][::-1],xpos), np.append(-ypos[1:][::-1],ypos)
    #nx, ny = len(xx), len(yy)

    d = 0.15*hmax
    R = np.sqrt(1-d**2/(d+hmax)**2)*Lx

    nx = int(np.round(2*R/Delta,0))
    nx = nx-nx%2
    print((nx,nx))
    xx, yy = np.linspace(-R,R,nx), np.linspace(-R,R,nx)
    yM, xM = np.meshgrid(yy,xx)
    hh, dhh_x, dhh_y = np.zeros_like(xM), np.zeros_like(xM), np.zeros_like(xM)

    mask = (xM)**2+(yM)**2>=R**2
    SQR = np.ma.masked_array(np.sqrt(np.abs(1-(xM/Lx)**2-(yM/Ly)**2)), mask=mask)
    hh = (hmax+d)*SQR-d
    dhh_x, dhh_y = -xM/Lx**2*(hmax+d)/SQR, -yM/Ly**2*(hmax+d)/SQR
    x = np.ma.masked_array(xM, mask=mask).flatten().compressed()
    y = np.ma.masked_array(yM, mask=mask).flatten().compressed()
    h = np.ma.masked_array(H0 + hh-np.min(hh), mask=mask).flatten().compressed()
    dh_x = np.ma.masked_array(dhh_x, mask=mask).flatten().compressed()
    dh_y = np.ma.masked_array(dhh_y, mask=mask).flatten().compressed()
    hh = np.ma.masked_array(H0 + hh - np.min(hh), mask=mask).filled(H0)
    dhh_x = np.ma.masked_array(dhh_x, mask=mask).filled(0.0)
    dhh_y = np.ma.masked_array(dhh_y, mask=mask).filled(0.0)

    print("min(dh_x) = "+str(np.min(np.abs(np.ma.masked_array(dh_x,mask=dh_x==0)))))
    print("max(dh_x) = "+str(np.max(np.abs(dh_x))))
    topo = dict([('delta',Delta),('x',x),('y',y),('h',h),('dh_x',dh_x),('dh_y',dh_y),
                 ('xx',xx),('yy',yy),('hh',hh),('dhh_x',dhh_x),('dhh_y',dhh_y)])
    np.save(path+topofile,topo,allow_pickle=True)
    return (nx,nx)


def generateSmoothPillbox(path,topofile,Delta,hmax,R,nR,s,H0):
        # Inputs : 
    #   'path'  directory in which to save topofile
    #   'topofile'      File with the topography description
    #   'Delta'         Cell size
    #   'hmax'          Maximum height of topo
    #   'H0'            Global height of the ocean

    #xpos, ypos = np.arange(0,Lx,Delta), np.arange(0,Ly,Delta)
    #xx, yy = np.append(-xpos[1:][::-1],xpos), np.append(-ypos[1:][::-1],ypos)
    #nx, ny = len(xx), len(yy)

    nx = int(np.round(2*nR*R/Delta,0))
    nx = nx-nx%2
    xx, yy = np.linspace(-nR*R,nR*R,nx), np.linspace(-nR*R,nR*R,nx)
    yM, xM = np.meshgrid(yy,xx)
    r = np.sqrt(xM**2+yM**2)

    mask = r>=nR*R
    hh = hmax/2/np.tanh(s*R)*(np.tanh(s*(r+R))-np.tanh(s*(r-R)))
    dhh_x = hmax/2/np.tanh(s*R)* s*xM/r *(np.tanh(s*(r-R))**2-np.tanh(s*(r+R))**2)
    dhh_y = hmax/2/np.tanh(s*R)* s*yM/r *(np.tanh(s*(r-R))**2-np.tanh(s*(r+R))**2)
    x = np.ma.masked_array(xM, mask=mask).flatten().compressed()
    y = np.ma.masked_array(yM, mask=mask).flatten().compressed()
    h = np.ma.masked_array(H0 + hh-np.min(hh), mask=mask).flatten().compressed()
    dh_x = np.ma.masked_array(dhh_x, mask=mask).flatten().compressed()
    dh_y = np.ma.masked_array(dhh_y, mask=mask).flatten().compressed()
    hh = np.ma.masked_array(H0 + hh - np.min(hh), mask=mask).filled(H0)
    dhh_x = np.ma.masked_array(dhh_x, mask=mask).filled(0.0)
    dhh_y = np.ma.masked_array(dhh_y, mask=mask).filled(0.0)

    print("min(dh_x) = "+str(np.min(np.abs(np.ma.masked_array(dh_x,mask=dh_x==0)))))
    print("max(dh_x) = "+str(np.max(np.abs(dh_x))))
    topo = dict([('delta',Delta),('x',x),('y',y),('h',h),('dh_x',dh_x),('dh_y',dh_y),
                 ('xx',xx),('yy',yy),('hh',hh),('dhh_x',dhh_x),('dhh_y',dhh_y)])
    np.save(path+topofile,topo,allow_pickle=True)
    return (nx,nx)




# hmax, H0 = 600, -3000
# mu = 14.25
# R = mu*np.abs(H0)*1/np.pi
# k = 6
# s = k*np.pi/mu/np.abs(H0)
# nR = 1+3/k
# print(R)
# path = './'
# topofile = "topographyPillbox.npy"
# print('Generating the topography ...')
# Delta = 0.021*mu*np.abs(H0)/np.pi
# nx, ny = generateSmoothPillbox(path,topofile,Delta,hmax,R,nR,s,H0)
# plotTopo(path+topofile,field='hh',savefig=True)
# plotTopo(path+topofile,field='dhh_x',savefig=False)
# plt.show()
