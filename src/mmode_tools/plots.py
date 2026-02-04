"""
Module for generating some of the characteristic mmode plots.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import AsinhNorm,Normalize,LogNorm
import cmasher as cmr

def plot_baseline_fringes(lstVec,covTensor,antPair,interferometer,figaxs=None,
                          scale='linear',title=None,xlim=None,ylim=None,
                          plotReal=False,plotImag=False,plotAmp=False):
    """
    Plots the visibility fringes for a given baseline over as a function of
    LST.

    Parameters
    ----------
    lstVec : 1D array
        Array of LST values [hours].
    covTensor : 3D array
        Covariance tensor of shape (Ntimes,Nants,Nants).
    antPair : tuple
        Tuple of antenna indices (ant1,ant2).
    interferometer : Interferometer object
        RadioArray object containing baseline information.ß
    figaxs : tuple, optional
        Tuple of (figure,axes) to plot on. If None, a new figure and axes are created.
    scale : str, optional
        Scale for the y-axis ('linear' or 'log'). Default is 'linear'.
    title : str, optional
        Title for the plot. If None, a default title is generated.
    xlim : tuple, optional
        Limits for the x-axis. If None, no limits are set.
    ylim : tuple, optional
        Limits for the y-axis. If None, no limits are set.

    Returns
    -------
    None    
    """
    if figaxs is None:
        _,axs = plt.subplots(1,figsize=(10,5))
    else:
        _,axs = figaxs

    # Getting the antenna indices.
    antInd1,antInd2 = antPair
    #
    if plotReal:
        axs.plot(lstVec,covTensor[:,antInd1,antInd2].real,
                 label='Real',color='tab:red')
    elif plotImag:
        axs.plot(lstVec,covTensor[:,antInd1,antInd2].imag,label='Imaginary',
             color='tab:blue')
    elif plotAmp:
        axs.plot(lstVec,np.abs(covTensor[:,antInd1,antInd2]),color='k',
                 zorder=1e3,label='Amplitude',linewidth=3,alpha=0.5)
    
    if not(plotReal) and not(plotImag) and not(plotAmp):
        axs.plot(lstVec,np.abs(covTensor[:,antInd1,antInd2]),color='k',
                 zorder=1e3,label='Amplitude',linewidth=3,alpha=0.5)
        axs.plot(lstVec,covTensor[:,antInd1,antInd2].real,
                 label='Real',color='tab:red')
        axs.plot(lstVec,covTensor[:,antInd1,antInd2].imag,
                 label='Imaginary',color='tab:blue')

    axs.set_xlabel('LST [hours]',fontsize=20)
    axs.set_ylabel('Amplitude',fontsize=20)
    if title is None:
        rbase = np.sqrt(interferometer.uu_m[antInd1,antInd2]**2 + \
                        interferometer.vv_m[antInd1,antInd2]**2)
        title = f'Data: u={interferometer.uu_m[antInd1,antInd2]:5.3f}'+\
                f', v={interferometer.vv_m[antInd1,antInd2]:5.3f}, ' +\
                f'r = {rbase:5.3f}, ' +\
                f'(ant1,ant2) = {antInd1,antInd2}'
    axs.set_title(title)

    [x.set_linewidth(2.) for x in axs.spines.values()]
    axs.grid()
    axs.set_yscale(scale)
    if xlim is not None:
        axs.set_xlim(xlim)
    if ylim is not None:
        axs.set_ylim(ylim)

    axs.tick_params(axis='x',labelsize=18)
    axs.tick_params(axis='y',labelsize=18)
    axs.legend(fontsize=14)


def coefficient_plot(coeffs,lmax=None,figaxs=None,cmap='viridis',norm='linear',
                     vmin=None,vmax=None,linear_width=10,plotreal=False,
                     plotimag=False,clab=None,title=None,fullPlot=True,
                     add_contours=False,colorBar=True,returnIm=False,
                     fontsize=14,**kwargs):
    """
    Generates coefficient plots, which are collquially referred to as teepee 
    plots.

    Parameters
    ----------
    coeffs : numpy array, float
        Coefficient array.
    figaxs : tuple, default=None
        Tuple object containing fig, and axs matplotlib objects. If not given,
        they are generated in the function.
    cmap : str, default='viridis'
        Colormap, use matplotlib supported colormaps.
    norm : str, default='linear'
        Colorbar normalisation, currently linear and asinh are supported.
    vmin : float, default=None
        Normalisation minimum.
    vmax : float, default=None
        Normalisation maximum.
    plotreal : bool, default=False
        If Ture plot the real values of the coefficients.
    plotimag : bool, defualt=False
        If True plot the imaginary values of the coefficients.

    Returns
    -------
    None
    """
    
    # Creating the figure and axis objects.
    if np.any(figaxs):
        fig,axs = figaxs
    else:
        if fullPlot:
            fig,axs = plt.subplots(1,figsize=(6,8))
        else:
            fig,axs = plt.subplots(1,figsize=(9,8))

    # Determining the normalisation.
    if norm == 'asinh':
        from matplotlib.colors import AsinhNorm
        norm = AsinhNorm(linear_width=linear_width,vmin=vmin,vmax=vmax)
    elif norm == 'linear':
        from matplotlib.colors import Normalize
        norm = Normalize(vmin=vmin,vmax=vmax)
    elif norm == 'log':
        from matplotlib.colors import LogNorm
        norm = LogNorm(vmin=vmin,vmax=vmax)
    
    if np.any(vmin) and np.any(vmax):
        extend='both'
    else:
        extend=None

    if coeffs.ndim == 2:
        print('Only positive coeffs, only plotting half the SH coeffs space.')
        fullPlot = False
    elif coeffs.ndim !=3 and coeffs.ndim !=2:
        raise ValueError("Coefficient array must be 2D or 3D.")

    # Combining the negative and positive mmodes together.
    if np.any(lmax):
        if fullPlot:
            coeff_Arr = np.hstack((coeffs[0,:lmax+1,:lmax+1][:,::-1],
                                coeffs[1,:lmax+1,1:lmax+1]))
        else:
            if coeffs.ndim ==3:
                coeff_Arr = coeffs[0,:lmax+1,:lmax+1]
            elif coeffs.ndim ==2:
                coeff_Arr = coeffs[:lmax+1,:lmax+1]
    else:
        if fullPlot:
            coeff_Arr = np.hstack((coeffs[0,:,:][:,::-1],coeffs[1,:,1:]))
        else:
            if coeffs.ndim ==3:
                coeff_Arr = coeffs[0,:,:]
            elif coeffs.ndim ==2:
                coeff_Arr = coeffs
        lmax = coeff_Arr.shape[0]-1

    if fullPlot:
        extent=[-lmax,lmax,lmax,0]
    else:
        extent=[0,lmax,lmax,0]

    #
    if plotreal and not(plotimag):
        if title is None:
            title = f"Real Coefficients"
        image = coeff_Arr.real
        if clab == None:
            clab = r'$\mathbb{R}[a_{l,m}]$ [arbitrary units]'
    elif plotimag and not(plotreal):
        if title is None:
            title = f"Imaginary Coefficients"
        image = coeff_Arr.imag
        if clab == None:
            clab = r'$\mathbb{I}[a_{l,m}]$ [arbitrary units]'
    else:
        if title is None:
            title = f"Absolute Coefficients"
        image = np.abs(coeff_Arr)
        if clab == None:
            clab = r'$|a_{l,m}|$ [arbitrary units]'

    # Setting the bad colours if any.
    cmap = matplotlib.cm.get_cmap(cmap)
    cmap.set_bad('lightgray',1.)

    axs.set_title(title)
    im = axs.imshow(image,cmap=cmap,norm=norm,aspect='auto',
                    extent=extent,**kwargs)
    
    if colorBar:
        cb = fig.colorbar(im,ax=axs,aspect=40,label=clab,
                            extend=extend)
    
    if add_contours:
        colorList = cmr.take_cmap_colors('cmr.neutral_r',5,return_fmt='hex')
        fmt = matplotlib.ticker.LogFormatterMathtext()
        fmt.create_dummy_axis()

        logMax = (10**int(np.log10(vmax)))
        Nlevels = 5
        levels = np.logspace(-Nlevels,-1,Nlevels)*logMax

        if fullPlot:
            Larr,Marr = np.meshgrid(np.arange(lmax+1),np.arange(-lmax,lmax+1))
        else:
            Larr,Marr = np.meshgrid(np.arange(lmax+1),np.arange(0,lmax+1))
        CS = axs.contour(Marr,Larr,coeff_Arr.T,levels=levels,
                        colors=colorList,alpha=1)
        axs.clabel(CS,fontsize=12,fmt=fmt)
        if colorBar:
            cb.add_lines(levels=levels,colors=colorList,
                        linewidths=[2,2,2,2,2])

    axs.set_xlabel(r'$m$',fontsize=fontsize)
    axs.set_ylabel(r'$\ell$',fontsize=fontsize)
    axs.set_xticklabels(axs.get_xticks().astype(int),fontsize=fontsize-2)
    axs.set_yticklabels(axs.get_yticks().astype(int),fontsize=fontsize-2)

    [x.set_linewidth(2.) for x in axs.spines.values()]
    if returnIm:
        return im


def plot_equatorial_map(skyMap,lon=None,lat=None,figsize=(16,10),norm='linear',
                        vmax=None,vmin=None,linear_width=None,
                        projection='mollweide',cmap='twilight_shifted',
                        shading='gouraud',grid=True,fontsize=20,ticks=True,
                        figaxs=None,xticks=False,title=None,transparent=False):
    """
    """
    # Get the normalisation.
    if norm == 'linear':
        norm = Normalize(vmin=vmin,vmax=vmax)
    elif norm == 'log':
        norm = LogNorm(vmin=vmin,vmax=vmax)
    elif norm == 'asinh':
        if linear_width==None:
            if vmax is None:
                vmax = np.nanmax(skyMap)
            if vmin is None:
                vmin = np.nanmin(skyMap)
            if vmax == vmin:
                vmin=0
            linear_width = np.abs(vmax-vmin)/100
            norm = AsinhNorm(linear_width=linear_width,vmin=vmin,vmax=vmax)
        else:
            norm = AsinhNorm(linear_width=linear_width,vmin=vmin,vmax=vmax)
    
    if np.any(figaxs):
        fig,axs = figaxs
    else:
        fig = plt.figure(figsize=figsize)
        axs = fig.add_subplot(111,projection=projection)

    if transparent:
        # If True set the background to be transparent.
        fig.set_facecolor('none')
        axs.set_facecolor('none')

    if np.any(lon) and np.any(lat):
        pass
    else:
        lon = np.linspace(-np.pi,np.pi,skyMap.shape[-1])
        lat = np.linspace(-np.pi/2.,np.pi/2.,skyMap.shape[0])

    im = axs.pcolormesh(lon,lat,skyMap[:,::-1],cmap=cmap,shading=shading,norm=norm)

    if ticks:
        cb = fig.colorbar(im,location='bottom',fraction=0.046, pad=0.04)
        cb.set_label('Amplitude',fontsize=fontsize)
        axs.tick_params('both',labelsize=fontsize*(1+1/6))
        cb.ax.tick_params(labelsize=fontsize)
    else:
        axs.set_yticklabels([])
    if not(xticks):
        axs.set_xticklabels([])
    else:
        axs.set_xticklabels(['10h','8h','6h','4h','2h','0h',
                             '22h','20h','18h','16h','14h'])

    if title:
        axs.set_title(title,fontsize=fontsize)

    if grid:
        axs.grid(ls='-.',alpha=0.25,color='k')
    
    [x.set_linewidth(2.) for x in axs.spines.values()]
    #plt.show()
