__author__ = "Jaiden Cook"
__credits__ = ["Jaiden Cook"]
__version__ = "1.0"
__maintainer__ = "Jaiden Cook"
__email__ = "Jaiden.Cook1@gmail.com"

import numpy as np
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyshtools import SHGrid,SHCoeffs
from mmode_tools.inversion import invert_tikh_multi_assym
from mmode_tools.inversion import calc_reg_matrix,restore_negmodes
from mmode_tools.functions import Gaussian2Dxy
from scipy.optimize import curve_fit
from scipy.linalg import cho_factor
from scipy.ndimage import generic_filter
from warnings import warn
from mmode_tools.skymap import SkyMap,calc_analytic_ps,convolve_model_map


def fit_restoring_beam(xdata_tuple,data,coord,returnPA=False):
    """
    Fit a single Gaussian to the PSF.

    Parameters:
    ----------
    xdata_Tuple : tuple
        Tuple containing the X-data and Y-data arrays.
    data : numpy
        Data array with the same shape as X-data and Y-data.
    coord : numpy array or list
        List or numpy array containing the (x,y) coordinates of the fit 
        Gaussian.
    returnPA : bool, optional
        If True, return the position angle of the fit Gaussian, by default False.
            
    Returns:
    ----------
    amp : float
        Amplitude of the fit.
    sigx : float
        x-axis sigma width.
    sigy : float
        y-axis sigma width.
    """

    xx,yy = xdata_tuple

    rms = np.std(data)
    peak = np.max(data)

    # Roughly 2-sigma condition.
    boolVec = data >= peak*0.01

    lowBound = [0,0,0,0,0,-np.pi]
    upBound = [np.inf,np.inf,np.inf,np.inf,np.inf,np.pi]
    # Perform the fitting.
    popt,_ = curve_fit(Gaussian2Dxy,(xx[boolVec]-coord[1],yy[boolVec]-coord[0]),
                       data[boolVec],p0=[1,0,0,1,1,0],bounds=(lowBound,upBound),
                       sigma=rms*np.ones(xx[boolVec].size))
    # Getting the
    amp = popt[0] 
    sigx = popt[3]
    sigy = popt[4]
    PA = popt[5]

    if returnPA:
        return amp,sigx,sigy,PA
    else:
        return amp,sigx,sigy

def calc_psf_map(pointCoord,weightsTensor,dirtyMapShape,returnCoeffs=False,
                 scale=1):
    """
    Function calculates the psf map for a given x and y pixel coordinate.

    Parameters:
    ----------
    pointCoord : tuple
        Tuple containing the x and y coordinates as array indices for the point
        source position.
    weightsTensor : np.ndarray, np.complex64
        This is the weights tensor that converts the PS to PSF coefficients.
    dirtyMapShape : tuple
        This is the x and y axis shape for the input dirty map.
    returnCoeffs : bool, default=False
        If True return the PS and PSF coefficients.

    Returns:
    ----------
    mapPSF : np.ndarray, np.float64
        PSF map
    apsf : np.ndarray, np.complex64
        PSF SH coefficients.
    aps : np.ndarray, np.complex64
        Point source SH coefficients.
    """
    from mmode_tools.inversion import restore_negmodes
    
    # Creating the point source map. We can also calculate this analytically 
    # as well using the RA and DEC position. We leave this as a TODO for next
    # time.
    xcoord,ycoord = pointCoord
    pointMap = np.zeros(dirtyMapShape)
    pointMap[int(ycoord),int(xcoord)] = 1*scale

    mapPrep = SHGrid.from_array(np.array(pointMap,dtype=np.complex64))
    # Set to zero for the next iteration.
    aps = mapPrep.expand(normalization='ortho',csphase=-1).coeffs
    # Get the coefficients.
    aps = aps[0,:,:]

    # Using a for-loop:
    apsf = np.zeros_like(aps)
    for m, rhsMatrix in enumerate(weightsTensor):
        apsf[:,m] = rhsMatrix @ aps[:,m]  # almMatrix[:, i] has shape (N,)

    apsf = restore_negmodes(apsf)

    # Expanding the map.
    psfCoeffs = SHCoeffs.from_array(apsf,normalization='ortho',csphase=-1)
    mapPSF = psfCoeffs.expand(grid='DH2',backend='ducc').data.real

    if returnCoeffs:
        aps = restore_negmodes(aps)

        return mapPSF,apsf,aps
    else:
        return mapPSF
    

def forward_model_psf(pointMap,almTensor,lMax=129,rtol=1e-16,verbosity=10,
                      damp=0.5,returnCoeffs=False):
    """
    This function takes in the input point source map, and expands it to into
    spherical harmonic coefficients, and then convolves with the instrument. The
    PSF sky coefficients are then solved for.

    Parameters:
    ----------
    pointMap : numpy array, float64
        Real 2D Cartesian map, containing a single pixel with value 1, and
        zeros elsewhere.
    almTensor : list or numpy array, complex64
        Array containing the beam fringe spherical harmonic coefficients. For 
        multi-system this is a list containing the beam fringe sh-coefficient
        tensors for each system.
    lMax : int, default=129
        lMax for the spherical harmonic expansion.
    rtol : float, default=1e-4
        Tolerance for the CGLS inversion step.
    verbosity : int, default=10
        Level of output by CGLS.
    damp : float, default=0.5
        CGLS dampening coefficient.
            
    Returns:
    ----------
    mapPSF : numpy array, float64
        Output PSF map.
    """
    mapPrep = SHGrid.from_array(np.array(pointMap,dtype=np.complex64))
    # Set to zero for the next iteration.
    mapCoef = mapPrep.expand(normalization='ortho',csphase=-1).coeffs

    # Get the coefficients.
    mapCoef = mapCoef[0,:,:]

    if isinstance(almTensor,list):
        # For multi-system CLEAN with different lmax values.
        invert = invert_tikh_multi_assym

        NbVec = np.array([alm.shape[0] for alm in almTensor]) # Nbaseline vector
        lVec = np.array([alm.shape[-1] for alm in almTensor]) # lMax vector

        NbSum = 0 # Running number of baselines sum.
        mmodeTensor = np.zeros([np.sum(NbVec),int(lMax+1)],dtype=np.complex64)
        
        for i,alm in enumerate(almTensor):
            # Getting the temp mmode tensor for system i
            tmpMmodeTensor = np.conj(np.einsum("blm,lm->bm",alm,
                                               mapCoef[:lVec[i],:lVec[i]],
                                               optimize='optimal'))
            # Assigning the mmode values to the appropriate baseline indices.
            mmodeTensor[NbSum:NbSum+NbVec[i],:lVec[i]] = tmpMmodeTensor
            # Increasing the running total baseline sum.
            NbSum += NbVec[i]
    elif isinstance(almTensor,np.ndarray):
        # For single system.
        invert = invert_tikh_multi_assym

        # Forward modelling the mmode tensor.
        mmodeTensor  = np.conj(np.einsum("blm,lm->bm",almTensor,
                                         mapCoef[:lVec[i],:lVec[i]],
                                         optimize='optimal'))

    # Solving for the sky modes.
    skyModes = invert(almTensor,np.conj(mmodeTensor),lmax=lMax,rtol=rtol,
                      verbosity=verbosity,damp=damp)
    
    # Expanding the sky modes to get the PSF map.
    psfCoeffs = SHCoeffs.from_array(skyModes,normalization='ortho',csphase=-1)
    mapPSF = psfCoeffs.expand(grid='DH2',backend='ducc').data

    if returnCoeffs:
        # Returned coefficients are useful for testing cases.
        return mapPSF,skyModes
    else:
        return mapPSF

def calc_bkg(skyCo,windowSizeDeg=6,returnWindow=False):
    """calc_bkg _summary_

    Parameters
    ----------
    skyCo : _type_
        _description_
    windowSizeDeg : int, optional
        _description_, by default 6
    returnWindow : bool, optional
        _description_, by default False

    Returns
    -------
    _type_
        _description_
    """
    lMax = skyCo.shape[-1] - 1

    if windowSizeDeg is None:
        skyCoeffs = SHCoeffs.from_array(skyCo,normalization='ortho',
                                           csphase=-1,lmax=lMax)
        skyMapOb = skyCoeffs.expand(grid='DH2',backend='ducc',lmax=lMax)
        bkgMap = np.median(skyMapOb.data.real)*np.ones_like(skyMapOb.data.real)
        windowSizePix = None

    else:
        Ncells = skyCo.shape[1]*4 + 1
        windowSizePix = int(windowSizeDeg/(360/Ncells)) + 1

        if windowSizePix % 2 == 0:
            windowSizePix += 1

        # Calculating the background estimate. Using a Gaussian to low pass 
        # filter the coefficients. In future can use a different filter.
        lsig = 2*np.pi*(1/(np.radians(360/Ncells)*windowSizePix))
        lVec = np.arange(lMax+1)
        skyCoLPF = np.copy(skyCo)
        skyCoLPF[:,:lMax+1,:lMax+1] *= np.exp(-0.5*(lVec/lsig)**2)[None,:,None]
        skyCoeffsLPF = SHCoeffs.from_array(skyCoLPF,normalization='ortho',
                                           csphase=-1,lmax=lMax)
        skyMapObjLPF = skyCoeffsLPF.expand(grid='DH2',backend='ducc',lmax=lMax)
        bkgMap = skyMapObjLPF.data.real

    if returnWindow:
        return bkgMap,windowSizePix
    else:
        return bkgMap

def make_thresh_maps(dirtyMap,skyCo,windowSizeDeg=6):
    """
    Makes the threshold map. Smooths the dirty image, subtracts the smoothed 
    image from the original dirty image to remove background estimate. 
    Calculates the local standard devation, and divides the residual map by the 
    std map to get the threshold map in units of sigma.

    Parameters:
    ----------
    dirtyMap : np.float64 np.ndarray
        Dirty map, as input 2D numpy array.
    relWindowSize : float, default=0.04
        Relative window size for determining the threshold window. This is a 
        scale invariant method. This is multiplied by the Naxis1 of the dirty
        image.

    Returns:
    ----------
    bkgMap : np.float64 np.ndarray
        Smoothed background map of the dirty image.
    stdMap : np.float64 np.ndarray
        Standard deviation map of the dirty image.
    threshMap : np.float64, np.ndarray
        Threshold map in units of sigma, used for peak detection.
    """
    bkgMap,windowSizePix = calc_bkg(skyCo,windowSizeDeg=windowSizeDeg,
                                    returnWindow=True)
    
    if windowSizePix is None:
        Ncells = skyCo.shape[1]*4 + 1
        windowSizePix = int(6/(360/Ncells)) + 1

    print(f"Window size = {windowSizePix}")

    diffMap = (dirtyMap.real-bkgMap)
    # Cal factor corrects for underestimate of the variance from using the 
    # median. Here we are using the MAD (Median Absolute Deviation) to 
    # estimate the local standard deviation.
    calFactor = 1.4826
    stdMap =  calFactor*generic_filter(np.abs(diffMap),np.median,
                                    size=windowSizePix)
    
    threshMap = diffMap/stdMap

    return bkgMap,stdMap,threshMap

def find_good_peaks(threshMap,thresh=4,cleanMask=None,
                    threshold_rel=None,threshold_abs=3):
    """
    Finds peaks for CLEAN using the threshold map.

    Parameters:
    ----------
    threshMap : np.float64, np.ndarray
        Threshold map in units of sigma, used for peak detection.
    DECgrid : np.float64 np.ndarray
        2D DEC grid.
    thresh : float, default=4
        Significance threshold in sigma for peak detection.
    cleanMask : bool np.ndarray
        Boolean numpy array with the same shape as the dirty image. Used to find
        peaks within only the mask region.
    threshold_rel : float or None, default=None
        Minimum intensity of peaks, calculated as max(image) * threshold_rel.
    threshold_abs : float or None, default=4
        Minimum intensity of peaks. By default, the absolute threshold is the 
        minimum intensity of the image. Threshold maps are in units of sigma,
        the default value is 4 sigma here.

    
    Returns:
    ----------
    coords : np.float64 np.ndarray
        2D numpy array containing peak xy-coordinates.
    """

    from skimage.feature import peak_local_max
    # If not None apply a CLEAN mask, only find point within the masked region.
    if cleanMask is not None:
        # Check the shapes are the same, required to filter coords outside the 
        # clean mask.
        if cleanMask.shape != threshMap.shape:
            errMsg = f"Clean mask shape {cleanMask.shape} not equal " +\
                     f"to map shape {threshMap.shape}"
            raise ValueError(errMsg)
        
        threshMap[cleanMask==False] = 0

    # Performing the peak detection on the masked threshold map.
    coords = peak_local_max(threshMap,threshold_rel=threshold_rel,
                            threshold_abs=threshold_abs)
    threshVec = threshMap[coords[:,0],coords[:,1]]
    coords = coords[threshVec>=thresh,:]

    #yGrid,xGrid = np.mgrid[0:threshMap.shape[0],0:threshMap.shape[1]]
    #coords = np.array([yGrid[threshMap>=thresh],xGrid[threshMap>=thresh]]).T


    return coords

def plot_dirty_image(dirtyMap,coords=None,figaxs=None,cmap='twilight_shifted',
                     vmin=None,vmax=None,linear_width=None,norm='linear',
                     title=None,patchList=None,**kwargs):
    """
    Plot the dirty image with peaks.

    Parameters:
    ----------
    dirtyMap : np.float64 np.ndarray
        Dirty map, as input 2D numpy array.
    coords : np.float64 np.ndarray
        2D numpy array containing peak xy-coordinates.
    cmap : str, default='twilight_shifted'
        Colormap style.
    vmin : float, default=None
        Min value of the colorbar scale.
    vmax : float, default=None
        Max value of the colorbar scale.
    linear_width : float, default=None
        Used for norm='asinh', defines the linear region of the scale.
    norm : str, default='linear'
        Colorbar normalisation method, options are 'linear','log' and 'asinh'.
    title : str, default=None
        Plot title.
    patchList : list, default=None
        List of patch regions, these are by default square patches.

    Returns:
    ----------
    """
    from matplotlib.colors import AsinhNorm
    from matplotlib.colors import Normalize,LogNorm
    import matplotlib.patches as patches

    if figaxs == None:
        fig,axs = plt.subplots(1,figsize=(11,5))
    else:
        fig,axs = figaxs

    # Get the normalisation.
    if norm == 'linear':
        norm = Normalize(vmin=vmin,vmax=vmax)
    elif norm == 'log':
        norm = LogNorm(vmin=vmin,vmax=vmax)
    elif norm == 'asinh':
        if linear_width==None:
            if vmax is None:
                vmax = np.nanmax(dirtyMap)
            if vmin is None:
                vmin = np.nanmin(dirtyMap)

            if vmax == vmin:
                vmin=0
            linear_width = np.abs(vmax-vmin)/100
            norm = AsinhNorm(linear_width=linear_width,vmin=vmin,vmax=vmax)
        else:
            norm = AsinhNorm(linear_width=linear_width,vmin=vmin,vmax=vmax)

    im = axs.imshow(dirtyMap[:,::-1],norm=norm,
                    cmap=cmap,aspect='auto',origin='lower',**kwargs)
    
    if np.any(coords):
        axs.scatter(dirtyMap.shape[1]-coords[:,1]-1,coords[:,0],
                    marker='x',color='c',s=10)
    cb = fig.colorbar(im)
    if title:
        axs.set_title(title)
    
    # If there are any flagged regions.
    if patchList:
        Naxis1 = dirtyMap.shape[1]
        for patch in patchList:
            if len(patch) < 4:
                winx,winy = patch[2],patch[2]
            elif len(patch) == 4:
                winx,winy = patch[2],patch[3]
            square = patches.Rectangle((Naxis1-patch[1]-winx-1,patch[0]),
                                       winx,winy,edgecolor='k',facecolor='none')
            axs.add_patch(square)

def make_resid_map(model,almTensor,mmodeTensor,weights=None,lMax=130,lMaxVec=None,
                   mMax=None,verbosity=1,damp=0.5,returnCoeffs=False,plotCond=False,
                   vmin=-1e6,vmax=1e6,linear_width=1e5):
    """
    Takes input model image, beam fringe coefficients, and mmode visibility data
    tensor. Performs a subtraction of the model from the data, and solves for 
    the residual image spherical harmonic coefficients and outputs the residual 
    image.

    Parameters:
    ----------
    model : np.float64 np.ndarray
        Real 2D map or 3D array of complex SH coefficients.
    almTensor : list or numpy array, complex64
        Array containing the beam fringe spherical harmonic coefficients. For 
        multi-system this is a list containing the beam fringe sh-coefficient
        tensors for each system.
    mmodeTensor : np.complex64 np.ndarray
        Mmode visibility data tensor.
    lMax : int, default=130
        Max l-mode.
    verbosity : int, default=1
        CGLS output order, goes from 1-10, 1 being less output, 10 being more.
    damp : float, default=0.5
        CGLS dampening coefficient.
    plotCond : bool, default=False
        Plot condition, if True output plots.

    Returns:
    ----------
    residDirtyMap : np.complex64 np.ndarray
        Output residual dirty map.
    """
    if model.ndim == 2:
        # If the input dimensions is 2 then the input model is an image/map.
        mapPrep = SHGrid.from_array(np.array(model,dtype=np.complex64))
        # Set to zero for the next iteration.
        modelCo = mapPrep.expand(normalization='ortho',csphase=-1).coeffs
        modelCo = modelCo[0,:,:]
    elif model.ndim == 3:
        # If number of dimensions is 3, then the input is the model coefficients.
        modelCo = model[0,:,:]

    invert = invert_tikh_multi_assym
    if isinstance(almTensor,list):
        # For multi-system CLEAN with different lmax values.
        # The number of baselines per system.
        NbVec = np.array([alm.shape[0] for alm in almTensor]) # Nbaseline vector

        # Getting the lMax vector if one is not provided. 
        if lMaxVec is None:
            lMaxVec = np.array([alm.shape[-1] for alm in almTensor])
            # Setting the maximum lMax to the provided one
            lMaxVec[lMaxVec > lMax] = lMax

        #
        NbSum = 0 # Running number of baselines sum.
        modelMmodeTensor = np.zeros([np.sum(NbVec),int(lMax+1)],
                                    dtype=np.complex64)
        
        for i,alm in enumerate(almTensor):
            lMaxTmp = lMaxVec[i]
            # Getting the temp mmode tensor for system i
            tmpMmodeTensor = np.conj(np.einsum("blm,lm->bm",
                                               alm[:,:lMaxTmp+1,:lMaxTmp+1],
                                               modelCo[:lMaxTmp+1,:lMaxTmp+1],
                                               optimize='optimal'))
            # Assigning the mmode values to the appropriate baseline indices.
            modelMmodeTensor[NbSum:NbSum+NbVec[i],:lMaxTmp+1] = tmpMmodeTensor
            # Increasing the running total baseline sum.
            NbSum += NbVec[i]

    elif isinstance(almTensor,np.ndarray):
        # For single system.
        # Forward modelling the mmode tensor.
        modelMmodeTensor  = np.conj(np.einsum("blm,lm->bm",
                                              almTensor[:,:lMax+1,:lMax+1],
                                              modelCo[:,:lMax+1],
                                              optimize='optimal'))

    # Calculating the residual mmode tensor.
    if mmodeTensor.ndim == 3:
        # Only need positive m-modes.
        residMmodeTensor = mmodeTensor[:,0,:lMax+1] - modelMmodeTensor
    elif mmodeTensor.ndim == 2:
        # Only need positive m-modes.
        residMmodeTensor = mmodeTensor[:,:lMax+1] - modelMmodeTensor

    # Solving for the sky modes.
    skyModes = invert(almTensor,np.conj(residMmodeTensor),lmax=lMax,
                      weights=weights,mMax=mMax,lMaxVec=lMaxVec,
                      verbosity=verbosity,damp=damp)

    sphericalCoeffs = SHCoeffs.from_array(skyModes,normalization='ortho',
                                          csphase=-1)
    griddedCoeffs = sphericalCoeffs.expand(grid='DH2',
                                           backend='ducc',lmax_calc=lMax)
    residDirtyMap = griddedCoeffs.data.real

    if plotCond:
        if model.ndim == 2:
            plot_dirty_image(model,linear_width=linear_width,
                            norm='asinh',title='Model Image')
        plot_dirty_image(residDirtyMap.real,norm='asinh',vmax=vmax,vmin=vmin,
                        linear_width=linear_width,title='Residual Dirty Image')

    if returnCoeffs:
        # Returned coefficients are useful for testing cases.
        return residDirtyMap,skyModes
    else:
        return residDirtyMap
    
def minor_iter(dirtySkyMap,modelMap,peakInterp,psfWeightsTensor,
               loopGain=0.1,thresh=7,windowSizeDeg=6,lMax=None,
               verbosity=0,peaks_kwargs=None,findNegPeaks=False,
               bkgMap=True):
    # Coefficients can change between iterations.
    #dirtySkyMap.expand_coeffs()
    dirtySkyMap.calc_thresh_map(windowSizeDeg=windowSizeDeg,lMax=lMax,
                                bkgMap=bkgMap)

    if peaks_kwargs is not None:
        peakCoords = dirtySkyMap.find_peaks(thresh=thresh,
                                            findNegPeaks=findNegPeaks,
                                            **peaks_kwargs)
    else:
        peakCoords = dirtySkyMap.find_peaks(thresh=thresh,
                                            findNegPeaks=findNegPeaks)
    
    if verbosity > 0:
        print(f"Number of peaks found = {peakCoords.shape[0]}")

    if not(np.any(peakCoords)):
        print("No peaks found...")
        return False

    # Creating a temporary model map.
    tmpModelMap = np.zeros_like(dirtySkyMap.skyMap)

    #
    if dirtySkyMap.stdMap is not None:
        stdRef = np.nanmedian(dirtySkyMap.stdMap)
        stdVec = dirtySkyMap.stdMap[peakCoords[:,0],peakCoords[:,1]]
        loopGain = np.abs(loopGain*stdRef/stdVec)
        # Set a minimum to the loop gain.
        loopGain[loopGain < 0.005] = 0.005

    # Calculating the amplitude for the model components.
    peakVec = dirtySkyMap.skyMap[peakCoords[:,0],peakCoords[:,1]]
    peakLatVec = dirtySkyMap.decVec[peakCoords[:,0]]
    ampVec = peakVec*loopGain/peakInterp(peakLatVec)
    dOmega = 4*np.pi/dirtySkyMap.skyMap.size
    ampVec = ampVec/dOmega/np.cos(np.radians(peakLatVec))

    if verbosity > 0:
        print(f"Max amplitude = {ampVec.max()}")
        print(f"Min amplitude = {ampVec.min()}")

    # Creating a temporary model map.
    tmpModelMap[peakCoords[:,0],peakCoords[:,1]] = ampVec

    # Convoling the model map.
    dirtyModelSkyMap = convolve_model_map(tmpModelMap,psfWeightsTensor)

    # Adding to the model map.
    modelMap[:,:] = modelMap[:,:] + tmpModelMap
    # Calculating the residual map.
    dirtySkyMap.coeffs = dirtySkyMap.coeffs - dirtyModelSkyMap.coeffs

    return True


def deep_minor_iter(dirtySkyMap,modelMap,peakInterp,psfWeightsTensor,
                    peakCoords,loopGain=0.1,thresh=7,
                    windowSizeDeg=6,lMax=None,verbosity=0,findNegPeaks=False,
                    bkgMap=True):
    """deep_clean Minor iteration down to a lower threshold. Only CLEANs the 
    model components.

    Parameters
    ----------
    dirtySkyMap : _type_
        _description_
    modelMap : _type_
        _description_
    peakInterp : _type_
        _description_
    psfWeightsTensor : _type_
        _description_
    loopGain : float, optional
        _description_, by default 0.1
    thresh : int, optional
        _description_, by default 7

    Returns
    -------
    _type_
        _description_
    """
    # Coefficients can change between iterations.
    #dirtySkyMap.expand_coeffs()
    dirtySkyMap.calc_thresh_map(windowSizeDeg=windowSizeDeg,lMax=lMax,
                                bkgMap=bkgMap)

    xVec,yVec = peakCoords

    # Only deep clean peaks in the mask.
    mask = dirtySkyMap.cleanMask[yVec,xVec]
    xVec = xVec[mask]
    yVec = yVec[mask]

    #
    if findNegPeaks:
        SNRvec = np.abs(dirtySkyMap.threshMap[yVec,xVec])
    else:
        SNRvec = dirtySkyMap.threshMap[yVec,xVec]

    if verbosity > 0:
        print("=========")
        print(SNRvec.mean())
        print(SNRvec.max())
        print(SNRvec.min())
    
    xVec = xVec[SNRvec>=thresh]
    yVec = yVec[SNRvec>=thresh]

    if not(np.any(xVec)):
        print("No model sources above threshold...")
        return False
    
    if verbosity > 0:
        print(f"Number of sources to CLEAN = {xVec.size}")

    try:
        # Need a lat grid, don't want to make it every function call.
        peakLatVec = dirtySkyMap.latGrid[yVec,xVec]
    except AttributeError:
        # If there is no lat grid, then make one, should persist.
        _,latGrid = np.meshgrid(dirtySkyMap.raVec,dirtySkyMap.decVec)
        dirtySkyMap.latGrid = latGrid

    # Creating a temporary model map.
    tmpModelMap = np.zeros_like(dirtySkyMap.skyMap)

    if dirtySkyMap.stdMap is not None:
        stdRef = np.nanmedian(dirtySkyMap.stdMap)
        stdVec = dirtySkyMap.stdMap[yVec,xVec]
        loopGain = np.abs(loopGain*stdRef/stdVec)
        # Set a minimum to the loop gain.
        loopGain[loopGain < 0.005] = 0.005

    #
    peakVec = dirtySkyMap.skyMap[yVec,xVec]
    ampVec = peakVec*loopGain/peakInterp(peakLatVec)
    dOmega = 4*np.pi/dirtySkyMap.skyMap.size
    ampVec = ampVec/dOmega/np.cos(np.radians(peakLatVec))

    # Assigning the amplitudes.
    tmpModelMap[yVec,xVec] = ampVec

    # Convoling the model map.
    dirtyModelSkyMap = convolve_model_map(tmpModelMap,psfWeightsTensor)

    # Adding to the model map.
    modelMap[:,:] = modelMap[:,:] + tmpModelMap
    # Calculating the residual map.
    dirtySkyMap.coeffs = dirtySkyMap.coeffs - dirtyModelSkyMap.coeffs

    SNRvec = dirtySkyMap.threshMap[yVec,xVec]

    if (SNRvec>=thresh).sum() == 0:
        print("No more model sources above threshold...")
        return False
    
    return True
    

def major_iter(skyCoeffs,modelMap,peakInterp,psfWeightsTensor,
               almTensor,mmodeTensor,damp,weights=None,lMax=None,
               lMaxVec=None,verbosity=0,cleanMask=None,maskList=None,
               DECthresh=(90,-90),Nminor=1000,loopGain=0.1,thresh=7,sigThresh=2,
               windowSizeDeg=6,plotCond=False,peaks_kwargs=None,
               findNegPeaks=False,deepClean=True,minorLoop=True,bkgMap=True,
               stdMapMode='MAD'):
    
    dirtySkyMap = SkyMap(coeffs=skyCoeffs)
    if verbosity > 0:
        print("Calculating the median deviation map...")
    dirtySkyMap.calc_background_map(windowSizeDeg=windowSizeDeg)
    dirtySkyMap.calc_std_map(windowSizeDeg=windowSizeDeg,mode=stdMapMode)

    if verbosity > 0:
        print("Calculating the clean mask...")
    # 
    dirtySkyMap.calc_mask(initialMask=cleanMask,DECthresh=DECthresh,
                          maskList=maskList,plotCond=plotCond)

    minorLoopCounter = 0
    if minorLoop:
        # There may be conditions where there are no more minor loops needed.
        if verbosity > 0:
            print(f"CLEANing down to thresh {thresh}...")
        # Find peaks to CLEAN to a shallow PSF in the dirty Image.
        for i in tqdm(range(Nminor)):
            if i%200 == 0: 
                printCond = 1
            else:
                printCond = 0
            #
            loopCond = minor_iter(dirtySkyMap,modelMap,peakInterp,
                                  psfWeightsTensor,loopGain=loopGain,
                                  thresh=thresh,verbosity=printCond,
                                  windowSizeDeg=windowSizeDeg,lMax=lMax,
                                  findNegPeaks=findNegPeaks,
                                  peaks_kwargs=peaks_kwargs,
                                  bkgMap=bkgMap)

            if not(loopCond):
                break
            else:
                minorLoopCounter += 1

    # 
    _,latGrid = np.meshgrid(dirtySkyMap.raVec,dirtySkyMap.decVec)
    yGrid,xGrid = np.mgrid[:dirtySkyMap.decVec.size,:dirtySkyMap.raVec.size]
    xVec = xGrid[modelMap!=0]
    yVec = yGrid[modelMap!=0]
    modelCoords = np.vstack((xVec,yVec))
    dirtySkyMap.latGrid = latGrid
    
    if plotCond:
        dirtySkyMap.plot_cart_map(norm='asinh',vmin=-1e6,vmax=1e6,
                                  linear_width=1e4)
        dirtySkyMap.plot_cart_map(img=modelMap,norm='asinh',vmin=-1e6,vmax=1e6,
                                linear_width=1e4)
        

    # Perform a deep clean using only the model parameters.
    deepCleanCounter = 0
    if deepClean:
        # You may not want to perform a deep CLEAN, and just find peaks.
        if verbosity > 0:
            print(f"CLEANing model down to thresh {sigThresh}...")
            print(f"Number of sources to CLEAN = {xVec.size}")

        for i in tqdm(range(Nminor)):
            if i % 200 == 0:
                printCond = 1
            else:
                printCond = 0
            #
            loopCond = deep_minor_iter(dirtySkyMap,modelMap,peakInterp,
                                    psfWeightsTensor,modelCoords,
                                    loopGain=loopGain,thresh=sigThresh,
                                    verbosity=printCond,
                                    windowSizeDeg=windowSizeDeg,lMax=lMax,
                                    findNegPeaks=findNegPeaks,
                                    bkgMap=bkgMap)

            if not(loopCond):
                break
            else:
                deepCleanCounter += 1

    if plotCond:
        #dirtySkyMap.plot_cart_map(coords=np.array([yVec,xVec]).T,norm='asinh',
        #                          vmin=-1e6,vmax=1e6,linear_width=1e4)
        dirtySkyMap.plot_cart_map(norm='asinh',vmin=-1e6,vmax=1e6,
                                  linear_width=1e4)

    #
    SNRmap = np.zeros_like(modelMap)
    SNRmap[modelMap!=0] = dirtySkyMap.threshMap[modelMap!=0]
    #print("SNR_VECTOR")
    #print("-------------------------")
    #print(dirtySkyMap.threshMap[modelMap!=0])
    # Setting model coefficients that have signal to noise maps less than 0 to 
    # zero.
    print(f"Number of model sources = {modelMap[modelMap!=0].size}")
    #modelMap[SNRmap < 0] = 0
    #print(f"Number of model sources = {modelMap[modelMap!=0].size}")

    # Making the skyMap object
    modelSkyMap = SkyMap(skyMap=modelMap)
    # Calculating the residual sky modes through a major iteration.

    if deepCleanCounter + minorLoopCounter > 0:
        if verbosity > 0:
            print(f"Calculating the residual sky modes...")
        _,skyModes = make_resid_map(modelSkyMap.coeffs,almTensor,mmodeTensor,
                                    lMax=lMax,lMaxVec=lMaxVec,weights=weights,
                                    damp=damp,returnCoeffs=True,
                                    verbosity=verbosity)
    else:
        if verbosity > 0:
            print(f"No CLEANing performed, returning the original sky modes...")
        skyModes = skyCoeffs
    
    return skyModes
    

def make_restored_map(residDirtyMap,modelMap,paramsArr,xygrid,
                      returnConvMap=False):
    """
    Creates the restored CLEAN image for cartesian map projection.
    
    Parameters:
    ----------
    residDirtyMap : np.complex64 np.ndarray
        Output residual dirty map.
    modelMap : np.float64 np.ndarray
        Real 2D map containing the model points.
    paramsArr : np.float64 np.ndarray
        2D array containing the fitted PSF Gaussian parameters. Needed to make
        the final resotred map.
    xygrid : tuple,np.float64 np.ndarray
        Tuple containing the 2D xy-grid numpy arrays.
    returnConvMap : bool, default=False
        If True return the convolved model map.
    
    Returns:
    ----------
    restoredMap : float, np.ndarray
        Restored map.
    modelConvMap : float np.ndarray, optional
        Convolved model map.
    """

    if residDirtyMap.shape != modelMap.shape:
        # Make sure these are the same shape, if not raise error.
        errMsg = f"Clean mask shape {residDirtyMap.shape} not equal " +\
                     f"to map shape {modelMap.shape}"
        raise ValueError(errMsg)
    # Creating the DEC grid. Has to go from 90 to -90
    Npoint = residDirtyMap.shape[0]
    DECVec = np.linspace(90,-90,Npoint)

    # Getting the xy-grid arrays.
    xx,yy= xygrid
    # Getting the boolean mask for all model point sources.
    modelInds = modelMap > 0

    # Determining the Gaussian restoring beam size. Should be the min fit 
    # Gaussian.
    sigMin = np.min(paramsArr[:,3:5][paramsArr[:,3:5]>=1])
    xcoords = xx[modelInds]
    ycoords = yy[modelInds]
    sigxVec = np.ones(ycoords.size)*sigMin/np.cos(np.radians(DECVec[ycoords]))
    sigyVec = np.ones(ycoords.size)*sigMin
    ampVec = modelMap[ycoords,xcoords]/(2*np.pi*sigxVec*sigyVec)

    # Creating the source parameter array. Used to create 2D Gaussian maps.
    srcParams = np.array([ampVec,xcoords,ycoords,sigxVec,sigyVec]).T

    # Iterate through and add Gaussians to image.
    modelConvMap = np.zeros(residDirtyMap.shape)
    for params in srcParams:
        modelConvMap += Gaussian2Dxy((xx,yy),*params)

    restoredMap = residDirtyMap.real + modelConvMap

    if returnConvMap:
        return restoredMap,modelConvMap
    else:
        return restoredMap


def make_mask_box(maskParams,RAgrid,DECgrid):
    """
    Function for making a masking box.

    Parameters:
    ----------
    RA : np.ndarray
        Vector of RA values for each of the masks.
    DEC : np.ndarray
        Vector of DEC values for each of the masks.
    size : np.ndarray
        Vector of mask sizes in degrees.
    RAgrid : np.ndarray
        RA grid, for determining the pixel location.
    DECgrid : np.ndarray
        DEC grid, for determining the pixel location.

    Returns:
    ----------
    maskList : list
        List of mask parameters, each entry is a tuple of size 3, with elements
        being ycorner, xcorner indices, and the size of the square mask in pix.

    """
    _,RA,DEC,size = zip(*maskParams)

    RA = np.array(RA)
    DEC = np.array(DEC)
    size = np.array(size)

    # Expected pixel size in degrees.
    dtheta = 360/RAgrid.shape[1]
    # Converting angular size from degrees to pixel size.
    sizePix = (size/dtheta).astype(int)

    # Calculating the RA and DEC corner values.
    RAcorner = RA - size/2
    DECcorner = DEC - size/2

    # Adjusting values in if any fall outside of the grid.
    DECcorner[DECcorner < -90] = -90
    RAcorner[RAcorner < 0] = 180 + RAcorner[RAcorner < 0] # Wraps.

    # Getting the ravel indices for each of the grid points.
    indexVec = np.zeros(RA.shape,dtype=int)
    for ind,ra in enumerate(RAcorner):
        dec = DECcorner[ind]
        indexVec[ind] = np.argmin(np.sqrt((RAgrid-ra)**2 + (DECgrid-dec)**2))
    
    # Getting the y and x ind corner unravelled indices
    yCorner,xCorner = np.unravel_index(indexVec,RAgrid.shape)

    # Zipping the values together in a list. This list can be read by the CLEAN
    # functions.
    #maskList = list(zip(yCorner,xCorner,sizePix))
    maskList = list(zip(yCorner.tolist(),xCorner.tolist(),sizePix.tolist()))

    return maskList


def calc_clean_mask(skyCo,initalMask=None,DECthresh=(90,-90),maskList=None,
                    GPthresh=0,GPthreshFlip=False,plotCond=False,maskFlip=False):
    """
    Function calculates a clean mask from input threshold parameters.

    Parameters:
    ----------
    skyCo : np.ndarray, np.complex64
        Array of SH-coefficients for the dirty map. Used to construct the grid.
    initalMask : np.ndarray, bool, default=None
        Initial clean mask grid, if given this is added to the clean mask list.
    DECthresh : tuple, default=(41,-80)
        Tuple containing the min and maximum declinations. Values above and 
        below are masked.
    maskList : list, default=None
        List of box mask regions, if not none these are calculated and added to 
        the clean mask list.
    GPthresh : float, default=0
        Galactic Plane latitude threshold. If > 0 then all values below this
        cutoff are masked. Great for masking the GP, when only wanting to CLEAN
        extra-galactic sources.
    plotCond : bool, default=False
        If True plot the mask. Use this when you want to ensure that outputs 
        makse sense.
    maskFlip : bool, default=False
        If True flip the False and True values in the mask.

    Returns:
    ----------
    cleanMask : bool, np.ndarray
        2D grid of boolean values, used to mask the dirty image when performing
        peak detection.
    """
    from astropy.coordinates import SkyCoord
    from astropy import units as u

    coeffsObj = SHCoeffs.from_array(skyCo,normalization='ortho',csphase=-1)
    coeffsObjExp = coeffsObj.expand(grid='DH2',backend='ducc')
    RAVec = coeffsObjExp.lons()
    DECVec = coeffsObjExp.lats()
    RAVecNew = np.roll(np.copy(RAVec),int(RAVec.size/2))
    RAgrid,DECgrid = np.meshgrid(RAVecNew,DECVec[::-1])

    # Making the declination mask.
    #decMask = (DECgrid >= DECthresh[0]) | (DECgrid < DECthresh[1])
    decMask = (DECgrid >= DECthresh[0]) | (DECgrid < DECthresh[1]) == False
    
    # Initialising the CLEAN
    cleanMaskList = [decMask]

    # If mask list is not None then create mask grid and add to cleanmask list.
    if maskList is not None:
        maskGrid = np.ones_like(RAgrid)
        for mask in maskList:
            yInd = mask[0]
            xInd = mask[1]
            size = mask[2]
            maskGrid[yInd:yInd+size,xInd:xInd+size] = 0
        # Converting to True False map.
        maskGrid = maskGrid.astype(bool)
        cleanMaskList.append(maskGrid)

    # Creating a mask for the coords in the GP.
    if GPthresh > 0:
        cArr = SkyCoord(RAgrid*u.deg,DECgrid*u.deg)
        GPlatGrid = cArr.galactic.b.value
        GPmask = np.abs(GPlatGrid) >= GPthresh
        if GPthreshFlip:
            # If True Flag all latitudes outside the GP lat cuttoff.
            GPmask = GPmask == False
        cleanMaskList.append(GPmask)

    # If initial mask is provided we can add this to the mask list.
    if initalMask is not None:
        if initalMask.shape == RAgrid.shape:
            # Check that the shape of the initial mask is the same as the
            # grid.
            errMsg = f"initalMask.shape {initalMask.shape} != RAgrid.shape +"
            f" {RAgrid.shape}"
            raise ValueError(errMsg)
        else:
            cleanMaskList.append(initalMask)

    # Making the final clean mask by multiplying all masks together.
    cleanMask = np.copy(cleanMaskList[0])
    for ind,maskGrid in enumerate(cleanMaskList):
        if ind > 0:
            cleanMask *= maskGrid

    if maskFlip:
        cleanMask = cleanMask == False

    # If True plot the clean mask for visual inspection.
    if plotCond:
        plot_dirty_image(cleanMask,cmap='grey',norm='linear',title='CLEAN mask')

    return cleanMask

def calc_psf_weights_matrix(m,B,almTensorList,lMaxVec,lMax=None,damp=0.01,
                            weights=None):
    """
    Calculates the psf weights given by equation:
    
    W_m,psf = (B_m^* @ B_m + Q_m^* @ Q_m)^-1 B_m^* @ B_m

    which is used to calculate the psf coefficients from point source 
    coefficients by the equation:

    a_m,psf = W_m,psf @ a_mps

    This function calculates this for a single m-mode. 

    Parameters:
    ----------
    m : int
        m-mode to calculate the weight matrix.
    B : np.complex64, np.ndarray
        Array to assign the beam fringe values.
    almTensorList : list
        List of beam fringe coefficients, each item is for a different 
        instrument.
    damp : float, or np.ndarray, default=0.01
        The damping coefficient or regularisation parameters used in the image
        inversion.
    weights : float, np.ndarray, default=None
        If given these are assumed to be the noise weights for the inversion.

    Returns:
    ----------
    Wm : np.complex64, np.ndarray
        Output weights matrix that converts the point source coefficients to the
        psf coefficients.

    """
    if lMax is None:
        # If no lmax is given then calculate from the almTensor list.
        lMax = lMaxVec.max()

    NbaseSum = 0
    Bm = B[:,m:]
    for i,almTensor in enumerate(almTensorList):
        Nbase = int(almTensor.shape[0])
        # Assigning the beam fringes from each array to a total tensor.
        if m <= lMaxVec[i]:
            if lMaxVec[i] < lMax:
                Bm[NbaseSum:NbaseSum+Nbase,:(lMaxVec[i]+1)-m] = \
                    almTensor[:,m:(lMaxVec[i]+1),m]
            else:
                Bm[NbaseSum:NbaseSum+Nbase,:(lMaxVec[i]+1)] = \
                    almTensor[:,m:(lMaxVec[i]+1),m]
        # Increment the baseline number sum.
        NbaseSum += Nbase

    #
    if np.any(weights):
        if weights.size != Bm.shape[0]:
            raise ValueError(f'Weights shape {weights.size} should match B' +\
                             f' axis 0 size {Bm.shape[0]}')
        else:
            Bm = np.matrix(Bm*weights[:,None])
    else:
        Bm = np.matrix(Bm)

    # Regularisation matrix.
    R = calc_reg_matrix(damp,m,lMax)
    Lam = Bm.H @ Bm
    Lam_prime = Lam + R

    # Decomposing the Lambda prime matrix into upper triangular matrix.
    U,_ = cho_factor(Lam_prime,lower=False)
    U = np.matrix(np.triu(U))

    # Calculating the Weights matrix.
    Wm =  U.I @ U.H.I @ Bm.H @ Bm

    return Wm


def calc_psf_weights_tensor(almTensorList,damp=0.01,weights=None,lMax=None,
                            lMaxVec=None):
    """
    Calculates the psf weights for each m-mode given by equation:
    
    W_m,psf = (B_m^* @ B_m + Q_m^* @ Q_m)^-1 B_m^* @ B_m

    which is used to calculate the psf coefficients from point source 
    coefficients by the equation:

    a_m,psf = W_m,psf @ a_mps

    Returns the W_m,psf tensor, where the first axix is the m-mode, and the 
    second and third have dimension lmax + 1.


    Parameters:
    ----------
    B : float, np.complex64
        Array to assign the beam fringe values.
    almTensorList : list
        List of beam fringe coefficients, each item is for a different 
        instrument.
    damp : float, or np.ndarray, default=0.01
        The damping coefficient or regularisation parameters used in the image
        inversion.
    weights : float, np.ndarray, default=None
        If given these are assumed to be the noise weights for the inversion.

    Returns:
    ----------
    WmTensor
    """

    if lMaxVec is not None:
        if len(lMaxVec) != len(almTensorList):
            raise ValueError('len(lMaxVec) != len(almTensorList)')
    else:
        lMaxVec = np.array([int(almTensor.shape[-1])-1 \
                            for almTensor in almTensorList])
    
    if lMax is None:
        # If no lmax is given then calculate from the almTensor list.
        lMax = lMaxVec.max()
    else:
        if lMax > lMaxVec.max():
            warningMsg = f"lmax {lMax} > {np.max(lMaxVec)}, must be strictly smaller"+\
                  f" or equal. Setting lmax to {np.max(lMaxVec)}"
            warn(warningMsg)
            lMax = lMaxVec.max()

    #
    WmTensor = np.zeros((lMax+1,lMax+1,lMax+1),dtype=np.complex64)
    NbaseTot = np.array([alm.shape[0] for alm in almTensorList]).sum()

    #
    B = np.zeros((NbaseTot,lMax+1),dtype=np.complex64)
    for mmode in tqdm(range(lMax+1)):
        Wm=calc_psf_weights_matrix(mmode,B,almTensorList,lMaxVec,lMax=lMax,
                                   damp=damp,weights=weights)
        #   
        WmTensor[mmode,mmode:,mmode:] = Wm

    
    return WmTensor