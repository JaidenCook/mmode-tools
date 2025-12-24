import numpy as np
import tqdm
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyshtools import SHGrid,SHCoeffs
from mmode_tools.inversion import invert_tikh_multi_assym
from mmode_tools.inversion import invert_CGLS_multi_pylops_assym
from mmode_tools.inversion import calc_reg_matrix,restore_negmodes
from mmode_tools.functions import Gaussian2Dxy
from scipy.optimize import curve_fit
from scipy.linalg import cho_factor
from scipy.ndimage import generic_filter
from warnings import warn


def fit_restoring_beam(xdata_tuple,data,coord):
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

    lowBound = [0,0,0,0,0]
    upBound = [np.inf,np.inf,np.inf,np.inf,np.inf]
    # Perform the fitting.
    popt,_ = curve_fit(Gaussian2Dxy,(xx[boolVec]-coord[1],yy[boolVec]-coord[0]),
                       data[boolVec],p0=[1,0,0,1,1],bounds=(lowBound,upBound),
                       sigma=rms*np.ones(xx[boolVec].size))
    # Getting the
    amp = popt[0] 
    sigx = popt[3]
    sigy = popt[4]

    return amp,sigx,sigy

def calc_psf_map(pointCoord,weightsTensor,dirtyMapShape,returnCoeffs=False):
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
    pointMap[int(ycoord),int(xcoord)] = 1

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
    Ncells = dirtyMap.shape[1]
    windowSizePix = int(windowSizeDeg/(360/Ncells)) + 1

    if windowSizePix % 2 == 0:
        windowSizePix += 1
    
    # Calculating the background estimate. Using a Gaussian to low pass filter
    # the coefficients. In future can use a different filter.
    lsig = 2*np.pi*(1/(np.radians(360/Ncells)*windowSizePix))
    lMax = skyCo.shape[-1] - 1
    lVec = np.arange(lMax+1)
    skyCoLPF = np.copy(skyCo)
    skyCoLPF[:,:lMax+1,:lMax+1] *= np.exp(-0.5*(lVec/lsig)**2)[None,:,None]
    skyCoeffsLPF = SHCoeffs.from_array(skyCoLPF,normalization='ortho',csphase=-1,
                                   lmax=lMax)
    skyMapObjLPF = skyCoeffsLPF.expand(grid='DH2',backend='ducc',lmax=lMax)
    bkgMap = skyMapObjLPF.data.real
    
    print(f"Window size = {windowSizePix}")

    diffMap = (dirtyMap.real-bkgMap)
    stdMap =  np.sqrt(generic_filter(diffMap**2,np.median,size=windowSizePix))
    
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
    # TODO: Refactor the masking. This can be greatly simplified.

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

def make_resid_map(modelMap,almTensor,mmodeTensor,lMax=130,lMaxVec=None,
                   verbosity=1,damp=0.5,plotCond=False,vmin=-1e6,vmax=1e6,
                   linear_width=1e5):
    """
    Takes input model image, beam fringe coefficients, and mmode visibility data
    tensor. Performs a subtraction of the model from the data, and solves for 
    the residual image spherical harmonic coefficients and outputs the residual 
    image.

    Parameters:
    ----------
    modelMap : np.float64 np.ndarray
        Real 2D map containing the model points.
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
    mapPrep = SHGrid.from_array(np.array(modelMap,dtype=np.complex64))
    # Set to zero for the next iteration.
    modelCoeff = mapPrep.expand(normalization='ortho',csphase=-1).coeffs
    modelCoeff = modelCoeff[0,:,:]

    if isinstance(almTensor,list):
        # For multi-system CLEAN with different lmax values.
        #invert = invert_CGLS_multi_pylops_assym
        invert = invert_tikh_multi_assym

        NbVec = np.array([alm.shape[0] for alm in almTensor]) # Nbaseline vector
        lVec = np.array([alm.shape[-1] for alm in almTensor]) # lMax vector

        NbSum = 0 # Running number of baselines sum.
        modelMmodeTensor = np.zeros([np.sum(NbVec),int(lMax+1)],
                                    dtype=np.complex64)
        
        for i,alm in enumerate(almTensor):
            # Getting the temp mmode tensor for system i
            tmpMmodeTensor = np.conj(np.einsum("blm,lm->bm",alm,
                                               modelCoeff[:lVec[i],:lVec[i]],
                                               optimize='optimal'))
            # Assigning the mmode values to the appropriate baseline indices.
            modelMmodeTensor[NbSum:NbSum+NbVec[i],:lVec[i]] = tmpMmodeTensor
            # Increasing the running total baseline sum.
            NbSum += NbVec[i]

    elif isinstance(almTensor,np.ndarray):
        # For single system.
        invert = invert_tikh_multi_assym

        # Forward modelling the mmode tensor.
        modelMmodeTensor  = np.conj(np.einsum("blm,lm->bm",almTensor,modelCoeff,
                                    optimize='optimal'))

    # Calculating the residual mmode tensor.
    if len(mmodeTensor.shape) == 3:
        # Only need positive m-modes.
        residMmodeTensor = mmodeTensor[:,0,:] - modelMmodeTensor
    elif len(mmodeTensor.shape) == 2:
        # Only need positive m-modes.
        residMmodeTensor = mmodeTensor - modelMmodeTensor

    # Solving for the sky modes.
    skyModes = invert(almTensor,np.conj(residMmodeTensor),lmax=lMax,
                      lMaxVec=lMaxVec,verbosity=verbosity,damp=damp)

    sphericalCoeffs = SHCoeffs.from_array(skyModes,normalization='ortho',
                                          csphase=-1)
    griddedCoeffs = sphericalCoeffs.expand(grid='DH2',
                                           backend='ducc',lmax_calc=lMax)
    residDirtyMap = griddedCoeffs.data.real

    if plotCond:
        plot_dirty_image(modelMap,linear_width=linear_width,
                        norm='asinh',title='Model Image')
        plot_dirty_image(residDirtyMap.real,norm='asinh',vmax=vmax,vmin=vmin,
                        linear_width=linear_width,title='Residual Dirty Image')

    return residDirtyMap

def minor_iteration(dirtyMap,dirtyPeakMap,modelMap,psfWeightsTensor,coords,
                    psfCoeffCube,stdMap,bkgMap,loopGain=0.1,sigThresh=2,
                    verbosity=1,plotCond=False):
    """
    Performs the minor iteration.

    Parameters:
    ----------
    dirtyMap : np.float64 np.ndarray
        Dirty map, as input 2D numpy array.
    dirtyPeakMap : np.float64 np.ndarray
        Background subtracted dirty map, as input 2D numpy array.
    modelMap : np.float64 np.ndarray
        Real 2D map containing the model points.
    coords : np.float64 np.ndarray
        2D numpy array containing peak xy-coordinates.
    psfCube : np.float64 np.ndarray
        3D numpy array, each slice has the same dimensions as the dirty map,
        contains all the PSF maps for each declination.
    stdMap : np.float64 np.ndarray
        Standard deviation map of the dirty image.
    poptArr : np.float64 np.ndarray
        2D array containing the fitted PSF Gaussian parameters. Needed to make
        the final resotred map.
    bkgMap : np.float64 np.ndarray
        Smoothed background map of the dirty image.
    xygrid : tuple,np.float64 np.ndarray
        Tuple containing the 2D xy-grid numpy arrays.
    almTensor : list or numpy array, complex64
        Array containing the beam fringe spherical harmonic coefficients. For 
        multi-system this is a list containing the beam fringe sh-coefficient
        tensors for each system.
    loopGain : float, default=0.1
        Fraction of peak to subtraction from CLEAN component.
    lMax : int, default=130
        Max l-mode.
    sigThresh : float, default=2
        Significance threshold as a sigma multiple to CLEAN down towards. Lower
        means deaper clean.
    verbosity : int, default=1
        CGLS output order, goes from 1-10, 1 being less output, 10 being more.

    Returns:
    ----------
    """
    peaks = dirtyMap[coords[:,0],coords[:,1]].real
    peakInd = np.argmax(np.abs(peaks))
    ycoord,xcoord = coords[peakInd,:]
    xcent = int(dirtyMap.shape[1]/2)

    # Making the PSF.
    if np.sum(psfCoeffCube[ycoord,:,:]) == 0:
        # Used to model the psf:
        pointMap = np.zeros(dirtyMap.shape)
        pointMap[int(ycoord),xcent] = 1
        psfMap,apsf,_ = calc_psf_map((xcent,ycoord),psfWeightsTensor,
                                     dirtyMap.shape,returnCoeffs=True)

        # Assign to PSF coefficient cube for later use.
        psfCoeffCube[ycoord,:,:] = apsf[0,:,:] # Only need the positive m-modes.
    else:
        # If the coefficients already exist then we can expand the map.
        apsf = psfCoeffCube[ycoord,:,:] # psf coefficients.
        apsf = restore_negmodes(apsf) # Restoring negative modes.
        psfCoeffs = SHCoeffs.from_array(apsf,normalization='ortho',csphase=-1)
        psfMap = psfCoeffs.expand(grid='DH2',backend='ducc').data.real # psf map.

    #
    if plotCond:
        plot_dirty_image(psfMap,norm='linear',title='PSF')

    bkg = bkgMap[ycoord,xcoord]
    std = stdMap[ycoord,xcoord]

    psfMap = np.roll(psfMap,xcoord-xcent,axis=1)
    #peak = dirtyMap[ycoord,xcoord].real
    peak = dirtyPeakMap[ycoord,xcoord].real

    dirtyMap[:,:] = dirtyMap[:,:]-psfMap*peak*loopGain
    dirtyPeakMap[:,:] = dirtyPeakMap[:,:]-psfMap*peak*loopGain

    modelMap[ycoord,xcoord] += peak*loopGain
    #modelMap[ycoord,xcoord] += peak*loopGain*psfMap[ycoord,xcoord]
    #print(psfMap[ycoord,xcoord]*peak*loopGain,sigThresh*std+bkg,sigThresh*std,np.abs(dirtyPeakMap[ycoord,xcoord].real))

    if np.abs(dirtyPeakMap[ycoord,xcoord].real) <= (sigThresh*std):
        if verbosity > 0:
            print('Point source reached threshold:')
            print(f'(y,x) = ',ycoord,xcoord)
            print(f'bkg = {bkg:5.3f}')
            print(f'std = {std:5.3f}')
            print(f'peak = {peak:5.3f}')
            print(psfMap[ycoord,xcoord]*peak*loopGain,sigThresh*std+bkg)
        # If threshold reached then delete the source from the list.
        coords = np.delete(coords,peakInd,axis=0)

    return coords

def major_iteration(mmodeTensor,almTensor,residDirtyMap,modelMap,
                    psfCoeffCube,psfWeightsTensor,skyCoeffs,coords=None,
                    Nminor=20000,plotCond=False,thresh=4,lMax=130,lMaxVec=None,
                    loopGain=0.1,sigThresh=2,verbosity=1,damp=0.5,
                    cleanMask=None,windowSizeDeg=7,vmin=-1e6,vmax=1e6,
                    linear_width=1e5):
    """
    Performs the major iteration. Finds CLEAN components with minor loops, and
    then subtracts the model from the mmode visibility tensor.

    Parameters:
    ----------
    mmodeTensor : np.complex64 np.ndarray
        Mmode visibility data tensor.
    almTensor : list or numpy array, complex64
        Array containing the beam fringe spherical harmonic coefficients. For 
        multi-system this is a list containing the beam fringe sh-coefficient
        tensors for each system.
    residDirtyMap : np.complex64 np.ndarray
        Output residual dirty map.
    modelMap : np.float64 np.ndarray
        Real 2D map containing the model points.
    paramsArr : np.float64 np.ndarray
        2D array containing the fitted PSF Gaussian parameters. Needed to make
        the final resotred map.
    psfCube : np.float64 np.ndarray
        3D numpy array, each slice has the same dimensions as the dirty map,
        contains all the PSF maps for each declination.
    DECgrid : np.float64 np.ndarray
        2D DEC grid.
    xygrid : tuple,np.float64 np.ndarray
        Tuple containing the 2D xy-grid numpy arrays.
    Nminor : int, default=10000
        Number of minor loop iterations.
    plotCond : bool, default=False
        Plot condition, if True output plots.
    thresh : float, default=4
        Significance threshold in sigma for peak detection.
    lMax : int, default=130
        Max l-mode.
    loopGain : float, default=0.1
        Fraction of peak to subtraction from CLEAN component.
    sigThresh : float, default=2
        Significance threshold as a sigma multiple to CLEAN down towards. Lower
        means deaper clean.
    verbosity : int, default=1
        CGLS output order, goes from 1-10, 1 being less output, 10 being more.
    damp : float, default=0.5
        CGLS dampening coefficient.
    DECthresh : tuple, default=(41,-80)
        Tuple containing the DEC limits (max and min) which to not CLEAN outside.
    maskList : list, default=None
        List containing mask tuples, each mask is a tuple of size 3 or 4, 
        containing the (x,y,winx,winy) values (coordinates and window size).
    cleanMask : bool np.ndarray, default=None
        Boolean numpy array with the same shape as the dirty image. Used to find
        peaks within only the mask region.
    windowSizeDeg : float, default=7
        Window size in degrees for calculating the background.
    

    Returns:
    ----------
    """
    # Get the background, standard deviation and threshold maps.
    bkgMap,stdMap,threshMap = make_thresh_maps(residDirtyMap,skyCoeffs,
                                               windowSizeDeg=windowSizeDeg)
    if verbosity > 0:
        print('Background, standard deviation, and threshold maps created...')
    # Subtract the background from the residual image.
    dirtyPeakMap = residDirtyMap.real-bkgMap
    
    if coords is None:
        if verbosity > 0:
            print('Performing peak detection...')
        # Perform peak detection on the threshold map, apply masks if available.
        coords = find_good_peaks(threshMap,thresh=thresh,cleanMask=cleanMask,
                                 threshold_abs=thresh)
    # If no sources found then exit.
    if coords.size == 0:
        print('No sources found.')
        return False
    
    if verbosity > 0:
        print(f"{coords.shape[0]} peaks found.")

    if plotCond:
        # Plot the bkg, dirty map, and threshold map with coords overlaid.
        plot_dirty_image(residDirtyMap.real,coords=coords,
                         linear_width=linear_width,
                         norm='asinh',title='Dirty Image',
                         vmax=vmax,vmin=vmin)
        if verbosity > 0:
            # If verbosity is greater than zero and plot cond is true, plot
            # the threshold maps.
            plot_dirty_image(bkgMap,linear_width=linear_width,
                            norm='asinh',title='Background')
            plot_dirty_image(dirtyPeakMap,linear_width=linear_width,
                            norm='asinh',title='Background-image')
            plot_dirty_image(stdMap,linear_width=linear_width,
                            norm='asinh',title='Std')
            plot_dirty_image(threshMap,coords=coords,vmax=10,vmin=0,
                            norm='linear',title='Threshold')

    for i in tqdm(range(Nminor)):
        # If no more sources to loop through we can cancel.
        if coords.size == 0:
            print(f'Minor loops finished at {i}')
            break

        if i == 0 and plotCond:
            plotCondMinor = True
        else:
            plotCondMinor = False

        #
        coords = minor_iteration(residDirtyMap,dirtyPeakMap,modelMap,
                                 psfWeightsTensor,coords,psfCoeffCube,stdMap,
                                 bkgMap,loopGain=loopGain,sigThresh=sigThresh,
                                 verbosity=verbosity,plotCond=plotCondMinor)
    # Calculate the residual dirty map.
    residDirtyMap[:,:] = make_resid_map(modelMap,almTensor,mmodeTensor,
                                        lMax=lMax,lMaxVec=lMaxVec,
                                        verbosity=verbosity,damp=damp,
                                        plotCond=plotCond,vmin=vmin,vmax=vmax,
                                        linear_width=linear_width)
    return True

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


def calc_clean_mask(skyCo,initalMask=None,DECthresh=(41,-80),maskList=None,
                    GPthresh=0,GPthreshFlip=False,plotCond=False):
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