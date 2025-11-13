import numpy as np
import tqdm
import pyshtools
from tqdm import tqdm
from astropy.io import fits
import matplotlib.pyplot as plt
from pyshtools import SHGrid,SHCoeffs
from mmode_tools.inversion import invert_CGLS_multi_pylops
from mmode_tools.inversion import invert_CGLS_multi_pylops_assym
from mmode_tools.functions import Gaussian2Dxy
from scipy.optimize import curve_fit

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
        invert = invert_CGLS_multi_pylops_assym

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
        invert = invert_CGLS_multi_pylops

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

def make_thresh_maps(dirtyMap,relWindowSize=0.04):
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
    from skimage.filters import threshold_local
    from scipy.stats import iqr

    # Scale invariant window size.
    windowSize = int(relWindowSize*dirtyMap.shape[1])
    if windowSize % 2 == 0:
        windowSize += 1
    
    print(f"Window size = {windowSize}")

    bkgMap = threshold_local(dirtyMap.real,windowSize,mode='wrap',
                             method='gaussian')
    stdMap = threshold_local(dirtyMap.real-bkgMap,windowSize,mode='wrap',
                             method='generic',param=iqr)/1.35

    threshMap = ((dirtyMap.real-bkgMap)/stdMap)

    return bkgMap,stdMap,threshMap

def find_good_peaks(threshMap,DECgrid,
                    thresh=4,DECthresh=(41,-80),maskList=None,cleanMask=None,
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
    DECthresh : tuple, default=(41,-80)
        Tuple containing the DEC limits (max and min) which to not CLEAN outside.
    maskList : list, default=None
        List containing tuples of (x,y,window) coordinates for rectangular
        masks.
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
    # Remove low latitude regions, and high latitude regions where we have poor 
    # sensitivity.
    decMask = (DECgrid >= DECthresh[0]) | (DECgrid < DECthresh[1])

    threshMap[decMask] = 0
    # Performing the peak detection on the masked threshold map.
    coords = peak_local_max(threshMap,threshold_rel=threshold_rel,
                            threshold_abs=threshold_abs)
    threshVec = threshMap[coords[:,0],coords[:,1]]
    coords = coords[threshVec>=thresh,:]

    # Finding sources which are in the Sun and Cygnus A sidelobe regions. 
    # Sources in these locations are potentially spurious artefacts.
    if np.any(maskList):
        for i,mask in enumerate(maskList):
            boolInds = ((coords[:,1]>mask[1])&(coords[:,1]<=mask[1]+mask[2]))*\
            ((coords[:,0]>mask[0])&(coords[:,0]<=mask[0]+mask[2]))
            # Subset coords for all sources not in the masked region.
            coords = coords[boolInds==False,:]
    
    if np.any(cleanMask):
        # Check the shapes are the same, required to filter coords outside the 
        # clean mask.
        if cleanMask.shape != threshMap.shape:
            errMsg = f"Clean mask shape {cleanMask.shape} not equal " +\
                     f"to map shape {threshMap.shape}"
            raise ValueError(errMsg)
        
        print('Appling a CLEAN mask.')
        # Calculate the 1D index values of the coordinates.
        ravelCoords = np.ravel_multi_index((coords[:,0],coords[:,1]),
                                           cleanMask.shape)

        # Create a flat index vector and apply the clean mask to get the mask 
        # index vector.
        indVec = np.arange(cleanMask.size)
        maskInds = indVec[cleanMask.flatten()]

        # Find all the mask indices that are the ravelCoords vector.
        coordMask = np.isin(ravelCoords,maskInds)

        # Apply the mask to coords.
        coords = coords[coordMask,:]

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

def make_resid_map(modelMap,almTensor,mmodeTensor,lMax=130,verbosity=1,damp=0.5,
                   plotCond=False,rtol=1e-4,vmin=-1e6,vmax=1e6,
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
        invert = invert_CGLS_multi_pylops_assym

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
        invert = invert_CGLS_multi_pylops

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
    skyModes = invert(almTensor,np.conj(residMmodeTensor),lmax=lMax,rtol=rtol,
                      verbosity=verbosity,damp=damp)

    sphericalCoeffs = SHCoeffs.from_array(skyModes,normalization='ortho',
                                          csphase=-1)
    griddedCoeffs = sphericalCoeffs.expand(grid='DH2',
                                           backend='ducc',lmax_calc=lMax)
    residDirtyMap = griddedCoeffs.data

    if plotCond:
        plot_dirty_image(modelMap,linear_width=linear_width,
                        norm='asinh',title='Model Image')
        plot_dirty_image(residDirtyMap.real,norm='asinh',vmax=vmax,vmin=vmin,
                        linear_width=linear_width,title='Residual Dirty Image')

    return residDirtyMap

def minor_iteration(dirtyMap,dirtyPeakMap,modelMap,coords,psfCube,stdMap,
                    poptArr,bkgMap,xygrid,almTensor,loopGain=0.1,lMax=130,
                    sigThresh=2,verbosity=1,damp=0.5,plotCond=False):
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
    if np.sum(psfCube[ycoord,:,:]) == 0:
        # Used to model the psf:
        pointMap = np.zeros(dirtyMap.shape)
        pointMap[int(ycoord),xcent] = 1
        psfMap = forward_model_psf(pointMap,almTensor,lMax=lMax,
                                   verbosity=verbosity,damp=damp).real

        # Assign to PSF cube:
        psfCube[ycoord,:,:] = psfMap

        # Getting the fit PSF params.
        _,sigx,sigy = fit_restoring_beam(xygrid,psfMap,np.array([ycoord,xcent]))
        
        # Assigning the fit PSF params.
        poptArr[ycoord,:] = np.array([1/(2*np.pi*sigx*sigy),xcoord,ycoord,
                                    sigx,sigy])
    else:
        if dirtyMap.shape[1] > psfCube[ycoord,:,:].shape[1]:
            # For larger maps, we only stor the 5 sigma around the PSF.
            psfMap = np.zeros(dirtyMap.shape,dtype=np.float64)
            dN = int((psfMap.shape[1] - psfCube[ycoord,:,:].shape[1])/2)
            psfMap[:,dN:-dN-1] = psfCube[ycoord,:,:]
        else:
            psfMap = psfCube[ycoord,:,:]

    if plotCond:
        plot_dirty_image(psfMap.real,norm='linear',title='PSF')

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

def major_iteration(mmodeTensor,almTensor,residDirtyMap,modelMap,paramsArr,
                    psfCube,DECgrid,xygrid,coords=None,Nminor=20000,
                    plotCond=False,thresh=4,lMax=130,loopGain=0.1,sigThresh=2,
                    verbosity=1,damp=0.5,DECthresh=(41,-80),maskList=None,
                    cleanMask=None,relWindowSize=0.04,vmin=-1e6,vmax=1e6,
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
    relWindowSize : float, default=0.04
        Relative window size for determining the threshold window. This is a 
        scale invariant method. This is multiplied by the Naxis1 of the dirty
        image. 
    

    Returns:
    ----------
    """
    # Get the background, standard deviation and threshold maps.
    bkgMap,stdMap,threshMap = make_thresh_maps(residDirtyMap,
                                               relWindowSize=relWindowSize)
    if verbosity > 0:
        print('Background, standard deviation, and threshold maps created...')
    # Subtract the background from the residual image.
    dirtyPeakMap = residDirtyMap.real-bkgMap
    
    if np.any(coords) == None:
        if verbosity > 0:
            print('Performing peak detection...')
        # Perform peak detection on the threshold map, apply masks if available.
        coords = find_good_peaks(threshMap,DECgrid,thresh=thresh,
                                 DECthresh=DECthresh,maskList=maskList,
                                 cleanMask=cleanMask,threshold_abs=thresh)
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
                         norm='asinh',title='Dirty Image',patchList=maskList,
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

        coords = minor_iteration(residDirtyMap,dirtyPeakMap,modelMap,coords,
                                 psfCube,stdMap,paramsArr,bkgMap,xygrid,
                                 almTensor,loopGain=loopGain,lMax=lMax,
                                 sigThresh=sigThresh,verbosity=verbosity,
                                 damp=damp,plotCond=plotCondMinor)
    # Calculate the residual dirty map.
    residDirtyMap[:,:] = make_resid_map(modelMap,almTensor,mmodeTensor,
                                        lMax=lMax,verbosity=verbosity,damp=damp,
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
    modInds = modelMap > 0

    # Determining the Gaussian restoring beam size. Should be the min fit 
    # Gaussian.
    sigMin = np.min(paramsArr[:,3:5][paramsArr[:,3:5]>=1])
    xcoords = xx[modInds]
    ycoords = yy[modInds]
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