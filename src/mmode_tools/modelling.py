import numpy as np
from numpy import exp,pi
from mmode_tools.constants import c,MRO

def calc_lmn(altVec,azVec,degrees=True):
    """
    Function for calculating the direction cosines (l,m,n) for a given altitude
    and azimuth vector. The inputs are assumed to be in degrees.

    Parameters
    ----------
    altVec : numpy np.ndarray, float
        Vector of altitude values.
    azVec : numpy np.ndarray, float
        Vector of azimuth values.
    degrees : bool, default=True
        If True, inputs are assumed to be in degrees.

    Returns
    -------
    lVec : numpy np.ndarray, float
        Vector of direction cosine l values.
    mVec : numpy np.ndarray, float
        Vector of direction cosine m values.
    nVec : numpy np.ndarray, float
        Vector of direction cosine n values.
    """
    if degrees:
        altVec = np.radians(altVec)
        azVec = np.radians(azVec)

    lVec = np.cos(altVec)*np.sin(azVec)
    mVec = np.cos(altVec)*np.cos(azVec)
    nVec = np.sqrt(1 - lVec**2 - mVec**2)

    return lVec,mVec,nVec

def radec2lmn(tgpsVec,ra,dec,location=MRO,nan_below_horizon=True):
    """
    Function to convert ra and dec coordinate for a given source, to direction
    cosines (l,m,n) for a given set of utc times in the GPS format. This assumes
    the location of the site is the MRO, and that all positions are in degrees.

    Parameters
    ----------
    tgpsVec : numpy np.ndarray, float
        Vector of utc times in gps format.
    ra : float
        Right ascension position value of source in degrees.
    dec : float
        Declination position value of source in degrees.

    Returns
    -------
    lVec : numpy np.ndarray, float
        Vector of direction cosine l values.
    mVec : numpy np.ndarray, float
        Vector of direction cosine m values.
    nVec : numpy np.ndarray, float
        Vector of direction cosine n values.
    """
    ## TODO
    # Make this work for multiple input sources.
    from astropy.time import Time
    from astropy.coordinates import AltAz,SkyCoord
    from astropy import units as u

    if isinstance(ra,np.ndarray)==False:
        if isinstance(ra,(float,np.float32,np.float64))==False:
            if isinstance(ra,(int,np.int32,np.int64)):
                ra = float(ra)
            else:
                print(np.dtype(ra))
                print(ra)
                raise TypeError("ra should be float or np.float64 dtype.")
        
    if isinstance(dec,np.ndarray)==False:
        if isinstance(dec,(float,np.float32,np.float64))==False:
            if isinstance(dec,(int,np.int32,np.int64)):
                dec = float(dec)
            else:
                print(dec)
                raise TypeError("dec should be float or np.float64 dtype.")

    t = Time(tgpsVec,format="gps",scale="ut1")
    sky_posn = SkyCoord(ra*u.deg,dec*u.deg)
    altaz = sky_posn.transform_to(AltAz(obstime=t,location=location))

    alt,az = altaz.alt.deg,altaz.az.deg
    lVec,mVec,nVec = calc_lmn(alt,az,degrees=True)

    if nan_below_horizon:
        lVec[alt < 0] = np.nan
        mVec[alt < 0] = np.nan
    else:
        lVec[alt < 0] = 1
        mVec[alt < 0] = 1

    return lVec,mVec,nVec

def point_mod(interferometer,lam,lMod,mMod,nMod,Sapp,verbose=False):
    """
    Takes input point source l, m and n values (as a function of LST), as well 
    as the apparent source brightness, and computes a model visibility 
    covariance tensor.

    Parameters
    ----------
    Array : mmod_tools.RadioArray object
        Array for the observations, used to calculate the phases.
    lam : float
        Wavelength in m, of the observation.
    lMod : numpy np.ndarray, float
        Vector of model direction cosine l values.
    mMod : numpy np.ndarray, float
        Vector of model direction cosine m values.
    nMod : numpy np.ndarray, float
        Vector of model direction cosine n values.
    Sapp : numpy np.ndarray, float
        Vector of apparent model flux density values.

    Returns
    -------
    VisCovTensor : numpy np.ndarray, np.complex64
        Visibility model covariance tensor [Nlst,Nant,Nant].
    """
    
    if np.any(np.isnan(lMod)):
        # If there are any nonsense values then set the l,m,n values to 0,
        # no phase rotation. Nan values are the source below the horizon.
        Sapp[np.isnan(lMod)] = 0
    elif np.any(np.isnan(mMod)):
        Sapp[np.isnan(mMod)] = 0
    #
    uu_lmod = interferometer.uu_m[None,:,:]*lMod[:,None,None]/lam
    vv_mmod = interferometer.vv_m[None,:,:]*mMod[:,None,None]/lam
    ww_nmod = interferometer.ww_m[None,:,:]*(nMod[:,None,None]-1)/lam

    if verbose:
        print(uu_lmod.shape)
        print(vv_mmod.shape)
        print(ww_nmod.shape)

    VisCovTensor = Sapp[:,None,None]*exp(-2*pi*1j*(uu_lmod+vv_mmod+ww_nmod))

    return VisCovTensor



def point_mod2(interferometer,tgpsVec,freq,raSrc,decSrc,srcFlux,
              beamFilePath=None,verbose=False):
    """
    Takes input point source l, m and n values (as a function of LST), as well 
    as the apparent source brightness, and computes a model visibility 
    covariance tensor.

    Parameters
    ----------
    Array : mmod_tools.RadioArray object
        Array for the observations, used to calculate the phases.
    lam : float
        Wavelength in m, of the observation.
    lMod : numpy np.ndarray, float
        Vector of model direction cosine l values.
    mMod : numpy np.ndarray, float
        Vector of model direction cosine m values.
    nMod : numpy np.ndarray, float
        Vector of model direction cosine n values.
    Sapp : numpy np.ndarray, float
        Vector of apparent model flux density values.

    Returns
    -------
    VisCovTensor : numpy np.ndarray, np.complex64
        Visibility model covariance tensor [Nlst,Nant,Nant].
    """
    from mmode_tools.beam import beam_vec_calc
    from astropy.coordinates import EarthLocation
    from astropy import units as u
    # Getting the array latitude in radians.
    latRad = interferometer.lat
    lonRad = interferometer.lon
    meanHeight = np.mean(interferometer.height)
    interferometerLocation = EarthLocation(lat=latRad*u.deg,lon=lonRad*u.deg,
                                           height=meanHeight*u.m)
    lam = c/freq
    lVec,mVec,nVec = radec2lmn(tgpsVec,raSrc,decSrc,
                               location=interferometerLocation,
                               nan_below_horizon=False)

    # Calculating the apparent source brightness.
    beamVals = beam_vec_calc(interferometer,raSrc,decSrc,tgpsVec,
                             beamFilePath=beamFilePath)
    if isinstance(srcFlux,(int,float)):
        srcFlux = srcFlux*np.ones(tgpsVec.size)
    srcFlux = srcFlux*beamVals

    if np.any(np.isnan(lVec)):
        # If there are any nonsense values then set the l,m,n values to 0,
        # no phase rotation. Nan values are the source below the horizon.
        srcFlux[np.isnan(lVec)] = 0
    elif np.any(np.isnan(mVec)):
        srcFlux[np.isnan(mVec)] = 0
    #
    uu_lmod = interferometer.uu_m[None,:,:]*lVec[:,None,None]/lam
    vv_mmod = interferometer.vv_m[None,:,:]*mVec[:,None,None]/lam
    ww_nmod = interferometer.ww_m[None,:,:]*(nVec[:,None,None]-1)/lam

    phaseTerm = -2*np.pi*1j*(uu_lmod+vv_mmod+ww_nmod)

    del uu_lmod,vv_mmod,ww_nmod

    VisCovTensor = srcFlux[:,None,None]*np.exp(phaseTerm)

    return VisCovTensor

def phase_rot_tensor(Array,lam,lMod,mMod,nMod):
    """
    Calculates the phase rotation tensor.

    Parameters
    ----------
    Array : mmod_tools.RadioArray object
        Array for the observations, used to calculate the phases.
    lam : float
        Wavelength in m, of the observation.
    lMod : numpy np.ndarray, float
        Vector of model direction cosine l values.
    mMod : numpy np.ndarray, float
        Vector of model direction cosine m values.
    nMod : numpy np.ndarray, float
        Vector of model direction cosine n values.

    Returns
    -------
    phaseTensor : numpy np.ndarray, np.complex64
        Phase rotation tensor [Nlst,Nant,Nant].
    """
    if np.any(np.isnan(lMod)):
        # If there are any nonsense values then set the l,m,n values to 0,
        # no phase rotation.
        nMod[np.isnan(lMod)] = 1
        mMod[np.isnan(lMod)] = 0
        lMod[np.isnan(lMod)] = 0
    elif np.any(np.isnan(mMod)):
        nMod[np.isnan(mMod)] = 1
        lMod[np.isnan(mMod)] = 0
        mMod[np.isnan(mMod)] = 0


    uu_m_lmod = Array.uu_m[None,:,:]*lMod[:,None,None]/lam
    vv_m_mmod = Array.vv_m[None,:,:]*mMod[:,None,None]/lam
    ww_m_nmod = Array.ww_m[None,:,:]*(nMod[:,None,None]-1)/lam

    phaseTensor = exp(2*pi*1j*(uu_m_lmod+vv_m_mmod+ww_m_nmod))

    return phaseTensor

def phase_back_rot_tensor(Array,lam,lMod,mMod,nMod):
    """
    Calculates the back phase rotation tensor.

    Parameters
    ----------
    Array : mmod_tools.RadioArray object
        Array for the observations, used to calculate the phases.
    lam : float
        Wavelength in m, of the observation.
    lMod : numpy np.ndarray, float
        Vector of model direction cosine l values.
    mMod : numpy np.ndarray, float
        Vector of model direction cosine m values.
    nMod : numpy np.ndarray, float
        Vector of model direction cosine n values.

    Returns
    -------
    phaseTensor : numpy np.ndarray, np.complex64
        Phase rotation tensor [Nlst,Nant,Nant].
    """
    if np.any(np.isnan(lMod)):
        # If there are any nonsense values then set the l,m,n values to 0,
        # no phase rotation.
        nMod[np.isnan(lMod)] = 1
        mMod[np.isnan(lMod)] = 0
        lMod[np.isnan(lMod)] = 0
    elif np.any(np.isnan(mMod)):
        nMod[np.isnan(mMod)] = 1
        lMod[np.isnan(mMod)] = 0
        mMod[np.isnan(mMod)] = 0

    uu_m_lmod = Array.uu_m[None,:,:]*lMod[:,None,None]/lam
    vv_m_mmod = Array.vv_m[None,:,:]*mMod[:,None,None]/lam
    ww_m_nmod = Array.ww_m[None,:,:]*(nMod[:,None,None]-1)/lam

    phaseTensor = exp(-2*pi*1j*(uu_m_lmod+vv_m_mmod+ww_m_nmod))

    return phaseTensor

def get_sun_lst_range(tgpsVec,verbose=False,returnAltAz=False):
    """
    Function which returns the lst values of the sun.
    
    Parameters
    ----------
    tgpsVec : float, np.ndarray
        time vector containing the UTC times for each covariance slice. in GPS
        format.

    Returns
    -------
    sunInds : numpy np.ndarray, bool
        Boolean array of indices when the sun is above the horizon.
    """
    from astropy.time import Time
    import astropy.coordinates
    from astropy.coordinates import AltAz,EarthLocation

    t = Time(tgpsVec,format="gps",scale="ut1")
    sunpos = astropy.coordinates.get_sun(t)
    location = EarthLocation.of_site('MWA')
    sunelaz = sunpos.transform_to(AltAz(obstime=t,location=location))

    sunInds = (sunelaz.alt.degree > 0.)
    #sunInds = (sunelaz.alt.degree > 0.1)

    if verbose:
        print(sunelaz.alt.degree)
    
    if returnAltAz:
        return sunInds,sunelaz
    else:
        return sunInds

def forward_model_mmode_vis(modelMap,blmTensor,lMax=129):
    """
    Parameters
    ----------

    Returns
    -------
    """
    from pyshtools import SHGrid

    mapPrep = SHGrid.from_array(np.array(modelMap,dtype=np.complex64))
    # Set to zero for the next iteration.
    mapCoef = mapPrep.expand(normalization='ortho',csphase=-1).coeffs

    # Get the positive map coefficients.
    mapCoef = mapCoef[0,:,:]
    if isinstance(blmTensor,list):
        # For multi-system CLEAN with different lmax values.
        NbVec = np.array([alm.shape[0] for alm in blmTensor]) # Nbaseline vector
        lVec = np.array([alm.shape[-1] for alm in blmTensor]) # lMax vector

        NbSum = 0 # Running number of baselines sum.
        mmodeTensor = np.zeros([np.sum(NbVec),int(lMax+1)],dtype=np.complex64)
        
        for i,blm in enumerate(blmTensor):
            # Getting the temp mmode tensor for system i
            tmpMmodeTensor = np.conj(np.einsum("blm,lm->bm",blm,
                                               mapCoef[:lVec[i],:lVec[i]],
                                               optimize='optimal'))
            # Assigning the mmode values to the appropriate baseline indices.
            mmodeTensor[NbSum:NbSum+NbVec[i],:lVec[i]] = tmpMmodeTensor
            # Increasing the running total baseline sum.
            NbSum += NbVec[i]
    elif isinstance(blmTensor,np.ndarray):
        # For single system.
        
        # Forward modelling the mmode tensor.
        mmodeTensor  = np.conj(np.einsum("blm,lm->bm",
                                         blmTensor[:,:lMax+1,:lMax+1],
                                         mapCoef[:lMax+1,:lMax+1],
                                         optimize='optimal'))
    
    return mmodeTensor


def phase_resid_squared(phiVec,visVec):
    """
    Function that calculates the chi squared value between the amplitude
    and the real value of a visibility for a range of phase (phiVec) 
    offsets.

    Parameters
    ----------
    phiVec : numpy np.ndarray, float
        Vector of phase offsets from -pi to pi.
    visVec : numpy np.ndarray, int
        Visibility vector.
    """

    visArr = np.ones((visVec.size,phiVec.size))*visVec[:,None]

    residSqdArr = (np.abs(visArr)-(visArr*np.exp(1j*phiVec[None,:])).real)**2

    return np.sum(residSqdArr,axis=0)

def calc_phase_correction(covTensor,indices,Nsamp=1000):
    """
    Find the phase that minimises the residual squared of the visibility 
    amplitude, and the real value of the visibility. This should account 
    for some of the phase offsets present in the data.

    Parameters
    ----------
    covTensor : numpy np.ndarray, float
        CovTensor shape (time,ant1,ant2)
    indices : numpy np.ndarray, int
        Vector of covTensor indices.
    Nsamp : int, default=1000
        How many phase samples.

    Returns
    -------
    phaseCorMatrix : numpy np.ndarray, float
        Phase correction matrix (ant1,ant2).
    """

    antIndVec = np.arange(covTensor.shape[1])

    phiVec = np.linspace(-np.pi,np.pi,Nsamp)

    phaseCorMatrix = np.zeros(covTensor[0,:,:].shape)
    for ant1 in antIndVec:
        for ant2 in antIndVec:
            if ant1 != ant2:
                tempVisVec = covTensor[indices,ant1,ant2]
                residVec = phase_resid_squared(phiVec,tempVisVec)
                phaseCorMatrix[ant1,ant2] = phiVec[np.argmin(residVec)]
            else:
                phaseCorMatrix[ant1,ant2] = 0

    return phaseCorMatrix

