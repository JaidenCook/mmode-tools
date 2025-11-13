import numpy as np
from tqdm import tqdm
import pyshtools
from astropy.io import fits
import mwa_hyperbeam
import datetime
import h5py as h5
import importlib.resources as resources
import toml
import warnings
import os

from mmode_tools.constants import c,MRO

configFile = "default_config.toml"
mmodeConfigPath = "mmode_tools.config"

with resources.files(mmodeConfigPath).joinpath(configFile).open("r") as f:
    paths = toml.load(f).get("files", {})
    MWA_BEAM_FILE = paths["MWA_BEAM_FILE"]


def radec2azel(ra,dec,lat):
    """
    Function to convert RA, Dec into Azimuth and Elevation given an observatory 
    location, assuming the LST is 12 hours.

    Parameters
    ----------
    ra : float
        Right Ascension in radians.
    dec : float
        Declination in radians.
    lat : float
        Observatory latitude in radians.

    Returns
    -------
    az : float
        Azimuth in radians
    el : float
        Elevation in radians
    """
    #Note this change, based on Eqs in 
    # https://astrogreg.com/convert_ra_dec_to_alt_az.html
    ha = ra - np.pi 
    az = np.arctan2(np.sin(ha),
                    (np.cos(ha)*np.sin(lat) - np.tan(dec)*np.cos(lat)))+np.pi
    el = np.arcsin(np.sin(lat)*np.sin(dec) + np.cos(lat)*np.cos(dec)*np.cos(ha))
    return az,el

def gen_lmn(lat,Ncells):
    """
    Function to generate the lmn tensor in equatorial cartesian ('CAR') 
    coordinates 

    Parameters
    ----------
    lat : float
        Observatory latitude in radians.
    Ncells : int
        Ncells.

    Returns
    -------
    lmn_array : numpy array, float
        The lmn array.
    """
    lmn_array = np.zeros([3,Ncells,Ncells])
    ra = np.linspace(0,2*np.pi,Ncells)
    dec = np.linspace(-np.pi/2,np.pi/2,Ncells)
    #
    for r_index in range(Ncells):
        for d_index in range(Ncells):
            az, el = radec2azel(ra[r_index],dec[d_index],lat)
            #
            lmn_array[0,d_index,r_index] = np.cos(el)*np.sin(az)
            lmn_array[1,d_index,r_index] = np.cos(el)*np.cos(az)
            lmn_array[2,d_index,r_index] = np.sin(el)

    return lmn_array

def bline2alm(baselines,beam,freq,lat,lMax,almCoeffsTensor=None):
    """
    Function to generate the alm coefficient tensor given a set of baselines, 
    telescope primary beam and the required frequency. Uses a relatively slower 
    method, but is better in terms of memory usage. 

    Parameters
    ----------
    baselines : numpy array, float
        Array of baselines in metres. Expects a [Nbaselines, 3] sized array
    beam : numpy array, float
        2D telescope primary beam in in equatorial cartesian ('CAR') coordinates 
    freq : float
        Frequency in Hz
    lat_rad : float
        Observatory latitude in radians
    Ncells : int
        Ncells
    order : str, default='C'
        This is the array data ordering. For parallel processing use 'F' for
        Fortran, otherwise the default is 'C'.

    Returns
    -------
    almCoeffsTensor : numpy array, float
        The 4-dimensional alm coefficient tensor. The ordering is baselines, 
        [0 or 1] for +ve or -ve m, l, m. This can be quite large !
    """
    Ncells = int(2*lMax+2)
    Nblines = baselines.shape[0]
    lmnGrid = gen_lmn(lat,Ncells)
    lam = c/freq
    # baselines * m/-m * l * m
    if np.any(almCoeffsTensor):
        returnCond = False
    else:
        almCoeffsTensor = np.zeros([Nblines,2,lMax+1,lMax+1],
                                     dtype=np.complex64)
        returnCond = True
    
    for blineInd in tqdm(range(Nblines)): 
        fringeMap = np.exp(-2j*np.pi*np.einsum('ijk,i->jk',lmnGrid, 
                                                baselines[blineInd,:], 
                                                optimize='greedy')/lam)
        beamFringeMap = np.einsum('ij,ij->ij',beam,fringeMap, 
                                    optimize='greedy')
        beamFringeGrid = pyshtools.SHGrid.from_array((beamFringeMap))
        alm = beamFringeGrid.expand(normalization='ortho',csphase=-1,
                                 lmax_calc=int(lMax),backend='ducc')
        #+m and -m coefficients copied at the same time
        almCoeffsTensor[blineInd,:,:,:] = alm.coeffs[:,:,:] 
    del lmnGrid,fringeMap,beamFringeMap,beamFringeGrid
    if returnCond:
        return almCoeffsTensor

def bline2alm_h5py(baselines,antPairs,beam,freq,lat,lMax,outFilePath=None,
                   chunks=False,compression="lzf",telescope=None):
    """
    Function to generate the alm coefficient tensor given a set of baselines, 
    telescope primary beam and the required frequency. Uses a relatively slower 
    method, but is better in terms of memory usage. 

    Updated function for writing out the almCoeffsTensor to a hdf5 file. This 
    avoids the necessity of keeping a numpy array in memory.

    Parameters
    ----------
    baselines : numpy array, float
        Array of baselines in metres. Expects a [Nbaselines, 3] sized array
    beam : numpy array, float
        2D telescope primary beam in in equatorial cartesian ('CAR') coordinates 
    freq : float
        Frequency in Hz
    lat : float
        Observatory latitude in radians
    lMax : int
        The maximum spherical harmonic order.
    outFilePath : str, default=None
        If None then the output file path for the hdf5 file is not set.

    Returns
    -------
    None
    """

    #
    Ncells = int(2*lMax+2)
    lam = c/freq
    Nbase = baselines.shape[0]
    lmnGrid = gen_lmn(lat,Ncells)
    # baselines * l * m
    almShape = (Nbase,lMax+1,lMax+1)

    # This should be a default.
    if outFilePath == None:
        outPath = "/data/M-MODE/beam_fringe_maps/"
        outName = "beam_fringe_coeffs"
        if telescope:
            outName += f"-{telescope}"
        if chunks:
            outName += f"-chunked-lMax{lMax}"
        if compression:
            outName += f"-{compression}.hdf5"
        else:
            outName += f".hdf5" 
        outFilePath = outPath+outName

    hf = h5.File(outFilePath,'w')
    g = hf.create_group('data')
    if chunks:
        print(f"Chunking set to true... Using compression {compression}...")
        chunk = (1,lMax+1,1)
        almCoeffsTensor = g.create_dataset('almCoeffTensor',almShape,
                                        dtype=np.complex64,chunks=chunk,
                                        compression=compression)
    else:
        almCoeffsTensor = g.create_dataset('almCoeffTensor',almShape,
                                        dtype=np.complex64)
    g.create_dataset('antPairs',data=antPairs)
    g.create_dataset('blineID',data=np.arange(Nbase))

    for blineInd in tqdm(range(Nbase)): 
        fringeMap = np.exp(-2j*np.pi*np.einsum('ijk,i->jk',lmnGrid, 
                                                baselines[blineInd,:], 
                                                optimize='greedy')/lam)
        beamFringeMap = np.einsum('ij,ij->ij',beam,fringeMap, 
                                    optimize='greedy')
        beamFringeGrid = pyshtools.SHGrid.from_array((beamFringeMap))
        alm = beamFringeGrid.expand(normalization='ortho',csphase=-1,
                                 lmax_calc=int(lMax),backend='ducc')
        
        # We only care about the positive (or the negative) m-modes.
        almCoeffsTensor[blineInd,:,:] = alm.coeffs[0,:,:]

    g.attrs['lMax'] = lMax
    g.attrs['Ncell'] = Ncells
    g.attrs['timestamp'] = str(datetime.datetime.now())

    hf.close()

    print(f"File saved to {outFilePath}")

    return None

def MWA_beam_calc(freq,Az,Alt,pol='X',tgpsVec=None,dipoleInd=10,
                  norm_to_zenith=True,delays=np.zeros(16,dtype=np.int16),
                  degrees=True,beam=None,normalise=True):
    """
    For a given frequency, altitude and azimuth, return the MWA beam. The beam
    can be for a single dipole or a collection of dipoles. Furthermore, this
    function only returns the beam for a single instrumental polarisation. 

    Parameters
    ----------
    freq : float
        Frequency in Hz
    Az : float, np.ndarray
        Azimuth or RA values in degrees by default or radians. If tgpsVec given
        then azimuth is interpretted as RA.
    Alt : float, np.ndarray
        Altitude or DEC values in degrees by default or radians. If tgpsVec 
        given, then altitude is interpretted as DEC.
    pol : str
        Polarisation - X or Y. Default is X
    dipoleInd : int, default=10
        The dipole number. Default is 10, checked against the data.
    norm_to_zenith : bool, default=True
        Dipole amplitudes normalised to zenith direction.
    delays : np.ndarray, default=np.zeros(16,dtype=np.int16)
        Array of delays for the dipoles. Zero only array is phased to zenith.
    degrees : bool, default=True
        If True then the alt and az parameters are in degrees.
    beam : mwa_hyperbeam.object
        beam object used to calculate the beam values.
    normalise : bool, default = False
        If Ture normalise the beam to the max value.

    Returns
    -------
    beamVec : np.ndarray
        Array of beam values for the associated az and alt values.
    """

    if degrees:
        Az = np.radians(Az)
        Alt = np.radians(Alt)
    
    if np.any(tgpsVec):
        # If tgpsvec given then alt and az are interpretted as ra and dec.
        # Calculate the Alt and Az for a given time series.
        from astropy.time import Time
        from astropy.coordinates import AltAz,SkyCoord
        from astropy import units as u

        t = Time(tgpsVec,format="gps",scale="ut1")
        sky_posn = SkyCoord(Az*u.deg,Alt*u.deg)
        altaz = sky_posn.transform_to(AltAz(obstime=t,location=MRO))

        Alt,Az = np.radians(altaz.alt.deg),np.radians(altaz.az.deg)

    amps = np.zeros(16)
    if isinstance(dipoleInd,int):
        amps[int(dipoleInd)] = 1
    elif isinstance(dipoleInd,np.ndarray):
        amps[dipoleInd.astype(int)] = 1

    Zen = np.pi/2 - Alt
    # 
    if beam == None:
        if MWA_BEAM_FILE == "":
            warnings.warn("MWA_BEAM_FILE path not set in config file, " +\
                          "using the analytic MWA beam model instead.")
            beam = mwa_hyperbeam.AnalyticBeam()
        else:
            if os.path.exists(MWA_BEAM_FILE):
                beam = mwa_hyperbeam.FEEBeam(MWA_BEAM_FILE)
            else:
                raise FileNotFoundError(f"MWA_BEAM_FILE path {MWA_BEAM_FILE} "+\
                                        f"not found.")
        

    # Calculating the Jones matrix tensor.
    jonesTensor = beam.calc_jones_array(Az,Zen,freq,delays,amps,norm_to_zenith)
    
    if pol == 'X' or pol == 'x' or pol == 'XX' or pol == 'xx':
        beamVec = np.abs(jonesTensor[:,0])**2 + np.abs(jonesTensor[:,1])**2
    elif pol =='Y' or pol =='y' or pol == 'YY' or pol == 'yy':
        beamVec = np.abs(jonesTensor[:,2])**2 + np.abs(jonesTensor[:,3])**2
    else:
        raise ValueError(f"Polarisation state {pol} not recognised.")
    
    if normalise:
        beamVec /= np.max(beamVec)

    return beamVec


def analytic_length_func(lam,L=1):

    from scipy.special import sici
    gamma = 0.5772156649 # Eulers constant.

    k = 2*np.pi/lam

    Si,Ci = sici(k*L)
    Si2,Ci2 = sici(2*k*L)

    term = gamma + np.log(k*L) - Ci + 0.5*np.sin(k*L)*(Si2-2*Si) +\
            0.5*np.cos(k*L)*(gamma+np.log(k*L/2) + Ci2 - 2*Ci)

    return term

def analytic_dipole_beam(freq,latRad,L=1,sampling='DH1',lMax=130):
    """
    Physical dipole model.
    """
    lam = c/freq
    k = 2*np.pi/lam
    f = analytic_length_func(lam,L=L)


    if sampling == 'DH1':
        Ncells = int(2*lMax+2)
        raVec = np.linspace(0,2*np.pi,Ncells)
        decVec = np.linspace(-np.pi/2,np.pi/2,Ncells)
    elif sampling == 'DH2':
        raVec = np.linspace(0,2*np.pi,4*lMax+1)
        decVec = np.linspace(-np.pi/2,np.pi/2,2*lMax+1)
    
    raGrid,decGrid = np.meshgrid(raVec,decVec)
    # Calculating the resulting Az and El grids from RA DEC grid, along with 
    # lat.
    azGrid,elGrid = radec2azel(raGrid,decGrid,latRad)
    # Calculating the new direction cosines for the RA DEC grid.
    #lObsVals = np.cos(elGrid)*np.sin(azGrid)
    #mObsVals = np.cos(elGrid)*np.cos(azGrid)

    #elGrid = np.roll(elGrid + np.pi/2,elGrid.shape[1]//2,axis=1)
    #elGrid = np.roll(elGrid,elGrid.shape[1]//2,axis=1)
    #elGrid = elGrid + np.pi/2
    elGrid[elGrid<0] = 0

    #beamTerm = ((np.cos(0.5*k*L*np.cos(elGrid) - \
    #                    np.cos(0.5*k*L)))/np.sin(elGrid))**2
    beamTerm = np.sin(elGrid)**2

    return beamTerm




def MWA_dipolebeam(freq,lat,Ncells,pol='X',dipoleInd=10):
    """
    Function to generate the MWA single dipole mode equatorial cartesian ('CAR') 
    beam using hyperbeam

    Parameters
    ----------
    freq : float
        Frequency in Hz
    lat_rad : float
        Observatory latitude in radians
    Ncells : int
        Ncells
    pol : str
        Polarisation - X or Y. Default is X
    dipoleInd : int, default=10
        The dipole number. Default is 10, checked against the data.

    Returns
    -------
    beam_MWA : numpy array, float
        The (reprojected) interpolated MWA single dipole beam, either X or Y
    """

    ra = np.linspace(0,2*np.pi,Ncells)
    dec = np.linspace(-np.pi/2,np.pi/2,Ncells)

    beamMWAx = np.zeros([Ncells,Ncells])
    beamMWAy = np.zeros([Ncells,Ncells])

    #
    if MWA_BEAM_FILE == "":
        warnings.warn("MWA_BEAM_FILE path not set in config file, " +\
                        "using the analytic MWA beam model instead.")
        beam = mwa_hyperbeam.AnalyticBeam()
    else:
        if os.path.exists(MWA_BEAM_FILE):
            beam = mwa_hyperbeam.FEEBeam(MWA_BEAM_FILE)
        else:
            warnings.warn(f"MWA_BEAM_FILE path {MWA_BEAM_FILE} " +\
                         f"not found, using analytic beam instead.")
            beam = mwa_hyperbeam.AnalyticBeam()


    amps = np.zeros(16)
    amps[int(dipoleInd)] = 1
    delays = np.zeros(16,dtype=np.int16)
    norm_to_zenith=True

    # Refactor this to use beam.calc_jones_array, this will eliminate the need
    # to use nested for loops.
    for raInd in tqdm(range(Ncells)):
        for decInd in range(Ncells):
            az,el = radec2azel(ra[raInd],dec[decInd],lat)
            if el>0*np.pi/180:
                
                jmatrixMWA = beam.calc_jones(az_rad=az,za_rad=np.pi/2-el,
                                             freq_hz=freq,delays=delays,
                                             amps=amps,
                                             norm_to_zenith=norm_to_zenith)
                
                beamMWAx[decInd,raInd] = np.abs(jmatrixMWA[0])**2 + \
                                         np.abs(jmatrixMWA[1])**2
                beamMWAy[decInd,raInd] = np.abs(jmatrixMWA[2])**2 + \
                                         np.abs(jmatrixMWA[3])**2
    

    if pol=='Y' or pol=='y':
        print ("Generated Y pol beam")
        beamMWAy /= np.max(beamMWAy)
        return beamMWAy
    else:
        print ("Generated X pol beam")
        beamMWAx /= np.max(beamMWAx)
        return beamMWAx
 
def beam_vec_calc(interferometer,ra,dec,tgpsVec,beamFilePath=None):
    """
    Function for calculating the beam values for a fixed ra and dec, for a 
    given time series, beam model and interferometer location.

    Parameters
    ----------
    beamFilePath : str
        The SIN projected beam FITS filename
    interferometer : mmode_tools.interferometers.RadioArray
        The interferometer object
    ra : float
        Right Ascension in degrees.
    dec : float
        Declination in degrees.
    tgpsVec : np.ndarray
        Array of GPS format utc time values.
    
    Returns
    -------
    interpolated_beam : numpy array, float
        The (reprojected) interpolated beam
    """
    from scipy.interpolate import RegularGridInterpolator
    from astropy.io import fits
    from astropy.coordinates import EarthLocation
    import astropy.units as u
    from mmode_tools.modelling import radec2lmn
    
    # Getting the array latitude in radians.
    latRad = interferometer.lat
    lonRad = interferometer.lon
    meanHeight = np.mean(interferometer.height)
    interferometerLocation = EarthLocation(lat=latRad*u.deg,lon=lonRad*u.deg,
                                           height=meanHeight*u.m)

    lVec,mVec,_ = radec2lmn(tgpsVec,ra,dec,location=interferometerLocation)
    boolVec = ~(np.isnan(lVec)) # Nan values are below the horizon.

    # Opening the FITS file.
    if beamFilePath is not None:
        with fits.open(beamFilePath) as hdu:
            beamMapFits = hdu[0].data[:,:]
            NAXIS1 = hdu[0].header['NAXIS1']
            NAXIS2 = hdu[0].header['NAXIS2']

        # Creating the direction cosine grids.
        lGrid = np.linspace(-1,1,NAXIS1)
        mGrid = np.linspace(-1,1,NAXIS2)
        # Defining an interpolator relative to the grid.
        beamInterp = RegularGridInterpolator(points=(lGrid,mGrid),
                                            values=beamMapFits)

        beamVals = np.zeros(lVec.size)
        # Calculating the interpolated beam values.
        
        beamVals[boolVec] = beamInterp((mVec[boolVec],lVec[boolVec]))
    else:
        print("Assuming simple dipole beam...")
        nVec = np.sqrt(1-lVec**2 - mVec**2)
        beamVals = nVec**2
        beamVals[~boolVec] = 0
    
    return beamVals

def FITS2beam(beamFilePath,latRad,lMax,sampling='DH1'):
    """
    Function to generate the equatorial cartesian ('CAR') primary beam from a 
    SIN projected beam FITS file. Interpolates !

    Parameters
    ----------
    beamFilePath : str
        The SIN projected beam FITS filename
    latRad : float
        Observatory latitude in radians
    Ncells : int
        Ncells

    Returns
    -------
    interpolated_beam : numpy array, float
        The (reprojected) interpolated beam
    """
    from scipy.interpolate import RegularGridInterpolator

    # Opening the FITS file.
    with fits.open(beamFilePath) as hdu:
        beamMapFits = hdu[0].data[:,:]
        NAXIS1 = hdu[0].header['NAXIS1']
        NAXIS2 = hdu[0].header['NAXIS2']

    # Creating the direction cosine grids.
    lGrid = np.linspace(-1,1,NAXIS1)
    mGrid = np.linspace(-1,1,NAXIS2)
    # Defining an interpolator relative to the grid.
    beamInterp = RegularGridInterpolator(points=(lGrid,mGrid),
                                         values=beamMapFits)
    # Defining the RA and DEC Grids.
    # DH1 and DH2 sampling is now supported. Doesn't change anything.
    if sampling == 'DH1':
        Ncells = int(2*lMax+2)
        raVec = np.linspace(0,2*np.pi,Ncells)
        decVec = np.linspace(-np.pi/2,np.pi/2,Ncells)
    elif sampling == 'DH2':
        raVec = np.linspace(0,2*np.pi,4*lMax+1)
        decVec = np.linspace(-np.pi/2,np.pi/2,2*lMax+1)
    
    raGrid,decGrid = np.meshgrid(raVec,decVec)
    # Calculating the resulting Az and El grids from RA DEC grid, along with 
    # lat.
    azGrid,elGrid = radec2azel(raGrid,decGrid,latRad)
    # Calculating the new direction cosines for the RA DEC grid.
    lObsVals = np.cos(elGrid)*np.sin(azGrid)
    mObsVals = np.cos(elGrid)*np.cos(azGrid)
    # Calculating the interpolated beam values.
    interpolatedBeam = beamInterp((mObsVals,lObsVals))
    
    interpolatedBeam[elGrid<0] = 0


    return interpolatedBeam
