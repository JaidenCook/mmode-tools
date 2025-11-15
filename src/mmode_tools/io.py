import numpy as np
import astropy.io.fits as fits
from mmode_tools.beam import radec2azel
from mmode_tools.flag import split_baseline,apply_auto_flags,apply_flags
from tqdm import tqdm
import h5py as h5
from warnings import warn
import toml
import os
from mmode_tools.constants import c,MRO,ONSALA


def get_config_directory(pathName=None):
    """
    Helper function for getting the output paths for the various mmode_tools 
    data products. If pathName=None, then the directory key names and paths are 
    printed. The key names are the inputs to this function.

    Parameters
    ----------
    pathName : str, default=None
        The key name of the path e.g covTensorPath. These are listed in the 
        default_config.toml file in the config directory.
    
    Returns
    -------
    path : str
        Absolute path.
    """
    import importlib.resources as resources

    HOMEDIR = os.path.expanduser("~")
    configFile = "default_config.toml"
    mmodeConfigPath = "mmode_tools.config"

    with resources.files(mmodeConfigPath).joinpath(configFile).open("r") as f:
        config = toml.load(f)
        directoryDict = config.get("paths", {})

    if pathName is not None:
        path = HOMEDIR + directoryDict[pathName]
        return path
    else:
        print("List of mmode_tools directories by key:")
        for pathKey in list(directoryDict.keys()):
            print(f"{pathKey}: " +HOMEDIR+directoryDict[pathKey])
        
        return None

    

def get_blines_auto(ant_locs,bline_max=np.inf):
    """
    Function to generate baselines from an antenna locations (ENU) array. Autos 
    are included.

    Parameters
    ----------
    ant_locs : numpy array, float
        Antenna locations as an Nants x 3 array in East, North, Up format
    bline_max : float
        Maximum baseline to return in metres. Default is infinity, so selects 
        all the baselines

    Returns
    -------
    baselines : numpy array, float
        generated baselines
    antpairs : numpy array, float
        The corresponding antenna pairs (useful to chop the correlation tensor)
    """

    ant_locs = ant_locs - np.mean(ant_locs, axis=0)
    Nants = ant_locs.shape[0]

    blines = np.zeros([int(Nants*(Nants+1)/2), 3])
    antpairs = np.zeros([int(Nants*(Nants+1)/2), 2], dtype=np.int32)
    index = 0
    for ant1 in range(Nants):
        for ant2 in range(ant1, Nants):
            blines[index, :] = ant_locs[ant1, :] - ant_locs[ant2, :]
            antpairs[index,:] = np.array([ant1, ant2])
            index = index+1
    print("Warning. This function does baseline selection!")
    blines_abs = np.sqrt(blines[:,0]**2 + blines[:,1]**2 + blines[:,2]**2)
    short_bline_indices = np.where(blines_abs < bline_max)[0]
    blines = blines[short_bline_indices, :]
    antpairs = antpairs[short_bline_indices, :]
    return blines, antpairs

def writeFITS(map_array,fname,overwrite=False):
    """
    Function to write out Cartesian coordinate FITS maps.

    Parameters
    ----------
    map_array : str
        numpy 2D array with the map
    fname : str
        File path and name
    overwrite : bool
        Flag to enable overwrite

    """
    
    dec_span = map_array.shape[0]-1
    ra_span = map_array.shape[1]-1
    # map_array should be real.
    #hdu = fits.PrimaryHDU(data=np.fliplr(map_array.real),dtype=np.float32)
    hdu = fits.PrimaryHDU(data=np.array(np.fliplr(map_array.real),
                                        dtype=np.float32))

    hdul = fits.HDUList([hdu])
    hdul[0].header['CRVAL1'] = 0
    hdul[0].header['CRPIX1'] = int(ra_span/2)+1
    hdul[0].header['CDELT1'] = -360/(ra_span)
    hdul[0].header['CTYPE1'] = 'RA---CAR'

    hdul[0].header['CRVAL2'] = 0
    hdul[0].header['CRPIX2'] = int(dec_span/2)+1
    hdul[0].header['CDELT2'] = 180/(dec_span)
    hdul[0].header['CTYPE2'] = 'DEC--CAR'
    hdu.writeto(fname,overwrite=overwrite)

def writeCovTensor(lstVec,covTensor,tVecGPS,filePath,flagInds=None,
                   flagBlines=None,stokes=None,overwrite=False,diffVis=False):
    """
    Function for writing the covariance tensor to a HDF5 file format.

    Parameters
    ----------
    lstVec : float, np.ndarray
        Lst vector containing the LST times for each covariance slice.
    covTensor : complex, np.ndarray
        Covariance tensor for the array. Shape [Nlst,Nant,Nant] or 
        [Nlst,Nant,Nant,Npol] if multiple polarisations.
    filepath : str
        File path for outputting the hdf5 file.
    flagInds : numpy array, bool
        Number boolean vector of flag indices.
    flagBlines : list, tuples, default=None
        List of tuples, containing the antenna ID's for a problem baseline.
    stokes : str, default=None
        String defining the instrumental polarisation (XX,YY,XY, or YX) 
        being written out.
    overwrite : bool, default=False
        If True, and data already exists overwrite the data in the hdf5 file.
    diffVis : bool, default=False
        If True, then input covariance tensor is assumed to be difference 
        visibility noise estimate tensor. This is saved to a different data 
        field.
    

    Returns
    -------
    None
    """
    ## TODO: Refactor this function, can be more concise.
    import h5py as h5
    from mmode_tools.flag import make_flag_matrix, append_flags

    if diffVis:
        prefix = 'vis_diff_'
    else:
        prefix = ''

    hf = h5.File(filePath,'a')
    try:
        g1 = hf.create_group('data') 
    except ValueError:
        print('Group data already exists.')
        g1 = hf['data']
        
    if len(covTensor.shape) == 3:
        if stokes:
            if stokes == "XX" or stokes == "xx":
                stokes = "xx"
            elif stokes == "YY" or stokes == "yy":
                stokes = "yy"
            elif stokes == "XY" or stokes == "xy":
                stokes = "xy"
            elif stokes == "YX" or stokes == "yx":
                stokes = "yx"
        else:
            stokes = ''
        
        try:
            g1.create_dataset(f'{prefix}covt{stokes}',data=covTensor)
        except ValueError:
            if overwrite:
                print('Overwriting data values...')
                g1[f'{prefix}covt{stokes}'][...] = covTensor
    elif len(covTensor.shape) == 4:
        print('Full instrumental stokes...')
        # AIPS Memo 117 pg70 polarisation convention for uvfits is xx,yy,xy,yx.
        try:
            g1.create_dataset(f'{prefix}covtxx',data=covTensor[:,0,:,:])
            g1.create_dataset(f'{prefix}covtyy',data=covTensor[:,1,:,:])
            g1.create_dataset(f'{prefix}covtxy',data=covTensor[:,2,:,:])
            g1.create_dataset(f'{prefix}covtyx',data=covTensor[:,3,:,:])
        except ValueError:
            if overwrite:
                print('Overwriting data values...')
                g1[f'{prefix}covtxx'][...] = covTensor[:,0,:,:]
                g1[f'{prefix}covtyy'][...] = covTensor[:,1,:,:]
                g1[f'{prefix}covtxy'][...] = covTensor[:,2,:,:]
                g1[f'{prefix}covtyx'][...] = covTensor[:,3,:,:]
    else:
        # Case where the shape does not match raise this as an error.
        errMsg = f"covTensor.shape length is not equal to 3 or 4:" +\
            f" {covTensor.shape}"
        raise ValueError(errMsg)
    
    try:
        g1.create_dataset('lst',data=lstVec)
        g1.create_dataset('tGPS',data=tVecGPS)
    except ValueError:
        print('tGPS, and lst already exist.')
        if overwrite:
            print('Overwriting data values...')
            g1['lst'][...] = lstVec
            g1['tGPS'][...] = tVecGPS

    hf.close()

    Nant = covTensor.shape[-1]
    print(f"File saved to {filePath}")
    if np.any(flagInds):
        flagMatrix = make_flag_matrix(Nant,flagInds,flagBlines=flagBlines)
        append_flags(filePath,flagMatrix,Nant=Nant,flagBlines=flagBlines,
                     overwrite=overwrite)
        print('Flags written to file...')
    else:
        # If no flag given, check if any antennas are zero.
        if len(covTensor.shape) == 3:
            diagInds = np.diag(covTensor[0,:,:]).real == 0
        elif len(covTensor.shape) == 4:
            diagInds = np.diag(covTensor[0,0,:,:]).real == 0
        # Get the flag inds.
        flagInds = np.arange(Nant)[diagInds]
        
        if len(flagInds) > 1:
            flagMatrix = make_flag_matrix(Nant,flagInds,flagBlines=flagBlines)
        else:
            flagMatrix = make_flag_matrix(Nant,None,flagBlines=flagBlines)

        # Appending the flags.
        append_flags(filePath,flagMatrix,Nant=Nant,flagBlines=flagBlines,
                     overwrite=overwrite)
        print('Flags written to file...')


def read_uvfits(filepath,returnuv=False):
    """
    Function to read uvfits files. This was written to read MWA uvfits after
    calibrated with MIRIAD. Might need updating in the future. 

    Parameters
    ----------
    filepath : uvfits file
        uvfits object.
    returnuv : bool, default=False
        If True return the UU,VV,WW vectors.
    
    Returns
    -------
    tjd : float
        Julian time of the observation, average if more than one time step
        is present.
    ant1Vec : int, ndarray
        Numpy array containing the antenna 1 ID for each baseline.
    ant2Vec : int, ndarray
        Numpy array containing the antenna 2 ID for each baseline.
    visData : ndarray
        Complex array containing the visibility information.
    """
    from astropy.io import fits
    
    c = 299792458 # speed of light m/s
    with fits.open(filepath) as hdu:
        #
        uvTable = hdu[0].data
        visData = uvTable['DATA']
        Ncor = hdu[0].data.size
        #telescope = hdu[0].header['TELESCOP']
        try:
            # Try this first, uvfits has flexibility in the formats of how
            # antenna IDs are stored.
            ant1Vec = uvTable['ANTENNA1'].astype(int)
            ant2Vec = uvTable['ANTENNA2'].astype(int)
        except KeyError:
            baselineIDs = uvTable['BASELINE']
            ant1Vec,ant2Vec = split_baseline(baselineIDs)
        
        # Julian date time.
        tjd = uvTable['DATE']

    if returnuv:
        try:
            # Try this first, uvfits has flexibility in the formats of how
            # antenna IDs are stored.
            UU = uvTable['UU']*c
            VV = uvTable['VV']*c
            WW = uvTable['WW']*c
        except KeyError:
            UU = np.empty(Ncor)
            VV = np.empty(Ncor)
            WW = np.empty(Ncor)
            for i in range(Ncor):
                UU[i] = uvTable[i][0]*c
                VV[i] = uvTable[i][1]*c
                WW[i] = uvTable[i][2]*c
        return tjd,ant1Vec,ant2Vec,visData,UU,VV,WW
    else:
        return tjd,ant1Vec,ant2Vec,visData

def read_uvfits_dates(filepath):
    """
    Function to read uvfits files. Returns the unique dates for the vis data.

    Parameters
    ----------
    filepath : uvfits file
        uvfits object.
    
    Returns
    -------
    tjd : float, ndarray
        Julian times of the observation.
    """
    from astropy.io import fits
    
    with fits.open(filepath) as hdu:
        # Julian date time.
        #print(hdu[0].data['DATE'])
        #tjdVec = np.unique(hdu[0].data['DATE'])
        tjdVec = hdu[0].data['DATE']

    return tjdVec


def make_covtensor(filepath,Nant=None,observatory='MRO'):
    """
    This function takes an input file which contains a list of uvfits files for
    different lst times, covering 24 hours of observations. It then reads in 
    each file, compiling them into a covtensor [Nlst,Nant,Nant,Npol] where Nlst
    is the number of time bines, Nant is the number of antennas, and Npol is the
    number of polarisations.

    Parameters
    ----------
    filepath : str or list
        Path and file name string, containing list of all observation uvfits 
        files. Can also be a list containing path and file names for each obs.
    Nant : int, default=None
        Number of antennas, if not given read from the file.
    observatory : astropy.EarthLocation
        Default location is the MRO, also supports ONSALA as an option.

    Returns
    -------
    lstVec : float, np.ndarray
        Lst vector containing the LST times for each covariance slice.
    tgpsVec : float, np.ndarray
        time vector containing the UTC times for each covariance slice. in GPS
        format.
    covTensor : complex, np.ndarray
        Covariance tensor for the array. Shape [Nlst,Nant,Nant] or 
        [Nlst,Nant,Nant,Npol] if multiple polarisations.
    """
    from astropy.time import Time
    from astropy.io import fits

    if isinstance(filepath,str):
        # If filepath is a string, then load in the files.
        filename = filepath.split('/')[-1]
        path = filepath.split(filename)[0]+"/"
        with open(filepath) as f:
            # Load the lines.
            files = f.readlines()
        with fits.open(files[0][:-1]) as hdu:
            if Nant == None:
                Nant = hdu[1].header['NAXIS2'] # Number of antennas.
            Npol = hdu[0].data['DATA'].shape[-2] # Number of instrument pols.
    elif isinstance(filepath,list):
        # If filepath is a list, then the list should contain all files.
        path = ''
        files = filepath
        with fits.open(files[0][:-1]) as hdu:
            if Nant == None:
                Nant = hdu[1].header['NAXIS2'] # Number of antennas.
            Npol = hdu[0].data['DATA'].shape[-2] # Number of instrument pols.
    else:
        errMsg = "Variable 'filepath' must be of 'list' or 'str' type."
        raise TypeError(errMsg)

    # Selecting the observing site, this is important to get the correct lst.
    if observatory == 'MRO':
        location = MRO
    elif observatory == 'ONSALA':
        location = ONSALA
    else:
        err = "Observing site not supported, choose either 'MRO' or 'ONSALA."
        raise ValueError(err)

    # Performing a check to ensure the number of time steps per file are equal.
    # For now we assume that they are, if they are not we raise an error.
    testFiles = [files[0],files[int(len(files)/2)],files[-1]]
    NtVec = [np.unique(read_uvfits_dates(file[:-1])).size for file in testFiles]
    if NtVec[0] != NtVec[1] or NtVec[0] != NtVec[2]:
        errMsg = f"Number of time steps per file is not equal, this function" +\
                f" assumes the same number of timesteps per file."
        raise ValueError(errMsg)
    else:
        Nt = NtVec[0] # Number of time steps per file. Assumed to be equal.

        if Nt > 1:
            # Create new file list, for each time step, common time steps will
            # have the same file name. Can't think of a better method at the 
            # moment.
            filesNew = []
            for file in files:
                filesNew.extend([file]*int(Nt))
            files = filesNew

    # Number of LST bins.
    Nlst = int(len(files))
    # Initialising the lst array.
    lstVec = np.zeros(Nlst)
    tgpsVec = np.zeros(Nlst,dtype=np.float64)
    
    # Initialising the covTensor
    covTensor = np.zeros([Nlst,Npol,Nant,Nant],dtype=np.complex64)

    # Looping through each file.
    counter = 0 # Loop through the timesteps
    for ind,file in enumerate(tqdm(files)):
        # Creating the temporary filepath.
        tmpFilePath = path + file[:-1]

        # Reading the UVFITS file.
        try:
            if (file == files[ind-1]) and (len(files) > 1):
                counter += 1
            else:
                tjd,ant1Vec,ant2Vec,visdata = read_uvfits(tmpFilePath)
                # converting jd time to LST and assinging to LST vector.
                tjd = np.unique(tjd)
                Ncor = int(ant1Vec.size/Nt) # Number of correlations per time.

                # Getting the lst time.
                obsTime = Time(tjd,format='jd',scale ='utc',location=location)
                lstTime = obsTime.sidereal_time('mean').value
                counter = 0

            # Assigning lst and gps times.
            lstVec[ind] = lstTime[counter]
            tgpsVec[ind] = np.float64(obsTime[counter].gps)

            # Number of visibilities includes autos doesn't include 
            # conjugate terms.
            Nlow = counter*Ncor
            Nhi = (counter+1)*Ncor
            ant1Ind = ant1Vec[Nlow:Nhi]-1
            ant2Ind = ant2Vec[Nlow:Nhi]-1
            for j in range(Npol):
                covTensor[ind,j,ant1Ind,ant2Ind] = visdata[Nlow:Nhi,0,0,0,j,0] + \
                                                   1j*visdata[Nlow:Nhi,0,0,0,j,1]
                # Conjugate term.
                covTensor[ind,j,ant2Ind,ant1Ind] = visdata[Nlow:Nhi,0,0,0,j,0] - \
                                                   1j*visdata[Nlow:Nhi,0,0,0,j,1]
        except OSError:
            # In the event that there is an error set that time steps values to
            # be nan.
            lstVec[ind] = np.nan
            tgpsVec[ind] = np.nan
            covTensor[ind,j,:,:] = np.nan
            
    # For consistency with older code.
    if Npol == 1:
        covTensor = covTensor[0,:,:,:]
    
    return lstVec,tgpsVec,covTensor


def read_LST_VisCube(filepath,returnAutos=False,applyFlags=True,flagMatrix=None,
                     verbose=False,lstSort=True,stokes='xx',diffVis=False,
                     reshape=False):
    """
    Read the visibilities from HDF5 file. If there are flags in the file
    apply the flags to the antennas/visibilities.

    Parameters
    ----------
    filepath : str
        File path and name.
    return_autos : bool, default=False
        If True only return the auto correlations as a function of LST.
    flag_autos : bool, default=False
        If True flag the visibilities or the auto-correlations. Flags must
        exist in the hdf5 file.
    verbose : bool, default=False
        If True print extra information.
    lstSort : bool, default=True
        If True sort LST vector, and the visibility tensor.
    stokes : str, default=xx
        Lowercase instrument polarisation. Choices are xx,xy,yy,yx.
    diffVis : bool, default=False
        If True, then input covariance tensor is assumed to be difference 
        visibility noise estimate tensor. This is saved to a different data 
        field.
    
    Returns
    -------
    lstVec : numpy array, float
        1D numpy array containing the LST values.
    vis_autos_arr : numpy array, float
        2D numpy array containing the autocorrelations for each
        antenna and LST.
    vis_data_cube : numpy array, complex64
        3D numpy array containing the visibilities for each LST.
    """
    if stokes == "XX":
        stokes = 'xx'
    elif stokes == "YY":
        stokes = 'yy'
    elif stokes == "XY":
        stokes = 'xy'
    elif stokes == "YX":
        stokes = 'yx'
    
    if diffVis:
        prefix = 'vis_diff_'
    else:
        prefix = ''

    with h5.File(filepath,'r') as f:
        if verbose:
            print(f.keys())
            print(f['data'].keys())
        
        # LST vector.
        lstVec = f['data']['lst'][:]
        # Check whether single polarisation or whether contains full 
        # polarisation.
        #if len(f['data'].keys()) == 3:
        if (len(f['data'].keys()) == 3) or (len(f['data'].keys()) == 2):
            try:
                # Old covtensors do not have multiple polarisations, this is a 
                # catch term.
                visCube = f[f'data/{prefix}covt'][:,:,:]
            except ValueError:
                visCube = f[f'data/{prefix}covt{stokes}'][:,:,:]
        elif len(f['data'].keys()) > 3:
            # Capitalisation convention is not necessarily standard, corrects 
            # for this input.
            print(f'Loading {stokes} instrumental polarisation.')
            visCube = f[f'data/{prefix}covt{stokes}'][:,:,:]
        else:
            # Raise error if the number of dimensions is not correct.
            Ndims = len(f[f'data/{prefix}covt'].shape)
            errMsg = f'Dimensions of covtensor is {Ndims}' +\
                    ' should be 3 or 4.'
            raise ValueError(errMsg)

        if lstSort:
            # Sorting so the covtensor starts at LST = 0.
            lstSortInd = np.argsort(lstVec)
            lstVec = lstVec[lstSortInd]
            visCube = visCube[lstSortInd,:,:]

        if np.any(flagMatrix):
            # Option for custom flagging matrix. Useful if flags from multiple
            # Polarisations are combined.
            print('Using custom flag matrix...')
            applyFlags = True
        else:
            try:
                flagMatrix = np.array(f['flags/autoFlags'])
            except KeyError:
                print('No flags')
                applyFlags = False

        Nant = visCube.shape[-1]
        if applyFlags:
            try:
                # Seeing if there are any baseline flags, if so we have to try 
                # a different flagging scheme.
                f['flags/autoFlags'].attrs['flag_baselines']
                print('Applying antenna and baseline flags...')
                visCube = apply_flags(visCube,flagMatrix)
            except KeyError:
                # Note apply_auto_flags reshapes the visCube arr.
                visCube = apply_auto_flags(visCube,flagMatrix,Nant=Nant,
                                           reshape=reshape)
            print('Flags applied...')

        if returnAutos:
            # Autos are the diagonals of the visibility matrix.
            autoInds = np.diag_indices(Nant)
            # Getting the visibility autos array.
            visAutosArr = visCube[:,autoInds[0],autoInds[1]].real
            
            if verbose:
                print(f'LST vector size/shape = {lstVec.shape}')
                print(f'Autos array shape = {visAutosArr.shape}')
                print(f'Nant = {Nant}')

            return lstVec,visAutosArr
        else:
            if verbose:
                print(f'LST vector size/shape = {lstVec.shape}')
                print(f'Visibility cube shape = {visCube.shape}')
                print(f'Nant = {Nant}')

            return lstVec,visCube

def get_covt_time(filepath,lstSort=True):
    """
    Basic function that returns the utc gps format time from the covariance 
    tensor.

    Parameters
    ----------
    filepath : str
        File path and name.
    lstSort : bool, default=True
        If True sort LST vector, and the visibility tensor.

    Returns
    -------
    tgpsVec : numpy array, float
        Numpy vector of ut1 gps format times associated with the lst.
    """
    with h5.File(filepath, 'r+') as f:
        tgpsVec = np.copy(f['data/tGPS'])
        if lstSort:
            # Sorting so the covtensor starts at LST = 0.
            lstVec = f['data']['lst'][:]
            lstSortInd = np.argsort(lstVec)
            tgpsVec = tgpsVec[lstSortInd]
    
    return tgpsVec

def read_flags(filepath,verbose=False):
    """
    Simple function that takes input filepath, and returns flags if they exist.

    Parameters
    ----------
    filepath : str
        File path and name.
    
    Returns
    -------
    goodAntInds : numpy array, int
        Numpy vector of good antenna indices.
    flagInds : numpy array, bool
        Number boolean vector of flag indices.
    flagMatrix : numpy array, bool
        2D numpy flag array of shape (Nant,Nant).
    """

    with h5.File(filepath, 'r') as f:
        if verbose:
            print(f.keys())
            print(f['data'].keys())
        try:
            goodAntInds = f['flags/autoFlags'].attrs['good_ant_inds']
            flagInds = f['flags/autoFlags'].attrs['flag_inds']
            flagMatrix = np.array(f['flags/autoFlags'])   
            try:
                flagBlines = f['flags/autoFlags'].attrs['flag_baselines']
            except KeyError:
                flagBlines = None
                print('No baseline flags.')
        except KeyError:
            print('No flags')
            return None,None,None
    if np.any(flagBlines):
        # If there are any individually flagged baselines, return those antenna 
        # pairs.  
        return goodAntInds,flagInds,flagMatrix,flagBlines
    else:
        return goodAntInds,flagInds,flagMatrix,None


def load_beam_fringe_coef(filePath,lmax=130,flagMatrix=None,autos=False,
                          verbose=False,memLim=128):
    """
    This function takes an input filepath to a hdf5 file which contains the 
    beam fringe coefficients for a given telescope. Provided an lmax and a 
    flag matrix for antennas, this returns the corresponding tensor.
    
    Parameters
    ----------
    filepath : str
        File path and name.
    lmax : int, default=130
        Maximum lmode order.
    flagMatrix : np.ndarray dtype=bool, default=None
        Antenna flag matrix.
    autos : bool, default=False
        If True auto-correlations are included. Currently not supported.
    verbose : bool, default=False
        If True print some useful information.
    memLim : int, default=128
        Memory limit in units of GB. If above some threshold, lazy load in the
        data.
    
    Returns
    -------
    almCoeffTensor : np.ndarray dtype=np.complex64
        The beam fringe lm coefficient tensor shape (Nbase,2,lmax+1,lmax+1).
    """
    if not isinstance(lmax,int):
        lmax = int(lmax)

    if np.any(flagMatrix):
        # Default is to assume that there are no flagged antennas.
        Nant = flagMatrix.shape[0]
        flatAutoInds = np.ravel_multi_index(np.diag_indices(Nant),
                                            dims=(Nant,Nant))
        # Matrix needs to be flattened.
        flatFlagMatrix = flagMatrix.flatten()
        if not(autos):
            # If autos not included delete them from the flatFlagMatrix.
            flatFlagMatrix = np.delete(flatFlagMatrix,flatAutoInds)
    
    # Preloading some important info, and performing some necessary checks.
    with h5.File(filePath,'r') as f:
        lmax0 = f['data'].attrs['lMax']
        blineIDs = np.array(f['data']['blineID'])
        if np.any(flagMatrix):
            blineIDs = blineIDs[flatFlagMatrix]
        
        # Check that the lmax is not larger than the max.
        if lmax > lmax0:
            errMsg = f"lmax {lmax} > lmax0 = {lmax0}, existing. Setting lmax " \
                    + f"to {lmax0}"
            warn(errMsg)
            lmax = lmax0

    # Loading in the coefficient tensor.
    with h5.File(filePath,'r') as f:
        dset = f['data']['almCoeffTensor']

        # Calculating the memory allocation.
        memAlloc = blineIDs.size*(lmax+1)*(lmax+1)*8/1024**3 # GB
        if verbose:
            dsetMem = dset.size*8/1024**3
            print(f'lmax= {lmax}')
            print(f'CoeffTensor.shape = {dset.shape}')
            print(f'CoeffTensor size = {dset.nbytes/1024**3:5.3f} GB')
            print(f'Dataset size = {dsetMem:5.3f} GB')
        # If memory allocation larger than available memory, lazy load the data.
        if memAlloc > memLim:
            almCoeffTensor = np.zeros((blineIDs.size,lmax+1,lmax+1),
                                      dtype=np.complex64)
            # Loop through each baseline. Faster if data chunked.
            for i,bind in enumerate(tqdm(blineIDs)):
                for m in range(lmax+1):
                    almCoeffTensor[i,:lmax+1,m] = dset[bind,:lmax+1,m]
        else:
            almCoeffTensor = dset[blineIDs,:lmax+1,:lmax+1]
    
    return almCoeffTensor

def write_data_config(outFilePath,dataFilePaths,interferometers,telescopes,
                      stokesList,beamFringeFilePaths,freq=160e6,dates=None,
                      lMaxList=None):
    """
    Function for writing a data configuration file. These are fed into various
    processing scripts. These take care of the administration of files.

    Parameters
    ----------
    outFilePath : str
        The output file and path location.
    dataFilePaths : list
        List of covtensor file paths.
    interferometers : dict
        Dictionary of the telescope interferometry RadioArray objects.
    telescopes : list
        List of telescopes for each obs.
    stokesList : list
        List of instrumental stokes parameters for each obs.
    beamFringeFilePaths : list
        List of file locations for the beam fringe paths.
    dates : list, default=None
        List of dates for the observations.
    lMaxList : list, default=None
        List of instrumental lMax for each instrument.

    Returns
    -------
    None
    """
    interferometerFilePaths = []
    latList = []
    try:
        for telescope in telescopes:
            interferometer = interferometers[telescope]
            interferometerFilePaths.append(interferometer.filepath)
            latList.append(float(interferometer.lat))
    except KeyError:
        errMsg = f"{telescope} is not a key in the Array dict."
        raise KeyError(errMsg)
    
    if lMaxList is None:
        # If this is not given we calculate the maximum l degree for each of the
        # telescopes and we add this to the lMaxList.
        lam = c/freq
        lMaxList = []
        for telescope in telescopes:
            rMax = np.nanmax(np.sqrt(interferometer.uu_m**2 + \
                                     interferometer.vv_m**2))
            lMaxList.append(int(2*np.pi*rMax/lam)+1)


    # Creating the data configuration dictionary.
    dataConfig = {
        "params": {
            "dates" : dates, # Dates of the observations. Often used in naming.
            "stokes" : stokesList, # The instrumental stokes parameters for each obs.
            "telescopes" : telescopes, # The telescope for each obs.
            "telescope_config_files" : interferometerFilePaths, # Optional location for telescope config files.
            "freq" : freq, # Observation frequency.
            "latitudes" : latList,
            "lMaxList" : lMaxList
        },
        "data" : {
            "file_paths" : dataFilePaths, # Paths to the data files. These are hdf5 files.
        },
        "beams" : { 
            "file_paths" : beamFringeFilePaths, # Paths to the beam fringe coefficients.
        }
    }   

    # Outputting the configuration file to a .toml file.
    with open(outFilePath,"w") as f:
        toml.dump(dataConfig, f)

    print(f"Data config file written to {outFilePath}...")

    return None

def read_data_config(configPath,returnDates=False):
    """
    Read a data configuration file for various calculation scripts. Primarily
    for calculating coefficients for joint observations, and calculating the
    regularisation parameters.

    Parameters
    ----------
    configPath : str
        Configuration file path.
    returnDates : bool, default=False
        If True return the observation dates if they're available.

    Returns
    -------
    arrFilePaths : list
        List of array file paths used to create RadioArray objects.
    Arrays : dict
        Dictionary of the telescope interferometry RadioArray objects.
    telescopes : list
        List of telescopes for each obs.
    stokesList : list
        List of instrumental stokes parameters for each obs.
    beamFringeFilePaths : list
        List of file locations for the beam fringe paths.
    dates : list, optional
        List of dates for the observations.
    """
    from mmode_tools.interferometers import make_radio_array
    
    with open(configPath, 'r') as f:
        config = toml.load(f)
        arrFilePaths = config["data"]["file_paths"]
        beamFringeFilePaths = config["beams"]["file_paths"]
        arrayFilePaths = config["params"]["telescope_config_files"]
        telescopes = config["params"]["telescopes"]
        stokesList = config["params"]["stokes"]
        latList = config["params"]["latitudes"]
        try:
            dates = config["params"]["dates"]
        except KeyError:
            dates = None

    # creating the arrays from the configuration files:
    Arrays = {}
    for i,telescope in enumerate(telescopes):
        arrayFileExtension = os.path.splitext(arrayFilePaths[i])[-1]
        if arrayFileExtension != '.toml':
            lat = latList[i]
            Arrays[f'{telescope}'] = make_radio_array(filePath=arrayFilePaths[i],
                                                    lat=lat,telescope=telescope)
        else:
            Arrays[f'{telescope}'] = make_radio_array(filePath=arrayFilePaths[i])
    
    # Returning the output lists and dictionary.
    if dates is not None and returnDates:
        return arrFilePaths,Arrays,telescopes,stokesList,beamFringeFilePaths,dates
    else:
        return arrFilePaths,Arrays,telescopes,stokesList,beamFringeFilePaths


def map2fits(skyMap,freq,outFilePath,skyCoeffs=None,verbose=False):
    """
    Assumes that the projection is CAR. Should also include sky coefficients,
    for now this is added as an optional keyword argument.

    Parameters
    ----------
    skyMap : np.ndarray np.float64
        Output sky map, shouldt be in Cartesian projection.
    freq : float
        Observation frequency.
    outFilePath : str
        Location path and filename for the output fits file.
    skyCoeffs : np.ndarray np.complex64, default=None
        Sky coefficients.
        
    Returns
    -------
    None
    """
    from astropy.io import fits
    from astropy import wcs
    from astropy.table import Table

    # Initialsing the world coordinate system.
    w = wcs.WCS(naxis=2)

    # Pixel scale in degrees.
    cdelt = 360/skyMap.shape[1]

    # Setting the wcs parameters.
    w.wcs.crpix = [skyMap.shape[1]/2,skyMap.shape[0]/2]
    w.wcs.cdelt = np.array([-cdelt,cdelt])
    w.wcs.crval = [0,0]
    w.wcs.cunit = ["deg","deg"]
    w.wcs.ctype = ["RA---CAR","DEC--CAR"]
    #w.wcs.set_pv([(2, 1, 45.0)])

    # Generating the header.
    header = w.to_header()
    header['FREQ'] = freq

    # Create sample table data
    if skyCoeffs is not None:
        if len(skyCoeffs.shape) == 3:
            # Since the sky is real valued we only need the positive coefficients.
            # We can restore these later.
            skyCoeffs = skyCoeffs[0,:,:]
        skyCoeffsTable = Table({'flat_positive_coeffs_real': skyCoeffs.flatten().real,
                                'flat_positive_coeffs_imag': skyCoeffs.flatten().imag})

    # Create HDUs
    # Sky map has to be flipped about the y axis. 
    skyMapHDU = fits.PrimaryHDU(data=skyMap[::,::-1],header=header)
    skyCoeffsHDU = fits.BinTableHDU(skyCoeffsTable,name='SKY_COEFFS')

    # Create HDUList
    hdul = fits.HDUList([skyMapHDU,skyCoeffsHDU])

    if verbose:
        print("Printing hdu and header information...")
        print("===============================================================")
        print(hdul.info())
        print(repr(skyMapHDU.header))

    # Write to FITS file
    hdul.writeto(outFilePath,overwrite=True)
