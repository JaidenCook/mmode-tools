import importlib.resources as resources
from mmode_tools.beam import FITS2beam,analytic_dipole_beam
import numpy as np
import os
from mmode_tools.io import get_config_directory

dataPath = resources.files('mmode_tools.data')
siderealDay2SolarDay = 23.9345/24
interferometrPath = get_config_directory(pathName="interferometerPath")
beamFringePath = get_config_directory(pathName="beamFringePath")
covTensorPath = get_config_directory(pathName="covTensorPath")

def load_default_beam_model(LAT=0,pol='I',lMax=541,freq=150e6,
                            simple_dipole=True,L=1):
    """
    Function to load in the test model primary beam.

    Parameters
    ----------
    LAT : float
        Latitude of the array in degrees.
    pol : str
        Instrument polarisation, either 'XX', 'YY' or 'I'.
    lMax : int
        Maximum spherical harmonic degree.

    Returns
    -------
    """

    if pol == 'I':
        if simple_dipole:
            beamMap = analytic_dipole_beam(freq,np.radians(LAT),lMax=lMax,
                                           L=L)
        else:
            beamFileNameXX="beam_model_XX_150MHz.fits"
            beamMapXX = FITS2beam(dataPath.joinpath(beamFileNameXX),
                                  np.radians(LAT),lMax,sampling='DH1')
            beamFileNameYY="beam_model_YY_150MHz.fits"
            beamMapYY = FITS2beam(dataPath.joinpath(beamFileNameYY),
                                  np.radians(LAT),lMax,sampling='DH1')
            
            beamMap = (beamMapYY + beamMapXX)/2
            
            beamMap = beamMap/np.max(np.abs(beamMap))
    
    elif pol == 'XX' or pol == 'xx' or pol == 'X' or pol == 'x':
        if simple_dipole:
            beamMap = analytic_dipole_beam(freq,LAT,lMax=lMax)
        else:
            beamFileNameXX="beam_model_XX_150MHz.fits"
            beamMap = FITS2beam(dataPath.joinpath(beamFileNameXX),
                                np.radians(LAT),lMax,sampling='DH1')
            beamMap = beamMap/np.max(np.abs(beamMap))
    elif pol == 'YY' or pol == 'yy' or pol == 'Y' or pol == 'y':
        if simple_dipole:
            beamMap = analytic_dipole_beam(freq,LAT,lMax=lMax)
        else:
            beamFileNameYY="beam_model_YY_150MHz.fits"
            beamMap = FITS2beam(dataPath.joinpath(beamFileNameYY),
                                np.radians(LAT),lMax,sampling='DH1')
            beamMap = beamMap/np.max(np.abs(beamMap))
    
    return beamMap

def load_model_map(freq=150e6):
    """
    Function to load in the test model map.

    Parameters
    ----------
    freq : float
        Frequency in MHz.

    Returns
    -------
    haslamMap : numpy array
        The scaled Haslam map at the requested frequency.
    """
    
    from astropy.io import fits

    haslamPath = dataPath.joinpath('radio408MHz_fixheader.fits')

    # Frequency scaling.
    scale = (freq/408e6)**(-2.55)
    with fits.open(haslamPath) as hdu:
        haslamMap = hdu[0].data[:,:][0,:,:]*scale
        
    return haslamMap


def load_example_interferometer():
    """
    Loads the example 32 element psuedo random uniform array.
    
    Parameters
    ----------

    Returns
    -------
    interferometer: 
    """
    from mmode_tools.interferometers import make_radio_array

    interferometerFilePath = dataPath.joinpath("N32_config.toml")
    interferometer = make_radio_array(filePath=interferometerFilePath)

    return interferometer

def make_point_covtensor(freq=150e6,raSrc=0,decSrc=0,srcFlux=1.0,lMax=130):
    """
    This function makes the example point source model for the tutorial. This
    is also a useful template for the file layout.
    """
    import datetime
    from astropy.coordinates import EarthLocation
    from astropy import units as u
    from astropy.time import Time
    from mmode_tools.modelling import point_mod2
    from mmode_tools.examples import load_example_interferometer
    from mmode_tools.io import writeCovTensor,write_data_config

    # Loading in the example interferometer.
    interferometer = load_example_interferometer()
    # Getting the telescope name.
    telescopeName = interferometer.telescope

    # Output filepath for the covariance tensor.
    fileName = f"point_source_covtensor_{telescopeName}_150MHz.h5"
    outFilePath = covTensorPath + fileName

    # Checking that the example dataset does not already exist.
    if not os.path.exists(outFilePath):
        # Getting the current date, this is assumed to be the observing time.
        date = datetime.datetime.utcnow()

        Ntimes = int(4*lMax) # Only need to sample in time 2 as many m-modes.
        dtVec = np.linspace(0,1,Ntimes)*siderealDay2SolarDay
        dateVec = [(date+datetime.timedelta(days=dt)).isoformat() \
                   for dt in dtVec]

        arrayLoc = EarthLocation(lat=interferometer.lat*u.deg,
                            lon=interferometer.lon*u.deg,
                            height=np.mean(interferometer.height)*u.m)


        t = Time(dateVec,format='isot',scale='utc',location=arrayLoc)
        lstVec = t.sidereal_time('mean').hour
        tgpsVec = t.gps

        # Making source model for each instrumental polarisation.
        covTensorXX = point_mod2(interferometer,tgpsVec,freq,raSrc,decSrc,
                                 srcFlux)

        # Constructing the full covariance tensor.
        Npol = 4 # Number of instrumental pol, XX,YY,XY,YX ordering.
        covTensor = np.zeros((tgpsVec.size,Npol) + covTensorXX[0,:,:].shape,
                            dtype=np.complex64)

        covTensor[:,0,:,:] = covTensorXX # Assuming identical beams for now.
        covTensor[:,1,:,:] = covTensorXX # Assuming identical beams for now.

        del covTensorXX

        # Antennas to be flagged, this is just an example.
        flagInds = np.array([0,11]).astype(int)
        # Baselines to be flagged, this is just an example.
        flagBlines = np.array([[5,1],[2,3]]).astype(int)

        # If the covariance tensor doesn't exist we will write it out.
        writeCovTensor(lstVec,covTensor,tgpsVec,outFilePath,flagInds=flagInds,
                        flagBlines=flagBlines,overwrite=True)

        # Each instrumental polarisation is treated as a separate entry.
        stokesList = ["XX","YY"]
        arrFilePaths = [outFilePath,outFilePath]
        interferometers = {interferometer.telescope : interferometer,
                        interferometer.telescope : interferometer,}
        telescopes = [interferometer.telescope,interferometer.telescope]
        beamFringeFilePaths = [beamFringePath + "beam_fringe_coeffs-N32-150MHz-I-lMax130.hdf5",
                            beamFringePath + "beam_fringe_coeffs-N32-150MHz-I-lMax130.hdf5"]

        configOutPath = covTensorPath
        configFileName = f"point_source_covtensor_{telescopeName}_150MHz_config.toml"
        configFilePath = os.path.join(configOutPath,configFileName)

        if not os.path.exists(configFilePath):
            write_data_config(configFilePath,arrFilePaths,interferometers,
                              telescopes,stokesList,beamFringeFilePaths,
                              freq=150e6,lMaxList=[int(lMax),int(lMax)])

    return None


def make_haslam_covtensor(freq=150e6,lMax=130):
    """
    This function makes an example covariance tensor using the Haslam map as an
    input model, and using the example interferometer and default dipole beam.
    """

    from mmode_tools.examples import load_model_map,load_default_beam_model
    from pyshtools import SHGrid,SHCoeffs
    import datetime
    from astropy.coordinates import EarthLocation
    from astropy import units as u
    from astropy.time import Time
    from mmode_tools.modelling import point_mod2
    from mmode_tools.examples import load_example_interferometer
    from mmode_tools.io import writeCovTensor,write_data_config
    from tqdm import tqdm

    # Loading in the example interferometer.
    interferometer = load_example_interferometer()
    # Getting the telescope name.
    telescopeName = interferometer.telescope

    # Output filepath for the covariance tensor.
    fileName = f"haslam_covtensor_{telescopeName}_150MHz.h5"
    outFilePath = covTensorPath + fileName

    if not os.path.exists(outFilePath):

        date = datetime.datetime.utcnow()

        Ntimes = int(4*lMax) # Only need to sample in time 2 as many m-modes.
        dtVec = np.linspace(0,1,Ntimes)*siderealDay2SolarDay
        dateVec = [(date+datetime.timedelta(days=dt)).isoformat() \
                   for dt in dtVec]

        arrayLoc = EarthLocation(lat=interferometer.lat*u.deg,
                            lon=interferometer.lon*u.deg,
                            height=np.mean(interferometer.height)*u.m)


        t = Time(dateVec,format='isot',scale='utc',location=arrayLoc)
        lstVec = t.sidereal_time('mean').hour
        tgpsVec = t.gps

        haslamMap = load_model_map(freq)
        mapGrid = SHGrid.from_array(np.array(haslamMap,dtype=np.complex64))
        # Set to zero for the next iteration.
        almArr = mapGrid.expand(normalization='ortho',csphase=-1,
                                lmax_calc=lMax).coeffs

        
        # Constructing the full covariance tensor.
        Nant = interferometer.Nant
        Npol = 4 # Number of instrumental pol, XX,YY,XY,YX ordering.
        covTensor = np.zeros((tgpsVec.size,Npol,Nant,Nant),dtype=np.complex64)
        #if not os.path.exists(beamFringeFilePath):
        from mmode_tools.beam import bline2alm
        from mmode_tools.interferometers import RadioArray

        # Loading in the beam model.
        beamModel = load_default_beam_model(pol='I',lMax=lMax)
        blines,antPairs = RadioArray.get_baselines(interferometer,
                                                    calcAutos=False)
        # Calculating the beam fringe coefficients for each baseline.
        blmCoeffsTensor = bline2alm(blines,beamModel,freq,
                                    interferometer.lat,lMax)
        # l and m coefficient index values.            
        ldegVec = np.arange(lMax+1)
        mmodeVecNeg = -1*np.arange(lMax+1)
        mmodeVecPos = np.arange(lMax+1)

        # Getting the m-mode values for each coefficients.
        _,mmodeGridPos = np.meshgrid(ldegVec,mmodeVecPos,indexing='ij')
        _,mmmodeGridNeg = np.meshgrid(ldegVec,mmodeVecNeg,indexing='ij')
        mmCoArr = np.zeros((2,lMax+1,lMax+1))
        mmCoArr[0,:,:] = mmodeGridPos
        mmCoArr[1,:,:] = mmmodeGridNeg

        
        # The coefficient array is half zeros. 
        boolVec = almArr[:,:lMax+1,:lMax+1] != 0
        # Creating the mmode grid vector.
        mmodeGridVec = mmCoArr[boolVec]
        almVec = almArr[:,:lMax+1,:lMax+1][boolVec]
        blmCoeffsMatrix = np.conj(blmCoeffsTensor[:,boolVec])
        for tind,phi in enumerate(tqdm(2*np.pi*lstVec/24)):
            blmTmp = np.einsum('il,l->il',blmCoeffsMatrix,
                            np.exp(-1j*mmodeGridVec*phi),optimize='greedy')
            visVec = np.einsum('il,l->i',blmTmp,almVec,optimize='greedy')

            # For simplicity we assume XX and YY are the same.
            covTensor[tind,0,antPairs[:,0],antPairs[:,1]] = visVec #XX
            covTensor[tind,1,antPairs[:,0],antPairs[:,1]] = visVec #YY
        
        # Antennas to be flagged, this is just an example.
        flagInds = np.array([0,11]).astype(int)
        # Baselines to be flagged, this is just an example.
        flagBlines = np.array([[5,1],[2,3]]).astype(int)

        # If the covariance tensor doesn't exist we will write it out.
        writeCovTensor(lstVec,covTensor,tgpsVec,outFilePath,flagInds=flagInds,
                       flagBlines=flagBlines,overwrite=True)
        
        # Each instrumental polarisation is treated as a separate entry.
        stokesList = ["XX","YY"]
        arrFilePaths = [outFilePath,outFilePath]
        interferometers = {interferometer.telescope : interferometer,
                        interferometer.telescope : interferometer,}
        telescopes = [interferometer.telescope,interferometer.telescope]
        #dates = ["",""]
        beamFringeFilePaths = [beamFringePath + "beam_fringe_coeffs-N32-150MHz-I-lMax130.hdf5",
                            beamFringePath + "beam_fringe_coeffs-N32-150MHz-I-lMax130.hdf5"]

        configOutPath = covTensorPath
        configFileName = f"haslam_covtensor_{telescopeName}_150MHz_config.toml"
        configFilePath = os.path.join(configOutPath,configFileName)

        if not os.path.exists(configFilePath):
            write_data_config(configFilePath,arrFilePaths,interferometers,
                              telescopes,stokesList,beamFringeFilePaths,
                              freq=150e6,lMaxList=[int(lMax),int(lMax)])

    return None
