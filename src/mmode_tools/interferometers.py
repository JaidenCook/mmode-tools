import numpy as np
import matplotlib.pyplot as plt
import sys,os
import toml
from mmode_tools.constants import c,MRO,ONSALA
from warnings import warn
import importlib.resources as resources
from mmode_tools.io import get_config_directory

# Getting the output default path.
interferometerPath = get_config_directory(pathName="interferometerPath")

MRO_LAT = MRO.lat.value
ONSALA_LAT = ONSALA.lat.value
MRO_LON = MRO.lon.value
ONSALA_LON = ONSALA.lon.value

class RadioArray:
    """
    Class for defining radio interferometric arrays.

    Attributes
    ----------
    filepath : str, default=filepath
        Attribute set in the child classes.

    Methods
    -------
    apply_flags(self,flags)
        ...
    enh2xyz(self,lat=arrayLat)
        ...
    calc_baseline_matrix(self,HA=H0,dec=arrayLat)
        ...
    get_uvw(self,HA=H0,dec=arrayLat,calc_autos=False)
        ...
    uvw_lam(self,wavelength,uvmax=None)
        ...
    plot_arr(self,uvmax=None,figsize=(10,10),scale=1,figaxs=None,**kwargs)
        ...
    """
    H0 = 0.0 # [deg]
    def __init__(self,filepath=None,eastNorthHeight=None,arrayLat=0,
                 arrayLon=0,telescope=None,antIDs=None):
        # Loading array east north height.
        if eastNorthHeight is not None and filepath is None:
            # If eastNorthHeight is given, then we assume this is a numpy array
            # with the east, north, height coordinates.
            if isinstance(eastNorthHeight, list):
                eastNorthHeight = np.array(eastNorthHeight).T

            self.east = eastNorthHeight[:,0]
            self.north = eastNorthHeight[:,1]
            self.height = eastNorthHeight[:,2]
            if antIDs is None:
                self.antIDs = np.arange(self.east.size).astype(str)
            elif len(antIDs) != len(self.east):
                # Check these are the same size, if not raise a warning,
                # and set the antIDs to simply be the indices.
                warn(f'Length of antIDs != size of east. Using default method.')
                self.antIDs = np.arange(self.east.size).astype(str)
            else:
                self.antIDs = antIDs
        elif filepath is not None:
            filepath = str(filepath) # Ensuring type.
            # Default option where the array information is contained in a 
            # configuration .toml file or the east north height and antenna IDs.
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"File {filepath} does not exist.")
            else:
                # Path to the antenna locations.
                self.path = os.path.dirname(filepath)
                # Getting the file extension. We now support .txt, and .toml 
                # extnsions.
                fileExtension = os.path.splitext(filepath)[1]
                
                if fileExtension == '.txt':
                    self.east,self.north,self.height = \
                        np.loadtxt(filepath,usecols=(1,2,3),unpack=True)
                    self.antIDs = np.loadtxt(filepath,usecols=(0),unpack=True,
                                             dtype=str)
                elif fileExtension == '.toml':
                    with open(filepath, 'r') as f:
                        config = toml.load(f)
                        self.east = np.array(config['layout']['east'])
                        self.north = np.array(config['layout']['north'])
                        self.height = np.array(config['layout']['height'])
                        self.antIDs = config['params']['antIDs']
                        telescope = config['params']['telescope']
                        arrayLat = float(config['location']['lat'])
                        filepath = config['params']['arrayFilePath']
                else:
                    raise ValueError("Unsupported file extension: " +\
                                     f"{fileExtension}. Supported"+\
                                     " extensions are .txt and .toml.")

        # Number of antennas:
        self.Nant = len(self.antIDs)

        # Dictionary containing the antenna IDs and the associated antenna 
        # index.
        self.antDict = {ant: ind for ind,ant in enumerate(self.antIDs)}
        # Saving the telescope name, this is used in naming conventions.
        self.telescope = telescope

        # Shifting the east, north, height, for mean zero.
        self.filepath = filepath
        # Array latitude and longitude in degrees:
        self.lat = arrayLat
        self.lon = arrayLon

    def apply_flags(self,goodInds):
        """
        Flag antennas.

        Parameters
        ----------
        goodInds : numpy array int
            Array of good antenna indices.
        """
        try:
            goodIDsOld = self.goodAntIDs
            goodIndsOld = np.array([self.antDict[ID] for ID in goodIDsOld])
            # Create new flag vector. Unique gets rid of double copies.
            goodInds = np.unique(np.concatenate((goodInds,goodIndsOld)))
            self.east,self.north,self.height = np.loadtxt(self.filepath,
                                                          usecols=(1,2,3),
                                                          unpack=True)
            self.east -= np.nanmean(self.east)
            self.north -= np.nanmean(self.north)
            self.height -= np.nanmean(self.height)
        except AttributeError: 
            # If flagIDs doesn't exist, then apply flags and create the attribute.
            pass
        self.goodAntIDs = self.antIDs[goodInds]
        # Performing the flagging.
        self.east = self.east[goodInds]
        self.north = self.north[goodInds]
        self.height = self.height[goodInds]
        self.Nant = self.east.size

    def enh2xyz(self,lat=None):
        """
        Calculates local X,Y,Z using east,north,height coords,
        and the latitude of the array. Latitude must be in radians
        
        Author: J.Line

        Parameters
        ----------
        lat : float, default=MWA_lat
            Latitude of the array. Must be in radians.

        Returns
        -------
        """
        if lat == None:
            # If None assume zenith.
            lat = self.lat
        lat = np.radians(lat)
        
        sl = np.sin(lat)
        cl = np.cos(lat)
        self.X = -self.north*sl + self.height*cl
        self.Y = self.east
        self.Z = self.north*cl + self.height*sl

    def calc_baseline_matrix(self,HA=H0,dec=None):
        """
        Alternative method for calculating the (u,v,w) values for the baselines.
        This method puts the baselines in the same format as the visibilities.

        Parameters
        ----------
        HA : float, default=H0
            Hour angle of the array/observation. default is the MWA hour angle 
            for a zenith pointed observation.
        dec : float, default=MWA_lat
            Declination of the observation. Default is MWA_lat which indicates a 
            zenith pointed observation.

        Returns
        -------
        """
        x1,x2 = np.meshgrid(self.X,self.X)
        y1,y2 = np.meshgrid(self.Y,self.Y)
        z1,z2 = np.meshgrid(self.Z,self.Z)

        dx = x1-x2
        dy = y1-y2
        dz = z1-z2

        if dec == None:
            # If None assume zenith.
            dec = np.radians(self.lat)
        else:
            dec = np.radians(dec)

        HA = np.radians(HA)

        # Calculating the baseline matrix.
        self.uu_m = np.sin(HA)*dx + np.cos(HA)*dy
        self.vv_m = -np.sin(dec)*np.cos(HA)*dx + np.sin(dec)*np.sin(HA)*dy + \
                    np.cos(dec)*dz
        self.ww_m = np.cos(dec)*np.cos(HA)*dx - np.cos(dec)*np.sin(HA)*dy + \
                    np.sin(dec)*dz
        
    def calc_baseline_length(self,HA=H0):
        """
        This method calculates the projected baseline length in meters. Note
        this has to be done after running the method calc_baseline_matrix.

        Parameters
        ----------

        Returns
        -------
        """

        try:
            self.rr_m = np.sqrt(self.uu_m**2 + self.vv_m**2)
        
        except AttributeError:
            print('No uu_m or vv_m found, running method'+\
                    ' calc_baseline_matrix')
            dec = self.lat
            self.calc_baseline_matrix(HA=HA,dec=dec)
            # Calculating the projected baseline length matrix.
            self.rr_m = np.sqrt(self.uu_m**2 + self.vv_m**2)

    @staticmethod
    def get_baselines(Array,HA=H0,dec=None,calcAutos=False,blineMax=None,
                      blineMin=None,goodInds=None,flagMatrix=None):
        """
        Returns the baselines and antenna pairs in 1D vectors.

        Parameters
        ----------
        Array : RadioArray object
            Input radio array.
        HA : float, default=H0
            Hour angle of the array/observation. default is the MWA hour angle 
            for a zenith pointed observation.
        dec : float, default=MWA_lat
            Declination of the observation. Default is MWA_lat which indicates a 
            zenith pointed observation.
        calcAutos : bool, default=False
            If True return the conjugates and the autos. 
        blineMax : float, default=None
            Thresholding parameter, subsets for baselines below a certain length.
        blineMin : float, default=None
            Thresholding parameter, subsets for baselines above a certain length.

        Returns
        -------
        baselines : np.ndarray, float
            Baselines array.
        antpairs : np.ndarray, float
            Antenna pair array.
        """
        if np.any(flagMatrix):
            # Flag matrix should take precedent over the flagging indices.
            antIndVec = np.arange(Array.Nant)
        else:
            # Initialise the baseline vector.
            if np.any(goodInds):
                #antInd_vec = goodInds
                Array.apply_flags(goodInds)
                antIndVec = np.arange(Array.Nant)
            else:
                antIndVec = np.arange(Array.Nant)
        
        if dec is None:
            dec = Array.lat

        Array.enh2xyz(lat=dec)
        Array.calc_baseline_matrix(HA=HA,dec=dec)
        ant1Arr,ant2Arr = np.meshgrid(antIndVec,antIndVec)

        if calcAutos:
            # If True include the auto correlations in the output. The autos
            # are assumed to not be flagged in the flagMatrix case.
            if np.any(flagMatrix):
                boolVec = flagMatrix
                N = boolVec[boolVec].size
            else:
                # This assumes that antenna flagging has occurred, but should 
                # work if no antennas have been already flagged.
                boolVec = np.ones(Array.uu_m.size).astype(bool)    
        else:
            # Calculate the indices of the auto's and set these to false in the
            # flag matrix. Alternatively use np.diag_indices().
            noAutoInds = np.sqrt(Array.uu_m**2 + Array.vv_m**2) == 0
            if np.any(flagMatrix):
                flagMatrix[noAutoInds] = False
                boolVec = flagMatrix
            else:
                boolVec = noAutoInds == False
            N = boolVec[boolVec].size
            
        blines = np.zeros((N,3))
        antPairs = np.zeros((N,2))
        blines[:,0] = Array.uu_m[boolVec]
        blines[:,1] = Array.vv_m[boolVec]
        blines[:,2] = Array.ww_m[boolVec]
        antPairs[:,0] = ant1Arr[boolVec]
        antPairs[:,1] = ant2Arr[boolVec]

        if np.any(blineMax) or np.any(blineMin):
            baseLen = np.sqrt(np.sum(blines**2,axis=1))
            
            if blineMin == None:
                # Default setting the min baseline length if not provided.
                blineMin = 0
            
            if blineMax == None:
                # Default setting the max baseline length if not provided.
                blineMax = np.max(baseLen)

            blineInds = baseLen <= blineMax
            blineInds *= baseLen > blineMin

            blines = blines[blineInds,:]
            antPairs = antPairs[blineInds,:]

        return blines,antPairs.astype(int)

        
    def plot_uv_dist(self,uvmax=None,figsize=(10,10),scale=1,figaxs=None,
                     flagMatrix=None,fontsize=22,**kwargs):
        """
        Plots the MWA uv sample for a max uv cutoff. Default units are in meters.

        Parameters
        ----------
        uvmax : float, default=None
            Set the uv limits of the plot window.
        figsize : tuple, defualt=(10,10)
            Default window size.
        scale : float, default=1
            Scalar factor that controls the size of the window. 
        figaxs : tuple, default=None
            Tuple containing the fig, and axs matplotlib objects. Useful for
            plotting multiple array layouts onto the same figure object.
        **kwargs :
            Keyword arguments for axs.scatter(**kwargs).
        
        Returns
        -------
        None
        """
        fontsize = fontsize*np.sqrt(scale)
    
        if figaxs:
            # If figure and axis given.
            fig = figaxs[0]
            axs = figaxs[1]
        else:
            fig, axs = plt.subplots(1,figsize=figsize,dpi=75)

        if scale != 1:
            # If scale is not default, rescale the figure size.            
            figx = fig.get_figheight()*scale
            figy = fig.get_figwidth()*scale

            fig.set_figheight(figx)
            fig.set_figwidth(figy)
        try:
            if np.any(flagMatrix):
                uVec = self.uu_m[flagMatrix!=0]
                vVec = self.vv_m[flagMatrix!=0]
            else:
                uVec = self.uu_m
                vVec = self.vv_m
            axs.scatter(uVec,vVec,**kwargs)
            axs.set_xlabel(r'$u\,[m]$',fontsize=fontsize)
            axs.set_ylabel(r'$v\,[m]$',fontsize=fontsize)
        except AttributeError:
            self.enh2xyz()
            self.calc_baseline_matrix()
            axs.scatter(self.uu_m,self.vv_m,**kwargs)
            axs.set_xlabel(r'$u\,[m]$',fontsize=fontsize)
            axs.set_ylabel(r'$v\,[m]$',fontsize=fontsize)

        axs.tick_params('both',labelsize=fontsize)

        if uvmax:
            axs.set_xlim(-uvmax,uvmax)
            axs.set_ylim(-uvmax,uvmax)
        
    def plot_arr_layout(self,rmax=None,figsize=(10,10),scale=1,figaxs=None,
                        fontsize=22,**kwargs):
        """
        Plots the MWA uv sample for a max uv cutoff. Defualt units are in meters.

        Parameters
        ----------
        rmax : float, default=None
            Set the uv limits of the plot window.
        figsize : tuple, defualt=(10,10)
            Default window size.
        scale : float, default=1
            Scalar factor that controls the size of the window. 
        figaxs : tuple, default=None
            Tuple containing the fig, and axs matplotlib objects. Useful for
            plotting multiple array layouts onto the same figure object.
        **kwargs :
            Keyword arguments for axs.scatter(**kwargs).
        
        Returns
        -------
        None
        """
        fontsize = fontsize*np.sqrt(scale)
    
        if figaxs:
            # If figure and axis given.
            fig = figaxs[0]
            axs = figaxs[1]
        else:
            fig, axs = plt.subplots(1,figsize=figsize,dpi=75)

        # If scale is not default, rescale the figure size.            
        figx = fig.get_figheight()*scale
        figy = fig.get_figwidth()*scale

        fig.set_figheight(figx)
        fig.set_figwidth(figy)
    
        axs.scatter(self.east,self.north,**kwargs)
        axs.set_xlabel(r'$east\,[m]$',fontsize=fontsize)
        axs.set_ylabel(r'$north\,[m]$',fontsize=fontsize)

        axs.tick_params('both',labelsize=fontsize)

        if rmax:
            axs.set_xlim(-rmax,rmax)
            axs.set_ylim(-rmax,rmax)

    def generate_config_file(self,outPath=interferometerPath,verbose=False,
                             telescope=None,override=True):
        """
        Wrapper function for make_config_file.

        Parameters
        ----------
        outFilePath : str
            The output file and path location.
        
        Returns
        -------
        None
        """

        if self.telescope is None:
            self.telescope = telescope

        make_config_file(outPath=outPath,Array=self,verbose=verbose,
                         override=override)


def make_config_file(outPath=interferometerPath,Array=None,arrayLocs=None,LAT=0,
                     LON=0,HEIGHT=None,antIDs=None,telescope=None,
                     arrayLayout=None,verbose=False,override=True):
    """
    Function for creating a RadioArray configuration file. 

    Parameters
    ----------
    outFilePath : str
        The output file and path location.
    Array : RadioArray object, default=None
        Preferred configuration input method. This is an mmode_tools RadioArray
        object, and contains all the information to write the output config 
        file. 
    arrayLocs : tuple, default=None
        Tuple containing the east north height arrays or lists. Optionally given
        instead of the Array object.
    LAT : float
        The latitude of the array in degrees.
    LON : float
        The longitude of the array in degrees.
    HEIGHT : float
        The height of the array in meters.
    antIDs : list
        List conatining the IDs for each of the antennas. If not given the 
        IDs are assumed to be integer indices.
    arrayLayout : str, default=None
        Path to the array east north height.
    verbose : bool, default=False
        If given print output information.

    Returns
    -------
    None
    """

    if Array is not None:
        # If array is given use this first.
        east = Array.east
        north = Array.north
        height = Array.height
        LAT = Array.lat
        LON = Array.lon
        telescope = Array.telescope
        antIDs = Array.antIDs
        arrayLocs = None
        Nant = Array.Nant

    elif arrayLocs is not None:
        # Case for no array, but east north and heigh given in tuple.
        east,north,height = arrayLocs
        if antIDs is None:
            antIDs = np.arange(east.size).astype(str)
    
        Nant = east.size

    if telescope is None:
        telescope = f'N{Nant}'
    
    if HEIGHT is None:
        weights = 1/np.sqrt((east-east.mean())**2 + (north-north.mean())**2)
        # Perform weighted average, antennas closer to array centre should 
        # have a higher weighting.
        HEIGHT = np.average(height,weights=weights)
    
    # Create the output file path, use telescope name for naming convention.
    outName = f'{telescope}_config.toml'
    outFilePath = os.path.join(outPath,outName)

    if arrayLayout is None:
        # If this is not given or doesn't exist then this is effectively 
        # encapsulated in the output configuration file.
        arrayLayout = outFilePath
    # Creating the RadioArray configuration dictionary.
    telescopeConfig = {
        "location": {
            "lat": LAT,
            "lon": LON,
            "height": HEIGHT
        },
        "layout": {
            "east": east.tolist(),
            "north": north.tolist(),
            "height": height.tolist()
        },
        "params": {
            "telescope": telescope,
            "antIDs": antIDs.tolist(),
            "override": override,
            "arrayFilePath": arrayLayout
        }
    }

    # Open the file in write mode and dump the data
    with open(outFilePath,"w") as f:
        toml.dump(telescopeConfig, f)
    
    if verbose:
        print(f"Telescope: {telescope}")
        print(f"Latitude: {LAT} [rad]")
        print(f"Longitude: {LON} [rad]")
        print(f"Height: {HEIGHT} [m]")
        print(f"Array Layout File: {arrayLayout}")
        print("Configuration file contents:")
        print(f"Telescope configuration file saved to: {outFilePath}")
    
    return None



def make_radio_array(filePath=None,eastNorthHeight=None,lat=None,lon=None,
                     telescope=None):
    """
    Wrapper function that creates a RadioArray object from a given filePath and
    latitude variable. This then outputs the initialised object ready for use.

    Parameters
    ----------
    filePath : str
        The path to the file containing the array layout.
    lat : float
        The latitude of the array in degrees.

    Returns
    -------
    outClass : RadioArray
        An instance of the RadioArray class with the specified filePath and 
        latitude.
    """
    import warnings
    # If we have a config file, we shouldn't need include the lat as an argument.
    if filePath is not None and os.path.exists(filePath):
        if os.path.splitext(filePath)[1] == '.toml':
            with open(filePath,'r') as f:
                config = toml.load(f)
                lat = config['location']['lat']

    if lat is None:
        warnings.warn("No latitude provided, assuming 0 [deg].")
        lat = 0
    
    if lon is None:
        warnings.warn("No longitude provided, assuming 0 [deg].")
        lon = 0
    
    if eastNorthHeight is None and filePath is None:
        raise ValueError("Either eastNorthHeight or filePath must be provided.")

    class outClass(RadioArray):
        #filepath = str(filePath)
        arrayLat = lat
        arrayLon = lon
        def __init__(self,filepath=filePath,eastNorthHeight=eastNorthHeight,
                     arrayLat=arrayLat,telescope=telescope):
            super().__init__(filepath=filePath,eastNorthHeight=eastNorthHeight,
                             arrayLat=arrayLat,telescope=telescope)
            super().enh2xyz()
            super().calc_baseline_matrix()

    return outClass()


mwaFilepath = interferometerPath + 'antenna_locations_mwa_Feb_24.txt'
eda2Filepath = interferometerPath + 'antenna_locations_eda2.txt'
mwaextFilepath = interferometerPath + 'antenna_locations_mwaPHII_extnd.txt'
onsalaFilepath = interferometerPath + 'antenna_locations_SE607HBA_SEPTON.txt'

# Making child classes for a couple of regularly used arrays.
# These are what is exported.
# These are some useful default instruments that can be used by the user. 
EDA2array = make_radio_array(filePath=eda2Filepath,lat=MRO_LAT,lon=MRO_LON,
                             telescope='EDA2')
MWAPH2array = make_radio_array(filePath=mwaFilepath,lat=MRO_LAT,lon=MRO_LON,
                               telescope='MWA')
MWAEXTarray = make_radio_array(filePath=mwaextFilepath,lat=MRO_LAT,lon=MRO_LON,
                               telescope='MWA_EXT')
ONSALAarray = make_radio_array(filePath=onsalaFilepath,lat=ONSALA_LAT,
                               lon=ONSALA_LON,telescope='ONSALA')