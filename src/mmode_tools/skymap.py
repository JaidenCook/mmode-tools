import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pyshtools import SHGrid,SHCoeffs
from mmode_tools.inversion import restore_negmodes
from scipy.ndimage import generic_filter
from warnings import warn
from mmode_tools.plots import coefficient_plot
from mmode_tools.io import fits2skyCoeffs


class SkyMap:
    """ _summary_
    """
    def __init__(self,filePath=None,coeffs=None,skyMap=None,lMax=None,
                 dirty=False,weights=None,
                 model=False,unit='Jy'):
        """__init__ 
        read in HEALPIX style coefficients.
        
        Parameters
        ----------
        filePath : str, optional
            If given this is a FITS file that has been created with the map2fits
            function in mmode_toosl.io.
        coeffs : np.complex64, optional
            Numpy array of sky coefficients, should either be 2 dimensional
            or 3 dimensional. If only 2, this indicates only the positive
            m-modes. Should have shape [2,lMax+1,lMax+1].
        skyMap : _type_, optional
            Optional input instead of coefficients, inpur the sky map, the 
            coefficients are then determined by expanding the map into the 
            coefficient domain, by default None.
        dirty : bool, optional
            If True the coefficients are the dirtyMap coefficients, which are
            the sky-coefficients convolved with the instrument coefficents, 
            by default False.
        weights : np.float32, optional
            These are the regularisation weights, or the expected noise for the
            coefficients (can be both as in the case of the Fisher information).
        model : bool, optional
            If True the map or coefficients are a model, if False then assumed
            to be data. This attribute is used for checking purposes. By 
            default False.

        Raises
        ------
        ValueError
            _description_
        """
        # Initialising all the possible attributes.
        self.skyMap = None
        self.bkgMap = None
        self.windowSizeDeg = None
        self.windowSizePix = None
        self.stdMap = None
        self.coeffsGalactic = None
        self.skyMapGalactic = None
        self.colat = None
        self.colon = None
        self.raVec = None
        self.decVec = None
        self.dirty = dirty
        self.model = model # Is the map a model map if yes True, else False.
        self.unit = unit
        #
        if filePath is not None:
            coeffs,weights=fits2skyCoeffs(filePath,readRegParams=True)


        if coeffs is not None:
            if coeffs.ndim == 1:
                raise ValueError("Coefficient array should have dimension = 3.")
            elif coeffs.ndim == 2:
                # Assume that the coefficients are real valued and we have been 
                # provided the positive m coefficients.
                coeffs = restore_negmodes(coeffs)
        
            self.coeffs = coeffs
            self.lMax = coeffs.shape[-1] - 1

            if skyMap is not None:
                msg = "Coeffs and skyMap both provided. Discarding skyMap " +\
                      "and creating new one from coefficients."
            self.expand_coeffs()
        
        #
        if skyMap is not None:
            self.skyMap = skyMap
            mapPrep = SHGrid.from_array(np.array(self.skyMap,
                                                 dtype=np.complex64))
            # Set to zero for the next iteration.
            coeffs = mapPrep.expand(normalization='ortho',csphase=-1).coeffs
            self.coeffs = coeffs
            self.lMax = coeffs.shape[-1] - 1
            
        #
        if weights is not None:
            # Weights could be the expected noise for the modes, or related to 
            # the regularisation parameter for the modes. 
            if isinstance(weights,np.ndarray):
                if weights.shape != coeffs.shape:
                    msg = "Input weights shape should be equal to coefficient"+\
                          ", setting to None."
                    warn(warn)
                else:
                    self.weights = weights
            elif isinstance(weights,float):
                # If weight is a single value then create weight array of same
                # shape as the coefficients, and set all non-zero entries equal
                # to the input weight. Equivalent to a constant tikhonov factor.
                self.weights = np.zeros_like(self.coeffs)
                self.weights[self.coeffs!=0] = weights
        else:
            self.weights = None

    #
    def __sub__(self,other):
        """__sub__ Defining class subtraction. The coefficients are subtracted,
        and a new object is created.

        Parameters
        ----------
        other : _type_
            _description_
        """
        if isinstance(other,SkyMap):
            if self.dirty != other.dirty:
                raise ValueError("Both sets of coefficients must be dirty or clean.")
            
            if self.coeffs.shape != other.coeffs.shape:
                # Arrays must have the same shape.
                warn("SkyMap objects must have the same coeff shape. Reducing to minimum lMax.")
                lMax = np.min((self.coeffs.shape[-1]-1,
                               other.coeffs.shape[-1]-1))
                coeffsNew = self.coeffs[:,:lMax+1,:lMax+1] - other.coeffs[:,:lMax+1,:lMax+1]
            else:
                coeffsNew = self.coeffs - other.coeffs
            return SkyMap(coeffs=coeffsNew,dirty=self.dirty,model=self.model)
        else:
            raise TypeError("Object should both be SkyMap.")
    
    def __add__(self,other):
        """__sub__ Defining class addition. The coefficients are added,
        and a new object is created.

        Parameters
        ----------
        other : _type_
            _description_
        """
        if isinstance(other,SkyMap):
            if self.dirty != other.dirty:
                raise ValueError("Both sets of coefficients must be dirty or clean.")
            
            if self.dirty == other.dirty:
                if self.coeffs.shape != other.coeffs.shape:
                    # Arrays must have the same shape.
                    warn("SkyMap objects must have the same coeff shape. Reducing to minimum lMax.")
                lMax = np.min((self.coeffs.shape[-1]-1,
                               other.coeffs.shape[-1]-1))
                coeffsNew = self.coeffs[:,:lMax+1,:lMax+1] + other.coeffs[:,:lMax+1,:lMax+1]
            else:
                coeffsNew = self.coeffs + other.coeffs
            return SkyMap(coeffs=coeffsNew,dirty=self.dirty,model=self.model)
        else:
            raise TypeError("Object should both be SkyMap.")

    def lmax_check(self,lMax):
        """lmax_check Checking function. Some method allow for lMax inputs not
        equal to the original set by the data. This must be strictly less than
        the lmax set by the data. This method checks this condition, raises a 
        warning if not met. In the case lMax > self.lMax, lMax is set to 
        self.lMax.

        Parameters
        ----------
        lMax : _type_
            _description_

        Returns
        -------
        _type_
            _description_
        """

        msg = f'Input lMax = {lMax} > than self {self.lMax},' + \
              ' setting to {self.lMax} '
        if lMax is None:
            lMax = self.lMax
        else:
            if lMax > self.lMax:
                warn(msg)
                lMax = self.lMax
        
        return lMax

    def convert2Jy(self,freq=150e6):
        """convert2Jy Converts from units of temperature Sr to units of 
        Jy. This operates in the coefficient domain not the map domain.

        Parameters
        ----------
        freq : float, optional
            Frequency of the coefficients, by default 150e6 Hz.

        Raises
        ------
        ValueError
            _description_
        """
        from mmode_tools.constants import kb,c
        if self.unit != 'K':
            raise ValueError(f"unit value should be in Kelvin not {self.unit}.")
        
        Temp2Jy = 2*kb/(c/freq)**2

        self.coeffs = self.coeffs*Temp2Jy
        self.unit = 'Jy'

        if self.skyMap is not None:
            self.expand_coeffs()
    
    def freq_scale_ceoffs(self,freqOld,freqNew,alpha=-2.55):
        """freq_scale_ceoffs scale the coefficients to a new frequency. Assumes
        that the map scaling is a power law, with a spectral index of -2.55.

        Parameters
        ----------
        freqOld : float
            Old frequency.
        freqNew : float
            New frequency
        alpha : float, optional
            Spectral index for power law scaling, by default -2.55.
        """

        scale = (freqNew/freqOld)**alpha
        self.coeffs = self.coeffs*scale
        if self.skyMap is not None:
            self.expand_coeffs()

    #   
    def expand_coeffs(self,lMax=None,galactic=False):
        """expand_coeffs creates the cartesian map from the spherical harmonic
        coefficients using the pyshtools package.

        Parameters
        ----------
        lMax : _type_, optional
            _description_, by default None
        galactic : bool, optional
            _description_, by default False
        """
        if galactic:
            self.celestial2Galactic()
            # Expanding the sky modes to get the PSF map.
            coeffsObj = SHCoeffs.from_array(self.coeffsGalactic,
                                            normalization='ortho',csphase=-1)
        else:
            coeffsObj = SHCoeffs.from_array(self.coeffs,normalization='ortho',
                                            csphase=-1)
        # Performing check on new lMax input.            
        lMax = self.lmax_check(lMax=lMax)

        if galactic:
            self.skyMapGalactic = coeffsObj.expand(grid='DH2',backend='ducc',
                                                   lmax=lMax).data.real
        else:
            # Setting the colat, colon, RA and DEC grid vectors:
            self.colat = 90-coeffsObj.expand(grid='DH2',backend='ducc',
                                                   lmax=lMax).lats()
            self.colon = 360-coeffsObj.expand(grid='DH2',backend='ducc',
                                                   lmax=lMax).lons()
            self.raVec = coeffsObj.expand(grid='DH2',backend='ducc',
                                                   lmax=lMax).lons()
            self.decVec = coeffsObj.expand(grid='DH2',backend='ducc',
                                                   lmax=lMax).lats()[::-1]
            self.raVec = np.roll(self.raVec,int(self.raVec.size/2))
            self.skyMap = coeffsObj.expand(grid='DH2',backend='ducc',
                                           lmax=lMax).data.real

    def calc_map_mean(self):
        """calc_map_mean calculates the mean sky value, either in Jy/Sr or in 
        Kelvin.
        """
        _,coLatGrid = np.meshgrid(self.colon,self.colat)
        latGrid = coLatGrid - 90

        #dtheta = np.radians(360/latGrid.shape[1])
        dOmega = 4*np.pi/latGrid.size

        if self.skyMap is None:
            self.expand_coeffs()
        
        #meanSky = dtheta**2 *np.sum(self.skyMap*np.cos(np.radians(latGrid)))/np.sqrt(4*np.pi)
        meanSky = dOmega *np.sum(self.skyMap*np.cos(np.radians(latGrid)))#/(4*np.pi)

        if self.unit == 'K':
            print(f"Mean sky temperature = {meanSky:5.3f} [{self.unit}]")
        elif self.unit == 'Jy':
            print(f"Mean sky intensity = {meanSky:5.3f} [{self.unit}/Sr]")



    def calc_mask(self,initalMask=None,DECthresh=(90,-90),maskList=None,
                  GPthresh=0,GPthreshFlip=False,plotCond=False,maskFlip=False):
        """calc_mask _summary_

        Parameters
        ----------
        initalMask : _type_, optional
            _description_, by default None
        DECthresh : tuple, optional
            _description_, by default (90,-90)
        maskList : _type_, optional
            _description_, by default None
        GPthresh : int, optional
            _description_, by default 0
        GPthreshFlip : bool, optional
            _description_, by default False
        plotCond : bool, optional
            _description_, by default False
        maskFlip : bool, optional
            _description_, by default False

        Raises
        ------
        ValueError
            _description_
        """
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        
        if self.skyMap is None:
            self.expand_coeffs()

        RAgrid,DECgrid = np.meshgrid(self.raVec,self.decVec)

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

        self.cleanMask = cleanMask

        # If True plot the clean mask for visual inspection.
        if plotCond:
            self.plot_cart_map(img=cleanMask,
                               cmap='grey',norm='linear',title='Mask')

    #
    def calc_background_map(self,windowSizeDeg=6,lMax=None):
        """calc_background_map _summary_

        Parameters
        ----------
        windowSizeDeg : int, optional
            _description_, by default 6
        lMax : _type_, optional
            _description_, by default None
        """
        
        # Performing check on new lMax input.            
        lMax = self.lmax_check(lMax=lMax)

        if windowSizeDeg is None:
            # If no size given estimate background from whole image.
            if self.skyMap is None:
                # If sky-map is none, expand and save. Need this for the median.
                self.expand_coeffs(lMax=lMax)

            # Calculating the background.
            self.bkgMap = np.median(self.skyMap)*np.ones_like(self.skyMap)
            windowSizePix = None

        else:
            Ncells = self.coeffs.shape[1]*4 + 1
            windowSizePix = int(windowSizeDeg/(360/Ncells)) + 1

            if windowSizePix % 2 == 0:
                windowSizePix += 1

            # Calculating the background estimate. Using a Gaussian to low pass 
            # filter the coefficients. In future can use a different filter.
            # TODO: Make the filter type flexible. Doesn't have to be a 
            # Gaussian.
            lsig = 2*np.pi*(1/(np.radians(360/Ncells)*windowSizePix))
            lVec = np.arange(lMax+1)
            skyCoLPF = np.copy(self.coeffs)
            skyCoLPF[:,:lMax+1,:lMax+1] *= np.exp(-0.5*(lVec/lsig)**2)[None,:,None]
            coeffsObjLFP = SHCoeffs.from_array(skyCoLPF,normalization='ortho',
                                            csphase=-1,lmax=lMax)
            skyMapObjLPF = coeffsObjLFP.expand(grid='DH2',backend='ducc',
                                               lmax=lMax)
            self.bkgMap = skyMapObjLPF.data.real
        # 
        self.windowSizeDeg = windowSizeDeg
        self.windowSizePix = windowSizePix
    
    #
    def calc_std_map(self,windowSizeDeg=6,lMax=None):
        """calc_std_map _summary_

        Parameters
        ----------
        windowSizeDeg : int, optional
            _description_, by default 6
        lMax : _type_, optional
            _description_, by default None
        """

        # Performing check on new lMax input.            
        lMax = self.lmax_check(lMax=lMax)
        
        if self.bkgMap is None:
            # We need the background map to estime the std, if None, then we
            # calculate it.
            self.calc_background_map(windowSizeDeg=windowSizeDeg,lMax=lMax)
        
        
        self.stdMap = calFactor = 1.4826
        diffMap = self.skyMap - self.bkgMap
        self.stdMap =  calFactor*generic_filter(np.abs(diffMap),np.median,
                                                size=self.windowSizePix)

    #
    def celestial2Galactic(self):
        """celestial2Galactic Converts the sky coefficients from a celestial 
        coordinate frame to a Galactic Coordinate frame.
        """

        clat = 27.12825
        clon = (192.85948)
        lNCP = 122.93192
        alpha = clat/2
        beta = -(90.-clat)
        gamma = -lNCP

        sphericalCoeffs = SHCoeffs.from_array(self.coeffs,normalization='ortho',
                                              csphase=-1)
        self.coeffsGalactic = sphericalCoeffs.rotate(alpha,beta,gamma,
                                                     degrees=True).coeffs


    def plot_coefficients(self,lMax=None,figaxs=None,cmap='viridis',
                          norm='log',vmin=None,vmax=None,linear_width=10,
                          plotreal=False,plotimag=False,clab=None,title=None,
                          fullPlot=True,add_contours=False,colorBar=True,
                          fontsize=14,**kwargs):
        """plot_coefficients _summary_

        Parameters
        ----------
        lMax : _type_, optional
            _description_, by default None
        figaxs : _type_, optional
            _description_, by default None
        cmap : str, optional
            _description_, by default 'viridis'
        norm : str, optional
            _description_, by default 'linear'
        vmin : _type_, optional
            _description_, by default None
        vmax : _type_, optional
            _description_, by default None
        linear_width : int, optional
            _description_, by default 10
        plotreal : bool, optional
            _description_, by default False
        plotimag : bool, optional
            _description_, by default False
        clab : _type_, optional
            _description_, by default None
        title : _type_, optional
            _description_, by default None
        fullPlot : bool, optional
            _description_, by default True
        add_contours : bool, optional
            _description_, by default False
        colorBar : bool, optional
            _description_, by default True
        fontsize : int, optional
            _description_, by default 14
        """

        coefficient_plot(self.coeffs,interpolation="None",lmax=lMax,
                         figaxs=figaxs,cmap=cmap,norm=norm,vmin=vmin,vmax=vmax,
                         linear_width=linear_width,plotreal=plotreal,
                         plotimag=plotimag,clab=clab,title=title,
                         fullPlot=fullPlot,add_contours=add_contours,
                         colorBar=colorBar,returnIm=False,fontsize=fontsize,
                         **kwargs)


    # TODO make these functions wrappers for functions in plots. 
    def plot_cart_map(self,img=None,coords=None,figaxs=None,
                      cmap='twilight_shifted',vmin=None,vmax=None,
                      linear_width=None,norm='linear',title=None,patchList=None,
                      lMax=None,**kwargs):
        """plot_cart_map Function plots the cartesian form of the map.

        Parameters
        ----------
        coords : _type_, optional
            _description_, by default None
        figaxs : _type_, optional
            _description_, by default None
        cmap : str, optional
            _description_, by default 'twilight_shifted'
        vmin : _type_, optional
            _description_, by default None
        vmax : _type_, optional
            _description_, by default None
        linear_width : _type_, optional
            _description_, by default None
        norm : str, optional
            _description_, by default 'linear'
        title : _type_, optional
            _description_, by default None
        patchList : _type_, optional
            _description_, by default None
        lMax : _type_, optional
            _description_, by default None
        """
        from matplotlib.colors import AsinhNorm
        from matplotlib.colors import Normalize,LogNorm
        import matplotlib.patches as patches

        # Performing check on new lMax input.            
        lMax = self.lmax_check(lMax=lMax)


        if self.skyMap is None and img is None:
            self.expand_coeffs(lMax=lMax)
            img = self.skyMap
        elif self.skyMap is not None and img is None:
            img = self.skyMap

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
                    vmax = np.nanmax(img)
                if vmin is None:
                    vmin = np.nanmin(img)

                if vmax == vmin:
                    vmin=0
                linear_width = np.abs(vmax-vmin)/100
                norm = AsinhNorm(linear_width=linear_width,vmin=vmin,vmax=vmax)
            else:
                norm = AsinhNorm(linear_width=linear_width,vmin=vmin,vmax=vmax)

        im = axs.imshow(img[:,::-1],norm=norm,
                        cmap=cmap,aspect='auto',origin='lower',**kwargs)
        
        if np.any(coords):
            axs.scatter(img.shape[1]-coords[:,1]-1,coords[:,0],
                        marker='x',color='c',s=10)
        cb = fig.colorbar(im)
        if title:
            axs.set_title(title)
        
        # If there are any flagged regions.
        if patchList:
            Naxis1 = img.shape[1]
            for patch in patchList:
                if len(patch) < 4:
                    winx,winy = patch[2],patch[2]
                elif len(patch) == 4:
                    winx,winy = patch[2],patch[3]
                square = patches.Rectangle((Naxis1-patch[1]-winx-1,patch[0]),
                                           winx,winy,edgecolor='k',
                                           facecolor='none')
            axs.add_patch(square)
    
    #
    def plot_equatorial_map(self,lon=None,lat=None,galactic=False,lMax=None,
                            figsize=(16,10),norm='linear',vmax=None,vmin=None,
                            linear_width=None,projection='mollweide',
                            cmap='twilight_shifted',shading='gouraud',grid=True,
                            fontsize=20,ticks=True,figaxs=None,xticks=False,
                            title=None,transparent=False):
        """plot_equatorial_map _summary_

        Parameters
        ----------
        skyMap : _type_
            _description_
        lon : _type_, optional
            _description_, by default None
        lat : _type_, optional
            _description_, by default None
        figsize : tuple, optional
            _description_, by default (16,10)
        norm : str, optional
            _description_, by default 'linear'
        vmax : _type_, optional
            _description_, by default None
        vmin : _type_, optional
            _description_, by default None
        linear_width : _type_, optional
            _description_, by default None
        projection : str, optional
            _description_, by default 'mollweide'
        cmap : str, optional
            _description_, by default 'twilight_shifted'
        shading : str, optional
            _description_, by default 'gouraud'
        grid : bool, optional
            _description_, by default True
        fontsize : int, optional
            _description_, by default 20
        ticks : bool, optional
            _description_, by default True
        figaxs : _type_, optional
            _description_, by default None
        xticks : bool, optional
            _description_, by default False
        title : _type_, optional
            _description_, by default None
        transparent : bool, optional
            _description_, by default False
        """
        # Get the normalisation.
        from matplotlib.colors import AsinhNorm
        from matplotlib.colors import Normalize,LogNorm

        # Performing check on new lMax input.            
        lMax = self.lmax_check(lMax=lMax)
        img = None
        if galactic:
            if self.skyMapGalactic is None:
                self.expand_coeffs(galactic=True,lMax=lMax)
            img = self.skyMapGalactic
        else:
            if self.skyMap is None:
                self.expand_coeffs(lMax=lMax)
            img = self.skyMap

        if norm == 'linear':
            norm = Normalize(vmin=vmin,vmax=vmax)
        elif norm == 'log':
            norm = LogNorm(vmin=vmin,vmax=vmax)
        elif norm == 'asinh':
            if linear_width==None:
                if vmax is None:
                    vmax = np.nanmax(img)
                if vmin is None:
                    vmin = np.nanmin(img)
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
            lon = np.linspace(-np.pi,np.pi,img.shape[-1])
            lat = np.linspace(-np.pi/2.,np.pi/2.,img.shape[0])

        if img is not None:
            im = axs.pcolormesh(lon,lat,img[:,::-1],cmap=cmap,
                                shading=shading,norm=norm)
        else:
            return None

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


def convolve_model_map(model,weightsTensor,expandMap=True,lMax=None):
    """convolve_model_map _summary_

    Parameters
    ----------
    model : np.ndarray
        Model coefficients or model Cartesian image.
    weightsTensor : np.ndarray, np.complex64
        Weight tensor, should have shape (lMax+1,lMax+1,lMax+1).
    expandMap : bool, optional
        If True also expand the map after convolution, by default True.
    lMax : int, optional
        Maximum spherical harmonic degree to convolve to, by default None

    Returns
    -------
    SkyMap
        New convolved SkyMap object. 

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    if isinstance(model,SkyMap):
        if not(model.dirty):
            # Assume that the model is not already convolved with PSF.
            almModel = model.coeffs[0,:,:]
            almConv = np.zeros_like(almModel)
            weights = model.weights
    elif isinstance(model,np.ndarray):
        if model.ndim == 2:
            mapPrep = SHGrid.from_array(np.array(model,dtype=np.complex64))
            # Set to zero for the next iteration.
            almModel = mapPrep.expand(normalization='ortho',csphase=-1).coeffs
            weights = None
        elif model.ndim == 3:
            # If 3 dimensions assume model is the coefficients.
            almModel = model
            weights = None
        else:
            err = f"model dimension should be 2 or 3 not {model.ndim}."
            raise ValueError(err)

    if lMax is None:
        lMax = almModel.shape[-1] - 1

    # Checking that the arrays are compatible.
    lMaxWeightsTensor = weightsTensor.shape[-1] - 1
    if lMax > lMaxWeightsTensor:
        raise ValueError("Model alm shape mismatch with weights tensor.")
        

    # Performing the convolution.   
    for m, rhsMatrix in enumerate(weightsTensor[:lMax+1,:lMax+1,:lMax+1]):
        almConv[:lMax+1,m] = rhsMatrix @ almModel[:lMax+1,m] 

    # Resotring the negative m-modes.
    almConv = restore_negmodes(almConv[:lMax+1,:lMax+1])

    # Create output dirty model SkyMap object.
    dirtyModel = SkyMap(coeffs=almConv,weights=weights,dirty=True)
    if expandMap:
        dirtyModel.expand_coeffs()

    return dirtyModel

def haslam2pyshtools(filePath,freq=408e6,lmax=570):
    """haslam2pyshtools _summary_

    Parameters
    ----------
    filePath : _type_
        _description_
    freq : _type_, optional
        _description_, by default 408e6
    lmax : int, optional
        _description_, by default 570

    Returns
    -------
    _type_
        _description_
    """
    import healpy as hp
    from mmode_tools.inversion import restore_negmodes

    scale = (freq/408e6)**(-2.55)
    hp_map = hp.read_map(filePath)*scale
    
    alm_hp = hp.map2alm(hp_map,lmax=lmax)
    alm_pyshtools = np.zeros((lmax+1,lmax+1),dtype=complex)

    # Need to first rotate to Celestial coordinates then flip by 180 degrees.
    rot = hp.rotator.Rotator(rot=[0,180],coord=['G','C'])
    alm_hp = rot.rotate_alm(alm_hp,lmax=lmax)
    rot = hp.rotator.Rotator(rot=[180,0])
    alm_hp = rot.rotate_alm(alm_hp,lmax=lmax)

    # Looping through each l and m, and assigning to the pyshtools alm array.
    # We only need to do this for the positive m modes.
    for l in range(lmax+1):
        for m in range(l+1):
            alm_pyshtools[l,m] = alm_hp[hp.Alm.getidx(lmax=lmax,l=l,m=m)]
    
    # Restoring the negative m modes. And performing a parity operation.
    # The parity flip from -m to +m is equivalent to rolling the m-axes.
    rescale = np.sqrt(2) # This is required to match the scales for the PyGSM.
    alm_pyshtools = np.roll(restore_negmodes(alm_pyshtools),1,axis=0)*rescale
    return alm_pyshtools
