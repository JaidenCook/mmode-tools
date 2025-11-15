import numpy as np
import matplotlib.pyplot as plt
import sys,os
import h5py as h5

# Stores the array objects for the different radio telescopes.
from mmode_tools.interferometers import EDA2array

def split_baseline(baselineIDs):
    """
    Function for determining the antenna IDs from the baseline ID. Baseline
    ID is determined by ant1*256 + ant2.

    Parameters
    ----------
    baselineIDs : ndarray
        Numpy array containing the baseline IDs (ant1*256+ant2).

    Returns
    -------
    ant1 : ndarray
        Numpy array containing the the antenna1 ID.
    ant2 : ndarray
        Numpy array containing the the antenna2 ID.
    """    
    if np.max(baselineIDs) >= 65536:
        ant1 = ((baselineIDs - 65536) // 2048).astype(int)
        ant2 = ((baselineIDs - 65536) % 2048).astype(int)
    else:
        ant1 = (baselineIDs // 256).astype(int)
        ant2 = (baselineIDs % 256).astype(int)
    return ant1,ant2


def flag_autos(autosArr,thresh=3,verbose=False,plotFlags=False,
               plot_ants=False,save_plots=None):
    """
    Parameters
    ----------
    
    autos_arr : float, numpy array
        Array containing the autocorrelations for each antenna
        as a function of LST.
    thresh : float, default=3
        Signma threshold from which to flag. Default is set to 3. 
    verbose : bool, default=False
        If True print extra information.
    plotFlags : bool, default=False
        If True plot the flagged antennas.
    plot_ants : bool, default=False
        If True plot the good antennas. 
    save_plots : str, default=None
        Directory where to save the plots.
    
    Returns
    -------
    ant_flag_inds : numpy array, int
        Array of integers containing the flagged antennas.
    ant_good_inds : numpy array, int
        Array of integers containing the good antennas.
    """
    
    from scipy.stats import linregress
    import os

    # Number of antennas.
    Nant = autosArr.shape[1]

    # Summing the autos across LST. Dead antennas will have 
    # a sum of zero.
    autosArrSum = np.nansum(autosArr,axis=0)

    # Creating an antenna index vector.
    antInds = np.arange(Nant)

    # Creating two new index array. Might not need these.
    antZeroInds = antInds[autosArrSum == 0]
    antNonzeroInds = antInds[autosArrSum != 0]

    # Calculating the mean auto-correlation     
    meanAutoVec = np.nanmean(autosArr[:,antNonzeroInds],axis=1)
    autosArrFlagged = autosArr[:,antNonzeroInds]

    # Recalculating the number of antennas.
    NantNew = autosArrFlagged.shape[1]

    ## Rescaling the data.
    # Creating empty array.
    slopeIntArr = np.zeros((NantNew,2))

    for ind in range(NantNew):
        coTemp = np.polyfit(meanAutoVec,autosArrFlagged[:,ind],1)
        slopeIntArr[ind,0] = coTemp[0] # New gains relative to the mean.
        slopeIntArr[ind,1] = coTemp[1] # Mean reciever noise temperature.

    # Calculating the rescaled auto correlations.
    autosRescaledArr = autosArrFlagged/slopeIntArr[None,:,0]-slopeIntArr[None,:,1]

    autosStdVec = np.std(autosRescaledArr,axis=1)
    #
    autoDiff = np.abs(autosRescaledArr-meanAutoVec[:,None])/autosStdVec[:,None]

    # Getting each antenna above the threshold.
    autoDiffThresh = np.any(autoDiff>thresh,axis=0)

    # Getting the unflagged antennas.
    antGoodInds = antNonzeroInds[autoDiffThresh == False]
    antFlagInds = np.hstack((antNonzeroInds[autoDiffThresh],
                             antZeroInds))
    if verbose:
        print('Flagged antennas:')
        print(antFlagInds)
        print(f'Number of flagged antennas = {len(antFlagInds)}')

    if save_plots:
        # Dont plot if saving figs.
        import matplotlib
        matplotlib.use('Agg')

    if plotFlags or save_plots:
        ## Plotting flagged antennas.
        fig,axs = plt.subplots(1,figsize=(12,6),layout='constrained')
        plot_autos(np.linspace(0,24,meanAutoVec.size),
                   autosRescaledArr[:,autoDiffThresh],
                   mean_auto_vec=meanAutoVec,autos_std_vec=autosStdVec,
                   ant_labels=list(antNonzeroInds[autoDiffThresh]),
                   figaxs=(fig,axs))
        plot_autos(np.linspace(0,24,meanAutoVec.size),
                   autosArr[:,antZeroInds] + np.nanmean(meanAutoVec),
                   mean_auto_vec=None,autos_std_vec=None,figaxs=(fig,axs),
                   ant_labels=list(antZeroInds))
        axs.legend(ncol=2,bbox_to_anchor=(1.01,1),fontsize=12)
        axs.set_title('Flagged antennas')

        if save_plots:
            print(save_plots)
            os.makedirs(save_plots,exist_ok=True)
            fig.savefig(f'{save_plots}/Flagged_ants.png',bbox_inches='tight')

    if plot_ants or save_plots:
        fig,axs = plt.subplots(1,figsize=(8,6),layout='constrained')
        plot_autos(np.linspace(0,24,meanAutoVec.size),
                   autosRescaledArr[:,autoDiffThresh == False],
                   mean_auto_vec=meanAutoVec,figaxs=(fig,axs))
        axs.set_title('Unflagged atennas')

        if save_plots:
            os.makedirs(save_plots,exist_ok=True)
            fig.savefig(f'{save_plots}/Unflagged_ants.png',bbox_inches='tight')

    return antFlagInds,antGoodInds


def make_flag_matrix(Nant,flagInds,flagBlines=None):
    """
    Takes input autocorrelation antenna flag indices and returns a flag
    matrix.

    Parameters
    ----------
    Nant : int
        Number of antennas (not the number of flagged antennas.)
    flagInds : ndarray int
        List of flag indices for antennas. Not a list of antenna names. Must be
        zero indexed.
    flagBlines : list, tuples, default=None
        List of tuples, containing the antenna ID's for a problem baseline.


    Returns
    -------    
    flag_matrix : ndarray bool
        Flag matrix, has shape (Nant,Nant).
    """
    # Initialising the flag matrix.
    flagMatrix = np.ones((Nant,Nant),dtype=bool)

    if np.any(flagInds):
        # Creating a flag grid.
        flagxGrid,flagyGrid = np.meshgrid(np.arange(Nant),flagInds)

        # Setting the flag grid to False.
        flagMatrix[flagxGrid,flagyGrid] = False
        flagMatrix[flagyGrid,flagxGrid] = False
    
    # Flagging any problem baselines.
    if np.any(flagBlines):
        if isinstance(flagBlines,list) or isinstance(flagBlines,np.ndarray):
            for antPair in flagBlines:
                flagMatrix[antPair[0],antPair[1]] = False
                flagMatrix[antPair[1],antPair[0]] = False

    return flagMatrix


def write_flags(hf,flagMatrix,flagBlines=None,
                plotFlags=False):
    """
    Helper function for writing auto-correlation flags to hdf5 files. Saves
    copying the code several times.

    This function now additionally can write out individual baseline flages.

    Parameters
    ----------
    hf : hdf5 file
        hdf5 object.
    flag_matrix : ndarray bool
        2D boolean numpy array containing the auto-correlation flags. False 
        where flagged.
    flagBlines : list, tuples, default=None
        List of tuples, containing the antenna ID's for a problem baseline.
    plotFlags : bool, default=False
        If True plot 2D image of the flag matrx. Should be symmetric about the 
        diagonal.


    Returns
    -------
    None
    """
    import datetime

    if flagMatrix.shape[0] != flagMatrix.shape[1]:
        # Performing check to make sure the flag matrix is square.
        errMsg = f'Matrix shape is not square, should be ({Nant},{Nant}).'     
        raise ValueError(errMsg)
    
    Nant = flagMatrix.shape[0]

    # Getting the flag indices, so we can get the flag antenna IDs.
    flagInds = np.arange(Nant)[np.diag(flagMatrix)==False]
    goodInds = np.arange(Nant)[np.diag(flagMatrix)]
    
    if plotFlags:
        plt.imshow(flagMatrix,interpolation='None')
        plt.show()

    try:
        # Try and create the group if it doesn't exits. In future will add more
        # types of flags.
        gflags = hf.create_group('flags')
    except ValueError:
        gflags = hf['flags']

    # Creating dataset for the flag matix.
    flagData = gflags.create_dataset('autoFlags',data=flagMatrix)

    # Assigning metadata to the flags.
    flagData.attrs['flag_inds'] = flagInds
    flagData.attrs['good_ant_inds'] = goodInds
    flagData.attrs['dtype'] = f'dtype : ndarray {np.dtype(flagMatrix[0,0])}'
    flagData.attrs['structure'] = "structure : (Ant1,Ant2)"
    flagData.attrs['shape'] = flagMatrix.shape
    if np.any(flagBlines):
        flagData.attrs['flag_baselines'] = flagBlines

    # Append the date and time when the flags were created. Helps determine if
    # new flags were created or not.
    flagData.attrs['timestamp'] = str(datetime.datetime.now())
    print(flagData.attrs['timestamp'])
    

def append_flags(filepath,flagMatrix,flagBlines=None,
                 overwrite=False):
    """
    Function for appending the autocorrelation flag matrix to the parent hdf5
    file.

    Parameters
    ----------
    filepath : str
        Path and file location.
    flag_matrix : ndarray bool
        2D boolean numpy array containing the auto-correlation flags. False 
        where flagged.

    Returns
    -------
    None
    """
    Nant = flagMatrix.shape[0]
    
    if flagMatrix.shape[0] != flagMatrix.shape[1]:
        # Performing check to make sure the flag matrix is square.
        errMsg = f'Matrix shape is not square, should be ({Nant},{Nant}).'     
        raise ValueError(errMsg)

    with h5.File(filepath,'a') as hf:
        try:
            # Testing to see if the autocorrelation flags exist.
            _ = hf['flags']['autoFlags'].shape
        except KeyError:
            # If they don't exist append them to the file.
            write_flags(hf,flagMatrix,Nant=Nant,flagBlines=flagBlines)
            
            # If there were no prior flags set overwrite to default, incase it
            # was set to True.
            overwrite = False
        
    if overwrite:
        with h5.File(filepath,'a') as hf:
            # If flags exist, and overwrite is True, create new flags.
            del hf['flags']['autoFlags']
            write_flags(hf,flagMatrix,Nant=Nant,flagBlines=flagBlines)


def update_flags(badAnts,filepath,Interferometer,flagBlines=None,clearFlags=False,
                 plotFlags=False,verbose=False):
    """
    Function for updating the flags for EDA2 and MWA covtensor formats. Accepts
    either a file which is a list of Antenna names, or a list/nd.array of 
    antenna names.

    Parameters
    ----------
    badAnts : str or np.ndarray
        Either a filepath containing list of bad tiles names or antennas 
        (not indices) or a numpy array containing the names of bad antennas. 
    filepath : str
        Cov tensor file path and name. 
    Interferometer : RadioArray object
        Either MWAPH2array or EDA2array.
    flagBlines : list, tuples, default=None
        List of tuples, containing the antenna ID's for a problem baseline.
    clearFlags : bool, default=False
        If True clear any existing flags.
    plotFlags : bool, default=False
        If True plot the flagMatrix image.
     
    Returns
    -------
    None
    """
    from mmode_tools.io import read_flags

    Nant = Interferometer.Nant
    ArrDict = Interferometer.antDict
    
    if isinstance(badAnts,str):
        # Testing ifd string is file.
        if os.path.isfile(badAnts):
            flagIDs = np.loadtxt(badAnts,usecols=(0),unpack=True,dtype=str)
            if verbose:
                print('flag IDs:')
                print(flagIDs)
    elif isinstance(badAnts,np.ndarray) or isinstance(badAnts,list):
        # Alternative option is to provide a numpy arry of antenna IDs.
        flagIDs = badAnts
    
    badAntInds = np.array([ArrDict[ant] for ant in flagIDs])

    # Get the old flags.
    if clearFlags:
        # If True clear existing flags and replace with new flags.
        newFlagInds = np.unique(badAntInds)
    else:
        _,flagInds,_,flagBlinesOld = read_flags(filepath)

        # Creating new baselines flagging array. 
        if np.any(flagBlinesOld):
            if len(flagBlinesOld.shape) == 2:
                flagBlines = np.vstack((flagBlinesOld,flagBlines))
            
            # Calculating the baseline ID, and making sure there are no duplicates.
            baseIDvec = flagBlines[:,0]*256 + flagBlines[:,1]
            uniqueInds = np.unique(baseIDvec,return_index=True)[1]
            # Subsetting for the unique baslines.
            flagBlines = flagBlines[uniqueInds,:]

        # Create new flag indices.
        newFlagInds = np.unique(np.concatenate([flagInds,badAntInds]))

    # Create new flag matrix.
    newFlagMatrix = make_flag_matrix(Nant,newFlagInds,flagBlines=flagBlines)

    with h5.File(filepath,'a') as hf:
        try:
            # If there are no auto flags then deleting this group throws a 
            # key error. 
            del hf['flags']['autoFlags']
        except KeyError:
            pass
        write_flags(hf,newFlagMatrix,Nant=Nant,plotFlags=plotFlags,
                    flagBlines=flagBlines)


def apply_auto_flags(visCube,flagMatrix,reshape=True):
    """
    Applies the autocorrelation flags to the visibility data cube. 
    
    Note: this function reshapes the input visCube array, this works because the
    auto flagging, flags individual bad antennas, and not individual baselines.

    Parameters
    ----------
    visCube : ndarray complex64
        Visibility cube (LST,Nant,Nant).
    flagMatrix : ndarray bool
        2D boolean numpy array containing the auto-correlation flags. False 
        where flagged.


    Returns
    -------
    visCube_flagged : ndarray complex64
        Contains only the good antennas (LST,NantGood,NantGood), where 
        NantGood < Nant.
    """

    if visCube[0,:,:].shape != flagMatrix.shape:
        errMsg = f"visCube shape {visCube[0,:,:].shape} " +\
            f"not equal to flagMatrix shape {flagMatrix.shape}"
        raise ValueError(errMsg)
    Nant = visCube.shape[-1]
    NgoodAnts = np.arange(Nant)[np.diag(flagMatrix)].size

    if reshape:
        # Reshaping the visibility cube. 
        visCubeFlagged = visCube[:,flagMatrix].reshape(visCube.shape[0],
                                                       NgoodAnts,
                                                       NgoodAnts)
    else:
        visCubeFlagged = np.copy(visCube)
        visCubeFlagged[:,flagMatrix==False] = np.nan
        
    return visCubeFlagged
    
def apply_flags(visCube,flagMatrix):
    """
    Applies flags to the visCube, the resultant flagged viscube will have nan
    values in for flagged baselines. All good baselines can be determined using
    np.isnan(visCubeFlagged[0,:,:])==False.

    Note: This function does not change the shape of the inpu array visCube.

    Parameters
    ----------
    visCube : ndarray complex64
        Visibility cube (LST,ant1,ant2).
    flagMatrix : ndarray bool
        2D boolean numpy array containing the auto-correlation flags. False 
        where flagged.

    Returns
    -------
    visCube_flagged : ndarray complex64
        Contains only the good antennas (LST,ant1_good,ant2_good), where 
        ant1_good < ant1.
    """
    if visCube[0,:,:].shape != flagMatrix.shape:
        errMsg = f"visCube shape {visCube[0,:,:].shape} " +\
            f"not equal to flagMatrix shape {flagMatrix.shape}"
        raise ValueError(errMsg)
    
    visCubeFlagged = np.copy(visCube)
    visCubeFlagged[:,flagMatrix==False] = np.nan

    return visCubeFlagged


def plot_autos(lstVec,autos_arr,figaxs=None,mean_auto_vec=None,
               autos_std_vec=None,ant_labels=None,fontsize=20,**kwargs):
    """
    Plot the auto-correlations.

    Parameters
    ----------
    lstVec : float, numpy array
        Vector containing the LST values.
    autos_arr : float, numpy array
        Array containing the autocorrelations for each antenna
        as a function of LST.
    figaxs : tuple, default=None
        Tuple containing the figure and axs objects. Makes subplotting
        easier.
    mean_auto_vec : float, numpy array, default=None
        If given plot the mean auto vec.
    autos_std_vec : float, numpy array, default=None
        If autos std vector given plot the mean autos with Nsigma=3
        area.
    ant_labels : bool,default=True
        If True include the labels in the legend. TODO: make this a vector
        which contains all the actual antenna labels, these are different
        to the array indices. 
    """
    fontscale = fontsize/20

    if len(autos_arr.shape) > 1:
        Nant = autos_arr.shape[1]
        single_plot = False
    else:
        Nant = 1
        single_plot = True

    if figaxs:
        axs = figaxs[1]
    else:
        _,axs = plt.subplots(1,figsize=(8,5))

    ls = '-'
    label = None
    for ind in range(Nant):
        if ind > 10:
            ls = '--'
        if ind > 20:
            ls = '-.'
        if ind > 30:
            ls = ':'
        
        if ant_labels:
            if isinstance(ant_labels[ind],int):
                label = f'Ant ID = {ant_labels[ind]+1}'
            elif isinstance(ant_labels[ind],(str,np.str_)):
                label = f'Ant ID = {ant_labels[ind]}'
            else:
                label = None

        if single_plot:
            # Only one auto in the array.
            axs.plot(lstVec,autos_arr,ls=ls,label=label,**kwargs)
        else:
            axs.plot(lstVec,autos_arr[:,ind],ls=ls,label=label,**kwargs)

    if np.any(mean_auto_vec) or np.any(autos_std_vec):
        # If True calculate the mean autos and plot.
        axs.plot(lstVec,mean_auto_vec,
                label=f'Avg',ls='--',color='k',lw=3,zorder=1e6)
    
        if np.any(autos_std_vec):
            # If there is a std vector plot the std about the mean.
            Nsigma = 3
            axs.fill_between(lstVec,mean_auto_vec-Nsigma*autos_std_vec,
                             mean_auto_vec+Nsigma*autos_std_vec,
                             label=fr'${Nsigma}\sigma$',
                             alpha=0.4,zorder=1e5,color='grey')

    axs.set_xlabel('LST [hours]',fontsize=fontsize)
    axs.set_ylabel('Amplitude',fontsize=fontsize)

    if label:
        ncol = int(np.ceil(Nant*fontscale/16))
        axs.legend(ncol=ncol,bbox_to_anchor=(1.01,1),fontsize=12*fontscale)

    [x.set_linewidth(2.) for x in axs.spines.values()]

    #axs.set_xlim([-0.1,24.1])
    axs.set_yscale('log')
    axs.tick_params(axis='x',labelsize=18*fontscale)
    axs.tick_params(axis='y',labelsize=18*fontscale)

