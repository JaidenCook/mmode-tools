import os
import toml
import shutil
import importlib.resources as resources

configFile = "default_config.toml"
mmodeConfigPath = "mmode_tools.config"

def ensure_output_dirs():
    """
    ensure_output_dirs _summary_
    """
    from mmode_tools.io import BASEDIR
    # Load the default config file from package resources
    with resources.files(mmodeConfigPath).joinpath(configFile).open("r") as f:
        config = toml.load(f)
    #
    directoryDict = config.get("paths", {})
    #
    for _,path in directoryDict.items():
        tempPath = BASEDIR/path
        if not tempPath.exists():
            tempPath.mkdir(parents=True,exist_ok=True)
            print(f"Created directory: {tempPath._str}")


def copy_array_data_files():
    """
    Move data files to the appropriate directories.
    """
    from mmode_tools.io import BASEDIR
    with resources.files(mmodeConfigPath).joinpath(configFile).open("r") as f:
        config = toml.load(f)
    
    directoryDict = config.get("paths", {})

    interferometrPath = directoryDict["interferometerPath"]
    covTensorPath = directoryDict["covTensorPath"]
    beamPath = directoryDict["beamPath"]
    dataPath = resources.files('mmode_tools.data')
    for file in os.listdir(dataPath):
        # Example array layout text files.
        if file.endswith('.txt'):
            srcPath = dataPath.joinpath(file)
            destPath = BASEDIR / interferometrPath / file
            if not os.path.exists(destPath):
                shutil.copy2(srcPath,destPath)
        # Example interferometer configuration file.
        if file == "N32_config.toml":
            srcPath = dataPath.joinpath(file)
            destPath = BASEDIR / interferometrPath / file
            if not os.path.exists(destPath):
                shutil.copy2(srcPath,destPath)
        # Getting the example covariance tensors.
        if file.endswith('.toml') or file.endswith('.h5'):
            srcPath = dataPath.joinpath(file)
            destPath = BASEDIR / covTensorPath / file
            if not os.path.exists(destPath):
                shutil.copy2(srcPath,destPath)

    # Move the test beams as well. These are just EDA2 beam models.
    for file in ["beam_model_XX_150MHz.fits","beam_model_YY_150MHz.fits"]:
        srcPath = dataPath.joinpath(file)
        destPath = BASEDIR / beamPath / file
        if not os.path.exists(destPath):
            shutil.copy2(srcPath,destPath)
    