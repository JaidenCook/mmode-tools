import os
import toml
import shutil
import importlib.resources as resources

HOMEDIR = os.path.expanduser("~")
configFile = "default_config.toml"
mmodeConfigPath = "mmode_tools.config"

def ensure_output_dirs():
    # Load the default config file from package resources
    with resources.files(mmodeConfigPath).joinpath(configFile).open("r") as f:
        config = toml.load(f)

    directoryDict = config.get("paths", {})

    for _,path in directoryDict.items():
        if not os.path.exists(HOMEDIR+path):
            os.makedirs(HOMEDIR+path,exist_ok=True)
            print(f"Created directory: {HOMEDIR+path}")


def copy_array_data_files():
    """
    Move data files to the appropriate directories.
    """
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
            destPath = HOMEDIR + interferometrPath + file
            if not os.path.exists(destPath):
                shutil.copy2(srcPath,destPath)
        # Getting the example covariance tensors.
        if file.endswith('.toml') or file.endswith('.h5'):
            srcPath = dataPath.joinpath(file)
            destPath = HOMEDIR + covTensorPath + file
            if not os.path.exists(destPath):
                shutil.copy2(srcPath,destPath)

    # Move the test beams as well. These are just EDA2 beam models.
    for file in ["beam_model_XX_150MHz.fits","beam_model_YY_150MHz.fits"]:
        srcPath = dataPath.joinpath(file)
        destPath = HOMEDIR + beamPath + file
        if not os.path.exists(destPath):
            shutil.copy2(srcPath,destPath)
    