import typer

from .beam_coefficients import beam_coefficients_main
from .telescope_config import telescope_config_main
from .fit_map import fit_map_main

telescopeConfigApp = typer.Typer()
telescopeConfigApp.command()(telescope_config_main)

beamCoApp = typer.Typer()
beamCoApp.command()(beam_coefficients_main)

fitMapApp = typer.Typer()
fitMapApp.command()(fit_map_main)



if __name__ == "__main__":
    telescopeConfigApp()
    beamCoApp()
    fitMapApp()