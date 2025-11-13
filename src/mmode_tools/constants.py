from astropy.coordinates import EarthLocation
from astropy import units

c = 299792458
MRO = EarthLocation.of_site('Murchison Widefield Array')
ONSALA = EarthLocation.from_geocentric(3370272.092,712125.596,5349990.934,
                                       unit=units.m)

