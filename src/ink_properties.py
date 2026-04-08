"""
ink_properties.py
-----------------
Fluid and substrate property definitions.

ORIGINAL (generic fluid dynamics assignment):
    Fluid was defined as a single dict of arbitrary floats for a
    generic "liquid droplet in air" problem. No physical ink data.
    Substrate was just a flat surface with a fixed contact angle of 90 deg.

ADAPTED FOR INKJET (changes marked with [INK]):
    Replaced generic fluid dict with dataclasses for type safety.

    [INK] Added surface tension, which was not needed in the original
    drag-only model but is required for the spreading and fixation modules.

    [INK] Added substrate contact angle as a named parameter. The original
    assumed theta = 90 deg (neutral wetting) everywhere.

    [INK] Added latex solid content phi0, needed for Module 3.

    [INK] Added preset ink profiles for water-based and UV-curable inks,
    with properties verified against published literature.

Sources used to verify parameter values:
    [S1] Krainer et al. (2019) RSC Advances 9:31708
         https://pubs.rsc.org/en/content/articlelanding/2019/ra/c9ra04993b
    [S2] Fujifilm Dimatix Ink Formulation Tutorial (Cornell NanoScale Facility)
         Surface tension = 32-42 dynes/cm at jetting temperature
         https://www.cnfusers.cornell.edu/sites/default/files/Equipment-Resources/Ink%20formulation%20tutorial.pdf
    [S3] Hoath (2016) Fundamentals of Inkjet Printing, Wiley-VCH
         Viscosity ranges ch.2, Oh jetting window 0.07 < Oh < 1
    [S4] IMI Europe rheology notes
         https://imieurope.com/inkjet-blog/2017/2/28/rheology-of-inkjet-inks
    [S5] Li et al. (2015) BioResources 10(4):8135, uncoated paper contact angle ~32 deg
         https://www.researchgate.net/publication/283648308
    [S6] Filter paper wetting study, coated paper ~75 deg, uncoated ~22 deg
         https://www.researchgate.net/figure/Water-contact-angle-measurements-of-coated-and-uncoated-paper-surface
    [S7] ASTM D5725, papers with contact angles 90-110 deg work best for water-based inks
         https://www.astm.org/d5725-99r08.html
    [S8] PET contact angle ~57 deg for water-based ink on untreated PET
         https://www.researchgate.net/figure/The-contact-angle-of-the-GO-ink-on-different-substrates
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Air properties (unchanged from original, ambient air at 25 C)
# ---------------------------------------------------------------------------

RHO_AIR = 1.184     # kg/m3, density of air at 25 C, 1 atm
MU_AIR  = 1.849e-5  # Pa.s, dynamic viscosity of air at 25 C
G       = 9.81      # m/s2, gravitational acceleration


# ---------------------------------------------------------------------------
# [INK] Ink (fluid) properties
# ---------------------------------------------------------------------------

@dataclass
class Ink:
    """
    Physical properties of an inkjet ink.

    [INK] Replaces the original single-dict fluid definition.
    The original only needed density and viscosity for drag.
    Surface tension and latex solid content are new additions needed
    for the spreading and fixation modules.

    Verified ranges for water-based inkjet inks:
        density        : 1000-1100 kg/m3            (general)
        viscosity      : 3-15 mPa.s jettable range  [S3, S4]
        surface tension: 32-42 mN/m at jetting temp [S2]
        solid content  : 0.05-0.40 by volume        [S3]
    """
    name:  str   = "generic_water_based"
    rho:   float = 1050.0   # kg/m3, ink density
    mu:    float = 8e-3     # Pa.s, dynamic viscosity
    sigma: float = 0.033    # N/m, [INK] surface tension (new)
    phi0:  float = 0.15     # [INK] latex solid volume fraction (new)


@dataclass
class Substrate:
    """
    Physical properties of the print substrate.

    [INK] Entirely new. The original model had no substrate object,
    just a flat wall at y = 0. The contact angle theta_eq is now
    parameterised per substrate type, which lets us compare dot size
    across different media in Module 2.

    Verified equilibrium contact angles for water-based ink on:
        coated paper   : ~75 deg, coating resists wetting [S6]
        uncoated paper : ~30-40 deg, more absorbent       [S5, S6]
        PET film       : ~57 deg, untreated               [S8]
        glass          : ~20 deg, very wettable           (general)

    Note: these are dynamic/initial contact angles measured with
    water or dilute inks. Inks with surfactants will wet more
    aggressively so equilibrium values may be somewhat lower.
    """
    name:         str   = "coated_paper"
    theta_eq_deg: float = 75.0    # degrees, equilibrium contact angle [INK]
    T_platen:     float = 333.15  # K, platen temperature (60 C) [INK]


# ---------------------------------------------------------------------------
# [INK] Preset ink profiles (new, not in original assignment)
# Values verified against [S1], [S2], [S3]
# ---------------------------------------------------------------------------

INKS = {
    "water_based_low_visc": Ink(
        name="water_based_low_visc",
        rho=1020.0, mu=3e-3, sigma=0.035, phi0=0.10,
    ),
    "water_based_mid_visc": Ink(
        name="water_based_mid_visc",
        rho=1050.0, mu=8e-3, sigma=0.032, phi0=0.15,
    ),
    "water_based_high_visc": Ink(
        name="water_based_high_visc",
        rho=1080.0, mu=14e-3, sigma=0.030, phi0=0.20,
    ),
    "uv_curable": Ink(
        name="uv_curable",
        rho=1100.0, mu=12e-3, sigma=0.034, phi0=1.0,  # no solvent, solid after cure
    ),
}

# theta_eq values sourced from [S5], [S6], [S7], [S8]
SUBSTRATES = {
    "coated_paper":   Substrate("coated_paper",   theta_eq_deg=75.0, T_platen=333.15),
    "uncoated_paper": Substrate("uncoated_paper", theta_eq_deg=35.0, T_platen=333.15),
    "pet_film":       Substrate("pet_film",       theta_eq_deg=57.0, T_platen=323.15),
    "glass":          Substrate("glass",          theta_eq_deg=20.0, T_platen=298.15),
}


# ---------------------------------------------------------------------------
# Dimensionless number helpers
# ---------------------------------------------------------------------------

def reynolds(rho: float, v: float, d: float, mu: float) -> float:
    """Re = rho * v * d / mu, ratio of inertial to viscous forces."""
    return rho * v * d / mu


def weber(rho: float, v: float, d: float, sigma: float) -> float:
    """
    We = rho * v^2 * d / sigma, ratio of kinetic energy to surface energy.
    [INK] New, not needed in the original drag-only model.
    """
    return rho * v**2 * d / sigma


def ohnesorge(mu: float, rho: float, sigma: float, d: float) -> float:
    """
    Oh = mu / sqrt(rho * sigma * d), ratio of viscous forces to
    surface tension and inertia combined.
    [INK] New, used for jetting stability and satellite drop analysis.
    """
    return mu / (rho * sigma * d) ** 0.5
