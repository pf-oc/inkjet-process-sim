"""
impact.py
---------
Module 2: Droplet impact and spreading.

ORIGINAL (generic fluid dynamics assignment):
    This module did not exist. The original problem ended at landing,
    it only asked for impact velocity and position. A note suggested
    "dot size ~ 2x droplet diameter" with no model behind it.

ADDED FOR INKJET ([INK] throughout):
    Post-impact physics determines print quality. This module models:

    1. Spreading factor beta = D_final / d_droplet using the energy
       balance model of Pasandideh-Fard et al. (1996), which accounts
       for kinetic energy, surface energy, and viscous dissipation.

    2. Contact angle dependence on substrate, which lets you compare
       the same ink on different media.

    3. Satellite drop risk via Rayleigh-Plateau instability of the ink
       ligament, giving a jetting stability diagram in Oh-We space.

References:
    Pasandideh-Fard et al. (1996) Phys. Fluids 8(3):650-659.
    Rayleigh (1878) Proc. London Math. Soc. 10:4-13.
    Hoath (2016) Fundamentals of Inkjet Printing, Wiley-VCH.
"""

import numpy as np
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from ink_properties import Ink, Substrate, weber, reynolds, ohnesorge
from flight import _ink_color


# ---------------------------------------------------------------------------
# [INK] Energy balance spreading model
# ---------------------------------------------------------------------------

def spreading_factor(
    ink: Ink,
    substrate: Substrate,
    d: float,
    v_impact: float,
) -> float:
    """
    [INK] Predicts the maximum spreading factor beta = D_final / d_droplet.

    Uses the energy balance from Pasandideh-Fard et al. (1996):

        KE_initial + SE_initial = SE_final + W_viscous

    Written out:
        (pi/12) rho d^3 v^2  +  pi d^2 sigma
            = (pi/4) beta^2 d^2 sigma (1 - cos theta)
              + (1/4) beta^3 / Re^0.5

    The whole thing is normalised by pi d^2 sigma, which turns it into
    a polynomial in beta that brentq solves numerically.

    Higher impact velocity means more KE, so beta goes up.
    Higher viscosity dissipates more energy, so beta goes down.
    Lower contact angle (more wettable substrate) means beta goes up.

    Raises ValueError if brentq cannot find a root, which would indicate
    the input conditions are outside the model's valid range.

    Parameters
    ----------
    ink       : ink properties
    substrate : substrate contact angle
    d         : droplet diameter at impact [m]
    v_impact  : impact speed from Module 1 [m/s]

    Returns
    -------
    beta : spreading factor, typically 1.5-4 for inkjet
    """
    We    = weber(ink.rho, v_impact, d, ink.sigma)
    Re    = reynolds(ink.rho, v_impact, d, ink.mu)
    theta = np.radians(substrate.theta_eq_deg)

    def residual(beta):
        SE_final = (np.pi / 4.0) * beta**2 * (1.0 - np.cos(theta))
        W_visc   = (1.0 / 4.0) * beta**3 / Re**0.5
        lhs      = We / 12.0 + 1.0
        rhs      = SE_final + W_visc
        return lhs - rhs

    beta_max = max(We**0.5, 5.0)

    try:
        beta = brentq(residual, 1.001, beta_max, xtol=1e-6)
    except ValueError:
        raise ValueError(
            f"spreading_factor: brentq could not find a root for "
            f"We={We:.1f}, Re={Re:.1f}, theta={substrate.theta_eq_deg:.0f} deg. "
            f"Check that impact velocity and ink properties are in a realistic range."
        )

    return float(beta)


def dot_diameter(ink: Ink, substrate: Substrate, d: float, v_impact: float) -> float:
    """
    [INK] Final dot diameter on the substrate [m].

        D_dot = beta * d_droplet

    This is the key print quality output from Module 2.
    """
    return spreading_factor(ink, substrate, d, v_impact) * d


# ---------------------------------------------------------------------------
# [INK] Parameter sweeps
# ---------------------------------------------------------------------------

def sweep_viscosity(
    inks: list,
    substrate: Substrate,
    d: float,
    v_impact: float,
) -> dict:
    """
    [INK] Compute spreading factor for a list of inks at fixed impact
    conditions. Useful for showing how viscosity controls dot size.
    """
    results = {}
    for ink in inks:
        beta = spreading_factor(ink, substrate, d, v_impact)
        results[ink.name] = {
            "beta":     beta,
            "D_dot_um": beta * d * 1e6,
            "We":       weber(ink.rho, v_impact, d, ink.sigma),
            "Re":       reynolds(ink.rho, v_impact, d, ink.mu),
            "Oh":       ohnesorge(ink.mu, ink.rho, ink.sigma, d),
        }
    return results


def sweep_substrate(
    ink: Ink,
    substrates: list,
    d: float,
    v_impact: float,
) -> dict:
    """
    [INK] Compute dot diameter for one ink across multiple substrates.
    Same ink, same jetting conditions, different contact angles.
    """
    results = {}
    for sub in substrates:
        beta = spreading_factor(ink, sub, d, v_impact)
        results[sub.name] = {
            "beta":      beta,
            "D_dot_um":  beta * d * 1e6,
            "theta_deg": sub.theta_eq_deg,
        }
    return results


# ---------------------------------------------------------------------------
# [INK] Rayleigh-Plateau satellite drop analysis
# ---------------------------------------------------------------------------

def satellite_threshold(ink: Ink, d_nozzle: float) -> dict:
    """
    [INK] Estimates satellite drop risk from Rayleigh-Plateau instability
    of the ink ligament formed during jetting.

    Note on applicability: the Rayleigh-Plateau formula strictly applies
    to a free cylindrical jet in the inviscid limit. The ink ligament
    during jetting is not a perfect free jet, so this is an approximation.
    It gives a useful order-of-magnitude estimate and a qualitative
    stability classification, not a precise satellite diameter.

    A cylindrical ligament of radius r is unstable to perturbations with
    wavelength lambda > 2*pi*r. The fastest-growing mode (Rayleigh 1878):

        lambda_max ~ 9.02 * r

    This value (9.02) comes from maximising the growth rate in the
    linearised inviscid stability analysis. It is confirmed by multiple
    sources (MIT OCW 18.357, PNAS 2023, Cambridge JFM 2019).

    If the ligament is long enough to break before it retracts, it
    produces a satellite drop. The satellite volume comes from mass
    conservation assuming the residual ligament collapses into one drop:

        V_sat = pi * r^2 * (L_lig - lambda_max)
        d_sat = (6 * V_sat / pi)^(1/3)

    The ligament length is assumed to be 3 * lambda_max. This is a rough
    estimate with no rigorous derivation behind it. It is chosen to be
    long enough that breakup is possible, and gives a satellite diameter
    in a physically plausible range. Treat d_sat as indicative only.

    The Ohnesorge number determines which regime you are in (Hoath 2016):
        Oh < 0.1  : inertia dominated, satellite risk
        0.1-1.0   : good jetting window
        Oh > 1.0  : too viscous to jet

    Parameters
    ----------
    ink      : ink properties
    d_nozzle : nozzle diameter [m]

    Returns
    -------
    dict with Oh, lambda_max_um, d_sat_um, and a stability label
    """
    r          = d_nozzle / 2.0
    lambda_max = 9.02 * r   # fastest-growing Rayleigh mode, inviscid limit
    Oh         = ohnesorge(ink.mu, ink.rho, ink.sigma, d_nozzle)

    if Oh < 0.1:
        stability = "satellite risk (Oh too low, inertia dominated)"
    elif Oh > 1.0:
        stability = "no jetting (Oh too high, viscosity dominated)"
    else:
        stability = "stable jetting window"

    # L_lig = 3 * lambda_max is a rough assumption, not derived.
    # d_sat should be read as an order-of-magnitude estimate only.
    L_lig = 3.0 * lambda_max
    V_sat = np.pi * r**2 * max(L_lig - lambda_max, 0.0)
    d_sat = (6.0 * V_sat / np.pi) ** (1.0 / 3.0) if V_sat > 0 else 0.0

    return {
        "Oh":            Oh,
        "lambda_max_um": lambda_max * 1e6,
        "d_sat_um":      d_sat * 1e6,
        "stability":     stability,
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_dot_vs_we(ink: Ink, substrate: Substrate,
                   d: float, v_range: np.ndarray, ax=None):
    """[INK] Dot diameter vs. Weber number for one ink/substrate pair."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    We_vals = [weber(ink.rho, v, d, ink.sigma) for v in v_range]
    D_vals  = [dot_diameter(ink, substrate, d, v) * 1e6 for v in v_range]

    ax.plot(We_vals, D_vals, color=_ink_color(ink), linewidth=2,
            label=f"{ink.name} on {substrate.name}")
    ax.axhline(d * 1e6, color="gray", linestyle="--", linewidth=1,
               label="droplet diameter")

    ax.set_xlabel("Weber number  We = rho*v^2*d/sigma")
    ax.set_ylabel("Dot diameter [um]")
    ax.set_title("Dot diameter vs. impact Weber number")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_beta_vs_substrate(results: dict, ax=None):
    """[INK] Bar chart of spreading factor across substrate types."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    names  = list(results.keys())
    betas  = [results[n]["beta"] for n in names]
    thetas = [results[n]["theta_deg"] for n in names]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))
    bars   = ax.bar(names, betas, color=colors, edgecolor="white")

    for bar, theta in zip(bars, thetas):
        ax.text(bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.03,
                f"theta={theta:.0f} deg", ha="center", va="bottom", fontsize=8)

    ax.set_ylabel("Spreading factor  beta = D_dot / d_drop")
    ax.set_title("Dot size vs. substrate (same ink, same jetting conditions)")
    ax.set_ylim(0, max(betas) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    return ax


def plot_oh_we_diagram(inks: list, d_nozzle: float, ax=None):
    """
    [INK] Oh-We stability diagram for a set of inks at a fixed nozzle diameter.
    Each ink appears as a vertical line (Oh is fixed for given ink + nozzle).
    The operating point at v = 8 m/s is marked with a larger dot.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    v_range = np.linspace(1.0, 20.0, 200)

    for ink in inks:
        Oh      = ohnesorge(ink.mu, ink.rho, ink.sigma, d_nozzle)
        We_vals = [weber(ink.rho, v, d_nozzle, ink.sigma) for v in v_range]
        ax.scatter([Oh] * len(We_vals), We_vals, s=2,
                   color=_ink_color(ink), alpha=0.3, label=ink.name)
        We_op = weber(ink.rho, 8.0, d_nozzle, ink.sigma)
        ax.scatter(Oh, We_op, s=80, color=_ink_color(ink),
                   edgecolors="black", zorder=5)

    ax.axhspan(0,   4,   alpha=0.08, color="green",  label="We < 4: low splash risk")
    ax.axhspan(4,   100, alpha=0.05, color="orange", label="We > 4: splash risk")
    ax.axvspan(0,   0.1, alpha=0.08, color="red",    label="Oh < 0.1: satellite risk")
    ax.axvspan(0.1, 1.0, alpha=0.08, color="green")
    ax.axvspan(1.0, 10,  alpha=0.08, color="red",    label="Oh > 1: no jetting")

    ax.set_xlabel("Ohnesorge number  Oh = mu / sqrt(rho*sigma*d)")
    ax.set_ylabel("Weber number  We = rho*v^2*d/sigma")
    ax.set_title(f"Jetting stability diagram  (d_nozzle = {d_nozzle*1e6:.0f} um)")
    ax.set_xscale("log")
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(True, alpha=0.3)
    return ax
