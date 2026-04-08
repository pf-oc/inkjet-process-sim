"""
flight.py
---------
Module 1: Droplet flight, ballistic trajectory with aerodynamic drag.

ORIGINAL (generic fluid dynamics assignment):
    Simulated a liquid sphere falling through air under gravity and
    Stokes drag (F = 3*pi*mu*d*v). Stokes drag is only valid for Re << 1,
    which was fine for the original problem (raindrops, ~2 mm diameter).

    State vector : [x, y, vx, vy]
    Integrator   : scipy.solve_ivp with RK45
    Termination  : fixed time span, no event detection

ADAPTED FOR INKJET (changes marked with [INK]):
    [INK] Replaced Stokes drag with the Schiller-Naumann correlation,
    valid up to Re ~ 800. Inkjet drops (d ~ 20-80 um, v ~ 5-15 m/s)
    have Re ~ 10-100, well outside the Stokes regime. Using Stokes here
    would underestimate drag by ~30% at typical jetting conditions.

    [INK] Added event-based termination: integration stops when the
    droplet hits y = 0. The original just ran for a fixed time and
    you checked the output manually afterwards.

    [INK] Added a Monte Carlo function that runs many trajectories with
    slightly randomised velocity and angle to simulate nozzle-to-nozzle
    variation. The spread of landing positions is the dot placement error,
    which is a standard print quality metric.

    Droplet diameter changed from mm to um scale. Physics unchanged.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

from ink_properties import Ink, RHO_AIR, MU_AIR, G, reynolds


# Color palette for plots, keyed by ink name.
# Kept here rather than on the Ink dataclass to separate physics from plotting.
INK_COLORS = {
    "water_based_low_visc":  "steelblue",
    "water_based_mid_visc":  "navy",
    "water_based_high_visc": "darkblue",
    "uv_curable":            "purple",
}
DEFAULT_COLOR = "steelblue"


def _ink_color(ink: Ink) -> str:
    return INK_COLORS.get(ink.name, DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Drag coefficient
# ---------------------------------------------------------------------------

def drag_coefficient(Re: float) -> float:
    """
    Schiller-Naumann drag correlation for a sphere.

        Cd = (24 / Re) * (1 + 0.15 * Re^0.687)

    [INK] Replaces Stokes drag (Cd = 24/Re) from the original assignment.
    Stokes drag is the Re -> 0 limit of this expression, so the two agree
    at low Re and diverge as Re increases into the inkjet range (~10-100).

    Returns 0 if Re < 1e-10 to avoid division by zero. In practice the
    droplet is never stationary during flight, so this guard rarely triggers.

    Reference: Schiller & Naumann (1933), VDI Zeitung 77:318-320.
    """
    if Re < 1e-10:
        return 0.0
    return (24.0 / Re) * (1.0 + 0.15 * Re ** 0.687)


# ---------------------------------------------------------------------------
# ODE right-hand side
# ---------------------------------------------------------------------------

def flight_ode(t: float, state: np.ndarray, d: float, m: float) -> list:
    """
    Equations of motion for a droplet in 2D (x horizontal, y vertical up).

        m * dvx/dt = -Fd * (vx / |v|)
        m * dvy/dt = -Fd * (vy / |v|) - m*g

    Drag acts opposite to the velocity direction.
    Gravity acts straight down.

    Drag depends on air properties (RHO_AIR, MU_AIR), not ink properties.
    The ink only affects the mass m, which is computed once and passed in.

    Returns d/dt of [x, y, vx, vy].
    """
    x, y, vx, vy = state
    v_abs = np.sqrt(vx**2 + vy**2)

    if v_abs < 1e-12:
        return [vx, vy, 0.0, -G]

    Re = reynolds(RHO_AIR, v_abs, d, MU_AIR)
    Cd = drag_coefficient(Re)
    A  = np.pi * (d / 2.0) ** 2
    Fd = 0.5 * Cd * RHO_AIR * A * v_abs ** 2

    ax = -Fd / m * (vx / v_abs)
    ay = -Fd / m * (vy / v_abs) - G

    return [vx, vy, ax, ay]


def _hit_ground(t, state, *args):
    """
    Event function for solve_ivp. Returns the y-coordinate of the droplet.
    Integration stops when this crosses zero from above (droplet hits substrate).

    [INK] The original assignment had no event detection. It ran for a fixed
    time and you found the landing point by scanning the output array.
    """
    return state[1]

_hit_ground.terminal  = True  # stop on zero crossing
_hit_ground.direction = -1    # only trigger going downward


# ---------------------------------------------------------------------------
# Single-trajectory solver
# ---------------------------------------------------------------------------

def simulate_flight(
    ink: Ink,
    d: float,
    v0: float,
    angle_deg: float,
    x0: float = 0.0,
    y0: float = 1e-3,
    t_max: float = 1e-2,
) -> dict:
    """
    Simulate one droplet from nozzle to substrate.

    Angle convention: angle_deg = 90 means straight down (vx = 0, vy = -v0).
    Values below 90 angle the droplet forward, shifting the landing position.
    This is measured from horizontal, not from vertical, which is not the
    standard physics convention but matches how jetting angles are described
    in inkjet literature.

    Parameters
    ----------
    ink       : ink properties (density, viscosity)
    d         : droplet diameter [m]
    v0        : jetting speed [m/s]
    angle_deg : jetting angle below horizontal [deg], 90 = straight down
    x0, y0    : starting position [m], y0 defaults to 1 mm standoff height
    t_max     : max integration time [s]

    Returns
    -------
    dict with:
        t, x, y, vx, vy  : trajectory arrays
        x_land           : landing x-position [m]
        v_impact         : speed at impact [m/s], passed to Module 2
        Re_impact        : Reynolds number at impact
    """
    angle_rad = np.radians(angle_deg)
    vx0 = v0 * np.cos(angle_rad)
    vy0 = -v0 * np.sin(angle_rad)  # negative because y is up

    m = ink.rho * (4.0 / 3.0) * np.pi * (d / 2.0) ** 3

    sol = solve_ivp(
        flight_ode,
        t_span=(0, t_max),
        y0=[x0, y0, vx0, vy0],
        args=(d, m),
        events=_hit_ground,
        max_step=t_max / 1000,
        rtol=1e-8,
        atol=1e-10,
    )

    x_land    = float(sol.y_events[0][0, 0]) if sol.y_events[0].size > 0 else np.nan
    vx_land   = float(sol.y_events[0][0, 2]) if sol.y_events[0].size > 0 else np.nan
    vy_land   = float(sol.y_events[0][0, 3]) if sol.y_events[0].size > 0 else np.nan
    v_impact  = float(np.sqrt(vx_land**2 + vy_land**2)) if not np.isnan(x_land) else np.nan
    Re_impact = reynolds(ink.rho, v_impact, d, ink.mu) if not np.isnan(v_impact) else np.nan

    return {
        "t":         sol.t,
        "x":         sol.y[0],
        "y":         sol.y[1],
        "vx":        sol.y[2],
        "vy":        sol.y[3],
        "x_land":    x_land,
        "v_impact":  v_impact,
        "Re_impact": Re_impact,
    }


# ---------------------------------------------------------------------------
# [INK] Monte Carlo dot placement error
# ---------------------------------------------------------------------------

def placement_error_distribution(
    ink: Ink,
    d: float,
    v0_mean: float,
    v0_std: float,
    angle_mean_deg: float,
    angle_std_deg: float,
    n_drops: int = 500,
    y0: float = 1e-3,
    seed: int = 42,
) -> np.ndarray:
    """
    [INK] Runs n_drops trajectories with normally distributed velocity and
    angle to simulate variation between nozzles in a printhead. Returns an
    array of landing x-positions.

    The standard deviation of those positions is the dot placement error,
    which sets a lower bound on achievable print resolution. This has no
    equivalent in the original assignment.

    Typical nozzle-to-nozzle variation from literature:
        velocity CV  : 1-3% (coefficient of variation)
        angle spread : +/- 0.5 deg

    Note: this runs a Python loop over n_drops calls to simulate_flight.
    It is slow for large n_drops (>1000). A vectorised version would need
    a different ODE setup.
    """
    rng    = np.random.default_rng(seed)
    v0s    = rng.normal(v0_mean,        v0_std,        n_drops)
    angles = rng.normal(angle_mean_deg, angle_std_deg, n_drops)

    x_lands = np.empty(n_drops)
    for i, (v, a) in enumerate(zip(v0s, angles)):
        res = simulate_flight(ink, d, max(v, 0.1), a, y0=y0)
        x_lands[i] = res["x_land"]

    return x_lands


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_trajectory(result: dict, ink: Ink, d: float, ax=None, label: str = None):
    """Plot a single droplet trajectory."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    color = _ink_color(ink)
    lbl   = label or f"{ink.name}, d={d*1e6:.0f} um"
    ax.plot(result["x"] * 1e3, result["y"] * 1e3, color=color, label=lbl)
    if not np.isnan(result["x_land"]):
        ax.axvline(result["x_land"] * 1e3, color=color, linestyle="--", alpha=0.4)

    ax.set_xlabel("Horizontal position [mm]")
    ax.set_ylabel("Height [mm]")
    ax.set_title("Droplet trajectory")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_placement_error(x_lands: np.ndarray, ink: Ink, ax=None):
    """
    [INK] Histogram of landing positions from placement_error_distribution.
    Shows the scatter across a simulated nozzle population.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    color = _ink_color(ink)
    x_um  = (x_lands - np.nanmean(x_lands)) * 1e6  # centred, converted to um
    ax.hist(x_um, bins=30, color=color, alpha=0.7, edgecolor="white")
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Placement error [um]")
    ax.set_ylabel("Count")
    ax.set_title(f"Dot placement error  |  sigma = {np.nanstd(x_um):.2f} um  |  {ink.name}")
    ax.grid(True, alpha=0.3)
    return ax
