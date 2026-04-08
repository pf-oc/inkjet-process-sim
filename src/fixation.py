"""
fixation.py
-----------
Module 3: Fixation, solvent evaporation and latex film formation.

ORIGINAL (generic fluid dynamics assignment):
    This module has no equivalent in the original assignment. The
    original problem ended at droplet impact.

ADDED FOR INKJET ([INK] throughout):
    After a water-based ink droplet spreads on a heated substrate:

    1. Water evaporates, driven by the vapour pressure difference
       between the hot film surface and the ambient air.
    2. Latex particle volume fraction phi increases as the volume drops.
    3. When phi exceeds close-packing (~0.64), film formation begins
       as capillary forces drive latex particle coalescence.
    4. The film is considered fixed when phi reaches 1 (all solvent gone).

    Evaporation rate uses the uniform-rate approximation with the Antoine
    equation for temperature-dependent saturation vapour pressure.

    Thermal justification for T_film = T_platen:
        Thermal diffusivity of water: alpha ~ 1.4e-7 m^2/s
        Film thickness h ~ 1-5 um
        tau_thermal = h^2/alpha ~ 1e-8 to 1e-7 s
        Drying timescale tau_dry ~ 0.05-1 s
        tau_thermal << tau_dry, so isothermal at T_platen is valid.

References:
    Antoine (1888) constants for water: NIST WebBook
        https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Type=ANTOINE
    Routh & Russel (1998) AIChE J. 44(9) -- latex film formation
    Hoath (2016) Fundamentals of Inkjet Printing, ch. 9
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from dataclasses import dataclass

from ink_properties import Ink, Substrate
from flight import _ink_color


# ---------------------------------------------------------------------------
# Antoine equation for water vapour pressure
# ---------------------------------------------------------------------------

# Antoine constants for water (T in deg C, P in mmHg).
# Source: NIST WebBook. Valid range: 60-150 deg C.
ANTOINE_A = 8.07131
ANTOINE_B = 1730.63
ANTOINE_C = 233.426

MMHG_TO_PA = 133.322   # 1 mmHg in Pascals


def saturation_pressure(T_K: float) -> float:
    """
    Saturation vapour pressure of water [Pa] at temperature T [K].

    Uses the Antoine equation:
        log10(P_sat / mmHg) = A - B / (C + T_C)

    where T_C is temperature in Celsius.

    Valid for 333-423 K (60-150 deg C), which covers typical platen
    temperatures. Constants from NIST WebBook.
    """
    T_C   = T_K - 273.15
    log_P = ANTOINE_A - ANTOINE_B / (ANTOINE_C + T_C)
    return 10.0 ** log_P * MMHG_TO_PA


def evaporation_rate_coefficient(T_K: float, RH: float = 0.50,
                                  h_m: float = 0.10) -> float:
    """
    Evaporation rate coefficient k_evap [m/s].

    Models the film as a flat liquid surface losing volume by convective
    mass transfer to the air above it:

        k_evap = h_m * (c_sat(T_film) - c_ambient) / rho_liquid

    where:
        h_m        : convective mass transfer coefficient [m/s]
        c_sat      : saturation vapour concentration at film temperature [kg/m^3]
        c_ambient  : actual vapour concentration in ambient air [kg/m^3]
        rho_liquid : density of liquid water [kg/m^3]

    c_sat and c_ambient are computed from the Antoine equation and ideal
    gas law. Ambient is assumed to be at 25 deg C.

    Parameters
    ----------
    T_K : film/platen temperature [K]
    RH  : relative humidity (0-1)
    h_m : convective mass transfer coefficient [m/s]

    Typical h_m values:
        0.005-0.02  : still air
        0.05-0.15   : light forced airflow
        0.2-0.5     : strong forced convection (industrial dryer)

    The default value of 0.10 m/s is a rough estimate for light airflow.
    It comes from: h_m ~ D_wv / delta, where D_wv is the diffusivity of
    water vapour in air (~2.6e-5 m^2/s at 60 C) and delta is the
    boundary layer thickness above the substrate. For a printer with a
    moving substrate or fan, delta is roughly 0.3 mm, giving
    h_m ~ 2.6e-5 / 3e-4 ~ 0.087 m/s, rounded to 0.1. This is not from
    a paper. It is a back-of-envelope estimate. If you have measured
    airflow data for a specific printer, use that instead.
    """
    Mw        = 0.018015   # kg/mol, molar mass of water
    R         = 8.314      # J/(mol*K), gas constant
    rho_water = 1000.0     # kg/m^3

    T_amb     = 298.15     # K, assumed ambient temperature (25 deg C)

    P_sat_hot = saturation_pressure(T_K)
    P_sat_amb = saturation_pressure(T_amb)

    c_sat     = P_sat_hot * Mw / (R * T_K)         # kg/m^3
    c_ambient = RH * P_sat_amb * Mw / (R * T_amb)  # kg/m^3

    k_evap = h_m * max(c_sat - c_ambient, 0.0) / rho_water
    return k_evap


# ---------------------------------------------------------------------------
# [INK] Drying model
# ---------------------------------------------------------------------------

@dataclass
class DryingState:
    """
    Input parameters for one drying simulation.

    D_dot and h0 come from Module 2 (impact and spreading).
    phi0 comes from the ink properties.

    h_m is the convective mass transfer coefficient [m/s]. It controls
    how fast water vapour is carried away from the film surface and is
    the most important parameter for drying time after temperature.

    Approximate h_m values:
        0.005-0.02  : still air (diffusion-limited)
        0.05-0.15   : light forced airflow (fan or substrate movement)
        0.2-0.5     : strong forced convection (industrial dryer)

    Real inkjet printers use heated platens with active airflow, so h_m
    around 0.1-0.2 m/s is more realistic than the still-air case.
    Default is 0.10 m/s (light forced airflow).
    """
    D_dot:   float          # dot diameter [m], from Module 2
    h0:      float          # initial film thickness [m], from Module 2
    phi0:    float          # initial latex solid volume fraction
    T_K:     float          # platen temperature [K]
    RH:      float = 0.50   # relative humidity (0-1)
    h_m:     float = 0.10   # mass transfer coefficient [m/s]


def _drying_ode(t: float, state: np.ndarray, k_evap: float,
                A_contact: float) -> list:
    """
    ODE for the evaporating ink film. State is [V], the ink volume [m^3].

        dV/dt = -k_evap * A_contact

    This is the uniform-rate approximation: the evaporation rate is
    proportional to the contact area and constant in time. It is valid
    while the film is still liquid (phi < phi_close_packing). The
    transition to the skin-formation regime is handled in simulate_drying
    by stopping integration when V reaches V_solid.
    """
    dVdt = -k_evap * A_contact
    return [dVdt]


def simulate_drying(ds: DryingState) -> dict:
    """
    [INK] Simulate the drying of one ink dot on a heated substrate.

    Takes dot geometry (D_dot, h0) and ink properties (phi0) from
    Module 2 and returns the full drying curve V(t), phi(t), and t_dry.

    The ODE is a simple linear decay so solve_ivp is overkill here, but
    it keeps the structure consistent with Module 1 and makes it easy to
    extend with a non-constant rate later.

    Parameters
    ----------
    ds : DryingState

    Returns
    -------
    dict with: t, V, phi, t_dry, phase
    """
    A_contact = np.pi * (ds.D_dot / 2.0) ** 2
    V0        = A_contact * ds.h0
    V_solid   = ds.phi0 * V0

    k_evap = evaporation_rate_coefficient(ds.T_K, ds.RH, ds.h_m)

    if k_evap < 1e-15:
        return {
            "t":     np.array([0.0]),
            "V":     np.array([V0]),
            "phi":   np.array([ds.phi0]),
            "t_dry": np.inf,
            "phase": ["liquid"],
        }

    t_dry_est = (V0 - V_solid) / (k_evap * A_contact)

    # Integrate over 2x the estimated drying time with dense output.
    # Termination is detected manually by finding where V crosses V_solid.
    # This avoids scipy event API issues in scipy >= 1.14.
    t_eval = np.linspace(0, t_dry_est * 2.0, 1000)
    sol = solve_ivp(
        _drying_ode,
        t_span=(0, t_dry_est * 2.0),
        y0=[V0],
        args=(k_evap, A_contact),
        t_eval=t_eval,
        rtol=1e-6,
    )

    V_arr = sol.y[0]

    crossed = np.where(V_arr <= V_solid)[0]
    if crossed.size > 0:
        idx   = crossed[0]
        t_dry = float(sol.t[idx])
        sol_t = sol.t[:idx + 1]
        V_arr = V_arr[:idx + 1]
    else:
        t_dry = float(sol.t[-1])
        sol_t = sol.t

    phi_arr = ds.phi0 * V0 / np.maximum(V_arr, V_solid)

    PHI_CP = 0.64  # random close packing of spheres
    phase  = np.where(
        phi_arr < PHI_CP,
        "liquid",
        np.where(phi_arr < 0.95, "film forming", "fixed")
    )

    return {
        "t":     sol_t,
        "V":     V_arr,
        "phi":   phi_arr,
        "t_dry": t_dry,
        "phase": phase,
    }


# ---------------------------------------------------------------------------
# [INK] Parameter sweeps
# ---------------------------------------------------------------------------

def drying_time_vs_temperature(
    ink: Ink,
    substrate: Substrate,
    D_dot: float,
    h0: float,
    T_range: np.ndarray,
    RH: float = 0.50,
) -> np.ndarray:
    """
    [INK] Drying time as a function of platen temperature.
    This is the main design curve for a fixation unit.
    """
    t_drys = np.empty(len(T_range))
    for i, T in enumerate(T_range):
        ds = DryingState(D_dot=D_dot, h0=h0, phi0=ink.phi0, T_K=T, RH=RH)
        t_drys[i] = simulate_drying(ds)["t_dry"]
    return t_drys


def drying_time_vs_dot_size(
    ink: Ink,
    substrate: Substrate,
    D_dot_range: np.ndarray,
    T_K: float,
    RH: float = 0.50,
) -> np.ndarray:
    """
    [INK] Drying time as a function of dot diameter.

    Connects Module 2 to Module 3: the dot size from spreading sets the
    film geometry here. Film thickness is computed from volume conservation:

        h = V / A = (pi/6 * d^3) / (pi/4 * D^2) = 2*d^3 / (3*D^2)

    Larger dots have more area but thinner films, so the two effects
    partially cancel. The net result depends on the specific geometry.
    """
    d_drop = 30e-6  # reference droplet diameter [m]
    t_drys = np.empty(len(D_dot_range))
    for i, D in enumerate(D_dot_range):
        h0 = (2.0 * d_drop**3) / (3.0 * D**2)
        ds = DryingState(D_dot=D, h0=h0, phi0=ink.phi0, T_K=T_K, RH=RH)
        t_drys[i] = simulate_drying(ds)["t_dry"]
    return t_drys


def fixation_operating_window(
    ink: Ink,
    D_dot: float,
    h0: float,
    T_range: np.ndarray,
    RH_range: np.ndarray,
    t_dry_limit: float = 0.2,
) -> np.ndarray:
    """
    [INK] 2D map of which (T, RH) combinations meet a drying time target.
    Returns a boolean grid (True = meets the target).

    This is the type of operating window plot that process engineers use
    to define machine specifications.
    """
    window = np.zeros((len(RH_range), len(T_range)), dtype=bool)
    for i, RH in enumerate(RH_range):
        for j, T in enumerate(T_range):
            ds = DryingState(D_dot=D_dot, h0=h0, phi0=ink.phi0, T_K=T, RH=RH)
            window[i, j] = simulate_drying(ds)["t_dry"] < t_dry_limit
    return window


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_drying_curve(result: dict, ink: Ink, ax=None):
    """[INK] Plot phi(t) drying curve with phase annotations."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    t_ms = result["t"] * 1e3
    ax.plot(t_ms, result["phi"], color=_ink_color(ink), linewidth=2)
    ax.axhline(0.64, color="gray",  linestyle="--", linewidth=1,
               label="Close packing (phi = 0.64)")
    ax.axhline(1.0,  color="black", linestyle="--", linewidth=1,
               label="Full fixation (phi = 1)")
    ax.axvline(result["t_dry"] * 1e3, color="red", linestyle=":",
               linewidth=1.5, label=f"t_dry = {result['t_dry']*1e3:.1f} ms")

    ax.set_xlabel("Time [ms]")
    ax.set_ylabel("Latex volume fraction phi")
    ax.set_title(f"Drying curve  |  {ink.name}")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_drying_vs_temperature(T_range: np.ndarray, t_drys: np.ndarray,
                                ink: Ink, ax=None):
    """[INK] Drying time vs. platen temperature."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    ax.plot(T_range - 273.15, t_drys * 1e3, color=_ink_color(ink),
            linewidth=2, label=ink.name)
    ax.set_xlabel("Platen temperature [deg C]")
    ax.set_ylabel("Drying time [ms]")
    ax.set_title("Fixation time vs. platen temperature")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return ax


def plot_operating_window(T_range: np.ndarray, RH_range: np.ndarray,
                           window: np.ndarray, t_dry_limit: float, ax=None):
    """[INK] 2D operating window map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.contourf(T_range - 273.15, RH_range * 100, window.astype(float),
                levels=[-0.5, 0.5, 1.5], colors=["#ffcccc", "#ccffcc"])
    ax.contour(T_range - 273.15, RH_range * 100, window.astype(float),
               levels=[0.5], colors=["black"], linewidths=2)

    ax.set_xlabel("Platen temperature [deg C]")
    ax.set_ylabel("Relative humidity [%]")
    ax.set_title(
        f"Fixation operating window\n"
        f"Green: t_dry < {t_dry_limit*1e3:.0f} ms  |  Red: too slow"
    )
    ax.grid(True, alpha=0.3)
    return ax
