# Inkjet Droplet Dynamics Simulator

Physics-based simulation of the full inkjet printing process chain:
**droplet flight → impact & spreading → fixation**.

---

## Origin and Motivation

This project started as a **computational physics assignment** on ballistic
motion with aerodynamic drag a standard problem in numerical methods courses.
The original task: simulate a spherical droplet falling through air, compare
Stokes drag to higher-Re corrections, and report landing position and velocity.

After completing the course, I extended the simulation in two directions:

1. **Adapted** the flight model to the inkjet scale and reparameterised
   everything with real ink fluid properties.
2. **Added** two entirely new physical modules droplet impact/spreading and
   fixation/drying to model the complete printing process chain.

The motivation was to understand the physics behind inkjet printing, which
combines fluid mechanics, surface science, and heat transfer in a tightly
coupled sequence. Each module feeds into the next:

```
Jetting conditions
      │
      ▼
[Module 1: Flight]  ──→  landing position, impact velocity v_imp
                                    │
                                    ▼
              [Module 2: Impact]  ──→  dot diameter D, film thickness h
                                                │
                                                ▼
                        [Module 3: Fixation]  ──→  drying time t_dry
```

---

## What Was Changed (vs. Original Assignment)

### Module 1 Flight (adapted)

| Original | Adapted |
|---|---|
| Stokes drag: `F = 3πµdv` (valid Re << 1) | Schiller-Naumann correlation (valid Re < 800) |
| Fixed time span integration | Event-based termination at substrate contact |
| Generic fluid, mm-scale drops | Real ink properties, µm-scale drops |
| Single deterministic trajectory | Monte Carlo ensemble → dot placement error σ |

The core physics (Newton's 2nd law, `scipy.solve_ivp`, RK45) is unchanged.

### Module 2 Impact and Spreading (new)

Entirely new. The original assignment had no post-impact model dot size
was estimated as "roughly 2× droplet diameter." This module implements:

- **Energy balance spreading model** (Pasandideh-Fard et al. 1996): predicts
  spreading factor β = D_dot/d as a function of We, Re, and contact angle θ.
- **Substrate parameterisation**: θ_eq varies by media type (coated paper,
  uncoated paper, PET film, glass), enabling dot-size vs. media studies.
- **Satellite drop analysis**: Rayleigh-Plateau instability of the ink
  ligament, with an Oh–We jetting stability phase diagram.

### Module 3 Fixation (new)

Entirely new. After spreading, water-based inks must dry on a heated substrate.
This module models:

- **Evaporation kinetics**: uniform-rate model with Antoine equation for
  temperature-dependent saturation vapour pressure.
- **Latex film formation**: volume fraction φ(t) tracking through three
  phases: liquid → close-packed → fixed film.
- **Thermal justification**: 1-D heat conduction gives τ_thermal ~ 10⁻⁸ s <<
  τ_dry ~ 10⁻²–10⁰ s, justifying T_film = T_platen as a physically motivated
  simplification (not a lazy assumption).
- **Operating window**: 2-D (T_platen, RH) map of conditions that meet a
  drying time target the type of process specification map used in practice.

---

## Physics Summary

### Dimensionless numbers

| Number | Formula | Role |
|---|---|---|
| Reynolds | Re = ρvd/µ | Inertia vs. viscosity; selects drag regime |
| Weber | We = ρv²d/σ | Kinetic vs. surface energy; drives spreading |
| Ohnesorge | Oh = µ/√(ρσd) | Viscous vs. surface+inertia; jetting stability |

### Key models

- **Drag**: Schiller-Naumann `Cd = (24/Re)(1 + 0.15 Re^0.687)`
- **Spreading**: Pasandideh-Fard energy balance → quadratic in β, solved with `brentq`
- **Evaporation**: `dV/dt = -k_evap × A_contact`, where `k_evap` uses Antoine equation
- **Film formation**: φ(t) = φ₀ V₀ / V(t), three-phase model

---

## Project Structure

```
inkjet-droplet-sim/
├── src/
│   ├── ink_properties.py   # Ink/substrate dataclasses, dimensionless numbers
│   ├── flight.py           # Module 1: ballistic trajectory + placement error
│   ├── impact.py           # Module 2: spreading factor + satellite analysis
│   └── fixation.py         # Module 3: evaporation + latex film formation
├── notebooks/
│   ├── 01_flight.ipynb     # Flight model demo + drag comparison
│   ├── 02_impact.ipynb     # Spreading, substrate effects, Oh-We diagram
│   └── 03_fixation.ipynb   # Drying curves, operating window, module coupling
├── results/figures/        # Output plots
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/pf-oc/inkjet-droplet-sim
cd inkjet-droplet-sim
pip install -r requirements.txt
jupyter notebook notebooks/
```

---

## Dependencies

`numpy`, `scipy`, `matplotlib`, `pandas` standard scientific Python stack.
No domain-specific packages; everything is built from first principles.

---

## References

- Pasandideh-Fard et al. (1996) *Phys. Fluids* 8(3):650–659 spreading model
- Schiller & Naumann (1933) *Z. Ver. Dtsch. Ing.* 77:318 drag correlation
- Eggers (1993) *Phys. Rev. Lett.* 71(21):3458 Rayleigh-Plateau instability
- Hoath (2016) *Fundamentals of Inkjet Printing*, Wiley-VCH
- NIST WebBook Antoine constants for water
