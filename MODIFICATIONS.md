# Modifications in this fork

This repository is a fork of [nasa/turbo-design](https://github.com/nasa/turbo-design),
maintained by [Suni Muhendis](https://github.com/suni-muhendis). It is not
endorsed by NASA or by any prior contributor.

The changes below are Modifications of the Subject Software as defined by the
NASA Open Source Agreement version 1.3 (see `LICENSE-NOSA-1.3.txt`). Suni
Muhendis is their originator and consents to their characterization as
Modifications. They are distributed under the same agreement.

Each change is one commit on the `sm` branch, on top of upstream release 1.4.4
(`4be23d1`). Releases of this fork are tagged `v<upstream>-sm.<n>` and carry the
version `<upstream>+sm.<n>`.

## 1.4.4+sm.1 — 2026-09-29

Changes that alter results are marked **(changes results)**; the others leave
every result unchanged for inputs that worked before them.

1. **`TD3_HOME` set by the caller is kept.** `LossBaseClass` and
   `DeviationBaseClass` overwrote the variable on construction, discarding a
   cache directory chosen by the caller. (`loss/losstype.py`,
   `deviation/deviation_base.py`)
2. **Kacker-Okapuu works with spanwise row geometry.** The mean-line
   correlation cast `pitch_to_chord`, the trailing-edge ratio and the chord to
   scalars, which failed for more than one streamline; it now uses their span
   means and the chord from the span-averaged stagger, also in the Reynolds
   number. (`loss/turbine/kackerokapuu.py`)
3. **Outlet pressure profiles are kept.** `Outlet.init_static` and
   `Outlet.init_total` discarded all but the first value of a radial
   pressure profile. A single value is still applied uniformly; a length
   mismatch now raises. (`outlet.py`)
4. **Streamline adjustment runs.** `TurbineSpool` called
   `adjust_streamlines` with two arguments where three were required, and the
   function compared cumulative massflow (kg/s) with fractions in [0, 1]. The
   fraction now defaults to an even split and is scaled by the row massflow.
   (`solve_radeq.py`)
5. **Radial equilibrium uses the static pressure** **(changes results)**.
   `radeq` used the radial gradient of the static pressure where the static
   pressure itself belongs in dP0/dr. (`radeq.py`)
6. **DiffusionLoss uses the Lieblein diffusion factor** **(changes results)**.
   The loading term was dVt/|U| instead of |dVt|/(2 sigma V1), which made
   every stator loss zero. (`loss/compressor/diffusion.py`)
7. **CarterDeviation returns Carter's deviation in degrees**
   **(changes results)**. It defaulted the exit angle to the flow angle and so
   always returned zero, worked in radians where `BladeRow` expects degrees,
   and could not take a spanwise solidity. It now applies
   (gamma_i - gamma_e) / (4 sqrt(sigma)) to the row's metal angles, keeping
   Mattingly's form for a target exit flow angle.
   (`deviation/carter_deviation.py`)
