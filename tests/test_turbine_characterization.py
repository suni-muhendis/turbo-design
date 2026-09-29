# tests/test_turbine_characterization.py
"""Pin the current behaviour of the axial and radial turbine examples.

Why this file exists: `flow_math.compute_streamline_areas` computes streamtube areas with
`2*pi*C*(S/2*dx**2 + r*dx)`, where S is a radius difference in metres. The first term is a
cubic metre and the second a square metre, so the expression adds a volume to an area (see
tests/test_band_area_oracle.py). That function is shared -- `turbine_math.py:306` calls it
directly and `turbine_spool.py` reaches it through `compute_massflow` -- so correcting the
area formula will change axial turbine results too, and axial turbines are this library's
primary published use case.

These are characterization goldens: they record what the code does today, not what is
physically correct. `eee_hpt.py` is expected to move once the shared area fix lands; the
point of pinning it now is that the change is then visible rather than discovered.
`radial_turbine-1D.py` is pinned because it should move further: it is a radial machine, so
it takes the buggy branch on every band. Over its solve, `compute_streamline_areas` is
called 64 times and the radial branch body executes 256 times (64 calls x 4 bands at 5
streamlines), at meridional angles up to 89.96 degrees. The axial branch never runs.

Both sets of goldens were produced by running the examples on this tree (see
`tests/conftest.py::_run`).
"""

import pytest

RTOL = 1e-9

# RE-BASELINED in the Suni Muhendis fork (1.4.4+sm.1) for the radial-equilibrium fix:
# radeq used dP/dr where the static pressure P belongs in dP0/dr. Every golden that
# moved, moved because of that commit; see MODIFICATIONS.md.


# ---------------------------------------------------------------------------
# examples/EEE-HPT/eee_hpt.py -- 2-stage axial turbine, SLSQP multi-stage pressure
# balance (turbodesign/turbine_spool.py:_balance_pressure, fmin_slsqp branch).
#
# This example has no main() -- it is a flat top-level script -- so `example_spool`
# returns its module namespace directly. It defines `stator1`, `rotor1`, `stator2`,
# `rotor2` (the same BladeRow objects that are inside `spool.rows`, mutated in place
# by `spool.solve()`) and `spool` itself. Runtime is ~7-8s (SLSQP + 5 streamlines),
# hence @pytest.mark.slow.
#
# Re-baselined at issue #40; see the note above the radial-turbine goldens below.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_eee_hpt_is_five_streamlines(example_spool):
    spool = example_spool("EEE-HPT/eee_hpt.py")["spool"]
    assert spool.num_streamlines == 5


@pytest.mark.slow
def test_eee_hpt_massflow_attribute_is_the_input_echo(example_spool):
    """`spool.massflow` is set once at construction (turbine_spool.py:92) and is never
    written back to during `.solve()` in pressure-balance mode; only
    `solve_massflow_for_power`, which this example does not use, updates it. So the
    attribute is not the converged massflow -- it is the 20 kg/s guess the example passed
    in. The converged value is pinned separately below, from `spool.convergence_history`.
    """
    spool = example_spool("EEE-HPT/eee_hpt.py")["spool"]
    assert spool.massflow == pytest.approx(20.0, rel=RTOL)


@pytest.mark.slow
def test_eee_hpt_stator1_per_streamline(example_spool):
    row = example_spool("EEE-HPT/eee_hpt.py")["stator1"]
    assert row.P0 == pytest.approx([1230586.8909893231, 1230586.8909893231, 1230586.8909893231, 1230586.8909893231, 1230586.8909893231], rel=RTOL)
    assert row.T0 == pytest.approx([1587.0, 1587.0, 1587.0, 1587.0, 1587.0], rel=RTOL)
    assert row.M == pytest.approx(
        [0.9013025604420105, 0.8758399828279942, 0.8522029316054892, 0.8301925230055965, 0.8095729849451481],
        rel=RTOL,
    )
    assert row.Yp == pytest.approx([0.057, 0.057, 0.057, 0.057, 0.057], rel=RTOL)


@pytest.mark.slow
def test_eee_hpt_rotor1_per_streamline(example_spool):
    row = example_spool("EEE-HPT/eee_hpt.py")["rotor1"]
    assert row.P0 == pytest.approx(
        [566058.2902899861, 562501.8003719215, 559941.1458236862, 557918.5563441236, 556584.4802341909],
        rel=RTOL,
    )
    assert row.T0 == pytest.approx(
        [1337.7307157470466, 1335.5186126135973, 1333.892602188266, 1332.6020132305794, 1331.7284895551684],
        rel=RTOL,
    )
    assert row.M == pytest.approx(
        [0.41379522819051473, 0.38624209929488906, 0.3639823251420871, 0.3469842287577223, 0.33415874484081354],
        rel=RTOL,
    )
    assert row.Yp == pytest.approx([0.088, 0.088, 0.088, 0.088, 0.088], rel=RTOL)


@pytest.mark.slow
def test_eee_hpt_stator2_per_streamline(example_spool):
    row = example_spool("EEE-HPT/eee_hpt.py")["stator2"]
    assert row.P0 == pytest.approx(
        [552705.4620553899, 549394.3699416717, 547010.4005572646, 545127.3697517918, 543885.3448934446],
        rel=RTOL,
    )
    assert row.M == pytest.approx(
        [0.8744076047404354, 0.8270816363334431, 0.7860962086410946, 0.7506139303473779, 0.7193574624191791],
        rel=RTOL,
    )
    assert row.Yp == pytest.approx([0.069, 0.069, 0.069, 0.069, 0.069], rel=RTOL)


@pytest.mark.slow
def test_eee_hpt_rotor2_per_streamline(example_spool):
    row = example_spool("EEE-HPT/eee_hpt.py")["rotor2"]
    assert row.P0 == pytest.approx(
        [264603.7626407024, 261040.00533826486, 259125.33158472396, 258024.59371648557, 257833.66366584634],
        rel=RTOL,
    )
    assert row.T0 == pytest.approx(
        [1130.2068148039295, 1126.3244344821946, 1124.1410713558068, 1122.8295622459968, 1122.4912540564746],
        rel=RTOL,
    )
    assert row.M == pytest.approx(
        [0.4962649909319989, 0.45613552292012216, 0.42889965441205585, 0.4132249512076619, 0.4052460089703709],
        rel=RTOL,
    )
    assert row.Yp == pytest.approx([0.014, 0.014, 0.014, 0.014, 0.014], rel=RTOL)


@pytest.mark.slow
def test_eee_hpt_slsqp_tripwire(example_spool):
    """A coarse tripwire, deliberately looser than the row-level goldens above.

    `fmin_slsqp`'s own `x`/`fun` live inside `TurbineSpool._balance_pressure`
    (turbodesign/turbine_spool.py) and are never assigned to an attribute the example can
    see, so they cannot be pinned without reaching into library internals.
    `spool.convergence_history[-1]` records the same optimizer's final objective value
    (`massflow_std`, what `fmin_slsqp` was minimizing) and its iteration count, which
    answers the same question: did the solver converge to roughly the same place. For
    anything that needs to be exact, prefer the per-row assertions above.
    """
    spool = example_spool("EEE-HPT/eee_hpt.py")["spool"]
    assert len(spool.convergence_history) == 119
    last = spool.convergence_history[-1]
    assert last["massflow_std"] == pytest.approx(0.0011267336808817162, rel=1e-6)
    assert last["massflow"] == pytest.approx(29.263924040194947, rel=1e-6)


# ---------------------------------------------------------------------------
# examples/radial-turbine/radial_turbine-1D.py -- single-stage radial-inflow turbine,
# single-stage pressure balance (turbodesign/turbine_spool.py:_balance_pressure,
# minimize_scalar branch -- only one stage, so fmin_slsqp is not used here).
#
# Pinned because it is expected to move once the area fix lands: this passage takes the
# `S/2 * dx**2` branch of `compute_streamline_areas` (flow_math.py:40) on every band --
# 256 band evaluations over 64 calls -- at meridional angles up to 89.96 degrees, where
# the term being added is dimensionally a volume rather than an area. On an axial machine
# that term is small; here it is not. This example also has no main(); `example_spool`
# returns its module namespace, which defines `stator`, `rotor`, and `spool`.
# ---------------------------------------------------------------------------


def test_radial_turbine_is_five_streamlines(example_spool):
    spool = example_spool("radial-turbine/radial_turbine-1D.py")["spool"]
    assert spool.num_streamlines == 5


def test_radial_turbine_massflow_attribute_is_the_input_echo(example_spool):
    """As with EEE-HPT: `spool.massflow` is the unconverged 0.1 kg/s guess passed into the
    constructor, not the converged value. See
    `test_eee_hpt_massflow_attribute_is_the_input_echo`.
    """
    spool = example_spool("radial-turbine/radial_turbine-1D.py")["spool"]
    assert spool.massflow == pytest.approx(0.1, rel=RTOL)


# --------------------------------------------------------------------------------------
# RE-BASELINED at issue #40 (turbine rotor P0R carried isentropically with T0R).
#
# turbine_math.rotor_calc took the ideal exit P0R straight from the rotor inlet while T0R
# moved with U^2 through rothalpy, so a rotor whose exit radius differs from its inlet radius
# destroyed entropy. Same defect #35 fixed in compressor_math.py; see
# tests/test_turbine_ideal_relative_pressure.py. Both examples moved:
#   radial turbine: converged massflow 0.33576 -> 0.31552 kg/s (CFD: 0.293), rotor P0 -3.5..-5 %,
#                   15 -> 16 balance iterations. Relative-total ds across the rotor is now
#                   +17.3..+17.6 J/(kg K) on every streamline.
#   EEE-HPT:        rotor streamlines sit ~1 mm off their stator-exit radii, so the axial
#                   rows move too, but only by <= 0.2 % in P0 and -0.04 % in massflow;
#                   SLSQP 106 -> 122 iterations.
# The "pre-fix" comments below refer to the #27 area fix, not this one.
# --------------------------------------------------------------------------------------
# RE-BASELINED at PR #27 (`fix/radial-area-and-massflow-sign`).
#
# The three radial-turbine goldens below moved because `compute_streamline_areas` changed:
# the band area is now the exact frustum area pi*(r1 + r2)*slant instead of an expression
# that added a cubic metre to a square metre. See tests/test_current_area_formula.py.
#
# These are CHARACTERIZATION tests. They pin behaviour so that a change cannot pass
# unnoticed; they are not statements that the pinned numbers are correct. They fired exactly
# as designed, and the values are re-taken from the example rather than adjusted to fit.
# Only the radial cases move -- every axial golden in this file is untouched, which is the
# expected signature of a change to the RADIAL branch of the area calculation.
# --------------------------------------------------------------------------------------


def test_radial_turbine_stator_per_streamline(example_spool):
    row = example_spool("radial-turbine/radial_turbine-1D.py")["stator"]
    assert row.P0 == pytest.approx([536657.313099755, 536657.313099755, 536657.313099755, 536657.313099755, 536657.313099755], rel=RTOL)
    assert row.T0 == pytest.approx([1222.8779357137, 1222.8779357137, 1222.8779357137, 1222.8779357137, 1222.8779357137], rel=RTOL)
    # was 0.42080703647747797 before the area fix; P0/T0/Yp are unchanged by it.
    assert row.M == pytest.approx([0.4424214336315834, 0.4424214336315834, 0.4424214336315834, 0.4424214336315834, 0.4424214336315834], rel=RTOL)
    assert row.Yp == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0], abs=1e-12)


def test_radial_turbine_rotor_per_streamline(example_spool):
    """The golden that was flagged as most likely to move once the area fix landed. It did.

    `M` here is the absolute Mach number, and it exceeds 1 at the hub streamline
    (M[0] = 1.1472). That is not a choked passage: choking is set by the meridional Mach
    number, which is far lower at a near-radial station, and a swirling flow can carry an
    absolute Mach number above 1 without it. It is also exactly the kind of station where
    the corrected term in `compute_streamline_areas` is large rather than negligible --
    which is why this row moved and the axial rows did not.
    """
    row = example_spool("radial-turbine/radial_turbine-1D.py")["rotor"]
    # pre-fix P0: [443074.15148622065, 437651.6584138988, 428537.84499390324,
    #              418045.0960861983, 409980.2644715815]
    assert row.P0 == pytest.approx(
        [410941.57858797745, 407129.00856657594, 401943.79639491363, 398205.07437928714, 398256.32040030503],
        rel=RTOL,
    )
    # pre-fix T0: [1161.8429099481775, 1158.73767317465, 1153.2557718494477,
    #              1146.9118258635594, 1142.398479508869]
    assert row.T0 == pytest.approx(
        [1159.130304400639, 1156.3816565382936, 1152.612666536613, 1149.9091102636341, 1150.051217602391],
        rel=RTOL,
    )
    # pre-fix M: [1.269577634397641, 0.9756593824217092, 0.882204528319701,
    #             0.837409532372009, 0.7773593618539902]
    assert row.M == pytest.approx(
        [1.3172382788958101, 0.9816257651045844, 0.8193500544745681, 0.6876230181258103, 0.5562643898518076],
        rel=RTOL,
    )
    # Yp is set by the loss model, not the area, and is unchanged by the fix.
    assert row.Yp == pytest.approx([0.1358751871641363, 0.1358751871641363, 0.1358751871641363, 0.1358751871641363, 0.1358751871641363], rel=RTOL)


def test_radial_turbine_convergence_tripwire(example_spool):
    """Coarse tripwire on the pressure-balance solve (see the EEE-HPT equivalent
    above for why this is coarse rather than exact: `minimize_scalar`'s own result
    object is internal to `_balance_pressure` and is not exposed on `spool`).

    The iteration count was unchanged at 15 by the area fix -- it moved where the solve
    lands, not how hard it was to get there. The #40 rotor P0R fix took it to 16.
    """
    spool = example_spool("radial-turbine/radial_turbine-1D.py")["spool"]
    assert len(spool.convergence_history) == 12
    last = spool.convergence_history[-1]
    # pre-fix: massflow_std 2.7887726216979658e-05, massflow 0.29864834362636516
    assert last["massflow_std"] == pytest.approx(3.2089620011832576e-05, rel=1e-6)
    assert last["massflow"] == pytest.approx(0.3106551150615096, rel=1e-6)


@pytest.mark.slow
def test_the_turbine_examples_are_deterministic(example_spool):
    """If the baseline does not reproduce bit for bit, nothing pinned above means anything.
    Checked here on the radial turbine, the cheaper of the two examples to run twice.
    """
    from tests.conftest import EXAMPLES, _run

    a = _run(EXAMPLES / "radial-turbine" / "radial_turbine-1D.py")[
        "spool"
    ].convergence_history[-1]["massflow_std"]
    b = _run(EXAMPLES / "radial-turbine" / "radial_turbine-1D.py")[
        "spool"
    ].convergence_history[-1]["massflow_std"]
    assert a == b
