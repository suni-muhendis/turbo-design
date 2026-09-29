# tests/test_axial_characterization.py
"""Pin the current behaviour of the axial compressor examples.

These are characterization goldens: they record what the code does today, not what is
physically correct. Several are expected to move when known defects are fixed; the point of
pinning them is that the change is then visible and has to be stated.

Pinned per streamline (hub, mean, shroud). Both examples run num_streamlines = 3, the
CompressorSpool default. A meanline-only golden would be blind to any defect whose error
vanishes at constant radius -- and both examples hold the mean radius constant.
"""

import numpy as np
import pytest

RTOL = 1e-9

# RE-BASELINED in the Suni Muhendis fork (1.4.4+sm.1) for the radial-equilibrium fix:
# radeq used dP/dr where the static pressure P belongs in dP0/dr. Every golden that
# moved, moved because of that commit; see MODIFICATIONS.md.


def test_example_9_1_is_three_streamlines(example_spool):
    spool = example_spool("mattingly-axial-compressor/example9.1.py")["spool"]
    assert spool.num_streamlines == 3


def test_example_9_1_overall(example_spool):
    # Re-baselined: stator_calc previously never re-derived a stator's P0 from its
    # upstream rotor's converged P0 (only from initialize()'s one-time, pre-convergence
    # seed), so overall_pressure_ratio() - which reads the last row's P0 - was wrong
    # whenever that last row is a stator, as it is here.
    spool = example_spool("mattingly-axial-compressor/example9.1.py")["spool"]
    assert spool.massflow == pytest.approx(22.6796, rel=RTOL)
    assert spool.overall_pressure_ratio() == pytest.approx(1.321457267948512, rel=RTOL)


def test_example_9_1_rotor_per_streamline(example_spool):
    rotor = example_spool("mattingly-axial-compressor/example9.1.py")["rotor"]
    assert rotor.r == pytest.approx([0.2746123945920154, 0.30479999999999996, 0.3349876054079845], rel=RTOL)
    assert rotor.P0 == pytest.approx(
        [119629.97989086181, 134654.88903465192, 145126.51438789186], rel=RTOL
    )
    assert rotor.T0 == pytest.approx(
        [302.2111314185301, 312.6534734479435, 319.44681679481846], rel=RTOL
    )
    assert rotor.P0R == pytest.approx(
        [101472.48482057032, 106006.8605075036, 103382.84248874572], rel=RTOL
    )
    assert rotor.T0R == pytest.approx(
        [288.2635670314187, 291.9064016271137, 289.81352780407684], rel=RTOL
    )
    assert rotor.M == pytest.approx(
        [0.8300420002164457, 0.8836650083043835, 0.945431289004829], rel=RTOL
    )
    assert rotor.M_rel == pytest.approx(
        [0.6541418460007263, 0.6319777982814438, 0.591685602826366], rel=RTOL
    )


def test_example_9_1_stator_per_streamline(example_spool):
    # Re-baselined along with test_example_9_1_overall - see that test's comment.
    stator = example_spool("mattingly-axial-compressor/example9.1.py")["stator"]
    assert stator.P0 == pytest.approx(
        [120775.31835277236, 134654.88903465192, 146370.65704532174], rel=RTOL
    )
    assert stator.M == pytest.approx(
        [0.5928323588696403, 0.6888644229630325, 0.7466516192637238], rel=RTOL
    )


def test_example_9_2_overall(example_spool):
    # Re-baselined: see test_example_9_1_overall's comment - same stator P0 fix.
    spool = example_spool("mattingly-axial-compressor/example9.2.py")["spool"]
    assert spool.massflow == pytest.approx(22.68, rel=RTOL)
    assert spool.overall_pressure_ratio() == pytest.approx(1.2644703168039397, rel=RTOL)


def test_example_9_2_rotor_per_streamline(example_spool):
    rotor = example_spool("mattingly-axial-compressor/example9.2.py")["rotor"]
    assert rotor.r == pytest.approx([0.2739786879609573, 0.3048, 0.3356213120390427], rel=RTOL)
    assert rotor.P0R == pytest.approx(
        [98740.79796553294, 102918.78333557918, 100612.12353804], rel=RTOL
    )
    assert rotor.T0R == pytest.approx(
        [288.0527969659146, 291.85741830900224, 289.97441112999275], rel=RTOL
    )
    assert rotor.M_rel == pytest.approx(
        [0.6643904760444648, 0.6407612251546442, 0.6002531990666731], rel=RTOL
    )


def test_the_annulus_contracts_off_the_meanline(example_spool):
    """What the per-streamline pin above is for.

    Both examples hold the mean radius constant while contracting the annulus, so r2/r1 is
    exactly 1.000 at the mean and differs from 1 at hub and shroud. Any error that scales
    with (U2 - U1) is therefore absent from the meanline and present only off it.
    """
    for name in (
        "mattingly-axial-compressor/example9.1.py",
        "mattingly-axial-compressor/example9.2.py",
    ):
        loc = example_spool(name)
        inlet, rotor = loc["spool"].blade_rows[0], loc["rotor"]
        ratio = np.asarray(rotor.r, float) / np.asarray(inlet.r, float)
        assert ratio[1] == pytest.approx(1.0, abs=1e-12), f"{name}: mean radius moved"
        assert abs(ratio[0] - 1.0) > 1e-3, f"{name}: hub does not contract"
        assert abs(ratio[2] - 1.0) > 1e-3, f"{name}: shroud does not contract"


def test_the_examples_are_deterministic(example_spool):
    """If the baseline does not reproduce bit for bit, nothing pinned above means anything."""
    from tests.conftest import EXAMPLES, _run

    a = _run(EXAMPLES / "mattingly-axial-compressor" / "example9.1.py")[
        "spool"
    ].overall_pressure_ratio()
    b = _run(EXAMPLES / "mattingly-axial-compressor" / "example9.1.py")[
        "spool"
    ].overall_pressure_ratio()
    assert a == b
