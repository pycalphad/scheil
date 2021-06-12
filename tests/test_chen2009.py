"""Tests for the hypothetical A-B-C system laid out by S.-L. Chen et al. JPEDAV 30(5) (2009) 429-434"""

import os
import numpy as np
from pycalphad import Database, variables as v
from scheil import simulate_scheil_solidification
import pytest

DB_CHEN = Database(os.path.join(os.path.dirname(__file__), 'sl_chen.tdb'))


def test_binary_A_B():
    """Tests for the Scheil properties of the A-B binary peritectic system"""
    comp = {v.X('B'): 0.5}
    start = 800  # Kelvin
    step = 5.0

    sol_res = simulate_scheil_solidification(DB_CHEN, ['A', 'B'], ['ALPHA', 'BETA', 'LIQUID'], comp, start, step_temperature=step, stop=1e-8)
    print(f"Converged to stopping criteria: {sol_res.converged}")

    phase_amnts = sol_res.phase_amounts
    # Check that the first solid phase to form is ALPHA and occurs at ~768 K
    idx_first_solid = np.nonzero(np.array(sol_res.fraction_solid) > 0)[0][0]
    assert phase_amnts['ALPHA'][idx_first_solid] > 0
    assert np.isclose(phase_amnts['BETA'][idx_first_solid], 0)
    # increase step atol a little for it to be inclusive
    assert np.isclose(sol_res.temperatures[idx_first_solid], 768, atol=step * 1.1)

    # Check that the last solid phase to form is BETA only and terminates at 600 K
    assert np.isclose(phase_amnts['ALPHA'][-1], 0)
    assert phase_amnts['BETA'][-1] > 0
    if sol_res.converged:
        # May have converged prior to the eutectic, but not below
        assert sol_res.temperatures[-1] < 600 + step * 2 and sol_res.temperatures[-1] > 600
    else:
        # increase step atol a a litte, since it should have ended at the eutectic
        assert np.isclose(sol_res.temperatures[-1], 600, atol=step * 1.1)

    # Check that the temperature where the ALPHA to BETA switch occurs is close to the peritectic temperature of 647 K.
    idx_first_beta = np.nonzero(np.array(phase_amnts['BETA']) > 0)[0][0]
    idx_last_alpha = np.nonzero(np.array(phase_amnts['ALPHA']) > 0)[0][-1]
    assert (idx_last_alpha + 1) == idx_first_beta
    # increase step atol a little for it to be inclusive
    assert np.isclose(sol_res.temperatures[idx_first_beta], 647, atol=step * 1.1)

@pytest.mark.parametrize(
    'phases',
    (
        ['ALPHA', 'BETA', 'ORD_BETA', 'LIQUID'],
        ['ALPHA', 'ORD_BETA', 'LIQUID'],
        ['ALPHA', 'BETA', 'LIQUID'],
    )
)
def test_binary_A_B_with_ordering(phases):
    """Tests for the Scheil properties of the A-B binary peritectic system with an ordered phase ORD_BETA"""
    with open(os.path.join(os.path.dirname(__file__), 'sl_chen.tdb')) as fp:
        tdb_str = fp.read()
    # Append an ordered phase, 'ORD_BETA' with no ordering parameters
    ordered_phase_str = """
    TYPE_DEFINITION L GES A_P_D  ORD_BETA DIS_PART BETA   ,,,!
    PHASE ORD_BETA %L  2  0.75  0.25   !
    CONSTITUENT ORD_BETA :A,B,C:A,B,C: !
    """
    tdb_str += ordered_phase_str

    dbf = Database(tdb_str)

    comp = {v.X('B'): 0.5}
    start = 800  # Kelvin
    step = 5.0

    sol_res = simulate_scheil_solidification(dbf, ['A', 'B'], phases, comp, start, step_temperature=step, stop=1e-8)
    print(f"Converged to stopping criteria: {sol_res.converged}")

    phase_amnts = sol_res.phase_amounts
    # Check that the first solid phase to form is ALPHA and occurs at ~768 K
    idx_first_solid = np.nonzero(np.array(sol_res.fraction_solid) > 0)[0][0]
    assert phase_amnts['ALPHA'][idx_first_solid] > 0
    assert np.isclose(phase_amnts['BETA'][idx_first_solid], 0)
    # increase step atol a little for it to be inclusive
    assert np.isclose(sol_res.temperatures[idx_first_solid], 768, atol=step * 1.1)

    # Check that the last solid phase to form is BETA only and terminates at 600 K
    assert np.isclose(phase_amnts['ALPHA'][-1], 0)
    assert phase_amnts['BETA'][-1] > 0
    if sol_res.converged:
        # May have converged prior to the eutectic, but not below
        assert sol_res.temperatures[-1] < 600 + step * 2 and sol_res.temperatures[-1] > 600
    else:
        # increase step atol a a litte, since it should have ended at the eutectic
        assert np.isclose(sol_res.temperatures[-1], 600, atol=step * 1.1)

    # Check that the temperature where the ALPHA to BETA switch occurs is close to the peritectic temperature of 647 K.
    idx_first_beta = np.nonzero(np.array(phase_amnts['BETA']) > 0)[0][0]
    idx_last_alpha = np.nonzero(np.array(phase_amnts['ALPHA']) > 0)[0][-1]
    assert (idx_last_alpha + 1) == idx_first_beta
    # increase step atol a little for it to be inclusive
    assert np.isclose(sol_res.temperatures[idx_first_beta], 647, atol=step * 1.1)


def test_binary_A_C():
    """Tests for the Scheil properties of the A-C binary eutectic system"""
    comp = {v.X('C'): 0.5}
    start = 900  # Kelvin
    step = 5.0

    sol_res = simulate_scheil_solidification(DB_CHEN, ['A', 'C'], ['ALPHA', 'BETA', 'LIQUID'], comp, start, step_temperature=step, stop=1e-8)
    print(f"Converged to stopping criteria: {sol_res.converged}")

    phase_amnts = sol_res.phase_amounts
    # Check that the first solid phase to form is BETA and occurs at ~857 K
    idx_first_solid = np.nonzero(np.array(sol_res.fraction_solid) > 0)[0][0]
    assert phase_amnts['BETA'][idx_first_solid] > 0
    assert np.isclose(phase_amnts['ALPHA'][idx_first_solid], 0)
    # increase step atol a little for it to be inclusive
    assert np.isclose(sol_res.temperatures[idx_first_solid], 857, atol=step * 1.1)

    # Check that the last solid phases to form contain ALPHA and BETA eutectic and terminates at 820 K
    assert phase_amnts['ALPHA'][-1] > 0
    assert phase_amnts['BETA'][-1] > 0
    if sol_res.converged:
        # May have converged prior to the eutectic, but not below
        assert sol_res.temperatures[-1] < 820 + step * 2 and sol_res.temperatures[-1] > 820
    else:
        # increase step atol a a litte, since it should have ended at the eutectic
        assert np.isclose(sol_res.temperatures[-1], 820, atol=step * 1.1)


def test_ternary_A_B_C():
    """Tests for the Scheil properties of the A-B-C ternary system

    There is a transition point, t, where the solidification changes from the
    eutectic monovariant to peritetic-like. Most of these tests are focused
    around this point.
    """
    t_temp = 683.7  # Eutectic-like to peritectic-like transition temperature, in Kelvin
    comp = {v.X('B'): 0.25, v.X('C'): 0.1}
    start = 800  # Kelvin
    step = 5

    sol_res = simulate_scheil_solidification(DB_CHEN, ['A', 'B', 'C'], ['ALPHA', 'BETA', 'LIQUID'], comp, start, step_temperature=step, stop=1e-8)
    print(f"Converged to stopping criteria: {sol_res.converged}")

    phase_amnts = sol_res.phase_amounts
    # Check that the first solid phase to form is ALPHA and occurs above t
    idx_first_solid = np.nonzero(np.array(sol_res.fraction_solid) > 0)[0][0]
    assert phase_amnts['ALPHA'][idx_first_solid] > 0
    assert np.isclose(phase_amnts['BETA'][idx_first_solid], 0)
    assert sol_res.temperatures[idx_first_solid] > t_temp

    # Check that the last solid phases to form is BETA only and termintes near 600 K
    assert np.isclose(phase_amnts['ALPHA'][-1], 0)
    assert phase_amnts['BETA'][-1] > 0
    if sol_res.converged:
        # May have converged prior to the eutectic, but not below
        assert sol_res.temperatures[-1] < 600 + step * 2 and sol_res.temperatures[-1] > 600
    else:
        # increase step atol a a litte, since it should have ended at the eutectic
        assert np.isclose(sol_res.temperatures[-1], 600, atol=step * 1.1)

    # Check that the eutectic-like to peritectic-like transitions (ALPHA+BETA forming to BETA forming) occurs near t
    idx_last_alpha = np.nonzero(np.array(phase_amnts['ALPHA']) > 0)[0][-1]
    # At the last formation of ALPHA, both ALPHA and BETA should be forming in the monovariant, but after only beta
    assert phase_amnts['ALPHA'][idx_last_alpha] > 0
    assert phase_amnts['BETA'][idx_last_alpha] > 0
    assert np.isclose(phase_amnts['ALPHA'][idx_last_alpha + 1], 0)
    assert phase_amnts['BETA'][idx_last_alpha + 1] > 0

    # Temperature is close to t
    assert np.isclose(sol_res.temperatures[idx_last_alpha], t_temp, atol=step * 2)
    # Check that the compositions are resonably close as well, within 1%
    assert np.isclose(sol_res.x_liquid['B'][idx_last_alpha], 0.583, atol=0.03)
    assert np.isclose(sol_res.x_liquid['C'][idx_last_alpha], 0.059, atol=0.03)