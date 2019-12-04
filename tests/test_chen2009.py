"""Tests for the hypothetical A-B-C system laid out by S.-L. Chen et al. JPEDAV 30(5) (2009) 429-434"""

import os
import numpy as np
from pycalphad import Database, variables as v
from scheil import simulate_scheil_solidification

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
