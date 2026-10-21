import json
import numpy as np
import pytest
from pycalphad import Database, variables as v
from scheil import simulate_scheil_solidification, simulate_equilibrium_solidification, SolidificationResult
from fixtures import select_database, load_database


def test_scheil_solidification_matches_exact_solution_for_pure_solids():
    """Scheil solidification with pure-element solids and an ideal liquid should match the exact solution at every step"""
    entropy_of_fusion = 10.0
    melting_temperatures = {'A': 1000.0, 'B': 900.0, 'C': 850.0}
    solid_phases = {'A': 'ALPHA', 'B': 'BETA', 'C': 'GAMMA'}
    tdb_str = """
    ELEMENT A BLANK 0 0 0 !
    ELEMENT B BLANK 0 0 0 !
    ELEMENT C BLANK 0 0 0 !
    TYPE_DEFINITION % SEQ * !
    PHASE LIQUID %  1 1 !
    CONSTITUENT LIQUID :A,B,C: !
    PARAMETER G(LIQUID,A;0) 1 0; 10000 N !
    PARAMETER G(LIQUID,B;0) 1 0; 10000 N !
    PARAMETER G(LIQUID,C;0) 1 0; 10000 N !
    """
    for el, phase_name in solid_phases.items():
        tdb_str += f"""
    PHASE {phase_name} %  1 1 !
    CONSTITUENT {phase_name} :{el}: !
    PARAMETER G({phase_name},{el};0) 1 {entropy_of_fusion}*T-{entropy_of_fusion * melting_temperatures[el]}; 10000 N !
    """
    dbf = Database(tdb_str)
    x0 = {'A': 0.6, 'B': 0.25, 'C': 0.15}

    def saturated_liquid(el, T):
        return np.exp(entropy_of_fusion * (T - melting_temperatures[el]) / (float(v.R) * T))

    def exact_scheil(T):
        forming, fraction_liquid, x_liquid = [], 1.0, dict(x0)
        if saturated_liquid('A', T) < x0['A']:
            forming = ['ALPHA']
            fraction_liquid = (1 - x0['A']) / (1 - saturated_liquid('A', T))
            x_liquid = {'A': saturated_liquid('A', T), 'B': x0['B'] / fraction_liquid, 'C': x0['C'] / fraction_liquid}
            if x_liquid['B'] > saturated_liquid('B', T):
                forming = ['ALPHA', 'BETA']
                fraction_liquid = x0['C'] / (1 - saturated_liquid('A', T) - saturated_liquid('B', T))
                x_liquid = {'A': saturated_liquid('A', T), 'B': saturated_liquid('B', T), 'C': x0['C'] / fraction_liquid}
        assert x_liquid['C'] < saturated_liquid('C', T)
        solid_amounts = {solid_phases[el]: x0[el] - fraction_liquid * x_liquid[el] for el in x0}
        return forming, fraction_liquid, x_liquid, solid_amounts

    sol_res = simulate_scheil_solidification(dbf, ['A', 'B', 'C'], ['ALPHA', 'BETA', 'GAMMA', 'LIQUID'], {v.X('B'): x0['B'], v.X('C'): x0['C']}, 710, step_temperature=5)

    # skip the initial liquid and the final eutectic point
    for k in range(1, len(sol_res.temperatures) - 1):
        forming, fraction_liquid, x_liquid, solid_amounts = exact_scheil(sol_res.temperatures[k])
        assert [phase_name for phase_name in solid_phases.values() if sol_res.phase_amounts[phase_name][k] > 0] == forming
        assert np.isclose(sol_res.fraction_liquid[k], fraction_liquid, atol=1e-6)
        for el in x0:
            assert np.isclose(sol_res.x_liquid[el][k], x_liquid[el], atol=1e-6)
        for phase_name, amount in solid_amounts.items():
            assert np.isclose(sol_res.cum_phase_amounts[phase_name][k], amount, atol=1e-6)
        for el, phase_name in solid_phases.items():
            if phase_name in forming:
                assert np.isclose(sol_res.phase_compositions[phase_name][el][k], 1.0)

    # the ternary eutectic is between the last two temperatures
    assert sum(saturated_liquid(el, sol_res.temperatures[-2]) for el in x0) > 1
    assert sum(saturated_liquid(el, sol_res.temperatures[-1]) for el in x0) < 1
    assert sol_res.fraction_solid[-1] == 1.0
    for el, phase_name in solid_phases.items():
        assert np.isclose(sol_res.cum_phase_amounts[phase_name][-1], x0[el], atol=1e-6)

@select_database("sl_chen.tdb")
def test_scheil_solidification_raises_without_initial_liquid(load_database):
    """Scheil solidification should raise a clear error if no liquid is stable at the start temperature."""
    dbf = load_database()
    # For X(B)=0.5 the A-B system is fully solid below 600 K, so 500 K has no stable liquid
    with pytest.raises(ValueError, match="[Ll]iquid"):
        simulate_scheil_solidification(dbf, ['A', 'B'], ['ALPHA', 'BETA', 'LIQUID'], {v.X('B'): 0.5}, 500.0, step_temperature=5.0)


@select_database("sl_chen.tdb")
def test_equilibrium_solidification_raises_without_initial_liquid(load_database):
    """Equilibrium solidification should raise a clear error if no liquid is stable at the start temperature."""
    dbf = load_database()
    with pytest.raises(ValueError, match="[Ll]iquid"):
        simulate_equilibrium_solidification(dbf, ['A', 'B'], ['ALPHA', 'BETA', 'LIQUID'], {v.X('B'): 0.5}, 500.0, step_temperature=5.0)


@select_database("alzn_mey.tdb")
def test_scheil_solidification_result_properties(load_database):
    """Test that SolidificationResult objects produced by Scheil solidification have the required properties."""
    # Required properties are
    # 1. the shape of the output arrays are matching
    # 2. the final fraction of solid is 1.0 and fraction of liquid is 0.0
    # 3. the sum total of the final (solid) phase amounts is 1.0

    dbf = load_database()
    comps = ['AL', 'ZN', 'VA']
    phases = sorted(dbf.phases.keys())

    initial_composition = {v.X('ZN'): 0.3}
    start_temperature = 850

    sol_res = simulate_scheil_solidification(dbf, comps, phases, initial_composition, start_temperature, step_temperature=20.0, verbose=True)

    num_temperatures = len(sol_res.temperatures)
    assert num_temperatures == len(sol_res.fraction_liquid)
    assert num_temperatures == len(sol_res.fraction_solid)
    assert all([num_temperatures == len(nphase) for nphase in sol_res.phase_amounts.values()])
    assert all([num_temperatures == len(liq_comps) for liq_comps in sol_res.x_liquid.values()])
    assert all([num_temperatures == len(nphase) for nphase in sol_res.cum_phase_amounts.values()])

    # final phase amounts are correct
    assert sol_res.fraction_liquid[-1] == 0.0
    assert sol_res.fraction_solid[-1] == 1.0

    # The final cumulative solid phase amounts is 1.0
    assert np.isclose(np.sum([amnts[-1] for amnts in sol_res.cum_phase_amounts.values()]), 1.0)
    # The final instantaneous phase amounts is not 1.0 (only the amount of new solid phase added
    assert np.sum([amnts[-1] for amnts in sol_res.phase_amounts.values()]) < 1.0

    # Test serialization
    for ky, vl in sol_res.to_dict().items():
        print(ky, vl, type(ky), type(vl))
        json.dumps({ky: vl})
    json.dumps(sol_res.to_dict())

    # Test round tripping to/from dict
    rnd_trip_sol_res = SolidificationResult.from_dict(sol_res.to_dict())
    assert rnd_trip_sol_res.fraction_liquid == sol_res.fraction_liquid
    assert rnd_trip_sol_res.fraction_solid == sol_res.fraction_solid
    assert rnd_trip_sol_res.x_liquid == sol_res.x_liquid
    assert rnd_trip_sol_res.cum_phase_amounts == sol_res.cum_phase_amounts
    assert rnd_trip_sol_res.phase_amounts == sol_res.phase_amounts
    assert rnd_trip_sol_res.temperatures == sol_res.temperatures
    assert rnd_trip_sol_res.converged == sol_res.converged
    assert rnd_trip_sol_res.method == sol_res.method

    # Test to_dataframe doesn't raise
    sol_res.to_dataframe(include_zero_phases=True)
    sol_res.to_dataframe(include_zero_phases=False)



@select_database("alzn_mey.tdb")
def test_scheil_solidification_custom_properties(load_database):
    """Test that SolidificationResult objects produced by Scheil solidification calculations with custom outputs have the right shape."""
    dbf = load_database()
    comps = ['AL', 'ZN', 'VA']
    phases = sorted(dbf.phases.keys())

    initial_composition = {v.X('ZN'): 0.3}
    start_temperature = 850

    sol_res = simulate_scheil_solidification(dbf, comps, phases, initial_composition, start_temperature, step_temperature=20.0, verbose=True, output=["HM", "CPM", "NP(LIQUID)"])

    # Note, we aren't checking for correctess right now, just that the shapes are notionally correct
    num_temperatures = len(sol_res.temperatures)
    assert num_temperatures == len(sol_res.output["HM"])
    print("HM", sol_res.output["HM"])
    assert num_temperatures == len(sol_res.output["NP(LIQUID)"])
    print("NP(LIQUID)", sol_res.output["NP(LIQUID)"])
    assert num_temperatures == len(sol_res.output["CPM"])
    print("CPM", sol_res.output["CPM"])

    # Test serialization
    for ky, vl in sol_res.to_dict().items():
        print(ky, vl, type(ky), type(vl))
        json.dumps({ky: vl})
    json.dumps(sol_res.to_dict())

    # Test round tripping to/from dict
    rnd_trip_sol_res = SolidificationResult.from_dict(sol_res.to_dict())
    assert rnd_trip_sol_res.fraction_liquid == sol_res.fraction_liquid
    assert rnd_trip_sol_res.fraction_solid == sol_res.fraction_solid
    assert rnd_trip_sol_res.x_liquid == sol_res.x_liquid
    assert rnd_trip_sol_res.cum_phase_amounts == sol_res.cum_phase_amounts
    assert rnd_trip_sol_res.phase_amounts == sol_res.phase_amounts
    assert rnd_trip_sol_res.temperatures == sol_res.temperatures
    assert rnd_trip_sol_res.converged == sol_res.converged
    assert rnd_trip_sol_res.method == sol_res.method
    assert set(sol_res.output.keys()) == set(rnd_trip_sol_res.output.keys())
    for ky in sol_res.output.keys():
        np.testing.assert_almost_equal(rnd_trip_sol_res.output[ky], sol_res.output[ky])

    # Test to_dataframe doesn't raise
    sol_res.to_dataframe(include_zero_phases=True)
    df = sol_res.to_dataframe(include_zero_phases=False)
    assert "CPM" in df.columns
    assert "HM" in df.columns


@select_database("alzn_mey.tdb")
def test_equilibrium_solidification_result_properties(load_database):
    """Test that SolidificationResult objects produced by equilibrium have the required properties."""
    # Required properties are that the shape of the output arrays are matching
    # NOTE: final phase amounts are not tested because they are not guaranteed
    # to be 0.0 or 1.0 in the same way as in the Scheil simulations.

    dbf = load_database()
    comps = ['AL', 'ZN', 'VA']
    phases = sorted(dbf.phases.keys())

    initial_composition = {v.X('ZN'): 0.3}
    start_temperature = 850
    sol_res = simulate_equilibrium_solidification(dbf, comps, phases, initial_composition,
                                                  start_temperature=start_temperature,
                                                  step_temperature=20.0, verbose=True)

    num_temperatures = len(sol_res.temperatures)
    assert num_temperatures == len(sol_res.fraction_liquid)
    assert num_temperatures == len(sol_res.fraction_solid)
    assert all([num_temperatures == len(np) for np in sol_res.phase_amounts.values()])
    assert all([num_temperatures == len(liq_comps) for liq_comps in sol_res.phase_compositions[sol_res.liquid_phase_name].values()])
    assert all([num_temperatures == len(nphase) for nphase in sol_res.cum_phase_amounts.values()])
    # The final cumulative solid phase amounts is 1.0
    assert np.isclose(np.sum([amnts[-1] for amnts in sol_res.cum_phase_amounts.values()]), 1.0)
    # The final instantaneous phase amounts is not 1.0 (only the amount of new solid phase added
    assert np.sum([amnts[-1] for amnts in sol_res.phase_amounts.values()]) < 1.0

    # Test serialization
    for ky, vl in sol_res.to_dict().items():
        print(ky, vl, type(ky), type(vl))
        json.dumps({ky: vl})
    json.dumps(sol_res.to_dict())

    # Test round tripping to/from dict
    rnd_trip_sol_res = SolidificationResult.from_dict(sol_res.to_dict())
    assert rnd_trip_sol_res.fraction_liquid == sol_res.fraction_liquid
    assert rnd_trip_sol_res.fraction_solid == sol_res.fraction_solid
    assert rnd_trip_sol_res.phase_compositions == sol_res.phase_compositions
    assert rnd_trip_sol_res.cum_phase_amounts == sol_res.cum_phase_amounts
    assert rnd_trip_sol_res.phase_amounts == sol_res.phase_amounts
    assert rnd_trip_sol_res.temperatures == sol_res.temperatures
    assert rnd_trip_sol_res.converged == sol_res.converged
    assert rnd_trip_sol_res.method == sol_res.method

    # Test to_dataframe doesn't raise
    sol_res.to_dataframe(include_zero_phases=True)
    sol_res.to_dataframe(include_zero_phases=False)


@select_database("alzn_mey.tdb")
def test_equilibrium_solidification_custom_properties(load_database):
    """Test that SolidificationResult objects produced by equilibrium solidification calculations with custom outputs have the right shape."""
    dbf = load_database()
    comps = ['AL', 'ZN', 'VA']
    phases = sorted(dbf.phases.keys())

    initial_composition = {v.X('ZN'): 0.3}
    start_temperature = 850

    sol_res = simulate_equilibrium_solidification(dbf, comps, phases, initial_composition, start_temperature=start_temperature, step_temperature=20.0, verbose=True, output=["HM", "CPM", "NP(LIQUID)"])

    # Note, we aren't checking for correctness right now, just that the shapes are notionally correct
    num_temperatures = len(sol_res.temperatures)
    assert num_temperatures == len(sol_res.output["HM"])
    print("HM", sol_res.output["HM"])
    assert num_temperatures == len(sol_res.output["NP(LIQUID)"])
    print("NP(LIQUID)", sol_res.output["NP(LIQUID)"])
    assert num_temperatures == len(sol_res.output["CPM"])
    print("CPM", sol_res.output["CPM"])

    # Test serialization
    for ky, vl in sol_res.to_dict().items():
        print(ky, vl, type(ky), type(vl))
        json.dumps({ky: vl})
    json.dumps(sol_res.to_dict())

    # Test round tripping to/from dict
    rnd_trip_sol_res = SolidificationResult.from_dict(sol_res.to_dict())
    assert rnd_trip_sol_res.fraction_liquid == sol_res.fraction_liquid
    assert rnd_trip_sol_res.fraction_solid == sol_res.fraction_solid
    assert rnd_trip_sol_res.phase_compositions == sol_res.phase_compositions
    assert rnd_trip_sol_res.cum_phase_amounts == sol_res.cum_phase_amounts
    assert rnd_trip_sol_res.phase_amounts == sol_res.phase_amounts
    assert rnd_trip_sol_res.temperatures == sol_res.temperatures
    assert rnd_trip_sol_res.converged == sol_res.converged
    assert rnd_trip_sol_res.method == sol_res.method
    assert set(sol_res.output.keys()) == set(rnd_trip_sol_res.output.keys())
    for ky in sol_res.output.keys():
        np.testing.assert_almost_equal(rnd_trip_sol_res.output[ky], sol_res.output[ky])

    # Test to_dataframe doesn't raise
    sol_res.to_dataframe(include_zero_phases=True)
    df = sol_res.to_dataframe(include_zero_phases=False)
    assert "CPM" in df.columns
    assert "HM" in df.columns
