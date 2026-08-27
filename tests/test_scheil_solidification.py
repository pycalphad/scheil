import json
import numpy as np
import pytest
from pycalphad import variables as v
from scheil import simulate_scheil_solidification, simulate_equilibrium_solidification, SolidificationResult
from fixtures import select_database, load_database

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
