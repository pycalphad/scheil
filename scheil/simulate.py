
from pycalphad import equilibrium, variables as v
from pycalphad.core.utils import instantiate_models
from pycalphad.codegen.callables import build_callables
from scheil import SolidifcationResult
import numpy as np
from collections import defaultdict

NAN_DICT = defaultdict(lambda:np.nan)

def simulate_scheil_solidification(dbf, comps, phases, composition,
                                   start_temperature, step_temperature=1.0,
                                   liquid_phase_name='LIQUID', stop=0.0001, verbose=False):
    """Perform a Scheil-Gulliver solidification simulation.

    Parameters
    ----------
    dbf : pycalphad.Database
        Database object.
    comps : list
        List of components in the system.
    phases : list
        List of phases in the system.
    composition : dict
        Dictionary of independent `v.X` composition variables.
    start_temperature : float
        Starting temperature for simulation. Must be single phase liquid.
    step_temperature : float, optional
        Temperature step size. Defaults to 1.0.
    liquid_phase_name : str
        Name of the phase treated as liquid.

    Returns
    -------
    scheil.solidification_result.SolidifcationResult

    """
    models = instantiate_models(dbf, comps, phases)
    cbs = build_callables(dbf, comps, phases, models, additional_statevars={v.P, v.T, v.N})
    solid_phases = sorted(set(phases)-{'GAS', liquid_phase_name})
    temp = start_temperature
    independent_comps = sorted(composition.keys(), key=str)

    x_liquid = [composition]
    fraction_solid = [0.0]
    temperatures = [temp]
    phase_amounts = {ph: [0.0] for ph in solid_phases}

    while fraction_solid[-1] < 1:
        NS = fraction_solid[-1]
        NL = 1 - NS
        if NL < stop:
            break
        conds = {v.T: temp, v.P: 101325}
        conds.update(x_liquid[-1])
        eq = equilibrium(dbf, comps, phases, conds, callables=cbs, model=models)
        eq_phases = eq["Phase"].values.squeeze()
        if liquid_phase_name not in eq["Phase"].values.squeeze():
            break
        # TODO: Will break if there is a liquid miscibility gap
        liquid_vertex = sorted(np.nonzero(eq["Phase"].values.squeeze().flat == liquid_phase_name))[0]
        liquid_comp = {comp: float(eq["X"].isel(vertex=liquid_vertex).squeeze().sel(component=str(comp)[2:]).values) for comp in independent_comps}
        x_liquid.append(liquid_comp)
        if np.nansum(eq["Phase"].values != '') == 1:
            np_liq = np.nansum(eq.where(eq["Phase"] == liquid_phase_name).NP.values)
            if verbose:
                print('T: {}, NL: {:0.3f}, Liquid single phase, liquid NP {}, isclose to 1: {}, phases: {}'.format(temp, NL, np_liq, np.isclose(np_liq, 1.0, atol=2e-6), eq_phases))

        current_fraction_solid = float(fraction_solid[-1])
        for solid_phase in solid_phases:
            if solid_phase not in eq["Phase"].values.squeeze():
                phase_amounts[solid_phase].append(0.0)
                continue
            NP_eq = np.nansum(eq.where(eq["Phase"] == solid_phase)["NP"].values.squeeze())
            np_tieline = NP_eq
            delta_fraction_solid = (1-current_fraction_solid) * np_tieline
            # alternate method for calculate NP based on current phase fractions
            try:
                assert np.isclose(NP_eq, np_tieline)
            except:
                print('NP_eq', NP_eq, 'NP_tieline', np_tieline, conds)
            if verbose:
                print('T: {}, NL: {:0.3f}, NP({})={}, phases: {}'.format(temp, NL, solid_phase, NP_eq, eq_phases))
            current_fraction_solid += delta_fraction_solid
            phase_amounts[solid_phase].append(delta_fraction_solid)

        fraction_solid.append(current_fraction_solid)
        temperatures.append(temp)
        temp -= step_temperature

    if fraction_solid[-1] < 1:
        x_liquid.append(NAN_DICT)
        fraction_solid.append(1.0)
        temperatures.append(temp)
        # set the final phase amount to the phase fractions in the eutectic
        # this method gives the sum total phase amounts of 1.0 by construction
        for solid_phase in solid_phases:
            amount = np.nansum(eq["NP"].isel(T=0,P=0).where(eq["Phase"] == solid_phase).values)
            phase_amounts[solid_phase].append(float(amount)*(1-current_fraction_solid))

    # take the instantaneous phase amounts and make them cumulative
    phase_amounts = {ph: np.cumsum(amnts).tolist() for ph, amnts in phase_amounts.items()}

    return SolidifcationResult(x_liquid, fraction_solid, temperatures, phase_amounts)


def simulate_equilibrium_solidification(dbf, comps, phases, composition,
                                        start_temperature, end_temperature, step_temperature,
                                        liquid_phase_name='LIQUID', callables=None):
    # Compute the equilibrium solidification path
    solid_phases = sorted(set(phases)-{'GAS', liquid_phase_name})
    independent_comps = sorted(composition.keys(), key=str)
    callables = build_callables(dbf, comps, phases)
    conds = {v.T: (end_temperature, start_temperature, step_temperature), v.P: 101325}
    conds.update(composition)
    eq = equilibrium(dbf, comps, phases, conds, **callables)

    temperatures = eq["T"].values.tolist()
    x_liquid = []
    fraction_solid = []
    phase_amounts = {ph: [] for ph in solid_phases}
    for T_idx in reversed(range(len(temperatures))):
        curr_eq = eq.isel(T=T_idx, P=0)
        curr_fraction_solid = 0.0
        # calculate the phase amounts
        for solid_phase in solid_phases:
            amount = float(np.nansum(curr_eq.NP.where(curr_eq.Phase == solid_phase).values))
            phase_amounts[solid_phase].append(amount)
            curr_fraction_solid += amount
        fraction_solid.append(curr_fraction_solid)

        # liquid phase constitution
        if 'LIQUID' in curr_eq.Phase.values:
            # TODO: will break for liquid miscibility gap
            liquid_vertex = sorted(np.nonzero(curr_eq.Phase.values.flat == 'LIQUID'))[0]
            liquid_comp = {comp: float(curr_eq.X.isel(vertex=liquid_vertex).sel(component=str(comp)[2:]).values) for comp in independent_comps}
            x_liquid.append(liquid_comp)
        else:
            x_liquid.append(np.nan)

    return SolidifcationResult(x_liquid, fraction_solid, temperatures, phase_amounts)
