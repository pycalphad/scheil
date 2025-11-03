import sys
from typing import Mapping, List
from numpy.typing import ArrayLike
import numpy as np
from pycalphad import variables as v, Workspace
from pycalphad.codegen.phase_record_factory import PhaseRecordFactory
from pycalphad.core.calculate import _sample_phase_constitution
from pycalphad.core.utils import instantiate_models, unpack_species, filter_phases, point_sample
from pycalphad.property_framework import ComputableProperty
from .solidification_result import SolidificationResult, PhaseName
from .utils import local_sample, get_phase_amounts
from .ordering import create_ordering_records, rename_disordered_phases, _wks_ordering_rename_map


def is_converged(wks):
    """
    Return true if there are phase fractions that are non-NaN

    Parameters
    ----------
    eq : pycalphad.LightDataset

    """
    if np.any(~np.isnan(wks.get("NP(*)"))):
        return True
    return False


def _update_points(wks, points_dict, dof_dict, local_pdens=0, verbose=False):
    """
    Update the points_dict by appending new points.

    Parameters
    ----------
    eq : pycalphad.LightDataset
        Point equilibrium result. Incompatible with xarray.Dataset objects.
    points_dict : dict[str, np.ndarray]
        Map of phase name to array of points
    dof_dict : dict[str, list[int]]
        Map of phase name to the sublattice degrees of freedom.
    local_pdens : Optional[int]
        Point density for local sampling. If zero (the default) only the equilibrium site fractions will be added.
    verbose : Optional[bool]

    """
    # Update the points dictionary with local samples around the equilibrium site fractions
    for compset in wks.get_composition_sets():
        phase_name = compset.phase_record.phase_name
        pts = points_dict.get(phase_name)
        if pts is not None:
            if verbose:
                print(f'Adding points to {phase_name}. ', end='')
            dof = dof_dict[phase_name]
            eq_pts = np.asarray(compset.dof[:sum(dof)]).reshape(1, -1)
            if local_pdens > 0:
                points_dict[phase_name] = np.concatenate([pts, local_sample(eq_pts, dof, pdens=local_pdens)], axis=0)
            else:
                points_dict[phase_name] = np.concatenate([pts, eq_pts], axis=0)


def _get_stable_phases_with_multiplicities(wks: Workspace):
    multiplicity_aware_phase_names = []
    for phase_name, phase_multiplicity in wks._detect_phase_multiplicity().items():
        if phase_multiplicity == 0:
            continue
        elif phase_multiplicity == 1:
            multiplicity_aware_phase_names.append(phase_name)
        else:
            for multiplicity in range(1, phase_multiplicity + 1):
                multiplicity_aware_phase_names.append(f"{phase_name}#{multiplicity}")
    return multiplicity_aware_phase_names


def _update_phase_compositions(phase_compositions: Mapping[PhaseName, Mapping[str, List[float]]], wks: Workspace, ordering_phase_name_remap: dict[PhaseName, PhaseName]):
    """
    Parameters
    ----------
    phase_compositions : Mapping[PhaseName, Mapping[ComponentName, List[float]]]
    wks: Workspace
        Containing a PyCalphad point equilibrium
    """
    components = set(comp.name for comp in wks.components) - {"VA"}

    # get phase names local to this equilibrium, assumes wks represents a point equilibrium
    multiplicity_aware_phase_names = _get_stable_phases_with_multiplicities(wks)

    # we need to handle the fact that new phases can come up (due to multiplicities) and pad those out with NaN appropriately
    for phase_name in multiplicity_aware_phase_names:
        phase_name = ordering_phase_name_remap.get(phase_name, phase_name)  # we can update phase_name directly because we don't ever call into PyCalphad (using the unmangled name)
        if phase_name in phase_compositions:
            continue
        # at this point, we haven't modified phase_compositions at all, so the shapes should be the same
        # choose a phase arbitrarily
        num_recorded_phase_comps = len(list(list(phase_compositions.values())[0].values())[0])
        phase_compositions[phase_name] = {}
        for component in components:
            phase_compositions[phase_name][component] = np.full(num_recorded_phase_comps, np.nan).tolist()

    # append values for stable phases
    for phase_name in multiplicity_aware_phase_names:
        ord_aware_phase_name = ordering_phase_name_remap.get(phase_name, phase_name)  # we need special treatment because we call PyCalphad with the original composition set name
        for component in components:
            phase_compositions[ord_aware_phase_name][component].append(float(wks.get(f"X({phase_name},{component})")))

    # pad all other (unstable) phases with NaN
    stable_phases = set(multiplicity_aware_phase_names) - set(ordering_phase_name_remap.keys()) | set(ordering_phase_name_remap.values())
    for phase_name in set(phase_compositions.keys()) - stable_phases:
        for component in components:
            phase_compositions[phase_name][component].append(np.nan)


def simulate_scheil_solidification(dbf, comps, phases, composition,
                                   start_temperature, step_temperature=1.0,
                                   liquid_phase_name='LIQUID', eq_kwargs=None,
                                   stop=0.0001, verbose=False, adaptive=True,
                                   output: list[str | ComputableProperty] | None = None,
                                   ):
    """Perform a Scheil-Gulliver solidification simulation.

    Parameters
    ----------
    dbf : pycalphad.Database
        Database object.
    comps : list
        List of components in the system.
    phases : list
        List of phases in the system.
    composition : Dict[v.X, float]
        Dictionary of independent ``v.X`` composition variables.
    start_temperature : float
        Starting temperature for simulation. Must be single phase liquid.
    step_temperature : Optional[float]
        Temperature step size. Defaults to 1.0.
    liquid_phase_name : Optional[str]
        Name of the phase treated as liquid (i.e. the phase with infinitely
        fast diffusion). Defaults to 'LIQUID'.
    eq_kwargs: Optional[Dict[str, Any]]
        Keyword arguments for equilibrium
    stop: Optional[float]
        Stop when the phase fraction of liquid is below this amount.
    adaptive: Optional[bool]
        Whether to add additional points near the equilibrium points at each
        step. Only takes effect if ``points`` is in the eq_kwargs dict.
    output: list[str | ComputableProperty] | None,
        List of PyCalphad computable properties to access (via ``Workspace.get()``)
        at each temperature. For Scheil simulations, the outputs will contain
        properties for "instantaneous" (N=1) properties each temperature step
        and may need post-processing to be useful.

    Returns
    -------
    SolidificationResult

    """
    eq_kwargs = eq_kwargs or dict()
    STEP_SCALE_FACTOR = 1.2  # How much to try to adapt the temperature step by
    MAXIMUM_STEP_SIZE_REDUCTION = 5.0
    T_STEP_ORIG = step_temperature
    phases = filter_phases(dbf, unpack_species(dbf, comps), phases)
    ordering_records = create_ordering_records(dbf, comps, phases)
    if 'model' in eq_kwargs:
        raise ValueError("Use `models` in eq_kwargs instead of `model`")
    if 'models' not in eq_kwargs:
        eq_kwargs['models'] = instantiate_models(dbf, comps, phases)
        eq_kwargs['phase_record_factory'] = PhaseRecordFactory(dbf, comps, [v.N, v.P, v.T], eq_kwargs['models'])
    models = eq_kwargs['models']
    if output is None:
        output = []
    filtered_disordered_phases = {ord_rec.disordered_phase_name for ord_rec in ordering_records}
    solid_phases = sorted((set(phases) | filtered_disordered_phases) - {liquid_phase_name})
    temp = start_temperature
    independent_comps = sorted([str(comp)[2:] for comp in composition.keys()])
    pure_comps = sorted(set(comps) - {"VA"})
    fraction_solid = [0.0]
    temperatures = [temp]
    phase_amounts = {ph: [0.0] for ph in solid_phases}
    phase_compositions = {ph: {comp: [np.nan] for comp in pure_comps} for ph in sorted(set(solid_phases) | {liquid_phase_name})}
    # TODO: the initial custom outputs may not be the same shape (on the inner values) as the usual outputs
    custom_outputs = {str(out): [np.nan] for out in output}

    if adaptive:
        dof_dict = {phase_name: list(map(len, mod.constituents)) for phase_name, mod in models.items()}
        eq_kwargs.setdefault('calc_opts', {})
        # TODO: handle per-phase/unpackable points and pdens
        if 'points' not in eq_kwargs['calc_opts']:
            if verbose:
                print('generating points... ', end='')
            points_dict = {}
            for phase_name, mod in models.items():
                if verbose:
                    print(phase_name, end=' ')
                pdens = eq_kwargs['calc_opts'].get('pdens', 1000)
                # Assume no phase_local_conditions, this is probably okay since there's no option to add additional conditions here
                # And I don't think it would make too much sense to have phase local conditions for scheil/equilibrium solidification anyways
                points_dict[phase_name] = _sample_phase_constitution(mod, point_sample, True, pdens=pdens, phase_local_conditions={})
            eq_kwargs['calc_opts']['points'] = points_dict
            if verbose:
                print('done')

    converged = False
    phases_seen = {liquid_phase_name, ''}
    liquid_comp = composition
    wks = Workspace(dbf, comps, phases, calc_opts=eq_kwargs.get("calc_opts"))
    last_converged_wks = None
    while fraction_solid[-1] < 1:
        conds = {v.T: temp, v.P: 101325.0, v.N: 1.0}
        comp_conds = liquid_comp
        fmt_comp_conds = ', '.join([f'{c}={val:0.2f}' for c, val in comp_conds.items()])
        conds.update(comp_conds)
        wks.conditions = conds
        wks.calc_opts.update(eq_kwargs["calc_opts"])
        if not np.isnan(wks.get("GM")):
            last_converged_wks = wks.copy()
        if adaptive:
            _update_points(wks, eq_kwargs['calc_opts']['points'], dof_dict, verbose=verbose, local_pdens=100)

        ordering_phase_name_remap = _wks_ordering_rename_map(wks, ordering_records)
        eq_phases = set(phase_name for phase_name, multiplicity in wks._detect_phase_multiplicity().items() if multiplicity > 0)
        new_phases_seen = set(eq_phases).difference(phases_seen)
        if len(new_phases_seen) > 0:
            if verbose:
                print(f'New phases seen: {new_phases_seen}. ', end='')
            phases_seen |= new_phases_seen
        if liquid_phase_name not in eq_phases:
            found_ph = set(eq_phases) - {''}
            if verbose:
                print(f'No liquid phase found at T={temp:0.3f}, {fmt_comp_conds}. (Found {found_ph}) ', end='')
            if len(found_ph) == 0:
                # No phases found in equilibrium. Just continue on lowering the temperature without changing anything
                if verbose:
                    print(f'(Convergence failure) ', end='')
            if T_STEP_ORIG / step_temperature > MAXIMUM_STEP_SIZE_REDUCTION:
                # Only found solid phases and the step size has already been reduced. Stop running without converging.
                if verbose:
                    print('Maximum step size reduction exceeded. Stopping.')
                converged = False
                break
            else:
                # Only found solid phases. Try reducing the step size to zero-in on the correct phases
                if verbose:
                    print(f'Stepping back and reducing step size.')
                temp += step_temperature
                step_temperature /= STEP_SCALE_FACTOR
                temp -= step_temperature
                continue
        # TODO: liquid composition will be wrong if there is a liquid miscibility gap
        liquid_comp = {}
        for comp in independent_comps:
            liquid_comp[v.X(comp)] = float(wks.get(f"X({liquid_phase_name},{comp})"))
        _update_phase_compositions(phase_compositions, wks, ordering_phase_name_remap)
        np_liq = np.nansum(wks.get(f"NP({liquid_phase_name})"))
        current_fraction_solid = float(fraction_solid[-1])
        found_phase_amounts = [(liquid_phase_name, np_liq)]  # tuples of phase name, amount
        multiplicity_aware_stable_phases = _get_stable_phases_with_multiplicities(wks)
        for phase_name in multiplicity_aware_stable_phases:
            # we need special name handling here because we still need to use the multiplicity aware name for PyCalphad Workspace calls
            ord_aware_phase_name = ordering_phase_name_remap.get(phase_name, phase_name)
            if phase_name == liquid_phase_name:
                continue
            if ord_aware_phase_name not in phase_amounts:
                # new phase, found, we need to pad with zeros
                # choose an arbitrary phase to count
                num_steps = len(list(phase_amounts.values())[0])
                phase_amounts[ord_aware_phase_name] = np.full(num_steps, 0.0).tolist()
            np_tieline = float(wks.get(f"NP({phase_name})"))
            found_phase_amounts.append((phase_name, np_tieline))
            delta_fraction_solid = (1 - current_fraction_solid) * np_tieline
            current_fraction_solid += delta_fraction_solid
            phase_amounts[ord_aware_phase_name].append(delta_fraction_solid)
        stable_phases = set(multiplicity_aware_stable_phases) - set(ordering_phase_name_remap.keys()) | set(ordering_phase_name_remap.values())
        for unstable_phase_name in set(phase_amounts.keys()) - set(stable_phases):
            # pad instantaneous stable phases that we know about with zero
            phase_amounts[unstable_phase_name].append(0.0)
        fraction_solid.append(current_fraction_solid)
        temperatures.append(temp)
        NL = 1 - fraction_solid[-1]
        for out in output:
            custom_outputs[str(out)].append(wks.get(out))
        if verbose:
            phase_amnts = ' '.join([f'NP({ph})={amnt:0.3f}' for ph, amnt in found_phase_amounts])
            if NL < 1.0e-3:
                print(f'T={temp:0.3f}, {fmt_comp_conds}, ΔT={step_temperature:0.3f}, NL: {NL:.2E}, {phase_amnts} ', end='')
            else:
                print(f'T={temp:0.3f}, {fmt_comp_conds}, ΔT={step_temperature:0.3f}, NL: {NL:0.3f}, {phase_amnts} ', end='')
        if NL < stop:
            if verbose:
                print(f'Liquid fraction below criterion {stop} . Stopping at {fmt_comp_conds}')
            converged = True
            break
        if verbose:
            print()  # add line break
        temp -= step_temperature

    if fraction_solid[-1] < 1:
        if last_converged_wks is None:
            raise ValueError("No calculations converged.")
        _update_phase_compositions(phase_compositions, last_converged_wks, ordering_phase_name_remap)
        # set the final phase amount to the phase fractions in the eutectic
        # this method gives the sum total phase amounts of 1.0 by construction
        fraction_solid.append(1.0)
        temperatures.append(temp)
        for out in output:
            custom_outputs[str(out)].append(wks.get(out))
        multiplicity_aware_stable_phases = _get_stable_phases_with_multiplicities(last_converged_wks)
        for phase_name in multiplicity_aware_stable_phases:
            # we need special name handling here because we still need to use the multiplicity aware name for PyCalphad Workspace calls
            ord_aware_phase_name = ordering_phase_name_remap.get(phase_name, phase_name)
            if phase_name == liquid_phase_name:
                continue
            if ord_aware_phase_name not in phase_amounts:
                # new phase, found, we need to pad with zeros
                # choose an arbitrary phase to count
                num_steps = len(list(phase_amounts.values())[0])
                phase_amounts[ord_aware_phase_name] = np.full(num_steps, 0.0).tolist()
            amount = float(last_converged_wks.get(f"NP({phase_name})"))
            phase_amounts[ord_aware_phase_name].append(float(amount) * (1 - current_fraction_solid))
        stable_phases = set(multiplicity_aware_stable_phases) - set(ordering_phase_name_remap.keys()) | set(ordering_phase_name_remap.values())
        for unstable_phase_name in set(phase_amounts.keys()) - set(stable_phases):
            # pad instantaneous stable phases that we know about with zero
            phase_amounts[unstable_phase_name].append(0.0)

    return SolidificationResult(phase_compositions, fraction_solid, temperatures, phase_amounts, converged, "scheil", output=custom_outputs)


def simulate_equilibrium_solidification(dbf, comps, phases, composition,
                                        start_temperature, step_temperature=1.0,
                                        liquid_phase_name='LIQUID', adaptive=True, eq_kwargs=None,
                                        binary_search_tol=0.1,
                                        verbose=False, output: list[str | ComputableProperty] | None = None):
    """
    Compute the equilibrium solidification path.

    Decreases temperature until no liquid is found, performing a binary search to get the soildus temperature.

    dbf : pycalphad.Database
        Database object.
    comps : list
        List of components in the system.
    phases : list
        List of phases in the system.
    composition : Dict[v.X, float]
        Dictionary of independent ``v.X`` composition variables.
    start_temperature : float
        Starting temperature for simulation. Should be single phase liquid.
    step_temperature : Optional[float]
        Temperature step size. Defaults to 1.0.
    liquid_phase_name : Optional[str]
        Name of the phase treated as liquid (i.e. the phase with infinitely
        fast diffusion). Defaults to 'LIQUID'.
    eq_kwargs: Optional[Dict[str, Any]]
        Keyword arguments for equilibrium
    binary_search_tol : float
        Stop the binary search when the difference between temperatures is less than this amount.
    adaptive: Optional[bool]
        Whether to add additional points near the equilibrium points at each
        step. Only takes effect if ``points`` is in the eq_kwargs dict.
    output: list[str | ComputableProperty] | None
        List of PyCalphad computable properties to access (via ``Workspace.get()``)
        at each temperature. In equilibrium solidification simulations, the
        outputs will be the equilibrium ("cumulative") properties at each time
        step and may need post-processing to be useful.

    """
    eq_kwargs = eq_kwargs or dict()
    phases = filter_phases(dbf, unpack_species(dbf, comps), phases)
    ordering_records = create_ordering_records(dbf, comps, phases)
    filtered_disordered_phases = {ord_rec.disordered_phase_name for ord_rec in ordering_records}
    solid_phases = sorted((set(phases) | filtered_disordered_phases) - {liquid_phase_name})
    pure_comps = sorted(set(comps) - {"VA"})
    phase_compositions = {ph: {comp: [] for comp in pure_comps} for ph in sorted(set(solid_phases) | {liquid_phase_name})}
    independent_comps = sorted([str(comp)[2:] for comp in composition.keys()])
    if 'model' in eq_kwargs:
        raise ValueError("Use `models` in eq_kwargs instead of `model`")
    if 'models' not in eq_kwargs:
        eq_kwargs['models'] = instantiate_models(dbf, comps, phases)
        eq_kwargs['phase_record_factory'] = PhaseRecordFactory(dbf, comps, [v.N, v.P, v.T], eq_kwargs['models'])
    models = eq_kwargs['models']
    if output is None:
        output = []
    custom_outputs = {str(out): [] for out in output}
    if verbose:
        print('building PhaseRecord objects... ', end='')
    if verbose:
        print('done')
    conds = {v.P: 101325, v.N: 1.0}
    conds.update(composition)


    if adaptive:
        dof_dict = {phase_name: list(map(len, mod.constituents)) for phase_name, mod in models.items()}
        eq_kwargs.setdefault('calc_opts', {})
        # TODO: handle per-phase/unpackable points and pdens
        if 'points' not in eq_kwargs['calc_opts']:
            # construct a points dict for the user
            points_dict = {}
            for phase_name, mod in models.items():
                pdens = eq_kwargs['calc_opts'].get('pdens', 50)
                # Assume no phase_local_conditions, this is probably okay since there's no option to add additional conditions here
                # And I don't think it would make too much sense to have phase local conditions for scheil/eq solidification anyways
                points_dict[phase_name] = _sample_phase_constitution(mod, point_sample, True, pdens=pdens, phase_local_conditions = {})
            eq_kwargs['calc_opts']['points'] = points_dict

    temperatures = []
    fraction_solid = []
    phase_amounts = {ph: [] for ph in solid_phases}  # instantaneous phase amounts
    cum_phase_amounts = {ph: [] for ph in solid_phases}
    converged = False
    current_T = start_temperature
    if verbose:
        print('T=')
    while (fraction_solid[-1] < 1 if len(fraction_solid) > 0 else True) and not converged:
        sys.stdout.flush()
        conds[v.T] = current_T
        if verbose:
            print(f'{current_T} ', end='')
        wks = Workspace(dbf, comps, phases, conds, **eq_kwargs)
        if not is_converged(wks):
            if verbose:
                comp_conds = {cond: val for cond, val in conds.items() if isinstance(cond, v.X)}
                print(f"Convergence failure at T={conds[v.T]} X={comp_conds} ")
        if adaptive:
            # Update the points dictionary with local samples around the equilibrium site fractions
            _update_points(wks, eq_kwargs['calc_opts']['points'], dof_dict, verbose=verbose, local_pdens=100)
        ordering_phase_name_remap = _wks_ordering_rename_map(wks, ordering_records)
        _update_phase_compositions(phase_compositions, wks, ordering_phase_name_remap)
        stable_phases = set(phase_name for phase_name, multiplicity in wks._detect_phase_multiplicity().items() if multiplicity > 0)
        liquid_is_stable = liquid_phase_name in stable_phases
        if liquid_is_stable:
            for out in output:
                custom_outputs[str(out)].append(wks.get(out))
            temperatures.append(current_T)
            current_T -= step_temperature
        else:
            # binary search to find the solidus
            T_high = current_T + step_temperature  # High temperature, liquid
            T_low = current_T  # Low temperature, solids only
            if verbose:
                print(f'Found phases {stable_phases - {""}}. Starting binary search between T={(T_low, T_high)} ', end='')
            while (T_high - T_low) > binary_search_tol:
                bin_search_T = (T_high - T_low) * 0.5 + T_low
                conds[v.T] = bin_search_T
                wks = Workspace(dbf, comps, phases, conds, **eq_kwargs)
                if not is_converged(wks):
                    if verbose:
                        comp_conds = {cond: val for cond, val in conds.items() if isinstance(cond, v.X)}
                        print(f"Convergence failure at T={conds[v.T]} X={comp_conds} ")
                if adaptive:
                    # Update the points dictionary with local samples around the equilibrium site fractions
                    _update_points(wks, eq_kwargs['calc_opts']['points'], dof_dict, verbose=verbose, local_pdens=100)
                # Check if liquid is present in this binary search step
                bin_search_stable_phases = set(phase_name for phase_name, multiplicity in wks._detect_phase_multiplicity().items() if multiplicity > 0)
                if liquid_phase_name in bin_search_stable_phases:
                    T_high = bin_search_T
                else:
                    T_low = bin_search_T
            converged = True
            conds[v.T] = T_low
            temperatures.append(T_low)
            wks = Workspace(dbf, comps, phases, conds, **eq_kwargs)
            if not is_converged(wks):
                if verbose:
                    comp_conds = {cond: val for cond, val in conds.items() if isinstance(cond, v.X)}
                    print(f"Convergence failure at T={conds[v.T]} X={comp_conds} ")
            final_stable_phases = set(phase_name for phase_name, multiplicity in wks._detect_phase_multiplicity().items() if multiplicity > 0)
            if verbose:
                found_phases = final_stable_phases - {''}
                phase_amounts_str = ", ".join([f"NP({ph})={float(wks.get(f'NP({ph})')):0.3f}" for ph in found_phases])
                print(f"Finished binary search at T={conds[v.T]} with phases={found_phases} and {phase_amounts_str}")
            if adaptive:
                # Update the points dictionary with local samples around the equilibrium site fractions
                _update_points(wks, eq_kwargs['calc_opts']['points'], dof_dict, verbose=verbose, local_pdens=100)
            # Recalculate ordering remap for the final converged workspace after binary search
            ordering_phase_name_remap = _wks_ordering_rename_map(wks, ordering_records)
            # Add custom outputs for the final converged point
            for out in output:
                custom_outputs[str(out)].append(wks.get(out))

        # Calculate fraction of solid and solid phase amounts
        current_fraction_solid = 0.0
        multiplicity_aware_stable_phases = _get_stable_phases_with_multiplicities(wks)

        # Calculate cumulative phase amounts for solid phases
        current_cum_phase_amnts = {ph: 0.0 for ph in solid_phases}
        for phase_name in multiplicity_aware_stable_phases:
            # Apply ordering-aware naming
            ord_aware_phase_name = ordering_phase_name_remap.get(phase_name, phase_name)
            if ord_aware_phase_name == liquid_phase_name:
                continue
            if ord_aware_phase_name in current_cum_phase_amnts:
                phase_amount = float(wks.get(f"NP({phase_name})"))
                current_cum_phase_amnts[ord_aware_phase_name] += phase_amount

        # Update phase amounts with instantaneous values
        for solid_phase in solid_phases:
            amount = current_cum_phase_amnts[solid_phase]
            # Since the equilibrium calculations always give the "cumulative" phase amount,
            # we need to take the difference to get the instantaneous.
            cum_phase_amounts[solid_phase].append(amount)
            if len(phase_amounts[solid_phase]) == 0:
                phase_amounts[solid_phase].append(amount)
            else:
                phase_amounts[solid_phase].append(amount - cum_phase_amounts[solid_phase][-2])
            current_fraction_solid += amount
        fraction_solid.append(current_fraction_solid)

    converged = True if np.isclose(fraction_solid[-1], 1.0) else False
    return SolidificationResult(phase_compositions, fraction_solid, temperatures, phase_amounts, converged, "equilibrium", output=custom_outputs)
