import itertools
from collections import defaultdict
import numpy as np
from scipy.stats import norm
from pycalphad.core.utils import unpack_components, generate_dof


def get_phase_amounts(eq_phases, phase_fractions, all_phases):
    """Return the phase fraction for each phase in equilibrium

    Parameters
    ----------
    eq_phases : Sequence[str]
        Equilibrium phases
    phase_fractions : Sequence[float]
        Phase amounts corresponding to the equilibrium phases
    all_phases : Sequence[str]
        All phases that we want to keep track of in the system.

    Returns
    -------
    Dict[str, float]

    """
    phase_amnts = {ph: 0.0 for ph in all_phases}
    for phase_name, fraction in zip(eq_phases, phase_fractions):
        if phase_name in all_phases:
            phase_amnts[phase_name] += float(fraction)
    return phase_amnts


def is_ordered(site_fracs, subl_dof, symmetric_subl_idx, **kwargs):
    """Return True if the site fraction configuration is ordered

    Parameters
    ----------
    site_fracs : numpy.ndarray[float,N]
        Site fraction array for a phase of length sum(subl_dof) (can be padded
        with arbitrary data, as returned by equilibrium calculations).
    subl_dof : list[int]
        List of the number of components active in each sublattice. Size should
        be equivalent to the number of sublattices in the phase.
    symmetric_subl_idx : list[list[int]]
        List of sets of symmetrically equivalent sublattice indexes (as a list).
        If sublattices at index 0 and index 1 are symmetric, as index 2 and
        index 3, then the symmetric_subl_idx will be [[0, 1], [2, 3]].
    kwargs :
        Additional keyword arguments passed to ``np.isclose`` to check if site
        fractions in symmetric sublattices are distinct.

    Returns
    -------
    bool

    Examples
    --------
    >>> dbf = Database('Fe-Ni-Ti.tdb')  # doctest: +SKIP
    >>> subl_dof = [4, 4, 1] # sublattice model is (Fe,Ni,Ti,Va)(Fe,Ni,Ti,Va)(Va)  # doctest: +SKIP
    >>> symm = [[0, 1]]  # sublattices 0 and 1 should be equivalent  # doctest: +SKIP
    >>> res = equilibrium(dbf, ['FE', 'NI', 'TI', 'VA'], ['BCC2'], {v.P: 101325, v.T: 1200, v.N: 1, v.X('FE'): 0.001, v.X('NI'): 0.001})  # doctest: +SKIP
    >>> is_ordered(res.Y.isel(vertex=0).values.squeeze(), subl_dof, symm)  # doctest: +SKIP
    False
    >>> res = equilibrium(dbf, ['FE', 'NI', 'TI', 'VA'], ['BCC2'], {v.P: 101325, v.T: 1200, v.N: 1, v.X('FE'): 0.25, v.X('NI'): 0.25})  # doctest: +SKIP
    >>> is_ordered(res.Y.isel(vertex=0).values.squeeze(), subl_dof, symm)  # doctest: +SKIP
    True

    """
    # For each sublattice, create a ``slice`` object for slicing the site
    # fractions of that particular sublattice from the site fraction array
    subl_slices = []
    for subl_idx in range(len(subl_dof)):
        start_idx = np.sum(subl_dof[:subl_idx], dtype=np.int_)
        end_idx = start_idx + subl_dof[subl_idx]
        subl_slices.append(slice(start_idx, end_idx))

    # For each set of symmetrically equivalent sublattices
    for symm_subl in symmetric_subl_idx:
        # Check whether the site fractions of each pair of symmetrically
        # equivalent sublattices are ordered or disordered
        for idx1, idx2 in itertools.combinations(symm_subl, 2):
            # A phase is ordered if any pair of sublattices does not have
            # equal (within numerical tolerance) site fractions
            pair_is_ordered = np.any(~np.isclose(site_fracs[subl_slices[idx1]], site_fracs[subl_slices[idx2]], **kwargs))
            if pair_is_ordered:
                return True
    return False


def order_disorder_dict(dbf, comps, phases):
    """Return a dictionary with the sublattice degrees of freedom and equivalent
    sublattices for order/disorder phases

    Parameters
    ----------
    dbf : pycalphad.Database
    comps : list[str]
        List of active components to consider
    phases : list[str]
        List of active phases to consider

    Returns
    -------
    dict

    Notes
    -----
    Phases which should be checked for ordered/disordered configurations are
    determined heuristically for this script.

    The heuristic for a phase satisfies the following:
    1. The phase is the ordered part of an order-disorder model
    2. The equivalent sublattices have all the same number of elements
    """
    species = unpack_components(dbf, comps)
    ord_disord_phases = {}
    for phase_name in phases:
        phase_obj = dbf.phases[phase_name]
        if phase_name == phase_obj.model_hints.get('ordered_phase', ''):
            # This phase is active and modeled with an order/disorder model.
            dof = generate_dof(dbf.phases[phase_name], species)[1]
            # Define the symmetrically equivalent sublattices as any sublattices
            # that have the same site ratio. Create a {site_ratio: [subl idx]} dict
            site_ratio_idxs = defaultdict(lambda: [])
            for subl_idx, site_ratio in enumerate(phase_obj.sublattices):
                site_ratio_idxs[site_ratio].append(subl_idx)
            equiv_sublattices = list(site_ratio_idxs.values())
            ord_disord_phases[phase_name] = {
                'subl_dof': dof,
                'symmetric_subl_idx': equiv_sublattices,
                'disordered_phase': phase_obj.model_hints['disordered_phase']
            }
    return ord_disord_phases


def order_disorder_eq_phases(eq_result, order_disorder_dict):
    """Return a list corresponding to the eq_result.Phase with order/disorder
    phases named correctly.

    Parameters
    ----------
    eq_result : pycalphad.LightDataset
    order_disorder_dict : Dict

    Returns
    -------
    List
    """
    eq_phases = []
    for vtx in eq_result.vertex.values:
        eq_phase = str(eq_result["Phase"].isel(vertex=vtx).values.squeeze())
        site_fracs = eq_result["Y"].isel(vertex=vtx).values.squeeze()
        if eq_phase in order_disorder_dict:
            odd = order_disorder_dict[eq_phase]
            is_ord = is_ordered(site_fracs, odd['subl_dof'], odd['symmetric_subl_idx'])
            if is_ord:
                eq_phases.append(eq_phase)
            else:
                eq_phases.append(odd['disordered_phase'])
        else:
            eq_phases.append(eq_phase)
    return eq_phases


def local_sample(sitefracs, comp_count, pdens=100, stddev=0.05):
    """Sample from a normal distribution around the optimal site fractions


    Parameters
    ----------
    sitefracs : np.ndarray[:, :]
        2d array of site fractions of shape (N, len(dof))
    comp_count : Sequence[int]
        Number of active components in each sublattice, e.g.
        (FE,NI,TI)(FE,NI)(FE,NI,TI) is [3, 2, 3]
    pdens : Optional[int]
        Number of points to add locally
    stddev : Optional[float]
        Standard deviation for the normal distribution to sample from

    Returns
    -------
    np.ndarray[:, :]
        Shape (pdens, len(dof))

    """
    # Need to take the absolute value, in order to make all the candidate site fractions positive
    # There's probably a better distribution, but this works. Vacancies may cause trouble with this kind of normal distribution, since they are often near zero.
    # Maybe like a gamma between Y=0 and Y=1, with a mean (or cdf of 0.5) optimal site fraction
    pts = np.concatenate(np.abs(norm(sitefracs, stddev).rvs([pdens, sitefracs.shape[0], sitefracs.shape[-1]])), axis=0)
    # add on the original points, since these are the actual minima
    pts = np.concatenate([pts, sitefracs], axis=0)
    # Need to normalize in each sublattice
    # comp_count is # of elements per sublattice, e.g.
    cur_idx = 0
    for ctx in comp_count:
        end_idx = cur_idx + ctx
        pts[:, cur_idx:end_idx] /= pts[:, cur_idx:end_idx].sum(axis=1)[:, None]
        cur_idx = end_idx
    return pts
