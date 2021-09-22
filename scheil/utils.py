import numpy as np
from scipy.stats import norm


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
