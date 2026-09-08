import numpy as np
from pycalphad import Workspace, variables as v
from scheil.simulate import _update_points
from fixtures import select_database, load_database


@select_database("alzn_mey.tdb")
def test_update_points_appends_equilibrium_site_fractions(load_database):
    """The point appended by _update_points must be the equilibrium site fractions, not the state variables."""
    dbf = load_database()
    wks = Workspace(dbf, ['AL', 'ZN', 'VA'], ['LIQUID', 'FCC_A1', 'HCP_A3'], {v.T: 700, v.P: 101325, v.N: 1, v.X('ZN'): 0.3})
    dof_dict = {ph: list(map(len, mod.constituents)) for ph, mod in wks.models.items()}
    points_dict = {ph: np.empty((0, sum(dof))) for ph, dof in dof_dict.items()}

    # Single-phase FCC_A1 region for this condition
    assert [cs.phase_record.phase_name for cs in wks.get_composition_sets()] == ['FCC_A1']

    _update_points(wks, points_dict, dof_dict, local_pdens=0)

    assert points_dict['FCC_A1'].shape == (1, 2)
    expected = np.array([wks.get(v.Y('FCC_A1', 0, 'AL')), wks.get(v.Y('FCC_A1', 0, 'ZN'))])
    appended = points_dict['FCC_A1'][0]
    np.testing.assert_allclose(appended, expected, rtol=1e-5)
    np.testing.assert_allclose(appended.sum(), 1.0)
    # The point is nudged so its composition does not coincide exactly with the equilibrium (and next-step condition)
    assert not np.array_equal(appended, expected)
    assert np.all(np.abs(appended - expected) > 0)
    # Other phases were not stable, so nothing should be appended
    assert points_dict['LIQUID'].shape == (0, 2)
    assert points_dict['HCP_A3'].shape == (0, 2)


@select_database("alzn_mey.tdb")
def test_update_points_local_samples_are_valid_site_fractions(load_database):
    """Locally sampled points must be valid site fractions and include the equilibrium point itself."""
    dbf = load_database()
    wks = Workspace(dbf, ['AL', 'ZN', 'VA'], ['LIQUID', 'FCC_A1', 'HCP_A3'], {v.T: 700, v.P: 101325, v.N: 1, v.X('ZN'): 0.3})
    dof_dict = {ph: list(map(len, mod.constituents)) for ph, mod in wks.models.items()}
    points_dict = {ph: np.empty((0, sum(dof))) for ph, dof in dof_dict.items()}


    _update_points(wks, points_dict, dof_dict, local_pdens=20)

    pts = points_dict['FCC_A1']
    assert pts.shape == (21, 2)  # 20 local samples + the (nudged) equilibrium point
    assert np.all(pts >= 0) and np.all(pts <= 1)
    np.testing.assert_allclose(pts.sum(axis=1), 1.0)
    expected = np.array([wks.get(v.Y('FCC_A1', 0, 'AL')), wks.get(v.Y('FCC_A1', 0, 'ZN'))])
    # The equilibrium point must always be present (within the nudge), never exactly coincident
    assert np.any(np.all(np.isclose(pts, expected, rtol=1e-5, atol=0), axis=1))
    assert not np.any(np.all(pts == expected, axis=1))


@select_database("alzn_mey.tdb")
def test_update_points_never_adds_exactly_coincident_point(load_database):
    """No appended point may coincide exactly with the equilibrium site fractions, at any local_pdens.

    The previous step's liquid composition becomes the next step's condition; a grid point exactly at the
    condition composition makes PyCalphad's convex hull search degenerate and ~20x slower.
    """
    dbf = load_database()
    wks = Workspace(dbf, ['AL', 'ZN', 'VA'], ['LIQUID', 'FCC_A1', 'HCP_A3'], {v.T: 800, v.P: 101325, v.N: 1, v.X('ZN'): 0.3})
    dof_dict = {ph: list(map(len, mod.constituents)) for ph, mod in wks.models.items()}
    stable = [cs.phase_record.phase_name for cs in wks.get_composition_sets()]
    assert 'LIQUID' in stable
    expected = np.array([wks.get(v.Y('LIQUID', 0, 'AL')), wks.get(v.Y('LIQUID', 0, 'ZN'))])
    for local_pdens in (0, 10, 100):
        points_dict = {ph: np.empty((0, sum(dof))) for ph, dof in dof_dict.items()}
        _update_points(wks, points_dict, dof_dict, local_pdens=local_pdens)
        pts = points_dict['LIQUID']
        assert pts.shape[0] == local_pdens + 1
        assert not np.any(np.all(pts == expected, axis=1))
        assert np.any(np.all(np.isclose(pts, expected, rtol=1e-5, atol=0), axis=1))
