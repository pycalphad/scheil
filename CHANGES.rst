=========
Changelog
=========

0.1.6 (2021-08-09)
==================

* Improve performance by building pycalphad phase records in simulations (:issue:`21`)

0.1.5 (2021-06-12)
==================

* Fix a floating point bug where equilibrium solidification could get stuck in an infinite loop after a binary search (:issue:`15`)
* Fix a bug where the disordered part of partitioned phases would not be counted as solidified solid if it became stable (:issue:`16`)

0.1.4 (2020-11-14)
==================

* Fix to PyPI distribution

0.1.3 (2020-11-14)
==================

This is a minor release containing some maintenance changes and bug fixes

* Don't automatically remove the "GAS" phase from the set of solid phases (:issue:`12`)
* Call filter_phases to remove phases that cannot exist (:issue:`11`)
* Developers: switch to GitHub Actions instead of Travis-CI

0.1.2 (2020-01-29)
==================

* Equilibrium solidification improvements
   * Make points updating adaptive
   * Convergence checking
   * Enable order-disorder deconvolution

0.1.1 (2020-01-23)
==================

* Packaging fixes
* Updated LICENSE

0.1 (2020-01-23)
==================

Initial release

* Perform Scheil-Gulliver and equilibrium solidification simulations
