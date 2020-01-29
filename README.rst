======
scheil
======

A Scheil-Gulliver simulation tool using `pycalphad`_.

.. image:: https://zenodo.org/badge/150358281.svg
   :target: https://zenodo.org/badge/latestdoi/150358281


.. _pycalphad: http://pycalphad.org

.. code-block:: python

   import matplotlib.pyplot as plt
   from pycalphad import Database, variables as v
   from scheil import simulate_scheil_solidification

   # setup the simulation parameters
   dbf = Database('alzn_mey.tdb')
   comps = ['AL', 'ZN', 'VA']
   phases = sorted(dbf.phases.keys())

   liquid_phase_name = 'LIQUID'
   initial_composition = {v.X('ZN'): 0.3}
   start_temperature = 850

   # perform the simulation
   sol_res = simulate_scheil_solidification(dbf, comps, phases, initial_composition, start_temperature, step_temperature=1.0)

   # plot the result
   for phase_name, amounts in sol_res.cum_phase_amounts.items():
       plt.plot(sol_res.temperatures, amounts, label=phase_name)
   plt.plot(sol_res.temperatures, sol_res.fraction_liquid, label='LIQUID')
   plt.ylabel('Phase Fraction')
   plt.xlabel('Temperature (K)')
   plt.title('Al-30Zn Scheil simulation, phase fractions')
   plt.legend(loc='best')
   plt.show()

.. image:: _static/Al-30Zn_Scheil_simulation.png
    :align: center
    :alt: Phase fraction evolution during a Scheil simulation of Al-30Zn

Installation
============

Anaconda (recommended)
----------------------

scheil is suggested to be installed from conda-forge.

.. code-block:: bash

    conda install -c pycalphad -c conda-forge --yes scheil

Development versions
--------------------

To make improvements to scheil, it is suggested to use
Anaconda to download all of the required dependencies. This
method installs scheil with Anaconda, removes specifically the
scheil package, and replaces it with the package from GitHub.

.. code-block:: bash

    git clone https://github.com/pycalphad/scheil.git
    cd scheil
    conda install scheil
    conda remove --force scheil
    pip install --no-deps -e .

Upgrading scheil later requires you to run ``git pull`` in this directory.


Theory
======

Uses classic Scheil-Gulliver theory (see G.H. Gulliver, *J. Inst. Met.* 9 (1913) 120–157 and Scheil, *Zeitschrift Für Met.* 34 (1942) 70–72.) with assumptions of

1. Perfect mixing in the liquid
2. Local equilibrium between solid and liquid
3. No diffusion in the solid


Getting Help
============

For help on installing and using scheil, please join the `pycalphad/pycalphad Gitter room <https://gitter.im/pycalphad/pycalphad>`_.

Bugs and software issues should be reported on `GitHub <https://github.com/pycalphad/scheil/issues>`_.

License
=======

scheil is MIT licensed. See LICENSE.


Citing
======

.. image:: https://zenodo.org/badge/150358281.svg
   :target: https://zenodo.org/badge/latestdoi/150358281


If you use the ``scheil`` package in your work, please cite the relevant version.

The following DOI, `doi:10.5281/zenodo.3630656 <https://doi.org/10.5281/zenodo.3630656>`_, will link to the latest released version of the code on Zenodo where you can cite the specific version that you haved used. For example, version 1.0 can be cited as:

``Bocklund, Brandon, Bobbio, Lourdes D., Otis, Richard A., Beese, Allison M., & Liu, Zi-Kui. (2020, January 29). pycalphad-scheil: 0.1.2 (Version 0.1.2). Zenodo. http://doi.org/10.5281/zenodo.3630657``

.. code-block::

   @software{bocklund_brandon_2020_3630657,
     author       = {Bocklund, Brandon and
                     Bobbio, Lourdes D. and
                     Otis, Richard A. and
                     Beese, Allison M. and
                     Liu, Zi-Kui},
     title        = {pycalphad-scheil: 0.1.2},
     month        = jan,
     year         = 2020,
     publisher    = {Zenodo},
     version      = {0.1.2},
     doi          = {10.5281/zenodo.3630657},
     url          = {https://doi.org/10.5281/zenodo.3630657}
   }

