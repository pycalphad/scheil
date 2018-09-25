================
pycalphad-scheil
================

A Scheil-Gulliver simulation tool using `pycalphad`_.

.. _pycalphad: http://pycalphad.org


Installation
============

Anaconda (recommended)
----------------------

pycalphad-scheil is suggested to be installed from conda-forge.

.. code-block:: bash

    conda install -c pycalphad -c msys2 -c conda-forge --yes pycalphad-scheil

Development versions
--------------------

To make improvements to pycalphad-scheil, it is suggested to use 
Anaconda to download all of the required dependencies. This
method installs pycalphad-scheil with Anaconda, removes specifically the
pycalphad-scheil package, and replaces it with the package from GitHub.

.. code-block:: bash

    git clone https://github.com/pycalphad/pycalphad-scheil.git
    cd pycalphad-scheil
    conda install pycalphad-scheil
    conda remove --force pycalphad-scheil
    pip install -e .

Upgrading pycalphad-scheil later requires you to run ``git pull`` in this directory.


Usage
=====


Getting Help
============

For help on installing and using pycalphad-scheil, please join the `pycalphad/pycalphad Gitter room <https://gitter.im/pycalphad/pycalphad>`_.

Bugs and software issues should be reported on `GitHub <https://github.com/pycalphad/pycalphad-scheil/issues>`_.

License
=======

pycalphad-scheil is MIT licensed. See LICENSE.



.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
