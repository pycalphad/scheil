Releasing scheil
================

Create a release of scheil
--------------------------
To release a new version of scheil:

These steps assume that ``0.1`` is the most recently tagged version number and ``0.2`` is the next version number to be released.
Replace their values with the last public release's version number and the new version number as appropriate.

#. Determine what the next version number should be using `semantic versioning <https://semver.org/>`_.
#. Resolve or defer all pull requests and issues tagged with the upcoming version milestone.
#. ``git stash`` to save any uncommitted work.
#. ``git checkout master``
#. ``git pull`` to make sure you haven't missed any last-minute commits. **After this point, nothing else is making it into this version.**
#. ``pytest`` to ensure that all tests pass locally.
#. ``sphinx-apidoc -f -H 'API Documentation' -o docs/api/ scheil`` to regenerate the API documentation.
#. Update ``CHANGES.rst`` with a human-readable list of changes since the last commit.
   ``git log --oneline --no-decorate --color 0.1^..master`` can be used to list the changes since the last version.
#. ``git add docs/api CHANGES.rst`` to stage the updated documentation.
#. ``git commit -m "REL: 0.2"`` to commit the changes.
#. ``git push origin master``
#. **Verify that all continuous integration test and build workflows pass.**
#. Create a release on GitHub

   #. Go to https://github.com/pycalphad/scheil/releases/new
   #. Set the "Tag version" field to ``0.2``.
   #. Set the branch target to ``master``.
   #. Set the "Release title" to ``scheil 0.2``.
   #. Leave the description box blank.
   #. If this version is a pre-release, check the "This is a pre-release" box.
   #. Click "Publish release".
#. The new version will be available on PyPI when the ``Build and deploy to PyPI`` workflow on GitHub Actions finishes successfully.

Deploy to PyPI (manually)
-------------------------

.. warning::

   DO NOT FOLLOW THESE STEPS unless the GitHub Actions deployment workflow is broken.
   Creating a GitHub release should trigger the ``Build and deploy to PyPI`` workflow on GitHub Actions that will upload source and platform-dependent wheel distributions automatically.

To release a source distribution to PyPI:

#. If deploying for the first time: ``pip install twine build``
#. ``rm -R dist/*`` on Linux/OSX or ``del dist/*`` on Windows
#. ``git checkout master`` to checkout the latest version
#. ``git pull``
#. ``git log`` to verify the repository state matches the newly created tag

#. ``python -m build --sdist``
#. **Make sure that the script correctly detected the new version exactly and not a dirty / revised state of the repo.**
#. ``twine upload dist/*`` to upload (assumes a `correctly configured <https://packaging.python.org/specifications/pypirc/>`_ ``~/.pypirc`` file)


Deploy to conda-forge (manually)
--------------------------------

The `conda-forge autotick bot`_ will automatically open a pull request in the
`conda-forge/scheil-feedstock`_ repository after the package has been uploaded
to PyPI. This usually happens in within an hour of the PyPI release. If the
build succeeds, the PR will be merged automatically and scheil will usually be
available in an hour or two.

.. warning::

   DO NOT FOLLOW THESE STEPS unless the pull request opened by the conda-forge
   autotick bot on the `conda-forge/scheil-feedstock`_ was not merged
   automatically and a new PR needs to be built manually.


Start with the commit checked out which was tagged with the new version.

1. Generate the SHA256 hash of the build artifact (tarball) submitted to PyPI.
   Alternatively, the hashes can be found by clicking the "View" button for the
   source distribution in the `PyPI download files table <https://pypi.org/project/scheil/#files>`_.
2. Fork the `conda-forge/scheil-feedstock`_ repo.
3. Update scheil version and sha256 strings in the ``recipe/meta.yaml`` file.
4. If any of the dependencies changed since the last release, make sure to update the ``recipe/meta.yaml`` file.
5. Submit a pull request to the main scheil feedstock repo.
6. Once the build completes successfully, merge the pull request.

.. _conda-forge autotick bot: https://github.com/regro-cf-autotick-bot
.. _conda-forge/scheil-feedstock: https://github.com/conda-forge/scheil-feedstock
