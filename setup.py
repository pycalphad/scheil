from setuptools import setup
import os
import versioneer

def readme(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='scheil',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author='Brandon Bocklund',
    author_email='brandonbocklund@gmail.com',
    description='Scheil-Gulliver simulations using pycalphad.',
    packages=['scheil'],
    license='MIT',
    long_description=readme('README.rst'),
    url='https://pycalphad.org/',
    install_requires=[
        'numpy',
        'scipy',
        'pycalphad>=0.8.1',
    ],
    extras_require={
        'dev': [
            'sphinx',
            'sphinx_rtd_theme',
            'pytest',
            'twine',
        ],
    },
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Chemistry',

        'License :: OSI Approved :: MIT License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)
