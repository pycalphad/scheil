from setuptools import setup
import os

def readme(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

setup(
    name='scheil',
    author='Brandon Bocklund',
    author_email='brandonbocklund@gmail.com',
    description='Scheil-Gulliver simulations using pycalphad.',
    packages=['scheil'],  # do not include _dev
    license='MIT',
    long_description=readme('README.rst'),
    long_description_content_type='text/x-rst',
    url='https://scheil.readthedocs.io/',
    install_requires=[
        'numpy',
        'scipy',
        'setuptools_scm[toml]>=6.0',
        'pycalphad>=0.8.1',
    ],
    extras_require={
        'dev': [
            'furo',
            'sphinx',
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
