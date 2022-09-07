from setuptools import find_packages, setup

NAME = 'NEMS'

VERSION = '0.0.1a'
GENERAL_REQUIRES = [
    'numpy==1.21.5',
    'scipy==1.9.0',
    'matplotlib==3.5.2',
    ]
# TODO: pycharm also requires tornado? Or was this a glitch?
# TODO: .ipynb files require jupyter, best way to specify this?
#       probably in EXTRAS, but which package specifically?

EXTRAS_REQUIRES = {
    # TODO: turn some of these back on, temporarily disabled for testing.
    # 'docs': ['sphinx', 'sphinx_rtd_theme', 'pygments-enaml', 'nbsphinx', 
    #          'pandoc', 'IPython', 'sphinx_copybutton'],
    # 'tensorflow': ['tensorflow==2.2', 'tensorboard'],
    # 'tests': ['pytest', 'pytest-benchmark'],
}

setup(
    name=NAME,
    version=VERSION,
    packages=find_packages(exclude=['tests']),
    include_package_data=True,
    zip_safe=True,
    author='LBHB',
    author_email='lbhb.ohsu@gmail.com',
    description='Neural Encoding Model System',
    url='http://neuralprediction.org',
    install_requires=GENERAL_REQUIRES,
    extras_require=EXTRAS_REQUIRES,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Natural Language :: English',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Bio-Informatics'
    ],
)