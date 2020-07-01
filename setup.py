"""Package Setup script for mlp."""

from __future__ import print_function

from setuptools import find_packages
from setuptools import setup

with open('requirements.txt') as fp:
  requirements = []
  for line in fp:
    requirements.append(line)


def _make_required_install_packages():
  return requirements


# Get version from version module.
with open('version.py') as fp:
  globals_dict = {}
  exec(fp.read(), globals_dict)  # pylint: disable=exec-used
__version__ = globals_dict['__version__']

setup(
    name='path_nn',
    version=__version__,
    author='',
    author_email='',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Operating System :: Linux',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    namespace_packages=[],
    install_requires=_make_required_install_packages(),
    python_requires='>2.7,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*,<4',
    packages=find_packages(),
    include_package_data=True,
    description='',
    requires=[]
    )
