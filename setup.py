from setuptools import setup, find_packages
import re

# https://stackoverflow.com/a/7071358/353278
VERSIONFILE="snafu/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

setup(name='snafu',
      version=verstr,
      description='Generate semantic networks from fluency data',
      url='https://github.com/AusterweilLab/snafu-py',
      author='The Austerweil Lab at UW-Madison',
      author_email='austerweil.lab@gmail.com',
      keywords=['fluency', 'networks'],
      packages=['snafu'],
      include_package_data=True,
      install_requires=['numpy','networkx','scipy'],
      zip_safe=False,
      classifiers=[
            'Programming Language :: Python :: 2',
            'Programming Language :: Python :: 2.7'
      ]
      )
