# Without thorough testing, this first attempt at a setup file with regard to
# dependency version requirements which will probably cause issues for some
# users. I expect it to work on Python 2.7, 2.8, and 2.9. It is not compatible
# with Python 2.6 or 3.0

from setuptools import setup

setup(name='snafu-py',
      version='1.0',
      description='Analyze your fluency data and build networks from there',
      python_requires='>=2.7,<3.0',
      url='https://github.com/AusterweilLab/snafu-py',
      author='The Austerweil Lab at UW-Madison',
      author_email='jeffzemla@gmail.com',
      #license='MIT',
      keywords=['fluency', 'networks'],
      packages=['rw'],
      install_requires = ['numpy','networkx','scipy','more_itertools'],
      classifiers=[ ]
      )
