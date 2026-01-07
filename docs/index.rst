.. SNAFU documentation master file, created by
   sphinx-quickstart on Sun Jun 29 02:36:05 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SNAFU: The Semantic Network and Fluency Utility
===============================================

What is SNAFU?
--------------

The semantic fluency task is frequently used in psychology by both researchers and clinicians. SNAFU is a tool that helps you analyze fluency data. It can help with:

- Counting cluster switches and cluster sizes
- Counting perseverations
- Detecting intrusions
- Calculating average age-of-acquisition and word frequency
- ...more!

SNAFU also implements multiple network estimation methods which allow you to perform network analysis on your data (see Zemla & Austerweil, 2018). These methods include:

- U-INVITE networks
- Pathfinder networks
- Correlation-based networks
- Naive random walk network
- Conceptual networks
- First Edge networks

.. _Zemla & Austerweil, 2018: https://link.springer.com/article/10.1007/s42113-018-0003-7

How do I use SNAFU??
-------------------

SNAFU can be used as a Python library or as a stand-alone GUI.  
The Python library is available at:

`https://github.com/AusterweilLab/snafu-py <https://github.com/AusterweilLab/snafu-py>`_

To install directly:

::

    pip install git+https://github.com/AusterweilLab/snafu-py

The GitHub repository contains several demo files, and a tutorial covering some basic usage is available in Zemla et al., 2020.

A graphical front-end is also available, though it contains fewer features than the Python library. Download it here:

**macOS**

- `SNAFU 2.4.1 for macOS (latest version) <https://alab.psych.wisc.edu/snafu/snafu-2.4.1-mac-x64.dmg>`_

**Windows**

- `SNAFU 2.2.0 for Windows <https://alab.psych.wisc.edu/snafu/snafu-2.2.0-win-x64.zip>`_

.. _Zemla et al., 2020: https://pmc.ncbi.nlm.nih.gov/articles/PMC7406526/

How can I reference SNAFU?
--------------------------

The primary citation for SNAFU is:

    Zemla, J. C., Cao, K., Mueller, K. D., & Austerweil, J. L. (2020).  
    *SNAFU: The semantic network and fluency utility.*  
    Behavior Research Methods, 52, 1681-1699.

Additional citations depending on specific data files used:

- **animals_snafu_scheme.csv**  
  Troyer (2000), Hills et al. (2012)

- **foods_snafu_scheme.csv**  
  Troyer (2000)

- **kuperman.csv (AoA norms)**  
  Kuperman et al. (2012)

- **subtlex-us.csv (Word frequency)**  
  Brysbaert & New (2009)

- **Dutch_animals_snafu_scheme.csv**  
  Rofes et al. (2023) `GitHub Link <https://github.com/jmjvonk/2022_Rofes_SMART-MR/blob/main/Dutch_animals_snafu_scheme.csv>`_

- **animals_Scheme_Greek_Karousou_v.01.2.csv**  
  Karousou et al. (2023)

- **animals_ESnoaccent_scheme.csv**  
  Neergaard et al. (2025)

- **animals_snafu_mexican_spanish.csv**  
  Adapted by Yamilka Garcia Avila and Yaira Chamorro

- **Italian scheme**  
  Provided by Sabia Costantini (Universität Potsdam)

Need help?
----------

Visit our `Google Group <https://groups.google.com/forum/#!forum/snafu-fluency>`_ or e-mail:

    snafu-fluency@googlegroups.com
 
.. toctree::
   :maxdepth: 2
   :caption: Contents:

   Overview
   snafu/index
   demos/index
   References

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
