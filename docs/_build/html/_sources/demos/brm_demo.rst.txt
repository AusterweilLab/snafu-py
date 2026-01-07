BRM Demo
========

This demo script showcases a comprehensive set of semantic fluency analyses using the `snafu` package.
The examples in this script are adapted from *Zemla, Cao, Mueller, & Austerweil (under revision)* and illustrate how SNAFU handles preprocessing, error detection, semantic metrics, and result export.

The script processes fluency data and demonstrates:

- Importing data by subject, category, or experiment group
- Applying schemes and spell files to clean responses
- Computing:
  - Cluster switches (static or fluid)
  - Switch rate (per item)
  - Cluster size (letter or semantic-based)
  - Perseverations and intrusions
  - Word frequency and age-of-acquisition
- Writing summarized metrics to a CSV file for downstream analysis

**Notable Functions Used:**

- `snafu.load_fluency_data`
- `snafu.clusterSwitch`
- `snafu.clusterSize`
- `snafu.perseverations` / `perseverationsList`
- `snafu.intrusions` / `intrusionsList`
- `snafu.wordFrequency`
- `snafu.ageOfAcquisition`

Output includes:

- `stats.csv`: Average metrics per participant
- `intrusions_list.pkl`: Serialized list of detected intrusions
- `intrusions_list_letter_a.pkl`: Intrusions specific to the letter 'A'

.. note::

   This demo is for illustration purposes and not all combinations of options (e.g., letter clustering with animal data) are semantically appropriate.

----

.. automodule:: demos.brm_demo
   :members:
   :undoc-members:
   :show-inheritance: