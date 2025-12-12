2.6.4
[2025-12-09]

Added
Logic requiring _version.py to update before allowing merge.

Changed
Restructured repository so /tests stores stable test inputs/outputs and /demos stores demo outputs only.

[2025-11-07]

Added
Static directory for unit-testing data.


[2025-10-30]

Fixed
_pci_lower_bound visibility issue causing failures in CBN tests.

[2025-10-23]

Added
Completed itemnum column logic in load_fluency_data:
Detects itemnum
Sorts by (subj, category, listnum, itemnum) when present.

[2025-10-17]

Changed
Added precise correlation-rounding (8f) fix for CBN unit test.

[2025-10-01]

Changed
Removed/deleted excess and outdated branches.

[2025-09-04]

Added
Final jump probability implementation with optional log-likelihood return.

[2025-08-26]

Added
Optional boolean flag to return log-likelihood from estimateJumpProbability.

[2025-08-19]

Added
Automation for documentation and unit tests on every push to the dev branch.

[2025-08-11]

Added
Completed and validated jump probability demo.
Fully implemented and validated itemnum sorting behavior.

[2025-08-04]

Added
Fully working Sphinx documentation automation using GitHub Actions.

[2025-07-29]

Added
NumPy-style docstrings throughout the codebase.
Working jump probability implementation (core logic complete).

Changed
Sphinx documentation hosted on GitHub Pages.

[2025-07-21]

Added
GitHub Pages + Sphinx build workflow file for documentation hosting.

[2025-07-14]

Added
Initial jump probability function logic integrated into the codebase.

[2025-06-30]

Added
Documentation overview and demos.
Initial automation for doc builds.

Initial jump probability demo.

[2.6.3]

[2025-06-22]
Added
Initial version of Read the Docs documentation (in progress).

Fixed issues related to uinvite network.

[2025-06-05]
Added
Planarity logic added to core.py (around line 471).

Created directories for test data and demo data.

[2025-06-04]
Changed
Updated project structure with a new directory for data.

Modified test scripts to accommodate directory changes.

[2025-05-27]
Changed
Replaced nx.from_numpy_matrix(usf_graph) with nx.from_numpy_array(usf_graph) in reconstruct_usf.py (line 18).

Added pytest scripts and CSV outputs for test results.
