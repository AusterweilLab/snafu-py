Estimate Networks Demo
======================

This demo estimates semantic networks from fluency data using a variety of modeling techniques implemented in SNAFU. It provides a comparative view of different network construction algorithms commonly used in semantic network research.

---

**Overview of Workflow:**

1. **Load animal fluency data** for a specific group (`Experiment1`), applying spell correction and flattening the data.
2. **Define fit parameters** using the `Fitinfo` object (mainly for Conceptual Network).
3. **Estimate semantic networks** using five different methods:
   - **Naive Random Walk (NRW)** – Random walk transition probabilities
   - **Conceptual Network (CN)** – Co-occurrence-based estimation from Goni et al. (2011)
   - **Pathfinder Network (PF)** – Based on distance metrics
   - **Correlation-Based Network (CBN)** – Based on word correlation
   - **First-Edge Network (FE)** – Based on order of item appearance

4. Save each network's edge list as a `.csv` file for further visualization or analysis.

---

**Functions Used:**

- `snafu.load_fluency_data`
- `snafu.Fitinfo`
- `snafu.naiveRandomWalk`
- `snafu.conceptualNetwork`
- `snafu.pathfinder`
- `snafu.correlationBasedNetwork`
- `snafu.firstEdge`
- `snafu.write_graph`

**Output Files:**

Each file contains an edge list in CSV format:

- `nrw_graph.csv`
- `cn_graph.csv`
- `pf_graph.csv`
- `cbn_graph.csv`
- `fe_graph.csv`

All files are saved in the `demos_data/` directory and labeled by group.

----

.. automodule:: demos.estimate_networks
   :members:
   :undoc-members:
   :show-inheritance: