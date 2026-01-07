Estimate Food Network Demo
==========================

This demo showcases how to estimate a semantic network from fluency data in the **"foods"** category using the **Conceptual Network** method. The script walks through data cleaning, error detection, and network construction based on co-occurrence logic.

---

**Steps Performed:**

1. Load fluency data grouped by participant and apply spell correction.
2. Detect and list:
   - **Intrusions**: Items not present in the category scheme
   - **Perseverations**: Repeated items within the same list
3. Flatten the data (convert hierarchical data to list-level) to prepare for network estimation.
4. Generate a **Conceptual Network** using item co-occurrence.
5. Export the resulting semantic network as an edge list (`foods_network.csv`).

---

**Key Parameters (in `Fitinfo`):**

- `cn_windowsize`: Size of the sliding window for co-occurrence
- `cn_threshold`: Minimum list frequency required for an item to be included
- `cn_alpha`: Significance level for co-occurrence edge inclusion

---

**Relevant Functions Used:**

- `snafu.load_fluency_data`
- `snafu.intrusions` / `intrusionsList`
- `snafu.perseverations` / `perseverationsList`
- `snafu.Fitinfo`
- `snafu.conceptualNetwork`
- `snafu.write_graph`

**Output:**

- `demos_data/foods_network.csv` — A full edge list of the estimated semantic network.

----

.. automodule:: demos.estimate_food_network
   :members:
   :undoc-members:
   :show-inheritance: