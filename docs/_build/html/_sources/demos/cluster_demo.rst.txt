Cluster Demo
=============================

This demo illustrates how to compute **item-level cluster switch labeling** for fluency data using the SNAFU library. It is useful for detailed analyses such as:

- Determining whether a word in a fluency list marks a **cluster switch**
- Identifying **intrusions** based on a semantic cluster scheme
- Exporting results at the item level for custom analysis


**Workflow Summary:**

1. **Load fluency data** using a defined semantic category (`animals`) and a spell correction file.
2. **Label each word** in the lists with cluster tags using a provided cluster scheme.
3. **Determine switch status** per item:
   - `1` = cluster switch
   - `0` = same cluster
   - `"intrusion"` = not found in scheme
4. Export all results to `demos_data/switches.csv` for further statistical analysis or visualization.


**Key Functions Used:**

- `snafu.load_fluency_data` — Load data by group and category
- `snafu.labelClusters` — Label each word with cluster(s)
- Custom logic — Determine whether each word initiates a cluster switch

**Output File:**

- `switches.csv`  
  Columns: `id`, `listnum`, `category`, `item`, `switch`  
  Each row corresponds to a word in the original fluency list.

----

.. automodule:: demos.cluster_demo
   :members:
   :undoc-members:
   :show-inheritance: