Reconstruct USF Network Demo
============================

This demo simulates semantic fluency data from a known semantic network (the USF animal subset) and evaluates how well various network estimation methods can **reconstruct** the original network.

It demonstrates the power and limitations of different modeling approaches as the number of simulated participants increases.

---

**What This Script Does:**

1. **Imports the USF semantic network** (Nelson et al., 1999)
2. **Generates simulated fluency data** using censored random walks over the USF network
3. **Fits new networks** to the simulated data using several methods:
   - Naive Random Walk
   - Conceptual Network
   - Pathfinder
   - Correlation-Based Network
   - *(Optionally)* U-INVITE
4. **Calculates similarity metrics** between the estimated networks and the original USF network
5. **Exports evaluation results** to a CSV for each simulation round

---

**Key Parameters:**

- `numsubs`: Number of pseudo-participants to simulate
- `listlength`: Number of items per fluency list
- `methods`: List of network estimation techniques to apply

**Performance Metrics:**

- **Cost**: Structural difference between estimated and true network
- **SDT (Signal Detection Theory) measures**: Hits, misses, false alarms, correct rejections

---

**SNAFU Functions Used:**

- `snafu.read_graph`
- `snafu.gen_lists`
- `snafu.naiveRandomWalk`, `conceptualNetwork`, `pathfinder`, `correlationBasedNetwork`
- `snafu.cost`, `costSDT`
- `snafu.DataModel`, `snafu.Fitinfo`

**Output:**

- `usf_reconstruction_results.csv`: A line-by-line record of how each method performed as participant count increased

**Note:**

Hierarchical U-INVITE is not supported in this demo, but code structure hints at how it could be added in future experiments.

----

.. automodule:: demos.reconstruct_usf
   :members:
   :undoc-members:
   :show-inheritance: