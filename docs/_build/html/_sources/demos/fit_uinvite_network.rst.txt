Fit U-INVITE Network Demo
=========================

This demo illustrates how to estimate semantic networks using the **U-INVITE** and **Hierarchical U-INVITE** models — two of the most powerful and computationally intensive methods available in the SNAFU framework.

---

**About U-INVITE:**

- U-INVITE uses a censored random walk model to infer latent semantic structure from fluency lists.
- Estimation can be slow, especially with larger vocabularies or participant pools.
- Hierarchical U-INVITE allows estimating both individual-level and group-level networks simultaneously.
- You can also fit U-INVITE using a static prior (e.g., from USF norms) for improved speed and interpretability.

---

**What This Demo Does:**

1. Loads animal fluency data (`Experiment1`) and prepares it for modeling.
2. Sets up the `DataModel` (e.g., jump rates, censoring behavior) and `Fitinfo` (e.g., initialization strategy, priors).
3. Runs **three network estimation methods**:
   - **Example 1:** Standard U-INVITE on a single participant's fluency lists
   - **Example 2:** Hierarchical U-INVITE across multiple participants
   - **Example 3:** U-INVITE with a static prior (USF semantic network)

4. Saves all estimated networks using `pickle`.

---

**Key SNAFU Functions Used:**

- `snafu.uinvite`
- `snafu.hierarchicalUinvite`
- `snafu.priorToGraph`
- `snafu.genGraphPrior`
- `snafu.load_network`

**Output Files:**

- `uinvite_network1.pkl`: Adjacency matrix for individual U-INVITE network
- `individual_graphs.pkl`: List of participant-level networks (hierarchical)
- `group_network.pkl`: Aggregated group network (hierarchical)
- `uinvite_network3.pkl`: U-INVITE network using USF prior

**Tips:**

- Consider reducing `prune_limit`, `triangle_limit`, and `other_limit` in `fitinfo` to reduce runtime.
- You can use a pre-defined network (like USF) as a prior for faster, guided fitting.

----

.. automodule:: demos.fit_uinvite_network
   :members:
   :undoc-members:
   :show-inheritance: