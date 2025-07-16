Compare Networks Demo
=====================

This demo illustrates how to simulate and compare semantic fluency data using two different networks — the original USF network and a perturbed version of it. This is useful for evaluating how well a model-generated or alternate network captures actual fluency behavior.

**Overview:**

1. Load a known semantic network (USF animal subset).
2. Create a perturbed version of the network by randomly flipping ~10% of edges.
3. Generate fluency lists from both networks using censored random walks.
4. Compute log-likelihoods of those lists under both networks.
5. Save likelihoods and data for further inspection.

**Key Concepts:**

- **Perturbation**: Randomly flip edges to simulate variation or error in network structure.
- **Fluency Simulation**: Generate fluency lists using random walks with optional jump probabilities.
- **Likelihood Comparison**: Assess how well each network explains data generated from itself and the other network.

**Functions Used:**

- `snafu.read_graph`
- `snafu.DataModel`
- `snafu.gen_lists`
- `snafu.probX`

**Output:**

- Prints log-likelihoods to console
- Saves rounded log-likelihoods in `demos_data/expected_likelihoods.pkl`

.. automodule:: demos.compare_networks
   :members:
   :undoc-members:
   :show-inheritance: