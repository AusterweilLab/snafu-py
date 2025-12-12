import csv
import math
import numpy as np
import networkx as nx
import snafu
import os
from pathlib import Path

# ------------------------------------------------------------
# Configure whether to print likelihoods from the function
# True  -> estimateJumpProbability prints and returns (best_jump, best_ll)
# False -> estimateJumpProbability does NOT print and returns (best_jump, np.nan)
# ------------------------------------------------------------

# ---------- Setup output directory ----------
out_dir = os.path.join(os.path.dirname(__file__), "demos_data")
os.makedirs(out_dir, exist_ok=True)
out_csv = os.path.join(out_dir, "jump_probability_results.csv")

# ---------- Build network ----------
G = nx.Graph()
with open(os.path.join("demos_data", "madrid_network.csv"), "r", newline="", encoding="utf-8") as f:
    reader = csv.reader(f)
    header = next(reader, None)
    for row in reader:
        if not row or len(row) < 4:
            continue
        _, item1, item2, edge = row[:4]
        if str(edge).strip() == "1":
            G.add_edge(item1, item2)
        else:
            G.add_node(item1)
            G.add_node(item2)

items = {inum: node for inum, node in enumerate(list(G.nodes()))}
group_network = nx.to_numpy_array(G)

# ---------- Load and align fluency data ----------
patient_data = snafu.load_fluency_data(os.path.join("demos_data", "PAFIP_animal.txt"))

revItems = snafu.reverseDict(items)  # label -> index
fluency_data = []
for lst in patient_data.labeledlists:
    fluency_data.append([revItems[label] for label in lst if label in revItems])

# ---------- Data model ----------
data_model = snafu.DataModel({
    "jump": 0.05,
    "jumptype": "uniform",
    "start_node": "uniform",
})

# ---------- Estimate best jump per subject/list ----------
best_jumps = []
log_likelihoods = []

for sub_idx, fluency_list in enumerate(fluency_data):
    if not fluency_list:
        best_jumps.append(np.nan)
        log_likelihoods.append(np.nan)
        continue

    best_jump, best_ll = snafu.estimateJumpProbability(
        group_network, data_model, fluency_list, return_ll=True
    )

    best_jumps.append(best_jump)
    log_likelihoods.append(best_ll)

# ---------- Save results ----------
Path(out_dir).mkdir(parents=True, exist_ok=True)

with open(out_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "best_jump_param", "log_lik"])
    for i, bj in enumerate(best_jumps):
        subj = patient_data.subs[i]
        ll = log_likelihoods[i]
        bj_cell = "" if (isinstance(bj, float) and math.isnan(bj)) else bj
        ll_cell = "" if (not isinstance(ll, (int, float)) or math.isnan(ll)) else ll
        w.writerow([subj, bj_cell, ll_cell])
