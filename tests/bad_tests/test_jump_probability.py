# demos/test_jump_probability_results_vs_sim_best_fit.py
import csv
from pathlib import Path

def _load_by_id(path: Path):
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"id", "best_jump_param", "log_lik"}
        missing = required.difference(set(reader.fieldnames or []))
        assert not missing, f"{path} missing column(s): {', '.join(sorted(missing))}"

        rows = {}
        for row in reader:
            sid = (row["id"] or "").strip()
            bj  = (row["best_jump_param"] or "").strip()
            ll  = (row["log_lik"] or "").strip()
            if sid:  # skip blank ids
                rows[sid] = (bj, ll)
        return rows

def test_jump_probability_results_matches_sim_best_fit():
    demos_dir = Path(__file__).resolve().parent
    demos_data = demos_dir / "../demos/demos_data"

    new_path = demos_data / "jump_probability_results.csv"
    base_path = Path("test_data/sim_best_fit.csv")

    assert new_path.exists(), f"Missing new results file: {new_path}"
    assert base_path.exists(), f"Missing baseline file: {base_path}"

    new_rows  = _load_by_id(new_path)
    base_rows = _load_by_id(base_path)

    # Ensure the same set of IDs
    new_ids, base_ids = set(new_rows), set(base_rows)
    assert new_ids == base_ids, (
        "ID set mismatch.\n"
        f"Only in new: {sorted(new_ids - base_ids)[:10]}\n"
        f"Only in baseline: {sorted(base_ids - new_ids)[:10]}"
    )

    # Exact-by-string comparison for each id (no tolerance)
    mismatches = []
    for sid in sorted(base_ids):
        bj_base,  ll_base  = base_rows[sid]
        bj_new,   ll_new   = new_rows[sid]
        if (bj_base != bj_new) or (ll_base != ll_new):
            mismatches.append((sid, bj_base, bj_new, ll_base, ll_new))

    if mismatches:
        preview = "\n".join(
            f"  id {sid}: best_jump_param expected {bj_b}, got {bj_n}; "
            f"log_lik expected {ll_b}, got {ll_n}"
            for sid, bj_b, bj_n, ll_b, ll_n in mismatches[:10]
        )
        raise AssertionError(
            f"{len(mismatches)} id(s) differ between baseline and new results.\n"
            f"First differences:\n{preview}"
        )
