import json
import os
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from qiskit import qpy


# -----------------------------
# IO: load task suite
# -----------------------------
def load_task_suite(folder: str, name: str) -> List[Dict[str, Any]]:
    """
    Load tasks from:
      - <folder>/<name>_circuits.qpy
      - <folder>/<name>_meta.jsonl

    Compatible metadata fields:
      - old: length_bin
      - new: difficulty_bin, difficulty_score
    """
    folder = Path(folder)
    qpy_path = folder / f"{name}_circuits.qpy"
    meta_path = folder / f"{name}_meta.jsonl"

    if not qpy_path.exists():
        raise FileNotFoundError(f"Missing circuits file: {qpy_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing meta file: {meta_path}")

    # circuits
    with qpy_path.open("rb") as f:
        circuits = list(qpy.load(f))

    # metadata rows
    metas = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            metas.append(json.loads(line))

    if len(circuits) != len(metas):
        raise ValueError(
            f"Mismatch: {qpy_path.name} has {len(circuits)} circuits, "
            f"but {meta_path.name} has {len(metas)} meta rows."
        )

    tasks = []
    for i, (qc, m) in enumerate(zip(circuits, metas)):
        task_id = m.get("task_id", f"task_{i}")
        n_qubits = int(m.get("n_qubits", qc.num_qubits))
        target_gates = int(m.get("target_gates", len(qc.data)))

        t = {
            "task_id": task_id,
            "qc": qc,
            "n_qubits": n_qubits,
            "target_gates": target_gates,
        }

        # bins: prefer difficulty_bin if present
        if "difficulty_bin" in m:
            t["difficulty_bin"] = m["difficulty_bin"]
        if "difficulty_score" in m:
            t["difficulty_score"] = int(m["difficulty_score"])

        # keep old field if present
        if "length_bin" in m:
            t["length_bin"] = m["length_bin"]

        # optional extras
        if "qc_text" in m:
            t["qc_text"] = m["qc_text"]
        if "qc_png" in m:
            t["qc_png"] = m["qc_png"]

        tasks.append(t)

    return tasks


# -----------------------------
# Summaries / sanity checks
# -----------------------------
def summarize(tasks: List[Dict[str, Any]], name: str) -> None:
    print(f"\n=== Suite: {name} ===")
    print("Total tasks:", len(tasks))
    if not tasks:
        return

    n_qubits_cnt = Counter([t.get("n_qubits", None) for t in tasks])
    print("n_qubits:", dict(sorted(n_qubits_cnt.items(), key=lambda x: (x[0] is None, x[0]))))

    # prefer difficulty_bin; fallback to length_bin; else 'NA'
    def get_bin(t):
        return t.get("difficulty_bin", t.get("length_bin", "NA"))

    bin_cnt = Counter([get_bin(t) for t in tasks])
    print("bin:", dict(bin_cnt))

    # joint distribution (n_qubits, bin)
    joint = Counter([(t.get("n_qubits", None), get_bin(t)) for t in tasks])
    print("joint (n_qubits, bin):")
    for (nq, b), c in sorted(joint.items(), key=lambda x: (x[0][0], str(x[0][1]))):
        print(f"  ({nq}, {b}): {c}")

    # if difficulty_score exists, show rough range by bin
    by_bin_scores = defaultdict(list)
    for t in tasks:
        b = get_bin(t)
        if "difficulty_score" in t:
            by_bin_scores[b].append(int(t["difficulty_score"]))
    if by_bin_scores:
        print("difficulty_score range by bin:")
        for b in sorted(by_bin_scores.keys()):
            arr = np.array(by_bin_scores[b], dtype=int)
            print(f"  {b}: min={arr.min()}  max={arr.max()}  mean={arr.mean():.1f}")


def check_physical_qubits(tasks: List[Dict[str, Any]], name: str, expect_num_qubits: int = 4) -> None:
    """Check qc.num_qubits distribution; report those not equal to expect_num_qubits."""
    nums = [int(t["qc"].num_qubits) for t in tasks]
    cnt = Counter(nums)
    print(f"\n[check_physical_qubits] {name}: qc.num_qubits distribution:", dict(cnt))

    bad = [t for t in tasks if int(t["qc"].num_qubits) != int(expect_num_qubits)]
    if bad:
        print(f"  WARNING: {len(bad)} tasks have qc.num_qubits != {expect_num_qubits}. Examples:")
        for t in bad[:10]:
            print(f"    {t['task_id']} qc.num_qubits={t['qc'].num_qubits} n_qubits(meta)={t.get('n_qubits')}")
    else:
        print(f"  OK: all tasks have qc.num_qubits == {expect_num_qubits}")


def sample_metadata(tasks: List[Dict[str, Any]], name: str, k: int = 2, seed: int = 0) -> None:
    """Sample k tasks per group (n_qubits, bin) and print quick info."""
    rng = np.random.default_rng(seed)

    def get_bin(t):
        return t.get("difficulty_bin", t.get("length_bin", "NA"))

    groups = defaultdict(list)
    for t in tasks:
        groups[(t.get("n_qubits", None), get_bin(t))].append(t)

    print(f"\n[sample_metadata] {name}: sample {k} per (n_qubits, bin)")
    for key in sorted(groups.keys(), key=lambda x: (x[0] is None, x[0], str(x[1]))):
        arr = groups[key]
        pick = rng.choice(len(arr), size=min(k, len(arr)), replace=False)
        print(f"  Group {key}, size={len(arr)}:")
        for i in pick:
            t = arr[int(i)]
            qc = t["qc"]
            msg = f"    {t['task_id']}: gates={len(qc.data)} depth={qc.depth()}"
            if "difficulty_score" in t:
                msg += f" score={t['difficulty_score']}"
            print(msg)


# -----------------------------
# Export PNGs
# -----------------------------
def _safe_filename(s: str) -> str:
    s = str(s)
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s[:200]


def export_all_png(tasks: List[Dict[str, Any]], out_dir: str, dpi: int = 140) -> None:
    """
    Export each circuit as a PNG:
      <out_dir>/<task_id>.png
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for t in tasks:
        qc = t["qc"]
        task_id = _safe_filename(t.get("task_id", "task"))
        png_path = out / f"{task_id}.png"

        # draw mpl
        fig = qc.draw(output="mpl", fold=-1, idle_wires=True)
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
        try:
            fig.clf()
        except Exception:
            pass
        plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main(names: list[str]):
    suite_dir = "task_suites"

    # change these names to your actual suite files
    # names = ["Final"]  # or ["train", "test", "spare"]

    for nm in names:
        tasks = load_task_suite(suite_dir, nm)
        summarize(tasks, nm)
        check_physical_qubits(tasks, nm, expect_num_qubits=4)
        sample_metadata(tasks, nm, k=2, seed=0)

        out_png = Path("circuit_png") / nm
        export_all_png(tasks, str(out_png), dpi=140)
        print(f"[export] {nm}: {len(tasks)} pngs -> {out_png}")


if __name__ == "__main__":
    main(names=["Final"])
