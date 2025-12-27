import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_meta_only(in_dir: str, name: str) -> list[dict]:
    meta_path = Path(in_dir) / f"{name}_meta.jsonl"
    tasks = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))
    return tasks


def sample_ids_by_bin(
    in_dir: str,
    name: str,
    per_bin: int,
    seed: int,
    out_path: str | None = None,
):
    """
    仅按 difficulty_bin 分层随机抽样任务 ID（不重新存 circuits）。
    - 同一 bin 下 random/common 任务都会参与抽样。
    - 每个 bin 最多抽 per_bin 条，不足则全取。
    - 输出为一个 JSON，结构：
        {
          "bin_stats": {"easy": 10, ...},
          "sampled": {
             "easy": ["random_0_2_4_4", "bell_2_2_2", ...],
             ...
          }
        }
    """
    rng = np.random.default_rng(int(seed))
    tasks = _load_meta_only(in_dir, name)

    buckets: dict[str | int, list] = defaultdict(list)
    for t in tasks:
        b = t.get("difficulty_bin", "NA")
        buckets[b].append(t["task_id"])

    sampled: dict[str | int, list[str]] = {}
    for b, lst in buckets.items():
        k = min(per_bin, len(lst))
        if k > 0:
            idx = rng.choice(len(lst), size=k, replace=False)
            sampled[b] = [lst[i] for i in idx]
        else:
            sampled[b] = []

    out_path = out_path or str(Path(in_dir) / f"{name}_sampled_bin{per_bin}.json")
    payload = {
        "bin_stats": {str(k): len(v) for k, v in buckets.items()},
        "sampled": {str(k): v for k, v in sampled.items()},
        "meta_source": f"{in_dir}/{name}_meta.jsonl",
        "per_bin": per_bin,
        "seed": seed,
    }
    Path(out_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Input tasks:", len(tasks))
    print("Bins:", {k: len(v) for k, v in buckets.items()})
    print("Sampled counts:", {k: len(v) for k, v in sampled.items()})
    print(f"Saved sampled IDs to: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample task IDs per difficulty_bin (no new circuits saved)")
    parser.add_argument("--in_dir", type=str, default="task_suites")
    parser.add_argument("--name", type=str, default="Final")
    parser.add_argument("--per_bin", type=int, default=10, help="max samples per difficulty_bin")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_path", type=str, default=None, help="output JSON path of sampled IDs")
    args = parser.parse_args()

    sample_ids_by_bin(
        in_dir=args.in_dir,
        name=args.name,
        per_bin=args.per_bin,
        seed=args.seed,
        out_path=args.out_path,
    )
