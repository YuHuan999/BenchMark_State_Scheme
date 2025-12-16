import json
import re
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any

from qiskit import qpy, QuantumCircuit
from qiskit.circuit.library import QFT

# ========= Bell (2 qubit) =========
qc_bell = QuantumCircuit(4)
qc_bell.h(0)
qc_bell.cx(0, 1)

# ========= GHZ (3,4 qubit) =========
qc_ghz_3 = QuantumCircuit(4)
qc_ghz_3.h(0)
qc_ghz_3.cx(0, 1)
qc_ghz_3.cx(0, 2)

qc_ghz_4 = QuantumCircuit(4)
qc_ghz_4.h(0)
qc_ghz_4.cx(0, 1)
qc_ghz_4.cx(0, 2)
qc_ghz_4.cx(0, 3)

# ========= Entanglement: linear chain (4 qubit) =========
# 0--1--2--3 的链式纠缠（结构明确，和 GHZ 不同）
qc_chain_4 = QuantumCircuit(4)
qc_chain_4.h(0)
qc_chain_4.cx(0, 1)
qc_chain_4.cx(1, 2)
qc_chain_4.cx(2, 3)

# ========= Entanglement: star (4 qubit) =========
# 以 1 为中心星型纠缠（拓扑不同）
qc_star_4 = QuantumCircuit(4)
qc_star_4.h(1)
qc_star_4.cx(1, 0)
qc_star_4.cx(1, 2)
qc_star_4.cx(1, 3)

# ========= Two Bell pairs in 4 qubits =========
# (0,1) 和 (2,3) 两对 Bell
qc_2bell_4 = QuantumCircuit(4)
qc_2bell_4.h(0); qc_2bell_4.cx(0, 1)
qc_2bell_4.h(2); qc_2bell_4.cx(2, 3)

# ========= QFT / inverse QFT =========
# 注意：这里用 library QFT，内部会包含 CP/Swap 等门。
# 如果你的 env 动作空间不支持这些门，建议不要把 QFT 作为 “rebuild target circuit” 任务，
# 或者后面用 transpile 分解到你支持的门集。
qc_qft_3 = QuantumCircuit(4)
qc_qft_3.append(QFT(3, do_swaps=False).to_instruction(), [0, 1, 2])
qc_iqft_3 = QuantumCircuit(3)
qc_iqft_3.append(QFT(3, do_swaps=False, inverse=True).to_instruction(), [0, 1, 2])

# ========= Toffoli-style structure demo (3 qubit) =========
# 如果你的 env 门集不含 CCX，这个任务就别用；
# 或者后面把 CCX 分解成 {H, T, CX, ...} 再作为 target。
qc_toffoli_3 = QuantumCircuit(4)
qc_toffoli_3.h(0)
qc_toffoli_3.h(1)
qc_toffoli_3.ccx(0, 1, 2)

# ========= Simple "ladder" circuit (4 qubit, slightly longer) =========
# 一个稍微长一点、但结构仍很清晰的 ladder
qc_ladder_4 = QuantumCircuit(4)
qc_ladder_4.h(0); qc_ladder_4.h(1)
qc_ladder_4.cx(0, 1)
qc_ladder_4.cx(1, 2)
qc_ladder_4.cx(2, 3)
qc_ladder_4.h(3)
qc_ladder_4.cx(0, 2)
qc_ladder_4.cx(1, 3)
Tasks_common = [
    {"task_id": "bell", "qc": qc_bell, "n_qubits": 2, "length_bin": "short", "target_gates": 2},
    {"task_id": "ghz_q3", "qc": qc_ghz_3, "n_qubits": 3, "length_bin": "short", "target_gates": 3},
    {"task_id": "ghz_q4", "qc": qc_ghz_4, "n_qubits": 4, "length_bin": "short", "target_gates": 4},
    {"task_id": "chain_q4", "qc": qc_chain_4, "n_qubits": 4, "length_bin": "short", "target_gates": 4},
    {"task_id": "star_q4", "qc": qc_star_4, "n_qubits": 4, "length_bin": "short", "target_gates": 4},
    {"task_id": "2bell_q4", "qc": qc_2bell_4, "n_qubits": 4, "length_bin": "short", "target_gates": 4},
    {"task_id": "qft_q3", "qc": qc_qft_3, "n_qubits": 3, "length_bin": "medium", "target_gates": 3},
    {"task_id": "iqft_q3", "qc": qc_iqft_3, "n_qubits": 3, "length_bin": "medium", "target_gates": 3},
    {"task_id": "toffoli_q3", "qc": qc_toffoli_3, "n_qubits": 3, "length_bin": "short", "target_gates": 3},
    {"task_id": "ladder_q4", "qc": qc_ladder_4, "n_qubits": 4, "length_bin": "medium", "target_gates": 4},
]





# ---------- Load ----------
def load_task_suite(folder: str, name: str) -> List[Dict[str, Any]]:
    folder = Path(folder)
    qpy_path = folder / f"{name}_circuits.qpy"
    meta_path = folder / f"{name}_meta.jsonl"

    with qpy_path.open("rb") as f:
        circuits = qpy.load(f)

    meta = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            meta.append(json.loads(line))

    assert len(circuits) == len(meta), f"Mismatch: {len(circuits)} circuits vs {len(meta)} meta"

    tasks = []
    for qc, m in zip(circuits, meta):
        tasks.append({
            "task_id": m["task_id"],
            "qc": qc,
            "n_qubits": int(m["n_qubits"]),
            "length_bin": m["length_bin"],
            "target_gates": int(m.get("target_gates", len(qc.data))),
        })
    return tasks


# ---------- Console summary (NO circuit drawings) ----------
def summarize(tasks: List[Dict[str, Any]], name: str):
    print(f"\n=== {name} ===")
    print("total:", len(tasks))

    c_qubits = Counter(t["n_qubits"] for t in tasks)
    c_bins = Counter(t["length_bin"] for t in tasks)
    joint = Counter((t["n_qubits"], t["length_bin"]) for t in tasks)

    print("by n_qubits:", dict(sorted(c_qubits.items())))
    print("by length_bin:", dict(sorted(c_bins.items())))
    print("by (n_qubits, bin):")
    for k in sorted(joint.keys()):
        print(f"  {k}: {joint[k]}")

def check_physical_qubits(tasks: List[Dict[str, Any]], name: str, expected: int = 4, strict: bool = False):
    """
    检查每个 task 的 qc 是否是 expected 个物理比特（你的设计里应该永远是 4）
    - 打印 qc.num_qubits 的分布
    - 打印不匹配的 task_id（最多前 20 个）
    - strict=True 时直接 raise
    """
    from collections import Counter

    dist = Counter(int(t["qc"].num_qubits) for t in tasks)
    bad = [t for t in tasks if int(t["qc"].num_qubits) != int(expected)]

    print(f"\n[check_physical_qubits] {name}: expected qc.num_qubits={expected}")
    print("  qc.num_qubits distribution:", dict(sorted(dist.items())))

    if bad:
        print(f"  BAD: {len(bad)} tasks not {expected}-qubit. Examples (up to 20):")
        for t in bad[:20]:
            print(f"   - task_id={t['task_id']}  qc.num_qubits={t['qc'].num_qubits}  n_qubits(active)={t['n_qubits']}")
        if strict:
            raise ValueError(f"{name}: found {len(bad)} circuits with qc.num_qubits != {expected}")
    else:
        print("  OK: all tasks are 4-qubit circuits.")

def sample_metadata(tasks: List[Dict[str, Any]], k_per_group: int = 1, seed: int = 0):
    """
    每个 (n_qubits, length_bin) 随机抽 k 条，只打印元信息，不打印电路图
    """
    rng = random.Random(seed)
    groups = defaultdict(list)
    for t in tasks:
        groups[(t["n_qubits"], t["length_bin"])].append(t)

    print(f"\n--- sample metadata (k_per_group={k_per_group}) ---")
    for key in sorted(groups.keys()):
        lst = groups[key][:]
        rng.shuffle(lst)
        picks = lst[: min(k_per_group, len(lst))]
        print(f"\nGroup {key}: show {len(picks)}/{len(lst)}")
        for t in picks:
            qc = t["qc"]
            print(
                f"- task_id={t['task_id']}  "
                f"target_gates={t['target_gates']}  depth={qc.depth()}  size={qc.size()}"
            )


# ---------- Export PNG ----------
def _safe_filename(task_id: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", task_id).strip("._-")
    return s or "task"


def export_all_png(tasks: List[Dict[str, Any]], out_dir: Path, dpi: int = 260, scale: float = 0.95):
    """
    导出所有电路为 PNG：
    out_dir/<task_id>.png

    美观参数：
    - output="mpl"：矢量风格更干净
    - fold=-1：不折行（长电路也保持一行，适合看结构）
    - idle_wires=False：隐藏未用到的线路（更清爽）
    - scale：整体缩放（字体/间距）
    - dpi：清晰度
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt  # 需要 matplotlib

    total = len(tasks)
    for i, t in enumerate(tasks, 1):
        qc = t["qc"]
        base = _safe_filename(t["task_id"])
        out_path = out_dir / f"{base}.png"

        # 如果 task_id 重名（很少见），避免覆盖
        if out_path.exists():
            out_path = out_dir / f"{base}__{i}.png"

        fig = qc.draw(
            output="mpl",
            fold=-1,
            idle_wires=True,
            scale=scale,
        )
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

        if i % 50 == 0 or i == total:
            print(f"[{out_dir.name}] exported {i}/{total}")


def main():
    suite_dir = "task_suites"
    out_root = Path("circuit_png")

    train = load_task_suite(suite_dir, "train")
    test = load_task_suite(suite_dir, "test")
    spare = load_task_suite(suite_dir, "spare")

    # 1) 保留之前的统计功能
    summarize(train, "train")
    summarize(test, "test")
    summarize(spare, "spare")
    # 2) 检查物理比特数
    check_physical_qubits(train, "train", expected=4, strict=False)
    check_physical_qubits(test,  "test",  expected=4, strict=False)
    check_physical_qubits(spare, "spare", expected=4, strict=False)

    # 3) 保留抽样检查（只输出元信息，不输出电路图）
    sample_metadata(train, k_per_group=1, seed=0)
    sample_metadata(test, k_per_group=1, seed=1)
    sample_metadata(spare, k_per_group=1, seed=2)

    # 4) 导出所有电路 PNG：按 train/test/spare 分文件夹
    export_all_png(train, out_root / "train", dpi=260, scale=0.95)
    export_all_png(test,  out_root / "test",  dpi=260, scale=0.95)
    export_all_png(spare, out_root / "spare", dpi=260, scale=0.95)

    print("\nDone. Output folders:")
    print(" -", (out_root / "train").resolve())
    print(" -", (out_root / "test").resolve())
    print(" -", (out_root / "spare").resolve())


if __name__ == "__main__":
    main()
