import numpy as np
from qiskit import QuantumCircuit

# ---------------------
# Defaults for length and depth bins
# ---------------------
DEFAULT_LENGTH_BINS = {
    "short": (4, 8),
    "medium": (9, 16),
    "long": (17, 24),
    "xlong": (25, 40),
    "xxlong": (41, 60),
}

# 默认让更长的任务占比低一些（总计 30%）
DEFAULT_LENGTH_BIN_PROBS = {
    "short": 0.25,
    "medium": 0.25,
    "long": 0.20,
    "xlong": 0.20,
    "xxlong": 0.10,
}

# 你可以自己改成你想要的深度档位（层数）
DEFAULT_DEPTH_BINS = {
    "shallow": (3, 6),
    "mid": (7, 12),
    "deep": (13, 20),
}

# ---------------------
# Gate set (match env)
# ---------------------
SINGLE_GATES = ["X", "Y", "Z", "H", "T"]
CANCEL_1Q = {"H", "X", "Y", "Z"}  # cancellation: 1q self-inverse
PAULI_REDUNDANT_PAIRS = {
    ("X", "Y"), ("Y", "X"),
    ("Y", "Z"), ("Z", "Y"),
    ("Z", "X"), ("X", "Z"),
}


def _all_actions(n_qubits: int):
    """
    枚举全部动作（与 env.action_mapping 的 dict 结构一致）：
      - 1q: {'gate': g, 'target': q, 'control': None}
      - CNOT: {'gate': 'CNOT', 'control': c, 'target': t}
    """
    actions = []
    for q in range(n_qubits):
        for g in SINGLE_GATES:
            actions.append({"gate": g, "target": q, "control": None})
    if n_qubits >= 2:
        for c in range(n_qubits):
            for t in range(n_qubits):
                if c != t:
                    actions.append({"gate": "CNOT", "control": c, "target": t})
    return actions


def _frontier_mask_is_legal(cand, qubit_stacks, op_table, n_qubits):
    """
    复刻你环境的 frontier-based mask 规则（只用到你目前的简化版）：
    - 1q cancellation: {H,X,Y,Z} 同门相邻
    - 1q redundancy: Pauli-Pauli (X/Y/Z) 两两组合冗余
    - CNOT cancellation: 同向相邻抵消（两条线 frontier 指向同一个 CNOT）
    - frontier 是 CNOT 时：1q 不触发 1q 规则（当作屏障）
    """
    def frontier_act(q):
        st = qubit_stacks[q]
        if not st:
            return None, None
        op_id = st[-1]
        return op_id, op_table[op_id]

    g = cand["gate"]

    # ===== A) 1q candidate =====
    if g in SINGLE_GATES:
        q = cand["target"]
        prev_id, prev = frontier_act(q)
        if prev is None:
            return True

        # frontier is CNOT: treat as barrier for 1q rules
        if prev["gate"] == "CNOT":
            return True

        prev_g = prev["gate"]

        # cancellation
        if g in CANCEL_1Q and prev_g == g:
            return False

        # redundancy (Pauli-Pauli)
        if (prev_g, g) in PAULI_REDUNDANT_PAIRS:
            return False

        return True

    # ===== B) CNOT candidate =====
    if g == "CNOT":
        c, t = cand["control"], cand["target"]
        pc_id, pc = frontier_act(c)
        pt_id, pt = frontier_act(t)

        # adjacent cancellation requires both frontiers point to same op_id
        if (pc is not None and pt is not None and pc_id == pt_id and
            pc["gate"] == "CNOT" and pc["control"] == c and pc["target"] == t):
            return False

        return True

    return True


def _depth_after_append(act, layers):
    """
    用“ASAP layering”方式增量维护 depth（层数）。
    Qiskit 的 qc.depth() 对于这种顺序 append 的 DAG depth，本质就是最长依赖路径，
    这套 layers 更新可以非常稳定地等价（且比每步 qc.depth() 快得多）。
    """
    g = act["gate"]
    if g in SINGLE_GATES:
        q = act["target"]
        new_layer = layers[q] + 1
        new_max = max(max(layers), new_layer)
        return new_layer, None, new_max
    elif g == "CNOT":
        c, t = act["control"], act["target"]
        base = max(layers[c], layers[t]) + 1
        new_max = max(max(layers), base)
        return base, base, new_max
    else:
        return None, None, max(layers)


def _apply_to_qiskit(qc: QuantumCircuit, act: dict):
    """把动作真正 append 到 Qiskit 电路里。"""
    g = act["gate"]
    if g == "X":
        qc.x(act["target"])
    elif g == "Y":
        qc.y(act["target"])
    elif g == "Z":
        qc.z(act["target"])
    elif g == "H":
        qc.h(act["target"])
    elif g == "T":
        qc.t(act["target"])
    elif g == "CNOT":
        qc.cx(act["control"], act["target"])
    else:
        raise ValueError(f"Unknown gate: {g}")


def _build_random_circuit_by_gates(n_qubits: int, gate_count: int, rng: np.random.Generator):
    """
    按门数生成：每一步只从“合法动作”中采样（遵守你的 action mask 规则）。
    """
    # qc = QuantumCircuit(n_qubits)
    qc = QuantumCircuit(4)
    action_mapping = _all_actions(n_qubits)

    op_table = []  # op_id -> act(dict)
    qubit_stacks = [[] for _ in range(n_qubits)]

    for _ in range(gate_count):
        legal_ids = [
            i for i, a in enumerate(action_mapping)
            if _frontier_mask_is_legal(a, qubit_stacks, op_table, n_qubits)
        ]
        if not legal_ids:
            raise RuntimeError("No legal actions available (gate mode).")

        aid = int(rng.choice(legal_ids))
        act = action_mapping[aid]

        # apply
        _apply_to_qiskit(qc, act)

        # update frontier table
        op_id = len(op_table)
        if act["gate"] == "CNOT":
            op_table.append({"gate": "CNOT", "control": act["control"], "target": act["target"]})
            qubit_stacks[act["control"]].append(op_id)
            qubit_stacks[act["target"]].append(op_id)
        else:
            op_table.append({"gate": act["gate"], "target": act["target"], "control": None})
            qubit_stacks[act["target"]].append(op_id)

    return qc


def _build_random_circuit_by_depth(n_qubits: int, target_depth: int, rng: np.random.Generator,
                                   max_steps: int = 20000):
    """
    按深度（层数）生成：不断采样合法动作，但额外约束“新增后 depth 不能超过 target_depth”。
    一旦达到 target_depth 就停止（电路深度恰好等于 target_depth）。
    """
    if target_depth <= 0:
        # return QuantumCircuit(n_qubits)
        return QuantumCircuit(4)

    # qc = QuantumCircuit(n_qubits)
    qc = QuantumCircuit(4)
    action_mapping = _all_actions(n_qubits)

    op_table = []
    qubit_stacks = [[] for _ in range(n_qubits)]

    # layers[q] 表示该 qubit 当前最外层所在的层号（从 0 开始计数）
    layers = [0] * n_qubits
    current_depth = 0

    # 我们需要达到 target_depth（层数），这里把“层数”定义为 max(layers)
    # 初始没有门 depth=0；加一个门后 depth 至少变成 1
    steps = 0
    while current_depth < target_depth:
        steps += 1
        if steps > max_steps:
            # 理论上很少；如果你把 mask 规则再加严可能会触发
            raise RuntimeError("Depth sampling stuck. Increase max_steps or loosen constraints.")

        feasible_ids = []
        for i, a in enumerate(action_mapping):
            # 先过 mask
            if not _frontier_mask_is_legal(a, qubit_stacks, op_table, n_qubits):
                continue

            # 再过 depth 预算：新增后不能超过 target_depth
            if a["gate"] in SINGLE_GATES:
                q = a["target"]
                new_layer = layers[q] + 1
                new_depth = max(current_depth, new_layer)
                if new_depth <= target_depth:
                    feasible_ids.append(i)
            elif a["gate"] == "CNOT":
                c, t = a["control"], a["target"]
                base = max(layers[c], layers[t]) + 1
                new_depth = max(current_depth, base)
                if new_depth <= target_depth:
                    feasible_ids.append(i)

        if not feasible_ids:
            # 这说明在当前状态下无法继续扩展到目标深度（极少见）
            # 最简单的处理：重来一次（rejection）
            return _build_random_circuit_by_depth(n_qubits, target_depth, rng, max_steps=max_steps)

        aid = int(rng.choice(feasible_ids))
        act = action_mapping[aid]

        # apply
        _apply_to_qiskit(qc, act)

        # update depth layers
        if act["gate"] in SINGLE_GATES:
            q = act["target"]
            layers[q] = layers[q] + 1
            current_depth = max(current_depth, layers[q])
        else:
            c, t = act["control"], act["target"]
            base = max(layers[c], layers[t]) + 1
            layers[c] = base
            layers[t] = base
            current_depth = max(current_depth, base)

        # update frontier
        op_id = len(op_table)
        if act["gate"] == "CNOT":
            op_table.append({"gate": "CNOT", "control": act["control"], "target": act["target"]})
            qubit_stacks[act["control"]].append(op_id)
            qubit_stacks[act["target"]].append(op_id)
        else:
            op_table.append({"gate": act["gate"], "target": act["target"], "control": None})
            qubit_stacks[act["target"]].append(op_id)

    return qc


def generate_task_suite(
    seed,
    n_tasks=60,
    n_qubits_choices=(2, 3, 4),
    length_bins=None,
    depth_bins=None,
    length_bin_probs=None,
    mode="gates",  # "gates" 或 "depth"
):
    """
    生成随机任务集，每个任务包含目标电路。

    mode="gates": 使用 length_bins 采样 gate_count，生成 gate_count 个门
    mode="depth": 使用 depth_bins 采样 target_depth，生成深度恰好为 target_depth 的电路

    返回: (train_tasks, test_tasks)，切分比例 70% / 30%。
    """
    rng = np.random.default_rng(seed)

    bins_g = length_bins or DEFAULT_LENGTH_BINS
    bins_d = depth_bins or DEFAULT_DEPTH_BINS

    tasks = []
    for idx in range(n_tasks):
        n_qubits = int(rng.choice(n_qubits_choices))

        if mode == "gates":
            bin_names = list(bins_g.keys())
            # 如果你用默认 length_bins（为 None），就采用默认分布；否则默认均匀采样
            if length_bin_probs is None:
                if length_bins is None:
                    p = np.array([DEFAULT_LENGTH_BIN_PROBS.get(k, 0.0) for k in bin_names], dtype=float)
                    if p.sum() <= 0:
                        p = None
                    else:
                        p = p / p.sum()
                else:
                    p = None
            else:
                # 允许传 dict: {"short":0.2,...} 或 list/np.array 与 bin_names 对齐
                if isinstance(length_bin_probs, dict):
                    p = np.array([float(length_bin_probs.get(k, 0.0)) for k in bin_names], dtype=float)
                else:
                    p = np.array(length_bin_probs, dtype=float)
                if p.sum() <= 0:
                    p = None
                else:
                    p = p / p.sum()

            bin_name = rng.choice(bin_names, p=p)
            low, high = bins_g[bin_name]
            gate_count = int(rng.integers(low, high + 1))
            target_qc = _build_random_circuit_by_gates(n_qubits, gate_count, rng)

            tasks.append(
                {
                    "task_id": f"task_{idx}",
                    "qc": target_qc,
                    "n_qubits": n_qubits,
                    "length_bin": bin_name,
                        "target_gates": int(len(target_qc.data)),
                }
            )

        elif mode == "depth":
            bin_name = rng.choice(list(bins_d.keys()))
            low, high = bins_d[bin_name]
            target_depth = int(rng.integers(low, high + 1))
            target_qc = _build_random_circuit_by_depth(n_qubits, target_depth, rng)

            # 这里 length_bin 仍然保留（兼容 env），但语义是 depth 档位
            tasks.append(
                {
                    "task_id": f"task_{idx}",
                    "qc": target_qc,
                    "n_qubits": n_qubits,
                    "length_bin": f"depth::{bin_name}",
                        "target_gates": int(len(target_qc.data)),
                }
            )

        else:
            raise ValueError("mode must be 'gates' or 'depth'.")

    # shuffle & split
    permutation = rng.permutation(len(tasks))
    tasks = [tasks[i] for i in permutation]

    split_idx = int(len(tasks) * 0.7)
    train_tasks = tasks[:split_idx]
    test_tasks = tasks[split_idx:]

    return train_tasks, test_tasks



# ---------------------
# Stratified task generation (exact counts per (n_qubits, length_bin))
# ---------------------
def generate_task_suite_stratified(
    seed: int,
    n_qubits_choices=(2, 3, 4),
    length_bins=None,
    per_group=(40, 20, 10),  # (train, test, spare) per (n_qubits, bin)
    mode="gates",            # currently supports "gates"; can be extended to "depth"
    include_text_preview: bool = False,
    preview_png_dir: str | None = None,
    preview_sample_per_group: int = 0,  # 0 means "no preview". Recommend 1~3.
):
    """
    分层生成任务：对每个 (n_qubits, length_bin) 组合，精确生成固定数量的任务，
    然后在组内 shuffle 并切分为 train/test/spare。

    任务 dict 格式（env 可直接读）：
      {
        "task_id": str,
        "qc": QuantumCircuit,
        "n_qubits": int,
        "length_bin": str,
        "target_gates": int,
        # 可选（用于快速肉眼检查/调试）：
        # "qc_text": str,  # ASCII 电路图
        # "qc_png": str,   # 预览图 PNG 路径
      }

    参数说明：
    - include_text_preview: 若 True，则为部分样本附加 qc_text（ASCII 图）。
    - preview_png_dir: 若不为 None，则为部分样本导出 PNG 预览图到该目录，并写入 qc_png 路径。
    - preview_sample_per_group: 每个 (n_qubits, bin) 组里生成多少个样本预览（默认 0 不生成）。
      建议 1~3 即可；生成所有任务的 PNG 会很占空间/很慢。

    例如：n_qubits_choices=(2,3,4)，length_bins=5 档，per_group=(40,20,10)
      => 总数 = 3 * 5 * (40+20+10) = 1050
      => train=600, test=300, spare=150
    """
    rng = np.random.default_rng(int(seed))

    if length_bins is None:
        length_bins = DEFAULT_LENGTH_BINS

    if mode != "gates":
        raise ValueError(f"Unsupported mode={mode}. Currently only supports 'gates'.")

    # bins_g: {"short": (4,8), ...}
    bins_g = dict(length_bins)

    n_train, n_test, n_spare = map(int, per_group)
    group_total = n_train + n_test + n_spare
    if group_total <= 0:
        raise ValueError("per_group sums to 0; nothing to generate.")

    # prepare preview dir if needed
    if preview_png_dir is not None:
        from pathlib import Path
        Path(preview_png_dir).mkdir(parents=True, exist_ok=True)

    def _maybe_add_previews(task: dict, qc, n_previews_done: int, max_previews: int):
        """Attach qc_text/qc_png for at most max_previews samples per group."""
        if max_previews <= 0 or n_previews_done >= max_previews:
            return False

        if include_text_preview:
            # ASCII circuit diagram (string)
            try:
                task["qc_text"] = qc.draw(output="text").single_string()
            except Exception:
                # fallback: basic repr
                task["qc_text"] = str(qc)

        if preview_png_dir is not None:
            # Lazy import: only if actually requested
            try:
                fig = qc.draw(output="mpl")
                png_path = f"{preview_png_dir}/{task['task_id']}.png"
                fig.savefig(png_path, dpi=200, bbox_inches="tight")
                # close figure to avoid memory leak
                try:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                except Exception:
                    pass
                task["qc_png"] = png_path
            except Exception:
                # If mpl drawing isn't available, just skip png
                pass

        return True

    train_tasks, test_tasks, spare_tasks = [], [], []
    global_idx = 0

    # 固定顺序遍历，保证 seed 可复现
    for n_qubits in list(n_qubits_choices):
        n_qubits = int(n_qubits)
        for bin_name in list(bins_g.keys()):
            low, high = bins_g[bin_name]

            group = []
            previews_done = 0

            for _ in range(group_total):
                gate_count = int(rng.integers(int(low), int(high) + 1))
                target_qc = _build_random_circuit_by_gates(n_qubits, gate_count, rng)

                task = {
                    "task_id": f"q{n_qubits}_{bin_name}_{global_idx}",
                    "qc": target_qc,
                    "n_qubits": n_qubits,
                    "length_bin": bin_name,
                    "target_gates": int(len(target_qc.data)),
                }

                # optional previews for a few samples in each group
                if _maybe_add_previews(task, target_qc, previews_done, int(preview_sample_per_group)):
                    previews_done += 1

                group.append(task)
                global_idx += 1

            # 组内 shuffle，再切分
            perm = rng.permutation(len(group))
            group = [group[i] for i in perm]

            train_tasks.extend(group[:n_train])
            test_tasks.extend(group[n_train : n_train + n_test])
            spare_tasks.extend(group[n_train + n_test :])

    # 最后再各自 shuffle 一下（不影响分层数量）
    def _shuffle(lst):
        if len(lst) <= 1:
            return lst
        p = rng.permutation(len(lst))
        return [lst[i] for i in p]

    train_tasks = _shuffle(train_tasks)
    test_tasks = _shuffle(test_tasks)
    spare_tasks = _shuffle(spare_tasks)

    return train_tasks, test_tasks, spare_tasks

# ----------------------------
# Storage helpers (Recommended)
# ----------------------------
def save_task_suite(out_dir: str, name: str, tasks: list[dict]) -> None:
    """
    Save tasks to disk in a robust way:
      - Circuits are stored as QPY:  <out_dir>/<name>_circuits.qpy
      - Metadata is stored as JSONL: <out_dir>/<name>_meta.jsonl

    Notes:
      - qc (QuantumCircuit) is NOT JSON serializable. We store qc separately in QPY.
      - JSONL includes: task_id, n_qubits, length_bin, target_gates, and optional qc_text/qc_png if present.
    """
    import json
    from pathlib import Path
    from qiskit import qpy

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    qpy_path = out / f"{name}_circuits.qpy"
    meta_path = out / f"{name}_meta.jsonl"

    circuits = [t["qc"] for t in tasks]
    with qpy_path.open("wb") as f:
        qpy.dump(circuits, f)

    with meta_path.open("w", encoding="utf-8") as f:
        for t in tasks:
            row = {
                "task_id": t["task_id"],
                "n_qubits": int(t["n_qubits"]),
                "length_bin": t["length_bin"],
                "target_gates": int(t.get("target_gates", len(t["qc"].data))),
            }
            if "qc_text" in t:
                row["qc_text"] = t["qc_text"]
            if "qc_png" in t:
                row["qc_png"] = t["qc_png"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_task_suite(in_dir: str, name: str) -> list[dict]:
    """
    Load tasks saved by save_task_suite().
    Returns a list of task dicts with the same keys:
      task_id, qc, n_qubits, length_bin, target_gates, (optional) qc_text, qc_png
    """
    import json
    from pathlib import Path
    from qiskit import qpy

    inp = Path(in_dir)
    qpy_path = inp / f"{name}_circuits.qpy"
    meta_path = inp / f"{name}_meta.jsonl"

    with qpy_path.open("rb") as f:
        circuits = qpy.load(f)

    meta = []
    with meta_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            meta.append(json.loads(line))

    if len(circuits) != len(meta):
        raise ValueError(f"Mismatch: circuits={len(circuits)} meta={len(meta)}")

    tasks = []
    for qc, m in zip(circuits, meta):
        t = {
            "task_id": m["task_id"],
            "qc": qc,
            "n_qubits": int(m["n_qubits"]),
            "length_bin": m["length_bin"],
            "target_gates": int(m.get("target_gates", len(qc.data))),
        }
        if "qc_text" in m:
            t["qc_text"] = m["qc_text"]
        if "qc_png" in m:
            t["qc_png"] = m["qc_png"]
        tasks.append(t)

    return tasks
