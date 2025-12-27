import math
import sys
from pathlib import Path
from qiskit import QuantumCircuit

# 确保项目根目录在 sys.path，兼容直接运行本文件或用 -m 方式
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 直接复用 prepare_tasks.py 的定义
from generate_task_suite.prepare_tasks import action_space_size

# 难度公式（最新版）：score = L * log10(A)
def difficulty_score(n_qubits: int, gate_count: int) -> float:
    A = action_space_size(n_qubits)
    return float(gate_count) * math.log10(A)


def _cz_via_h_cx_h(qc: QuantumCircuit, control: int, target: int) -> None:
    qc.h(target)
    qc.cx(control, target)
    qc.h(target)

def make_task_dict(qc: QuantumCircuit, prefix: str, n_qubits: int):
    L = int(len(qc.data))                    # gate_count = 实际 append 的门数
    sc = difficulty_score(n_qubits, L)       # 与随机任务使用同一难度公式
    task_id = f"{prefix}_{n_qubits}_{L}_{int(sc)}"  # 与保存时 difficulty_score(int) 保持一致
    return {
        "qc": qc,
        "task_id": task_id,
        "n_qubits": int(n_qubits),
        "target_gates": int(L),
        "difficulty_score": sc,
    }

# ---------- 常用电路：全部用基础门构造 + qc 固定 4 qubits ----------
def build_bell_fixed4(pair=(0, 1)) -> QuantumCircuit:
    qc = QuantumCircuit(4)
    a, b = pair
    qc.h(a)
    qc.cx(a, b)
    return qc

def build_ghz_fixed4(n_qubits: int) -> QuantumCircuit:
    assert 2 <= n_qubits <= 4
    qc = QuantumCircuit(4)
    qc.h(0)
    for i in range(1, n_qubits):
        qc.cx(0, i)
    return qc

def build_cluster_chain_fixed4(n_qubits: int) -> QuantumCircuit:
    assert 2 <= n_qubits <= 4
    qc = QuantumCircuit(4)
    for q in range(n_qubits):
        qc.h(q)
    for i in range(n_qubits - 1):
        _cz_via_h_cx_h(qc, i, i + 1)
    return qc

def build_graph_star_fixed4(n_qubits: int, center: int = 0) -> QuantumCircuit:
    assert 2 <= n_qubits <= 4
    qc = QuantumCircuit(4)
    for q in range(n_qubits):
        qc.h(q)
    for j in range(n_qubits):
        if j == center:
            continue
        _cz_via_h_cx_h(qc, center, j)
    return qc

def generate_common_tasks_qiskit(max_n: int = 4, include_star_graph: bool = True):
    tasks = []

    # Bell (2q)
    qc = build_bell_fixed4((0, 1))
    tasks.append(make_task_dict(qc, "bell", n_qubits=2))

    # n=3..max_n 的 GHZ / cluster / star
    for n in range(3, max_n + 1):
        qc = build_ghz_fixed4(n)
        tasks.append(make_task_dict(qc, "ghz", n_qubits=n))

        qc = build_cluster_chain_fixed4(n)
        tasks.append(make_task_dict(qc, "cluster_chain", n_qubits=n))

        if include_star_graph:
            qc = build_graph_star_fixed4(n, center=0)
            tasks.append(make_task_dict(qc, "graph_star_c0", n_qubits=n))

    return tasks
if __name__ == "__main__":
    tasks = generate_common_tasks_qiskit(max_n=4)
    for t in tasks:
        print(t["task_id"], "n=", t["n_qubits"], "target_gates=", t["target_gates"], "difficulty_score=", t.get("difficulty_score"))




