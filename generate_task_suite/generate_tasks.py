from collections import Counter
import json
from pathlib import Path

from prepare_tasks import (
    generate_tasks_by_score_distribution,   # ✅ 新版：按 score-bin + probs 生成 tasks
    build_equal_width_score_bins,           # ✅ 自动构造 score bins（可选）
    DEFAULT_SCORE_BIN_PROBS,                # ✅ easy/medium/... 的概率
    save_task_suite,                        # ✅ 保存 QPY + JSONL
)


def _bin_key(t: dict) -> str:
    # 兼容：优先 difficulty_bin，否则退回 length_bin，否则 NA
    return t.get("difficulty_bin", t.get("length_bin", "NA"))


def _print_suite_stats(name: str, tasks: list[dict]) -> None:
    print(f"\n=== {name} ===")
    print("num_tasks:", len(tasks))
    print("n_qubits:", dict(Counter([t.get("n_qubits") for t in tasks])))
    print("bin:", dict(Counter([_bin_key(t) for t in tasks])))


def main():
    # -----------------------
    # 你可以在这里改规模
    # -----------------------
    OUT_DIR = "task_suites"
    N_QUBITS_CHOICES = (2, 3, 4)

    N_DEV = 100     # 用于快速 smoke test / pipeline debug（可选）
    N_VAL = 200     # 用于 HPO 选超参
    N_EVAL = 600    # 用于最终报告表现（或更大）

    SEED_DEV = 0
    SEED_VAL = 1
    SEED_EVAL = 2

    MIN_GATES = 4
    MAX_GATES = 60
    MAX_TRIES = 2000   # 生成失败时重试上限；若你 bins 合理，一般不需要太大

    probs = DEFAULT_SCORE_BIN_PROBS

    # -----------------------
    # 关键：固定 score_bins，保证 Dev/Val/Eval 同一分布划分标准
    # 你也可以改成手动传你自己的 DEFAULT_SCORE_BINS
    # -----------------------
    score_bins = build_equal_width_score_bins(
        n_qubits_choices=N_QUBITS_CHOICES,
        min_gates=MIN_GATES,
        max_gates=MAX_GATES,
        labels=tuple(probs.keys()),
    )

    # -----------------------
    # 生成三套 tasks（同 bins、同 probs，不同 seed & 不同大小）
    # -----------------------
    dev_tasks, _ = generate_tasks_by_score_distribution(
        seed=SEED_DEV,
        n_tasks=N_DEV,
        n_qubits_choices=N_QUBITS_CHOICES,
        probs=probs,
        score_bins=score_bins,
        min_gates=MIN_GATES,
        max_gates=MAX_GATES,
        max_tries=MAX_TRIES,
    )

    val_tasks, _ = generate_tasks_by_score_distribution(
        seed=SEED_VAL,
        n_tasks=N_VAL,
        n_qubits_choices=N_QUBITS_CHOICES,
        probs=probs,
        score_bins=score_bins,
        min_gates=MIN_GATES,
        max_gates=MAX_GATES,
        max_tries=MAX_TRIES,
    )

    eval_tasks, _ = generate_tasks_by_score_distribution(
        seed=SEED_EVAL,
        n_tasks=N_EVAL,
        n_qubits_choices=N_QUBITS_CHOICES,
        probs=probs,
        score_bins=score_bins,
        min_gates=MIN_GATES,
        max_gates=MAX_GATES,
        max_tries=MAX_TRIES,
    )

    # -----------------------
    # 简单打印统计，检查分布是否大致符合 probs
    # -----------------------
    _print_suite_stats("Dev", dev_tasks)
    _print_suite_stats("Val", val_tasks)
    _print_suite_stats("Eval", eval_tasks)

    # -----------------------
    # 保存：QPY + JSONL（你已有 save_task_suite）
    # -----------------------
    save_task_suite(OUT_DIR, "Dev", dev_tasks)
    save_task_suite(OUT_DIR, "Val", val_tasks)
    save_task_suite(OUT_DIR, "Eval", eval_tasks)

    # -----------------------
    # 强烈建议：把 bins 和 probs 也存下来（复现/写论文用）
    # -----------------------
    out = Path(OUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    (out / "score_bins.json").write_text(
        json.dumps(score_bins, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out / "score_bin_probs.json").write_text(
        json.dumps(probs, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nSaved to: {OUT_DIR}/Dev_* , {OUT_DIR}/Val_* , {OUT_DIR}/Eval_*")
    print(f"Also saved: {OUT_DIR}/score_bins.json , {OUT_DIR}/score_bin_probs.json")


if __name__ == "__main__":
    main()
