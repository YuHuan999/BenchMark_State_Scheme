from collections import Counter
import json
from pathlib import Path

from prepare_tasks import (
    generate_tasks_by_score_distribution,   # ✅ 新版：按 score-bin + probs 生成 tasks
    build_equal_width_score_bins,           # ✅ 自动构造 score bins（可选）
    DEFAULT_SCORE_BIN_PROBS,                # ✅ easy/medium/... 的概率
    generate_tasks_from_guidance,           # ✅ 根据 guidance 生成 tasks
    save_task_suite,                        # ✅ 保存 QPY + JSONL
    DEFAULT_SCORE_BINS,
)
from prepare_common_task import generate_common_tasks_qiskit


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


def generate_save_by_guidance_random_common(
    guidance: list,
    per_guidance: int = 1,
    *,
    out_dir: str = "task_suites",
    set_name: str = "Guided",
    seed: int = 0,
    include_common: bool = True,
    common_max_n: int = 4,
    common_include_star: bool = True,
    score_bins: dict | None = None,
):
    tasks, bins = generate_tasks_from_guidance(
        seed=seed,
        guidance=guidance,
        per_guidance=per_guidance,   # 每条 guidance 生成几个任务
        max_tries=200,
        bins=None         # guidance 仍按 int(score) 分档
    )

    # 追加常用任务：按 score_bins 归到对应 bin；若无则默认 DEFAULT_SCORE_BINS
    if include_common:
        sbins = score_bins or DEFAULT_SCORE_BINS
        def _bin_for_score(sc: float) -> str | int:
            for name, (lo, hi) in sbins.items():
                if lo <= sc <= hi:
                    return name
            return int(sc)

        common_tasks = generate_common_tasks_qiskit(
            max_n=common_max_n,
            include_star_graph=common_include_star,
        )
        for t in common_tasks:
            sc = float(t["difficulty_score"])
            t["difficulty_bin"] = _bin_for_score(sc)
        tasks.extend(common_tasks)
        bins = sbins

    save_task_suite(out_dir, set_name, tasks)

def generate_save_by_guidance_random(
    guidance: list,
    per_guidance: int = 1,
    *,
    out_dir: str = "task_suites",
    set_name: str = "Guided",
    seed: int = 0,
):

    tasks, bins = generate_tasks_from_guidance(
        seed=seed,
        guidance=guidance,
        per_guidance=per_guidance,   # 每条 guidance 生成几个任务
        max_tries=200,
        bins=None         # 不用分档名，就用 int(score) 作为 difficulty_bin
    )

    save_task_suite(out_dir, set_name, tasks)


if __name__ == "__main__":
    # main()
# ---------------------
    Main_guidances = [
  [2, 1, 1.079181], [3, 1, 1.322219], [4, 1, 1.505150],
  [2, 2, 2.158362], [3, 2, 2.644439], [4, 2, 3.010300],
  [2, 3, 3.237544], [3, 3, 3.966658], [2, 4, 4.316725],
  [4, 3, 4.515450], [3, 4, 5.288877], [2, 5, 5.395906],
  [4, 4, 6.020600], [2, 6, 6.475087], [3, 5, 6.611096],
  [4, 5, 7.525750], [2, 7, 7.554269], [3, 6, 7.933316],
  [2, 8, 8.633450], [4, 6, 9.030900], [3, 7, 9.255535],
  [2, 9, 9.712631], [4, 7, 10.536050], [3, 8, 10.577754],
  [2, 10, 10.791812], [2, 11, 11.870994], [3, 9, 11.899973],
  [4, 8, 12.041200], [2, 12, 12.950175], [3, 10, 13.222193],
  [4, 9, 13.546350], [2, 13, 14.029356], [3, 11, 14.544412],
  [4, 10, 15.051500], [2, 14, 15.108537], [3, 12, 15.866631],
  [2, 15, 16.187719], [4, 11, 16.556650], [3, 13, 17.188851],
  [2, 16, 17.266900], [4, 12, 18.061800], [2, 17, 18.346081],
  [3, 14, 18.511070], [2, 18, 19.425262], [4, 13, 19.566950],
  [3, 15, 19.833289], [2, 19, 20.504444], [4, 14, 21.072100],
  [3, 16, 21.155508], [2, 20, 21.583625], [3, 17, 22.477728],
  [4, 15, 22.577250], [3, 18, 23.799947], [4, 16, 24.082400],
  [3, 19, 25.122166], [4, 17, 25.587550], [3, 20, 26.444385],
  [4, 18, 27.092700], [4, 19, 28.597850], [4, 20, 30.103000]
]

    guidance_tasks = [
    ## [n_qubits, n_gates, difficulty_score = L * log10(A)]
        [2, 1, 1.079181], 
        [3, 1, 1.322219], 
        [4, 1, 1.505150],
        [2, 2, 2.158362], 
        [3, 2, 2.644439], 
        [4, 2, 3.010300],
        [2, 3, 3.237544], 
        [3, 3, 3.966658], 
        [2, 4, 4.316725],
        [4, 3, 4.515450], 
        [3, 4, 5.288877], 
        [2, 5, 5.395906],
        [4, 4, 6.020600], 
        [2, 6, 6.475087], 
        [3, 5, 6.611096],
        [4, 5, 7.525750], 
        [2, 7, 7.554269], 
        [3, 6, 7.933316],
        [2, 8, 8.633450], 
        [4, 6, 9.030900], 
        [3, 7, 9.255535],
        [2, 9, 9.712631], 
        [4, 7, 10.536050], 
        [3, 8, 10.577754],
        [2, 10, 10.791812], 
        [2, 11, 11.870994], 
        [3, 9, 11.899973],
        [4, 8, 12.041200], 
        [2, 12, 12.950175], 
        [3, 10, 13.222193],
        [4, 9, 13.546350], 
        [2, 13, 14.029356], 
        [3, 11, 14.544412],
        [4, 10, 15.051500], 
        [2, 14, 15.108537], 
        [3, 12, 15.866631],]
    generate_save_by_guidance_random_common(guidance_tasks, per_guidance=20, out_dir="task_suites", set_name="Final", seed=77)