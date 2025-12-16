from prepare_tasks import (
    generate_task_suite_stratified,
    save_task_suite,
)

def main():
    train_tasks, test_tasks, spare_tasks = generate_task_suite_stratified(
        seed=0,
        n_qubits_choices=(2, 3, 4),
        per_group=(40, 20, 10),      # 每个 (n_qubits, length_bin) 组：40 train / 20 test / 10 spare
        mode="gates",

        # 可选：预览（默认都关）
        include_text_preview=False,  # True 会把 ASCII 电路图放在 task["qc_text"]
        preview_png_dir=None,        # 比如 "task_previews" 会导出 png 并写 task["qc_png"]
        preview_sample_per_group=0,  # 每组生成多少个预览（建议 1~3）
    )

    print(len(train_tasks), len(test_tasks), len(spare_tasks))  # 期望：600 300 150
    print(train_tasks[0].keys())  # 会包含 target_gates

    # ✅ 推荐存法：电路(QPY) + 元数据(JSONL)
    save_task_suite("task_suites", "train", train_tasks)
    save_task_suite("task_suites", "test", test_tasks)
    save_task_suite("task_suites", "spare", spare_tasks)

if __name__ == "__main__":
    main()
