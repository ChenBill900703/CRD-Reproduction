# 檔案導航

- `SOURCE_AUDIT.md`：論文／固定 commit 對照、研究決策及限制。
- `COMMANDS.md`：環境重建、準備、評估、smoke、正式訓練及 resume 命令。
- `PREFLIGHT_REPORT.md`：不依賴 CIFAR100 的程式查核結果（歷史查核階段）。
- `STAGE_AB_REPORT.md`：已完成的真實資料教師驗證、smoke、resume及資源估計；正式訓練尚未啟動。
- `../configs/`：author_code、paper_tau、smoke，固定設定與執行限制。
- `reproduce.py`：限定第一組 CIFAR100 的最小執行入口。
- `train_teacher.py --config ...`：vanilla supervised route；不需要 teacher。
- `train_student.py --config ...`：KD、CRD、CRD+KD route。
- `test_no_data.py`、`test_semantics.py`、`test_preflight.py`、`test_results.py`：研究語意、resume 及結果選取測試。
- `summarize.py`、`plot_metrics.py`：原始 metrics 的彙整及曲線。
- `../artifacts/`：來源 hashes／patch、環境 snapshot／lock、測試 logs、比較表。
- `../runs/`：實際執行的 config、environment、metrics、status、events、checkpoints。

本輪不代表整篇論文已重現，也沒有完成第一組正式5-seed比較。使用者指定的原始碼副本在 `../reference/RepDistiller` 保留原樣。
