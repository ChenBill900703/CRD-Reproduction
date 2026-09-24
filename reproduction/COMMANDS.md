# 操作命令（PowerShell）

工作目錄固定為 `E:/Contrastive Representation Distillation`。原始 reference 不變，執行 reproduction。目前兩個訓練入口只接受新的 `--config` 路徑；`--help` 可直接使用。legacy CLI 已明確停用，原碼仍保留於 reference，不宣稱支援其他蒸餾方法。

## 已建立環境的使用

```powershell
Set-Location 'E:/Contrastive Representation Distillation'
& ./.venv/Scripts/python.exe reproduction/test_no_data.py
# 以下需要下一階段授權；本輪不執行：
# & ./.venv/Scripts/python.exe reproduction/test_semantics.py Semantics.test_sampler_real
# & ./reproduction/prepare.ps1
```

重建相同 Python 3.12 環境（本輪測試精確版本 3.12.14；其他 Python 小版本未測）：

```powershell
# $python312 請設為本機 Python 3.12.14 的完整路徑
$python312 = 'C:/Users/ChenBill/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
& $python312 -m venv .venv_rebuilt
& ./.venv_rebuilt/Scripts/python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r artifacts/requirements-lock.txt
& ./.venv_rebuilt/Scripts/python.exe -m pip check
```

教師評估（避免覆盖，已存在的目錄請保留並改用新的輸出規約；此命令為首次執行用）：

```powershell
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_kd.json --seed 0 --trial teacher --evaluate-teacher
```

短程 smoke：真實 CIFAR100、正式 B64/K16384/50000 memory，2 個截短 epoch，各3 training batches、2 test batches（僅64筆 test）。教師評估仍使用完整10000筆。smoke 學生 accuracy 不可與論文正式比較。

```powershell
& ./.venv/Scripts/python.exe reproduction/train_teacher.py --config configs/smoke_vanilla.json --seed 0 --trial smoke_new
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/smoke_kd.json --seed 0 --trial smoke_new
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/smoke_crd.json --seed 0 --trial smoke_new
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/smoke_crd_kd.json --seed 0 --trial smoke_new
```

## 正式訓練命令（30 runs 已完成；重跑需先備份既有結果）

每個 JSON 包含完整固定設定，執行時寫入 run/config.json。run 已存在就拒絕覆盖。trial 可命名但不設定 seed。

```powershell
& ./.venv/Scripts/python.exe reproduction/train_teacher.py --config configs/author_code_vanilla.json --seed 0 --trial 1
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_kd.json --seed 0 --trial 1
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_crd.json --seed 0 --trial 1
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_crd_kd.json --seed 0 --trial 1
```

預先排序的全部30 runs（20 baseline + 10 paper_tau），執行前應閱讀資源估計：

```powershell
& ./reproduction/run_formal.ps1
```

paper_tau 命令（執行順序在全部 baseline 後）：

```powershell
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/paper_tau_crd.json --seed 0 --trial 1
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/paper_tau_crd_kd.json --seed 0 --trial 1
```

epoch 邊界暫停／接續範例（正式指令，**尚未執行**）：

```powershell
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_crd.json --seed 0 --trial 1 --stop-after-epoch 1
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_crd.json --seed 0 --trial 1 --resume runs/author_code_crd_seed0_trial1/resume.pt
```

resume.pt 僅載入本專案自己生成可信的 checkpoint。student_final.pt 為推論權重，不能拿來 resume。每 epoch 覆写目前 resume；並不儲存全部240份，以控制磁碟。若在 epoch 中中断，從上個完成 epoch 重跑。程式 hash/config/teacher checksum 改變會拒絕 resume，需另列版本處理。

```powershell
& ./.venv/Scripts/python.exe reproduction/summarize.py
& ./.venv/Scripts/python.exe reproduction/plot_metrics.py
```

comparison.csv 保留 smoke/failed/paused；eligible=False 的項目不計正式 gap/mean/std。std 採 ddof=1；只有 seeds 精確為0–4且各一筆、均 complete epoch240，formal_five_seed 才成立。best 只做紀錄，主比較來自 final。

## 2026-09-17 程式查核補充

`test_no_data.py` 不讀 CIFAR100；會使用已核對的教師 checkpoint 與合成影像，執行必要 forward/backward、模擬執行入口與完整 resume。合成測試結果存在 artifacts/preflight，不是正式研究 run。

resume checkpoint schema 現為2，加入完整 epoch metrics 歷史。checkpoint 是唯一提交點，metrics/status/推論檔可從它重建。若最後 epoch 已完成但結果發布中斷，再執行 resume 只修復結果檔，不追加訓練。舊 schema1 測試 checkpoint 不相容；尚無正式訓練 checkpoint 需要遷移。

限定 workers=0、預定正式 seeds0–4；未知設定欄位與不同 runtime environment 都拒絕接續。每步新增非有限 loss/gradient/parameter 檢查，不裁剪梯度、不改 loss。失敗時從上一個完整 epoch 的 checkpoint 接續；forward 已更新的 memory 不會單步回滾。記錄中的 seconds/elapsed_seconds 是 train+evaluation 時間，不含 checkpoint I/O 與初始教師評估。

## 真實資料驗證完成（2026-09-17）

教師評估與trial=stage_b/benchmark30的smoke已實際執行，詳見STAGE_AB_REPORT.md。既有run目錄不覆蓋；重跑smoke需另命名trial。正式30 runs已完成；本段其餘內容為階段B歷史紀錄。資料目錄由下載程序建立，受限帳號可能無法讀取；本轮使用可讀資料的使用者環境通過官方MD5、抽樣與訓練測試。

