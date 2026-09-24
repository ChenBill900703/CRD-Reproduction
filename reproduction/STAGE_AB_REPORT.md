# 階段 A/B 完成報告：真實 CIFAR-100 驗證

日期：2026-09-17。階段A來源查核、階段B相容性實作與必要驗證已完成；尚未啟動階段C的240-epoch正式訓練。本報告不代表整篇論文或第一組5-seed結果已重現。

## 教師與資料（實測）

- 官方checkpoint：resnet32x4_vanilla/ckpt_epoch_240.pth，檢查epoch=240並strict載入。
- 來源：http://shape2prog.csail.mit.edu/repo/resnet32x4_vanilla/ckpt_epoch_240.pth
- SHA-256：`22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f`。
- 完整官方test split 10,000張，無TTA：**top-1=79.42%、top-5=94.58%**；CE=0.8833264050。相對論文teacher top-1=79.42%的差距為**0.00百分點**。這只證實此teacher的評估，不預測學生精度。
- 詳細紀錄：`runs/teacher_evaluation/teacher.json`、config.json、environment.json、status.json。
- CIFAR100壓縮檔MD5：`eb9058c3a382ffc7106e4002c42a8d85`。train/test/meta的MD5均與torchvision官方清單相符，見`artifacts/stage_b/extracted-data-verification.json`。資料來源與整檔SHA-256見`artifacts/data-provenance.json`。

開始時受限執行帳號無法讀取先前下載建立的資料目錄，torchvision回報「不存在或損壞」。檢查期間亦出現PermissionError；重新校驗與寫出資料後，以可存取資料的使用者環境執行，抽樣與評估均通過。先前「解壓完成」未充分驗證受限帳號可讀性；不是論文資料或抽樣設定變更。最初失敗log保留在`artifacts/stage_b/real-sampler.log`及`real-sampler-retry.log`，成功log是`real-sampler-access.log`。一般使用者PowerShell可直接使用已下載資料；若由受限工具執行，需使用可讀該目錄的執行環境，勿因此更換資料來源。

## 短程smoke（實測，不能與論文accuracy比較）

以下均seed=0、trial=stage_b、tau=.07（vanilla/KD不使用CRD tau），B64、K16384、50000筆memory index、workers=0、FP32。每個截短epoch只有3 training batches（192筆），共2個截短epoch／6 steps；學生每次只評估64張test。教師仍評估完整10000張。

|方法|最終截短epoch|final top-1|final top-5|best top-1|狀態|
|---|---:|---:|---:|---:|---|
|vanilla|2|1.5625|6.25|1.5625|complete smoke|
|KD|2|0|4.6875|0|complete smoke|
|CRD|2|0|3.125|1.5625|complete smoke|
|CRD+KD|2|0|10.9375|0|complete smoke|

四條路徑均完成真實資料載入、augmentation、forward/backward、test evaluation、完整checkpoint、推論checkpoint和JSONL metrics，無非有限數值。極短程學生accuracy沒有研究比較意義，沒有據此改設定。

路徑：`runs/smoke_<method>_seed0_trialstage_b/`，method為vanilla/kd/crd/crd_kd；每個run都有config、environment、metrics.jsonl、status、events、resume.pt、student_final.pt和curves.png。原始console log在`artifacts/stage_b/smoke_<method>.log`。

## 真實資料resume（實測）

CRD另跑trial=stage_b_resume：在第一個截短epoch保存退出，再從resume.pt啟動新程序完成第二個截短epoch。與trial=stage_b的連續執行比較，student、雙projection、兩個memory、Z/params、optimizer、Python/NumPy/Torch CPU/CUDA/DataLoader RNG、global step、best及全部非計時metrics均**完全一致**（tensor rtol=0、atol=0）。證據：`artifacts/stage_b/real-resume-verification.json`。

這驗證workers=0的smoke邊界，不承諾任意batch中斷接續；前輪另以完整合成小資料epoch測試了尾batch與shuffle恢復。

## 計時與資源（短程實測 + 明確外推）

四方法各執行一次獨立30-step計時run（1920筆train），之後完整評估10000張test。trial=benchmark30，配置`configs/smoke_benchmark_<method>.json`。仍是smoke，不是正式epoch。測得學生top-1分別4.06/3.83/3.80/3.40%，只保留為流程紀錄，無論文差距或統計。

|方法|30 training steps 秒數|完整test秒數|peak allocated GiB|每run 240 epochs 外推小時|
|---|---:|---:|---:|---:|
|vanilla|1.326|1.960|0.323|2.43|
|KD|1.410|1.998|0.388|2.58|
|CRD|3.776|1.982|1.347|6.69|
|CRD+KD|3.852|2.020|1.347|6.83|

外推公式：`(train30_seconds/30 × 782 + test10000_seconds) × 240 / 3600`。782=ceil(50000/64)，尾batch保留16筆。這不是完整訓練實測，不含初始化/教師評估/checkpoint I/O，30步包含短程啟動效應，也不能反映長期溫度/背景程序/硬體穩定性。

- seed0四方法線性外推約18.54小時。
- author_code四方法×5 seeds共20 runs：約92.7 GPU小時。
- paper_tau兩方法×5 seeds共10 runs：約67.6 GPU小時；此處**假設tau=.1吞吐近似tau=.07，尚未用真實資料計時tau=.1**。
- 合計30 runs線性外推約160.3小時（6.7天）；加25–50%排程餘裕約200–240小時（8.4–10天）。不是保證工期，亦不因估計而改動batch/K/workers。
- GPU測試使用RTX3070Ti 8GB，所列為PyTorch peak allocated，非整卡總顯存；單GPU FP32正式B/K可行。
- 每run最新resume：vanilla/KD約9.94MB，CRD約61.67MB；另有student約4.97MB。30 runs最終checkpoint合计約1.5GB；原子保存需額外一份當前checkpoint暫存空間。建議另留至少5GB輸出餘裕，不含已安裝環境與資料。系統先前盤點RAM約31.8GiB，磁碟剩餘超過500GB。

機器可讀估計：`artifacts/stage_b/resource-estimates.json`。環境精確版本見各run/environment.json；依賴鎖定見`artifacts/requirements-lock.txt`。

## 比較表、命令與停止點

`artifacts/comparison.csv`列出全部9個smoke runs（包含resume核對run），全部eligible=False。`artifacts/summary.json`目前為空陣列，沒有冒充正式mean/std。run中的曲線已產生；CRD+KD兩個截短epoch曲線已視覺檢查。

正式命令（**提供但未執行**），在PowerShell中：

```powershell
Set-Location 'E:/Contrastive Representation Distillation'
& ./.venv/Scripts/python.exe reproduction/train_teacher.py --config configs/author_code_vanilla.json --seed 0 --trial 1
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_kd.json --seed 0 --trial 1
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_crd.json --seed 0 --trial 1
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_crd_kd.json --seed 0 --trial 1
```

預先排序的全30 runs腳本為`reproduction/run_formal.ps1`，尚未執行。先seed0四方法，再seeds1–4，最後paper_tau；trial和seed分開。完整準備、重建環境、評估、resume及彙整命令見`COMMANDS.md`。研究來源及差異見`SOURCE_AUDIT.md`，20項合成測試見`PREFLIGHT_REPORT.md`。

下一階段需由使用者明確啟動正式訓練。主要比較仍限定epoch240 final；不依best挑模型，不篩seed，不把CRD+KD低於CRD視為錯誤。本輪不改動已通過查核的Python訓練實作。
