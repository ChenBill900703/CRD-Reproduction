# CRD 論文重現：從原始碼到 30 次完整實驗

> **已完成**：CIFAR-100，`resnet32x4 → resnet8x4`，6 組設定 × 5 個 seeds × 240 epochs。  
> 每個數字都能追到設定、逐 epoch 紀錄與原作者實作；主要結果使用 **最後 epoch**，不是最好的一次。

![Final comparison](reports/final-comparison.png)

## 先看結論

CRD 可以讓小模型學得更好。本次公開程式設定下，學生從零訓練為 **72.488%**，CRD 為 **75.328%**，增加 **2.840 個百分點**。結果接近論文，但不宣稱精確複製原作者實驗，也不宣稱整篇論文已重現。

| 設定 | 方法 | 本次 final top-1，mean ± std | 論文參考 | 差距（百分點） |
|---|---|---:|---:|---:|
| author_code | Vanilla | 72.488 ± 0.272 | 72.50 | −0.012 |
| author_code | KD | 73.174 ± 0.206 | 73.33 ± 0.25 | −0.156 |
| author_code，τ=0.07 | CRD | 75.328 ± 0.201 | 75.51 ± 0.18 | −0.182 |
| author_code，τ=0.07 | CRD+KD | 75.330 ± 0.255 | 75.46 ± 0.25 | −0.130 |
| paper_tau，τ=0.1 | CRD | 75.208 ± 0.196 | 75.51 ± 0.18 | −0.302 |
| paper_tau，τ=0.1 | CRD+KD | 75.114 ± 0.223 | 75.46 ± 0.25 | −0.346 |

每組都是預先指定 seeds `[0,1,2,3,4]`，沒有刪掉較差的 run。標準差採 **ddof=1**；seeds 與這個計算慣例是本次新增規約，作者的 seeds／std 分母未確認。教師實測 **79.42% top-1、94.58% top-5**。

**怎麼解讀？** 公開程式基準的平均差距都小於 0.2 個百分點。CRD 與 CRD+KD 幾乎相同，不適合宣稱後者勝出。τ=0.1 在此配對略低，也不能據此說其他架構一定一樣。論文標準差不是硬性通過門檻。

- [完整逐 seed 結果與限制](reports/FINAL_REPORT.md)
- [機器可讀的最終查核](reports/final-audit.json)
- [彙整 JSON](artifacts/summary.json)／[逐 run CSV](artifacts/comparison.csv)
- [來源對照](reproduction/SOURCE_AUDIT.md)
- [相容性與語意測試](reproduction/PREFLIGHT_REPORT.md)／[真實資料 smoke 報告](reproduction/STAGE_AB_REPORT.md)

## CRD 到底在做什麼？

KD 讓學生模仿教師的分類機率；CRD 則讓學生模仿教師如何表示一張圖片。

論文第 3 節 Eq. (4)–(18) 用二元判別區分「同一樣本的師生表示」與「各自獨立抽取的表示」，從 joint distribution 與 product of marginals 推導 mutual-information lower bound。Eq. (19) 讓師生倒數第二層特徵各經線性投影、L2 normalization，再用內積與 temperature 建立 critic。

實際執行沿用官方 **雙向 NCE 二元對比損失 + 跨 batch memory banks**，不是 SimCLR／CLIP 的 batch softmax，也不是只比較當前 batch。兩方向 loss 相加，教師 backbone 固定，但教師 projection head 仍訓練。

### 為什麼有兩種 temperature？

論文 p.10 §4.5 說 CIFAR 使用 τ=0.1；本次固定 commit 的 `train_student.py` 預設卻是 0.07，CIFAR 範例沒有覆寫。因此先執行 `author_code` 0.07，再執行只改 τ 的 `paper_tau` 0.1。兩者都保留，沒有依結果選出「作者真值」。這個來源衝突尚未解決。

論文 Eq. (19) 的簡化表達也不能等同完整程式流程：官方保留首次 forward 決定的 Z normalization；CIFAR negatives 排除同類別，但 NCE 的 Pn 仍為 1/50000。

## 固定設定一覽

| 項目 | 設定與來源 |
|---|---|
| 官方版本 | HobbitLong/RepDistiller `b84f547c5db6a35318d4671d7d5c4de74c822403`，2023-10-16；不是已證實產生論文數值的歷史版本 |
| 模型 | 作者 CIFAR ResNet；256 維 pooled feature、100 類 logits；非 torchvision ImageNet ResNet |
| 資料 | CIFAR-100 train 50000 / test 10000；沒有另設 validation |
| 增強 | RandomCrop(32,padding=4)、RandomHorizontalFlip、ToTensor、Normalize；師生看同一張增強影像 |
| Normalize | mean=(0.5071,0.4867,0.4408)，std=(0.2675,0.2565,0.2761) |
| Batch | train 64，test 32；保留最後不足 batch |
| Optimizer | SGD，LR 0.05，momentum 0.9，weight decay 0.0005，240 epochs |
| LR | epochs 1–150: .05；151–180: .005；181–210: .0005；211–240: .00005 |
| KD | T=4，KL(teacher || student) × T²，sum / batch |
| CRD | Linear(256,128)+bias → L2；K=16384；momentum=.5；兩個 50000×128 memories |
| Sampling | exact positive=同一全域 index；異類 negatives，不放回；index shape=[B,16385] |
| Memory | uniform ±1/sqrt(128/3)，初始不先正規化；先算分數再於 forward 更新指定列 |
| Z / reduction | 每方向 Z 首次初始化後保留；Pn=1/50000；正負項 sum/B；雙向相加 |
| 本次規約 | seeds0–4，workers0，單 GPU FP32；cuDNN benchmark=False、deterministic=True，TF32=False |

`L = gamma × CE + alpha × KD + beta × CRD`：

| 方法 | gamma | alpha | beta |
|---|---:|---:|---:|
| Vanilla | 1 | 0 | 0 |
| KD | 0.1 | 0.9 | 0 |
| CRD | 1 | 0 | 0.8 |
| CRD+KD | 1 | 1 | 0.8 |

workers0 與 deterministic 設定是本次可重現性變更，並非作者原始 defaults。相容性修補包括 torchvision data/targets、device handling、Windows 路徑／命名、KD reduction 與 checkpoint API。細節見來源與測試報告。

## 專案地圖

```text
reproduction/       官方結構上的最小修補、訓練入口、測試與彙整
reference/          固定版本官方原始碼（獨立來源，不修改）
configs/            author_code / paper_tau / smoke 分開的設定
runs/               設定、環境、逐 epoch metrics、status、events、曲線
artifacts/          測試證據、環境鎖定、來源 manifest、console logs
reports/            最終報告、查核與比較圖
tools/              收尾報告工具；不屬於凍結訓練程式碼
```

本機 checkpoint、資料集、虛擬環境與中斷備份仍保留，但不放入 Git。Git 版本提供完整程式、設定及文字／圖表證據。30 runs 的完整 resume 與 student_final 權重另放在 [v1.0.0 Release](https://github.com/ChenBill900703/CRD-Reproduction/releases/tag/v1.0.0)，下載 formal-checkpoints.zip 與 SHA256SUMS.txt 核對後，在專案根目錄解壓。resume 仍需原環境與 config，跨路徑／硬體不保證直接接續。

## 如何重跑（Windows PowerShell）

已測環境：Windows 11、RTX 3070 Ti 8GB、Python 3.12.14、PyTorch 2.6.0+cu124、torchvision 0.21.0+cu124、NumPy 2.5.2。其他組合未驗證。完整套件版本在 [requirements-lock.txt](artifacts/requirements-lock.txt)。

官方 reference 以固定 commit submodule 保存，clone 後先執行 `git submodule update --init`。在專案根目錄執行；`python` 必須是你已安裝的 Python 3.12.14：

```powershell
python -m venv .venv
& ./.venv/Scripts/python.exe -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r artifacts/requirements-lock.txt
& ./.venv/Scripts/python.exe -m pip check
& ./reproduction/prepare.ps1
& ./.venv/Scripts/python.exe reproduction/test_no_data.py
```

prepare.ps1 取得教師並比對 SHA-256、下載官方 CIFAR-100。教師實際使用來源：
`http://shape2prog.csail.mit.edu/repo/resnet32x4_vanilla/ckpt_epoch_240.pth`

SHA-256：`22aa12e2b632b19ddde155ccf2388671842ddd304824b3095e0a7fff364beb4f`。
連結未來可能失效；校驗不符應停止，不要替換成未知權重。資料來源與 MD5 詳見 artifacts/data-provenance.json。

### 保留已發布結果，建立新的執行空間

本 repository 已包含 `runs/` 與 `artifacts/formal/queue.json` 的歷史紀錄。訓練入口會拒絕覆蓋同名 run。若要完整重跑，先把結果移到備份目錄，再執行；不要將新舊結果混在同一彙整中。

```powershell
Move-Item -LiteralPath runs -Destination published-runs
# 教師驗證
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_kd.json --seed 0 --trial teacher --evaluate-teacher
# 小規模 smoke，不能當論文數字
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/smoke_crd.json --seed 0 --trial smoke_new
# 完整 30 runs：預先固定順序，約需一週單 GPU 時間
& ./reproduction/run_formal.ps1
& ./.venv/Scripts/python.exe reproduction/plot_metrics.py
& ./.venv/Scripts/python.exe tools/finalize_report.py
```

`run_formal.ps1` 不會自行跳過失敗 run。上面的完整重跑命令是使用方式，不是另一輪已執行的實驗。歴史 logs 中的 E:/ 與使用者路徑反映當時機器；新的 run 會從程式根目錄產生新路徑。

### 單次執行與中斷恢復

```powershell
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_crd.json --seed 0 --trial 1
# 只對同一個、已有可信 resume.pt 的 run 使用：
& ./.venv/Scripts/python.exe reproduction/train_student.py --config configs/author_code_crd.json --seed 0 --trial 1 --resume runs/author_code_crd_seed0_trial1/resume.pt
```

Vanilla 使用 `train_teacher.py`；KD/CRD/CRD+KD 使用 `train_student.py`。CLI 以 JSON config 為主；`trial` 只是名稱，`seed` 才控制亂數。完整命令見 [COMMANDS.md](reproduction/COMMANDS.md)。

checkpoint 保存學生、雙 projection heads、memories/Z、optimizer、RNG、loader generator、config、環境、來源 hash 與 epoch history。支援完整 epoch 邊界恢復；半個 epoch 會重算。`student_final.pt` 是推論權重，不能當訓練 resume。

## 做過哪些驗證？

- 20 項不依賴 CIFAR 資料的語意／流程測試通過；部分測試需要已下載的教師權重。
- 官方 CRD 與修補版固定輸入數值、梯度、memory 更新比對；兩種 τ 都有覆蓋。
- exact positive、異類 negatives、全域 index、模型與 projection shapes、teacher BN 不變及 heads 更新。
- KD scaling、LR 邊界、Z 只初始化一次、完整 RNG/optimizer checkpoint round-trip。
- 正式 B64/K16384 GPU steps、真實 CIFAR smoke、教師全 test 評估與真實 epoch resume。
- 最終 30 runs、7200 筆 epoch 紀錄、六組五 seeds、教師 checksum 與凍結 Python hash 查核。

訓練曾遭遇兩次非預期關機與一次主動暫停重開，均從最後完整 checkpoint 接續。Windows 事件不足以證明根因；不將關機歸咎於訓練或硬體。恢復事件保存在各 run 的 events.jsonl。

## 資源與範圍

CRD 每 epoch 約 94–98 秒，一次 240 epochs 約 6.3 小時；框架記錄的 peak allocated memory 約 1.35 GiB，並非整張 GPU 的使用量。時間排除停機、初始化與 checkpoint I/O；不同硬體不能直接套用。

未完成範圍：其他模型配對、ImageNet、linear transfer、跨模態、ensemble、Deep Mutual Learning 與完整消融。原 repository 也未提供所有論文實驗的完整入口。後續需另立設定與來源核對，不能用這 30 runs 宣稱整篇論文已完成。

## 來源與授權

- [論文：Contrastive Representation Distillation](https://arxiv.org/abs/1910.10699)，ICLR 2020；本次核對 v3。
- [官方固定 commit](https://github.com/HobbitLong/RepDistiller/tree/b84f547c5db6a35318d4671d7d5c4de74c822403)。
- 官方程式 BSD-2-Clause 授權與版權保留於 [LICENSE](LICENSE)；本專案包含其衍生修改，非作者官方 repository。
- 論文 PDF、資料集與教師權重各依原來源授權，不因本 repository 程式授權而改變。


