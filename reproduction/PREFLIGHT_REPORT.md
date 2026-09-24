# 訓練前程式查核報告

日期：2026-09-17。結論：本次範圍內的20項不依賴CIFAR-100資料測試全部通過，0 failures、0 errors、0 skipped。未啟動CIFAR-100教師評估、真實資料smoke或240-epoch訓練。通過有限測試不代表長時間訓練或硬體故障不可能發生。

## 範圍與實測證據

- 測試命令：`.venv/Scripts/python.exe reproduction/test_no_data.py`，本次42.590秒。
- 入口：vanilla、KD、CRD、CRD+KD；實際作者模型、已核對教師checkpoint、optimizer、loss及memory，使用合成影像。沒有讀取已下載CIFAR檔案。測試產生的暫存run在artifacts/preflight內，結束即清理，不混入正式runs。
- `artifacts/preflight/tests.log`：完整20項結果；`test-result.json`：機器可讀摘要。
- `code-manifest.json`：本次實作檔SHA-256；`environment.json`：本次runtime與精度設定。
- 固定reference checkout仍無修改。更新後相容性差異位於`artifacts/compatibility.patch`。
- `compileall`通過；`pip check`無損壞依賴；PowerShell準備/正式訓練腳本僅做語法解析，未執行。

## 已修正的問題

1. **checkpoint與結果檔不同步**：原本metrics先寫、checkpoint後寫，中斷可能留下半筆JSON或缺漏final狀態。改為schema2 checkpoint包含完整epoch history，以原子replace提交；metrics/status/student_final由它重建。測試損壞三種結果檔及最後epoch提交後中斷，均能恢復且不多訓練一步。
2. **resume過早改寫既有檔案**：現先核對config、code hash、來源commit、teacher checksum、environment，失配就拒絕。加入OS排他檔案鎖避免兩個程序同時寫run。
3. **設定記錄與實際執行不一致**：驗證augmentation、workers、完整正式參數及所有未知欄位；正式seed限預定0–4；拒絕非法trial檔名、錯誤profile、負步數及無效batch。正式GPU配置不會靜默降成CPU。
4. **CLI路徑不穩定**：兩個入口統一使用--config，--help及--config=...可正常解析；方法放錯入口會明確拒絕，不再落入未安裝的legacy tensorboard_logger。
5. **非有限數值未及時發現**：新增loss、gradient、更新後參數、evaluation logits/loss的NaN/Inf檢查，空loader明確報錯。不改公式、不裁剪梯度；失敗時應回到上一完整epoch checkpoint。
6. **smoke多讀一個batch**：改用islice，限制步數後不額外取樣，避免額外消耗RNG。
7. **正式結果混用風險**：彙整要求epoch240、187680 optimizer steps、train50000/test10000、已知teacher checksum與完整teacher test評估；不同code/config/environment不混算正式mean；缺漏、損壞、failed、smoke均留紀錄但不算正式結果。
8. **曲線與空metrics處理**：繪圖可獨立測試，空檔略過，僅截短smoke顯示截短epoch標籤；以明確標記NOT_RESEARCH_RESULTS的合成fixture驗證輸出並視覺檢查。

## 研究語意檢查

- 兩種tau（.07/.1）完整CRDLoss對照固定commit原碼：loss、輸入gradient、雙投影頭gradient、兩memory與Z完全一致（rtol=0、atol=0）。原碼CPU對照僅中和建構時AliasMethod.cuda，不改計算。
- 官方抽樣器使用合成的50000筆/100類資料：B64×16385、positive=global index、異類negative、不放回、global index49999、49500負池；使用真正作者抽樣程式與augmentation。
- 256維pooled、128維projection、100類logits；teacher/student接收同一tensor；教師參數及BN running stats不變，學生與兩投影頭更新。
- memory uniform初始化範圍、未預先L2；Z只初始化一次；只更新指定列；NCE Pn/eps/reduction；CPU Alias fallback。
- KD方向、T²與sum/B；LR150/151、180/181、210/211。
- 四方法完整合成執行與CPU step；尾batch保留。連續與epoch邊界resume比較student、CRD、optimizer、Python/NumPy/CPU/CUDA/loader RNG，完全一致。
- final而非best、trial與seed分離、ddof=1、重複seed/混合版本拒絕作正式5-seed結果。

## GPU多步壓力測試

硬體RTX3070Ti 8GB，PyTorch2.6.0+cu124、torchvision0.21.0+cu124、FP32、TF32關閉。每種tau執行24個CRD+KD步，共48步，B64/K16384/兩個50000×128 memory banks。

|tau|步數|peak allocated|第3–24步已配置記憶體變化|非有限數值|
|---|---:|---:|---:|---|
|0.07|24|1,477,264,384 bytes（約1.38GiB）|0 bytes|未發現|
|0.1|24|1,477,264,384 bytes（約1.38GiB）|0 bytes|未發現|

數據見`artifacts/preflight/gpu-stress.json`。這是PyTorch allocated記憶體，不是整張顯示卡總占用；合成資料步速不能用來承諾真實CIFAR訓練工時或精度。

首次完整測試曾有1項memory穩定性斷言失敗（變動62,102,016 bytes）。單獨執行兩個profile均穩定；在各profile開始前回收前序測試不可達物件後，完整測試亦穩定。這是測試隔離問題的證據；未放寬1MiB門檻、未在每個training step強制GC、未改CRD公式。初次失敗log保留在`tests-before-memory-isolation.log`，單獨診斷在`memory-diagnostic.log`。

## 尚待下一步授權

真實CIFAR-100的完整教師accuracy、augmentation/data I/O端到端smoke與長時間訓練尚未驗證；其他模型/蒸餾方法、workers>0、AMP、多GPU、任意batch中斷不在本次已驗證範圍。當前設定只允許workers=0。程式停在訓練前，等待使用者指示「開始下一步」。
