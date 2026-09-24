# CRD 階段 A：來源查核與預先決策

查核日期：2026-09-16。原件 `1910.10699v3.pdf` 為 19 頁；版本日期採使用者提供的 2022-01-24。checkout HEAD 為 `b84f547c5db6a35318d4671d7d5c4de74c822403`，git status 空白。這是本次公開程式碼基準，不推定為論文數值所用歷史版本。reference 不修改；reproduction 是獨立副本。逐行原件見 artifacts/source-lines.txt，原檔 SHA-256 見 reference-manifest.json，差異見 compatibility.patch。

## 方法筆記

論文 pp.3–4 Eq.(4)–(7) 以 C=1 表示 joint p(T,S)，C=0 表示 product p(T)p(S)，正負先驗為 1:(N)。Bayes 得到二元 posterior；Eq.(8)–(9) 將其連到 MI 下界。Eq.(10)–(18) 加入非正的負對 log-likelihood，得到可同時優化 student 與 critic 的較弱 MI lower bound；並非要求每步求出完美 critic。理論所述獨立邊際抽樣，不等同實作的異類限制抽樣。

Eq.(19)，pp.4–5：倒數第二層 representation 經各自線性映射與 L2 normalization，內積除以 τ 後 exp，形成二元 critic。公開實作是兩方向 NCE binary loss **相加**，使用跨 batch memory；不是 batch softmax、InfoNCE、SimCLR 或 CLIP。程式額外使用兩個固定 Z，第一次 forward 計算 mean(exp_scores)*M，後續保留；Eq.(19) 沒有展示這項 Z。noise probability 仍為 1/50000，不能因異類池大小改為 1/49500。

## 來源對照（行號一律指未修改的固定 commit）

|事項|論文|程式碼與行號|判定|
|---|---|---|---|
|歷史環境、命令|—|README.md:32；scripts/run_cifar_distill.sh:5,29,32；scripts/run_cifar_vanilla.sh|Ubuntu16/Python3.5/Torch0.4 為歷史背景|
|teacher URL|—|scripts/fetch_pretrained_teachers.sh:19–21|resnet32x4_vanilla/ckpt_epoch_240.pth|
|240 epochs、SGD|p.14 §6.4|train_student.py:44–54,273–276；train_teacher.py:30–40,101–104|LR .05、momentum .9、WD .0005|
|LR 邊界|p.14 §6.4|helper/util.py:17–23|epoch > milestone；151/181/211 才衰減|
|augmentation/normalization|p.14 設定細節未逐項列出|dataset/cifar100.py:53–61,179–187|以作者程式碼為準|
|global index、negative|p.10 Table6|dataset/cifar100.py:107–169|exact positive、異類、K 小於池時不放回|
|test split/loader|p.6 Table1|dataset/cifar100.py:75–88,199–212|test 並非額外 validation|
|特徵架構|p.14 §6.3|models/resnet.py:127–128,191–200,233–238；models/__init__.py:1,17–18|256 pooled、100 logits|
|teacher 凍結、特徵位置|pp.3–5|helper/loops.py:74,115–117,130–134|同一份 augmented input，feat[-1]|
|雙投影頭 optimizer|p.5 Eq19 說明|train_student.py:190–198,273–279|教師投影仍可訓練|
|loss 組合|p.5 Eq20、p.15 §6.4|helper/loops.py:184；scripts/run_cifar_distill.sh:5,29,32|CE/KD/CRD 權重分開|
|KD|p.5 Eq20、p.15|distiller_zoo/KD.py:13–17|KL(teacher||student)*T²，sum/B|
|CRD loss|pp.3–5 Eq4–19|crd/criterion.py:42–48,59–76|Pn=1/M、eps=1e-7、每方向除 B|
|projection|pp.4–5 Eq19|crd/criterion.py:81–101|Linear 含 bias、128D、L2|
|memory/Z/update|p.5 Implementation|crd/memory.py:18–21,39–77|先 score，再 detached momentum 更新；未先 normalize 初始化|
|temperature 衝突|p.10 §4.5|train_student.py:84，範例不 override|author_code=.07；paper_tau=.1，未解決|
|final 非 best|p.6 Tables1/2；p.18/19 Tables10/11|train_student.py:333–335；train_teacher.py:161–163|正式比較 epoch240|
|trial 非 seed|作者 seed 未確認|train_student.py:71|另加 seed 管理|

## 固定設定與來源分類

- **論文**：第一組 resnet32x4→resnet8x4；Table1 5 runs、Table2 3 runs；Table10 KD 73.33±.25、CRD 75.51±.18、CRD+KD 75.46±.25；teacher79.42、vanilla72.50。std 分母和作者 seeds 未知。
- **作者程式碼**：CIFAR100 train50000/test10000；train B64/test B32、shuffle train、drop_last=False；crop32 padding4→flip→tensor→normalize mean(.5071,.4867,.4408),std(.2675,.2565,.2761)。單 GPU FP32 SGD .05/.9/.0005、240 epochs、150/180/210 字面 milestones、倍率 .1，不用 Nesterov/warmup。loss 權重 vanilla(1,0,0)、KD(.1,.9,0)、CRD(1,0,.8)、CRD+KD(1,1,.8)。KD T4；CRD dim128/K16384/memory momentum .5/exact/percent1、兩個 50000×128 banks，初始化 uniform ±1/sqrt(128/3)。
- **相容性修補**：data/targets 取代 train_data 等舊 API；AliasMethod 僅 fallback 抽樣時移到 feature device（不改 RNG 抽樣演算法）；KD sum reduction；教師明確指定架構，不從 Windows path 解析；run 名稱不含冒號；JSONL 取代舊 tensorboard_logger。--config 路徑不匯入舊 logger；未提供 --config 的 legacy 入口未納入驗證。
- **本次新增規約**：seeds[0,1,2,3,4]，trial 只是命名；std 使用 sample ddof=1。先 seed0 vanilla→KD→CRD→CRD+KD，再 seeds1–4 同順序，再 paper_tau CRD/CRD+KD seeds0–4。結果分組不挑優者。不依 test 調參或排除 runs。
- **本次新增規約**：workers=0（train/test），避免 Windows spawn/prefetch 狀態影響可測的 epoch 邊界 resume；因此吞吐不同於作者8/4。Python/NumPy/Torch CPU/CUDA 及 loader generator 都保存；worker_init_fn 可支援顯式 worker seed，但本輪 workers>0 未驗證。cuDNN benchmark=False、deterministic=True，TF32 全關，與原碼 benchmark=True 分開記錄；不保證跨硬體 bitwise 一致。
- **本次新增規約**：resume 保存完整 student/雙 heads/CRD buffers/optimizer/RNG/config/epoch/global_step/teacher checksum/來源 commit/所有實作檔 hash/environment；嚴格拒絕 config、code hash、teacher checksum 不同。只支援完成 epoch 的邊界；不支持任意 batch 精確接續。smoke 每個 epoch 僅3 batches，是截短的測試邊界，非完整50000筆訓練 epoch。

## 範圍與後續

本輪只 A/B，不啟動240 epochs。C 為第一組完整20個 baseline runs與10個 tau runs；D 才能進行正式差距分析。E 擴充 Table1/2/10/11、Table6/Fig5，再 ImageNet、遷移、跨模態、ensemble。公開入口僅 CIFAR100；不可聲稱其他流程完整公開。ImageNet Table3 是 error，CRD28.83/9.87、CRD+KD28.62/9.51。linear transfer 需凍結 backbone；跨模態與 segmentation 另建查核；Table9 DML 不混入固定教師。FT 論文 beta500 vs script200，AB/FSP 的初始化階段須獨立解釋，暫不實作。

## 2026-09-17 額外工程查核（本次新增規約）

完整 checkpoint schema2 保存 epoch history；寫入使用同目錄暫存檔、flush/fsync 與 replace。以最後完整 checkpoint 重建 metrics/status/student_final，可恢復發布中斷。run 使用 OS 檔案鎖避免同時寫入；resume 先檢查 config、code hash、commit、teacher checksum、environment，再修改結果。正式結果另外檢查240 epochs、187680 steps、完整split大小、teacher來源、code/config/environment一致性。未知 config 與未預定 seeds 不接受。

訓練入口現統一接受 --config；舊 tensorboard_logger 分支不再作為執行路徑。可用 --help 查看完整 CLI。加入 loss、gradient、更新後參數及評估輸出的 NaN/Inf 檢查，只偵測異常，不改有效數值或引入 gradient clipping。CRD 的 loss、Z 與 memory 公式維持原作者數值流程。
