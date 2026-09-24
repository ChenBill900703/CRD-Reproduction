# 正式訓練已啟動

2026-09-17：使用者授權開始正式訓練及定時監控。

- 執行器：artifacts/formal/run_queue.py（背景程序；不修改已凍結的訓練實作）。
- 順序：author_code seeds0–4各vanilla→KD→CRD→CRD+KD，再paper_tau seeds0–4各CRD→CRD+KD，共30個240-epoch runs。
- 目前狀態：queue.json；每個run console log位於本目錄；訓練metrics/checkpoint在runs內。
- 任一子程序非正常結束或未完成epoch240，佇列即停止並保存error，不跳過run。
- 定時監控：Codex heartbeat，ID crd，每15分鐘；正常無重要變化不通知，run完成、失敗、停滯或需處理時通知。
- 首次啟動確認：author_code_vanilla_seed0_trial1已完成epoch1（50000 train／10000 test、782 steps），test top1=14.68%，仍在訓練；不是最終比較結果。
- 保持電腦開機與Codex應用程式運行；電腦休眠會中斷計算或排程。未更動系統電源設定。
- 不要同時手動執行run_formal.ps1或再次啟動run_queue.py，以免重複工作。
