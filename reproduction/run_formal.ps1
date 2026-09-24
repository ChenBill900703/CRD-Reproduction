# 重跑入口（既有正式實驗已完成）：240 epochs，先 seed 0 四方法，再 seeds 1..4，再 paper_tau。
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot/..
$py = './.venv/Scripts/python.exe'
foreach ($seed in 0..4) {
    foreach ($method in @('vanilla','kd','crd','crd_kd')) {
        $entry = if ($method -eq 'vanilla') {'train_teacher.py'} else {'train_student.py'}
        & $py "reproduction/$entry" --config "configs/author_code_$method.json" --seed $seed --trial 1
        if ($LASTEXITCODE -ne 0) { throw "Run failed: $method seed $seed" }
    }
}
foreach ($seed in 0..4) {
    foreach ($method in @('crd','crd_kd')) {
        & $py reproduction/train_student.py --config "configs/paper_tau_$method.json" --seed $seed --trial 1
        if ($LASTEXITCODE -ne 0) { throw "Run failed: paper_tau $method seed $seed" }
    }
}
& $py reproduction/summarize.py

