$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot/..
New-Item -ItemType Directory -Force assets/resnet32x4_vanilla | Out-Null
$teacherPath = 'assets/resnet32x4_vanilla/ckpt_epoch_240.pth'
if (-not (Test-Path -LiteralPath $teacherPath)) {
    curl.exe -L --fail 'http://shape2prog.csail.mit.edu/repo/resnet32x4_vanilla/ckpt_epoch_240.pth' -o $teacherPath
    if ($LASTEXITCODE -ne 0) { throw 'Teacher download failed' }
}
if ((Get-FileHash -LiteralPath $teacherPath -Algorithm SHA256).Hash -ne '22AA12E2B632B19DDDE155CCF2388671842DDD304824B3095E0A7FFF364BEB4F') { throw 'Teacher checksum differs from audited artifact' }
& ./.venv/Scripts/python.exe -c "from torchvision.datasets import CIFAR100; CIFAR100('data',train=True,download=True); CIFAR100('data',train=False,download=True)"
if ($LASTEXITCODE -ne 0) { throw 'CIFAR-100 preparation failed' }
