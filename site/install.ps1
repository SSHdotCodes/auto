param(
    [ValidateSet('pi','opencode','hermes','all')][string]$Agent = 'pi',
    [ValidateSet('auto','cpu','cuda','cuda12','cuda-legacy','rocm')][string]$Backend = 'auto'
)
$ErrorActionPreference = 'Stop'
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host 'Installing uv from astral.sh to manage an isolated Python runtime…'
    Invoke-Expression (Invoke-RestMethod https://astral.sh/uv/install.ps1)
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
}
& uv python install 3.13
if ($LASTEXITCODE -ne 0) { throw 'Python installation failed' }
$AutoPython = (& uv python find 3.13).Trim()
$AutoScript = Join-Path ([System.IO.Path]::GetTempPath()) ([System.IO.Path]::GetRandomFileName() + '.py')
try {
    Invoke-WebRequest https://auto.ssh.codes/install.py -OutFile $AutoScript
    & $AutoPython $AutoScript --agent $Agent --backend $Backend
    if ($LASTEXITCODE -ne 0) { throw 'Auto installation failed. See the message above.' }
} finally {
    Remove-Item $AutoScript -ErrorAction SilentlyContinue
}
