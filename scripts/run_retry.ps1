# Restart-on-CUDA-error wrapper (the GPU is shared with other jobs).
#   powershell -File scripts\run_retry.ps1 -Script scripts/run_quant.py -Log results\quant_log.txt -Done "QUANT DONE"
# Retries only when the script exits with code 3 (CUDA error). Any other
# failure stops the loop, so a real bug is not retried 30 times.
param([string]$Script, [string]$Log, [string]$Done, [int]$Tries = 30)
Set-Location (Split-Path $PSScriptRoot -Parent)
for ($i = 1; $i -le $Tries; $i++) {
    "=== attempt $i  $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 $Log
    python -W ignore -u $Script 2>&1 | Out-File -Append -Encoding utf8 $Log
    $code = $LASTEXITCODE
    if (Select-String -Path $Log -Pattern $Done -Quiet -SimpleMatch) { break }
    if ($code -ne 3) { "=== stopped: exit code $code (not a CUDA error)" | Out-File -Append -Encoding utf8 $Log; exit $code }
    Start-Sleep -Seconds 60
}
