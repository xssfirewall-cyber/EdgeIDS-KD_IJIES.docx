# Restarts run_sweep.py after a CUDA error (the GPU is shared with other jobs).
# run_sweep.py skips finished runs, so each restart resumes where it stopped.
Set-Location (Split-Path $PSScriptRoot -Parent)
for ($i = 1; $i -le 30; $i++) {
    "=== attempt $i  $(Get-Date -Format s)" | Out-File -Append -Encoding utf8 results\sweep_log3.txt
    python -W ignore -u scripts/run_sweep.py 2>&1 | Out-File -Append -Encoding utf8 results\sweep_log3.txt
    if (Select-String -Path results\sweep_log3.txt -Pattern "SWEEP DONE" -Quiet) { break }
    Start-Sleep -Seconds 60
}
