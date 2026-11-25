# Evidence for the submission: one end-to-end query with per-query latency, chunk count and token usage.
# Run from problem1-rag/:   .\scripts\demo_query.ps1
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8  # read the CLI's UTF-8 output correctly
$q = "How long does the KX-200 fly with its maximum payload?"

Write-Host "`n=== rag query `"$q`" -k 5 ===" -ForegroundColor Cyan
.\.venv\Scripts\rag.exe query $q -k 5 2>&1 | ForEach-Object { "$_" } |
    Where-Object { $_ -match "INFO rag query|tokens\(|ANSWER|^\s+\[\d+\]|usage:" }

Write-Host "`n=== Same query as JSON log record (logs/queries.jsonl, last line) ===" -ForegroundColor Cyan
Get-Content logs\queries.jsonl -Tail 1
