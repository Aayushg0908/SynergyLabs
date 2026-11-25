# Evidence for the submission: re-ingesting the same corpus does not change the vector count.
# Run from problem1-rag/:   .\scripts\demo_reingest.ps1
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$keep = '"(files|new|updated|unchanged|chunks_written|vectors)"|INFO rag ingest'

Write-Host "`n=== 1) Vector count BEFORE re-ingest ===" -ForegroundColor Cyan
.\.venv\Scripts\rag.exe stats 2>$null | Select-String '"(documents|vectors)"'

Write-Host "`n=== 2) Re-ingest the same corpus (data/corpus) ===" -ForegroundColor Cyan
.\.venv\Scripts\rag.exe ingest 2>&1 | ForEach-Object { "$_" } | Select-String $keep

Write-Host "`n=== 3) Vector count AFTER re-ingest ===" -ForegroundColor Cyan
.\.venv\Scripts\rag.exe stats 2>$null | Select-String '"(documents|vectors)"'
