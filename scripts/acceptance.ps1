param(
    [int]$FrontendPort = 5173,
    [string]$ModelId,
    [string]$PredictionRowsPath
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    docker compose config --quiet
    if ($LASTEXITCODE -ne 0) { throw 'docker compose config failed.' }

    docker compose build
    if ($LASTEXITCODE -ne 0) { throw 'Docker image build failed.' }
    docker compose up -d
    if ($LASTEXITCODE -ne 0) { throw 'Docker Compose startup failed.' }

    $deadline = (Get-Date).AddMinutes(4)
    do {
        $services = docker compose ps --format json | ForEach-Object { $_ | ConvertFrom-Json }
        $backend = $services | Where-Object { $_.Service -eq 'backend' } | Select-Object -First 1
        $frontend = $services | Where-Object { $_.Service -eq 'frontend' } | Select-Object -First 1
        if ($backend.Health -eq 'healthy' -and $frontend.Health -eq 'healthy') { break }
        if ((Get-Date) -ge $deadline) { throw "Timed out waiting for container health. Backend=$($backend.Health), frontend=$($frontend.Health)" }
        Start-Sleep -Seconds 3
    } while ($true)

    $base = "http://localhost:$FrontendPort"
    $live = Invoke-RestMethod "$base/api/health/live"
    if ($live.status -ne 'alive') { throw 'Backend liveness check returned an unexpected response.' }
    $ready = Invoke-RestMethod "$base/api/health/ready"
    if (-not $ready.database_ready) { throw 'Backend database readiness check failed.' }
    try {
        $ml = Invoke-RestMethod "$base/api/health/ml"
    } catch {
        if ($_.Exception.Response.StatusCode.value__ -ne 503) { throw }
        $ml = @{ status = 'not_ready'; model_ready = $false }
    }

    # Prove the production frontend's same-origin /api proxy reaches the backend.
    Invoke-RestMethod -Method Post "$base/api/data/load-demo" | Out-Null
    $models = Invoke-RestMethod "$base/api/models"
    if ($ModelId) {
        if (-not $PredictionRowsPath) { throw 'PredictionRowsPath is required when ModelId is supplied.' }
        $rowsPath = (Resolve-Path -LiteralPath $PredictionRowsPath).Path
        $rows = Get-Content -LiteralPath $rowsPath -Raw | ConvertFrom-Json
        $prediction = Invoke-RestMethod -Method Post -Uri "$base/api/predict" -ContentType 'application/json' -Body (@{ model_id = $ModelId; rows = $rows } | ConvertTo-Json -Depth 30)
        if (-not $prediction.prediction_id -or -not $prediction.model_version) { throw 'Prediction response did not match the expected persisted schema.' }
        $retrieved = Invoke-RestMethod "$base/api/predict/$($prediction.prediction_id)"
        if ($retrieved.prediction_id -ne $prediction.prediction_id) { throw 'Persisted prediction retrieval did not match the submitted result.' }
        Write-Output "Inference and persisted retrieval passed for model $ModelId ($($prediction.model_version))."
    } else {
        Write-Output "Model readiness: $($ml.status). Available registered models: $($models.Count). Supply -ModelId and -PredictionRowsPath to run a real model inference against this deployment."
    }
    Write-Output "Liveness: $($live.status); database: $($ready.database_ready); frontend container: $($frontend.Health)."
    Write-Output 'Persistent named volumes are retained. Stop with: docker compose down'
} finally {
    Pop-Location
}
