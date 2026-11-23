param(
  [Parameter(Mandatory = $true)] [string] $ProjectId,
  [Parameter(Mandatory = $true)] [string] $Region,
  [Parameter(Mandatory = $true)] [string] $Zone,
  [Parameter(Mandatory = $true)] [string] $ClusterName,
  [Parameter(Mandatory = $true)] [string] $RepoName,
  [Parameter(Mandatory = $true)] [string] $SqlInstance,
  [Parameter(Mandatory = $true)] [string] $DbUser,
  [Parameter(Mandatory = $true)] [string] $DbPassword,
  [Parameter(Mandatory = $true)] [string] $DbName,
  # Vite build-time URLs (must match how users reach API + stream in prod).
  # For the current GCE Ingress path-based setup, stream is typically exposed as http(s)://INGRESS_IP/stream
  [Parameter(Mandatory = $false)] [string] $FrontendApiUrl = "",
  [Parameter(Mandatory = $false)] [string] $FrontendStreamUrl = ""
)

$ErrorActionPreference = "Stop"

function Invoke-GCloud {
  param(
    [Parameter(Mandatory = $true)] [string[]] $Args,
    [switch] $IgnoreStderr
  )

  # gcloud sometimes prints benign informational lines to stderr (PowerShell treats stderr as error records).
  # We still want real failures to stop the script, so we rely on $LASTEXITCODE.
  $prevEap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    if ($IgnoreStderr) {
      # Use for noisy commands where stderr is informational but exit code is authoritative.
      & gcloud @Args 2>$null
    } else {
      & gcloud @Args
    }
    if ($LASTEXITCODE -ne 0) {
      throw "gcloud failed (exit $LASTEXITCODE): gcloud $($Args -join ' ')"
    }
  } finally {
    $ErrorActionPreference = $prevEap
  }
}

function Ensure-GkeAuthPlugin {
  $pluginCmd = Get-Command gke-gcloud-auth-plugin.exe -ErrorAction SilentlyContinue
  if ($null -eq $pluginCmd) {
    Write-Host "==> Install gke-gcloud-auth-plugin (required by kubectl for GKE)"
    $copyPythonOutput = & gcloud components copy-bundled-python 2>$null
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($copyPythonOutput)) {
      $env:CLOUDSDK_PYTHON = ($copyPythonOutput | Select-Object -First 1).Trim()
    }
    Invoke-GCloud @("components", "install", "gke-gcloud-auth-plugin", "--quiet")
    $pluginCmd = Get-Command gke-gcloud-auth-plugin.exe -ErrorAction SilentlyContinue
    if ($null -eq $pluginCmd) {
      throw "gke-gcloud-auth-plugin.exe is still missing after install. Open Google Cloud SDK Shell once, run 'gcloud components install gke-gcloud-auth-plugin', then retry."
    }
  }

  # Force kubectl to use the gcloud auth plugin for GKE credentials.
  $env:USE_GKE_GCLOUD_AUTH_PLUGIN = "True"
}

Write-Host "==> Configure gcloud project"
Invoke-GCloud @("config", "set", "project", $ProjectId)

Write-Host "==> Enable required APIs"
Invoke-GCloud @(
  "services", "enable",
  "artifactregistry.googleapis.com",
  "cloudbuild.googleapis.com",
  "container.googleapis.com",
  "sqladmin.googleapis.com",
  "iam.googleapis.com"
)

Write-Host "==> Ensure Artifact Registry repository"
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  & gcloud artifacts repositories describe $RepoName --location=$Region 2>$null | Out-Null
  $exists = ($LASTEXITCODE -eq 0)
} finally {
  $ErrorActionPreference = $prevEap
}

if ($exists) {
  Write-Host "Artifact Registry repo already exists: $RepoName"
} else {
  $prevEap = $ErrorActionPreference
  $ErrorActionPreference = "Continue"
  try {
    & gcloud artifacts repositories create $RepoName `
      --repository-format=docker `
      --location=$Region `
      --description="Violence detection containers" 2>$null | Out-Null
  } finally {
    $ErrorActionPreference = $prevEap
  }

  if ($LASTEXITCODE -ne 0) {
    Write-Host "Artifact Registry repo creation skipped (likely already exists): $RepoName"
  }
}

Write-Host "==> Build and push images with Cloud Build"
# IMPORTANT (PowerShell): quote substitutions as a single argument, otherwise commas/spaces can break parsing.
$imageTag = ([guid]::NewGuid().ToString("N").Substring(0, 8))
if ([string]::IsNullOrWhiteSpace($FrontendApiUrl) -or [string]::IsNullOrWhiteSpace($FrontendStreamUrl)) {
  Write-Host ""
  Write-Host "NOTE: -FrontendApiUrl / -FrontendStreamUrl not provided."
  Write-Host "Cloud Build will use defaults from cloudbuild.yaml (often wrong for prod)."
  Write-Host "Recommended for current ingress path routing:"
  Write-Host '  -FrontendApiUrl "http://<INGRESS_IP>"'
  Write-Host '  -FrontendStreamUrl "http://<INGRESS_IP>/stream"'
  Write-Host ""
}

$substitutions = "_REGION=$Region,_REPO=$RepoName,_TAG=$imageTag"
if (-not [string]::IsNullOrWhiteSpace($FrontendApiUrl) -and -not [string]::IsNullOrWhiteSpace($FrontendStreamUrl)) {
  $substitutions += ",_FRONTEND_API_URL=$FrontendApiUrl,_FRONTEND_STREAM_URL=$FrontendStreamUrl"
}

Invoke-GCloud @(
  "builds", "submit",
  "--config=cloudbuild.yaml",
  "--substitutions=$substitutions",
  "--verbosity=info"
)

Write-Host "==> Latest Cloud Build (should not be empty if submit succeeded)"
Invoke-GCloud @("builds", "list", "--limit=3", "--format=table(id,status,createTime,logUrl)")

$tag = $imageTag

Write-Host "==> Create / configure GKE cluster (Autopilot)"
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
  & gcloud container clusters create-auto $ClusterName --region $Region 2>$null | Out-Null
} finally {
  $ErrorActionPreference = $prevEap
}
if ($LASTEXITCODE -ne 0) {
  Write-Host "GKE cluster create skipped (likely already exists): $ClusterName"
}
Ensure-GkeAuthPlugin
Invoke-GCloud @("container", "clusters", "get-credentials", $ClusterName, "--region", $Region)

Write-Host "==> Create namespace"
kubectl create namespace violence-prod --dry-run=client -o yaml | kubectl apply -f -

Write-Host "==> Create application secret (DATABASE_URL)"
$dbUrl = "postgresql://${DbUser}:${DbPassword}@127.0.0.1:5432/${DbName}"
kubectl -n violence-prod create secret generic app-secrets `
  --from-literal=DATABASE_URL="$dbUrl" `
  --dry-run=client -o yaml | kubectl apply -f -

Write-Host "==> Patch overlay placeholders and deploy"
$kustom = "k8s/overlays/gcp-production/kustomization.yaml"
$kustomContent = Get-Content $kustom -Raw
$kustomContent = $kustomContent `
  -replace 'value:\s+[^\r\n]*/backend:[^\r\n]+', "value: $Region-docker.pkg.dev/$ProjectId/$RepoName/backend:$tag" `
  -replace 'value:\s+[^\r\n]*/stream:[^\r\n]+', "value: $Region-docker.pkg.dev/$ProjectId/$RepoName/stream:$tag" `
  -replace 'value:\s+[^\r\n]*/frontend:[^\r\n]+', "value: $Region-docker.pkg.dev/$ProjectId/$RepoName/frontend:$tag" `
  -replace '[A-Za-z0-9\-]+:[a-z0-9\-]+:' + [Regex]::Escape($SqlInstance), "${ProjectId}:${Region}:${SqlInstance}"
Set-Content $kustom $kustomContent

$sa = "k8s/overlays/gcp-production/serviceaccount.yaml"
$saContent = Get-Content $sa -Raw
$saContent = $saContent -replace '@[A-Za-z0-9\-]+\.iam\.gserviceaccount\.com', "@$ProjectId.iam.gserviceaccount.com"
Set-Content $sa $saContent

kubectl kustomize k8s/overlays/gcp-production --load-restrictor LoadRestrictionsNone | kubectl apply -f -
if ($LASTEXITCODE -ne 0) {
  throw "kubectl kustomize/apply failed for gcp-production overlay."
}

Write-Host "==> Rollout refreshed images (avoid stale pods referencing old tags)"
$backendImage = "$Region-docker.pkg.dev/$ProjectId/$RepoName/backend:$tag"
$streamImage = "$Region-docker.pkg.dev/$ProjectId/$RepoName/stream:$tag"
$frontendImage = "$Region-docker.pkg.dev/$ProjectId/$RepoName/frontend:$tag"

kubectl -n violence-prod set image deployment/backend backend=$backendImage
kubectl -n violence-prod set image deployment/stream stream=$streamImage
kubectl -n violence-prod set image deployment/frontend frontend=$frontendImage

kubectl -n violence-prod rollout status deployment/backend
kubectl -n violence-prod rollout status deployment/stream
kubectl -n violence-prod rollout status deployment/frontend

Write-Host ""
Write-Host "Deployment submitted."
Write-Host "Check:"
Write-Host "  kubectl -n violence-prod get pods"
Write-Host "  kubectl -n violence-prod get ingress"

