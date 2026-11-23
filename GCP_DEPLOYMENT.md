# Google Cloud Deployment (Complete Process)

This guide deploys the project on Google Cloud with:
- **GKE (Autopilot)** for orchestration
- **Artifact Registry** for container images
- **Cloud SQL for PostgreSQL** for managed database
- **Ingress** for frontend/api/stream routing

## 1) Prerequisites

- Billing enabled (coupon applied)
- `gcloud`, `kubectl` installed and authenticated
- Docker not required locally if using Cloud Build

## 2) Choose environment values

Example:

- `PROJECT_ID=my-gcp-project`
- `REGION=europe-west1`
- `ZONE=europe-west1-b`
- `CLUSTER_NAME=violence-cluster`
- `REPO_NAME=violence-repo`
- `SQL_INSTANCE=violence-sql`
- `DB_USER=violence_user`
- `DB_PASSWORD=StrongPassword123`
- `DB_NAME=violence_prod`

## 3) Create Cloud SQL (managed PostgreSQL)

```bash
gcloud sql instances create violence-sql \
  --database-version=POSTGRES_16 \
  --cpu=2 --memory=8GB --region=europe-west1

gcloud sql users create violence_user --instance=violence-sql --password=StrongPassword123
gcloud sql databases create violence_prod --instance=violence-sql
```

## 4) Deploy everything (build + push + GKE apply)

From project root in PowerShell:

```powershell
.\scripts\gcp\deploy-gke.ps1 `
  -ProjectId "my-gcp-project" `
  -Region "europe-west1" `
  -Zone "europe-west1-b" `
  -ClusterName "violence-cluster" `
  -RepoName "violence-repo" `
  -SqlInstance "violence-sql" `
  -DbUser "violence_user" `
  -DbPassword "StrongPassword123" `
  -DbName "violence_prod"
```

## 5) Verify

```bash
kubectl -n violence-prod get pods
kubectl -n violence-prod get svc
kubectl -n violence-prod get ingress
```

## 6) DNS

Ingress uses placeholders:
- `app.example.com`
- `api.example.com`
- `stream.example.com`

Update these in:
- `k8s/overlays/gcp-production/ingress.yaml`

Then point DNS records to the ingress external IP.

## Notes

- In local/dev/test, `docker-compose` uses a PostgreSQL container.
- In Google Cloud production, use **Cloud SQL managed PostgreSQL**.
- `k8s/overlays/gcp-production` expects images in Artifact Registry.

