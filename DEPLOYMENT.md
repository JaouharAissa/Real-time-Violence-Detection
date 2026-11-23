# Docker + Kubernetes Deployment

## Architecture

Containers:
- `db` (PostgreSQL)
- `backend` (FastAPI + model inference)
- `frontend` (React/Vite build served by Nginx)
- `stream` (Socket.IO + OpenCV live stream)

## 3 Environments

- Development
- Test
- Production

---

## Docker Compose

### Development

```bash
docker compose --env-file .env.dev.example -f docker-compose.yml -f docker-compose.dev.yml up --build
```

### Test

```bash
docker compose --env-file .env.test.example -f docker-compose.yml -f docker-compose.test.yml up --build
```

### Production

```bash
docker compose --env-file .env.prod.example -f docker-compose.yml -f docker-compose.prod.yml up --build -d
```

---

## Kubernetes (Kustomize)

### Development

```bash
kubectl apply -k k8s/overlays/development
```

### Test

```bash
kubectl apply -k k8s/overlays/test
```

### Production

```bash
kubectl apply -k k8s/overlays/production
```

---

## Database Choice for Google Cloud

Recommended DB: **PostgreSQL**.

Why:
- Native support with **Google Cloud SQL**
- Easy migration from local Docker to managed cloud
- Works directly with `DATABASE_URL`

Local default:
- `sqlite:///./app.db` (no setup)

Cloud production:
- `postgresql://USER:PASSWORD@HOST:5432/DBNAME`

---

## Google Cloud End-to-End

For complete GCP process (Artifact Registry + GKE + Cloud SQL + Ingress), see:

- `GCP_DEPLOYMENT.md`
- `scripts/gcp/deploy-gke.ps1`

