# Heart Disease Classifier — End-to-End MLOps Assignment

A reproducible educational MLOps project for AIMLCZG523 Assignment 01. It includes UCI data acquisition, EDA, preprocessing, two models, cross-validation, MLflow tracking, FastAPI serving, tests, GitHub Actions, Docker, Kubernetes manifests, and Prometheus metrics.

**Responsible use:** this is a coursework demonstration, not a medical device. Do not use predictions for diagnosis, treatment, or clinical decisions.

## 1. Prerequisites

- Python 3.11
- Git
- Docker Desktop (Docker Engine + Compose)
- For local Kubernetes: Docker Desktop Kubernetes or Minikube with `kubectl`
- A GitHub account for CI/CD

Check versions:
```bash
python --version
git --version
docker --version
kubectl version --client   # optional, for Kubernetes
```

## 2. Create a virtual environment

Windows PowerShell:
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
$env:PYTHONPATH = "src"
```

macOS/Linux:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
export PYTHONPATH=src
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, or run `.venv\Scripts\python.exe` explicitly.

## 3. Download the data

```bash
python -m heart_ml.data_cli
```
The script fetches UCI Machine Learning Repository dataset ID 45 via `ucimlrepo`, converts the original severity target (0–4) into binary `target` (0 = no disease, 1 = disease present), and saves `data/raw/heart_disease.csv`. Internet access is required for the first download.

Inspect the schema and data quality:
```bash
python -c "import pandas as pd; d=pd.read_csv('data/raw/heart_disease.csv'); print(d.shape); print(d.isna().sum()); print(d.target.value_counts(dropna=False))"
```
The original UCI Cleveland data can contain missing values. They are imputed inside the model pipeline rather than filling the full dataset before splitting.

## 4. EDA

```bash
python -m heart_ml.eda
```
Plots and summaries are saved to `reports/figures/`: feature histograms, correlation heatmap, class balance, descriptive statistics, and missing values. Inspect them and use your own observations in the report.

## 5. Train and compare models

```bash
python -m heart_ml.train
```
This trains Logistic Regression and Random Forest, uses stratified five-fold CV on the training partition, evaluates on a held-out test partition, and logs each run to local MLflow. The selected model is the one with the best mean cross-validation ROC-AUC. The complete preprocessing/model pipeline is serialized to `artifacts/model.joblib`.

Outputs:
- `artifacts/model.joblib`
- `artifacts/metadata.json`
- `artifacts/metrics.json`
- model classification reports and plots in `artifacts/`
- MLflow runs under `mlruns/`

View experiment tracking in a second terminal:
```bash
mlflow ui --backend-store-uri ./mlruns --host 127.0.0.1 --port 5000
```
Open `http://127.0.0.1:5000`. Start MLflow UI after training if you want to inspect runs.

## 6. Run automated checks

```bash
ruff check src tests scripts
pytest -q
```
Tests are designed not to need the live UCI download. Training and the full end-to-end workflow do need the dataset.

## 7. Run API locally

```bash
uvicorn heart_ml.api:app --reload --host 127.0.0.1 --port 8000
```
Requires `PYTHONPATH=src` (or use the appropriate environment setup above). Open:
- API docs: http://127.0.0.1:8000/docs
- Health: http://127.0.0.1:8000/health
- Metrics: http://127.0.0.1:8000/metrics

Example request:
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  --data @sample_request.json
```
PowerShell alternative:
```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict -ContentType "application/json" -InFile sample_request.json
```
The response includes a binary prediction, class label, model confidence, class probabilities, and a disclaimer. The values are model outputs, not medical risk estimates validated for clinical use.

## 8. Docker: build and test the isolated API

**Run training first**, because the Docker image includes `artifacts/model.joblib`.
```bash
docker build -t heart-mlops-api:local .
docker run --rm -p 8000:8000 heart-mlops-api:local
```
In another terminal, test health, metrics, and prediction using the URLs and sample request above. Stop the container with Ctrl+C.

Compose alternative:
```bash
docker compose up --build
docker compose logs -f
docker compose down
```
Capture real screenshots of the successful build, running container, Swagger page, and prediction response.

## 9. GitHub Actions CI/CD

1. Create an empty GitHub repository.
2. Add this project and push it:
   ```bash
   git init
   git add .
   git commit -m "Implement end-to-end MLOps heart disease assignment"
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO.git
   git push -u origin main
   ```
3. In GitHub, open **Actions** and inspect `MLOps CI`.
4. The workflow installs dependencies, runs Ruff, runs Pytest, downloads UCI data, trains models, and uploads training artifacts.
5. If the workflow fails, open the failed step logs, fix the issue locally, run the same command, commit, and push again.
6. Save a screenshot of a green successful run. GitHub Actions uses hosted runner minutes and requires network access to fetch the dataset.

This CI workflow validates code and training; it does not automatically publish a cloud deployment. Add deployment automation only after choosing a cloud, setting up a registry and credentials securely, and testing the deployment manually.

## 10. Local Kubernetes deployment (no cloud bill)

### Option A — Minikube

1. Install Docker and Minikube, then start the cluster:
   ```bash
   minikube start --driver=docker
   ```
2. Build the image inside Minikube's Docker environment:
   ```bash
   minikube image build -t heart-mlops-api:local .
   ```
   If your Minikube version does not support `image build`, use `eval $(minikube docker-env)` in a Bash shell and run `docker build -t heart-mlops-api:local .`.
3. Apply the manifest:
   ```bash
   kubectl apply -f deployment/kubernetes/deployment.yaml
   kubectl rollout status deployment/heart-api
   kubectl get pods
   kubectl get services
   ```
4. Access the service:
   ```bash
   kubectl port-forward service/heart-api 8080:80
   ```
   Open http://127.0.0.1:8080/docs. Post `sample_request.json` to `/predict`.
5. For a Minikube LoadBalancer URL, in another terminal run:
   ```bash
   minikube tunnel
   ```
   Depending on OS/network permissions, tunnel support may vary. Port-forwarding is the reliable local test path.
6. Capture screenshots of ready pods, Service, API response, and `kubectl describe deployment heart-api`.

### Option B — Docker Desktop Kubernetes

1. Enable Kubernetes in Docker Desktop settings and wait until the cluster is running.
2. Ensure the local Docker image `heart-mlops-api:local` exists.
3. Apply the manifest and inspect the rollout:
   ```bash
   kubectl apply -f deployment/kubernetes/deployment.yaml
   kubectl rollout status deployment/heart-api
   kubectl get pods,svc
   ```
4. Access with `kubectl port-forward service/heart-api 8080:80`, then open `http://127.0.0.1:8080/docs`.

For local clusters, a LoadBalancer may not receive a public IP. The port-forward method proves service functionality but is not a publicly reachable endpoint. If your evaluator requires a public URL, deploy to a cloud.

## 11. Optional public cloud deployment (costs may apply)

The assignment allows a public cloud. A cloud provider is not preselected here because pricing, region, permissions, and account setup vary. A typical Kubernetes route is:
1. Build and test the Docker image locally.
2. Create a Kubernetes cluster (e.g., AKS, GKE, or EKS) and a container registry.
3. Push the version-tagged image to the registry.
4. Update `deployment/kubernetes/deployment.yaml` with the registry image name.
5. Authenticate `kubectl` to the cluster and apply the manifest.
6. Wait for rollout, obtain the Service external IP/hostname, and verify `/health`, `/docs`, and `/predict`.
7. Restrict network access where possible; do not expose sensitive data. Use HTTPS via ingress/load balancer configuration for a real production service.
8. Capture deployment and endpoint evidence.
9. Delete the cluster, load balancers, registry if no longer needed, and other billable resources when finished.

**Cloud-specific commands are deliberately not hard-coded** because the correct commands depend on your provider, subscription/project, region, cluster name, and permissions. If you tell me which provider you have (Azure, AWS, GCP) or confirm local Kubernetes, I can tailor the exact end-to-end deployment commands.

## 12. Monitoring and logging

- Every API request is logged as structured JSON to stdout.
- `/metrics` exports Prometheus request counts and latency histograms.
- `monitoring/prometheus/prometheus.yml` is a starter scrape config. To run Prometheus in Docker, configure it on the same Docker network as the API and use `heart-api:8000` as the target. For Kubernetes, use an appropriate service target or ServiceMonitor configuration.
- Grafana dashboard steps and useful PromQL expressions are in `monitoring/grafana/README.md`.
- Generate a few requests, inspect container logs, and capture actual metrics/dashboard screenshots.

## 13. Final submission checklist

- [ ] GitHub repository URL
- [ ] Clean setup from `requirements.txt`
- [ ] Dataset download script and generated cleaned CSV
- [ ] EDA figures and written interpretation
- [ ] At least two model runs and justified model selection
- [ ] MLflow tracking screenshots
- [ ] Saved model and preprocessing pipeline
- [ ] Pytest and Ruff pass
- [ ] Successful GitHub Actions run and artifacts
- [ ] Docker build/run proof and `/predict` response
- [ ] Kubernetes manifest, rollout, and service evidence (or public cloud URL)
- [ ] Monitoring/logging evidence
- [ ] Completed 10-page PDF/DOCX report
- [ ] Short video recording showing the overall pipeline

See `reports/FINAL_REPORT_TEMPLATE.md` for the report structure and `screenshots/README.md` for suggested evidence filenames.
