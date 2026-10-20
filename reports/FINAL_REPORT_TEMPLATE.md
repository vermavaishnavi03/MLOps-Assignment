# End-to-End MLOps for Heart Disease Classification
**Course:** AIMLCZG523 — Machine Learning Operations  
**Assignment:** 01  
**Student:** [Your name / ID]  
**Repository:** [Your GitHub URL]  
**Deployment URL or local access instructions:** [Fill in]

> This is a report template. Replace every bracketed instruction with your own measured results, screenshots, implementation decisions, and observations. Do not claim experiments or deployment steps you have not performed.

## 1. Abstract
Describe the problem, data, models compared, MLOps workflow, serving API, deployment target, and observed results. Insert measured metrics only after running the pipeline.

## 2. Problem statement and objectives
Build a reproducible binary classifier using the UCI Heart Disease dataset and expose it as an API. Explain that this is an educational experiment, not a medical diagnostic system.

## 3. System architecture
Insert your architecture diagram. Suggested flow:
UCI repository → acquisition/validation → EDA → preprocessing pipeline → cross-validation/model comparison → MLflow → serialized pipeline → FastAPI → Docker → Kubernetes/cloud → logs and Prometheus metrics → CI/CD.

## 4. Dataset acquisition and understanding
Document UCI dataset ID 45, selected Cleveland data, the 13 input features and binary target. Record row/column counts, missing-value counts, target distribution, and any schema decisions from your actual run.

## 5. EDA
Insert histograms, correlation heatmap, and class-balance chart. Explain visible distributions, relationships, potential data quality issues, and class imbalance based on the actual plots.

## 6. Preprocessing and feature engineering
Explain numeric median imputation and scaling; categorical most-frequent imputation and one-hot encoding. The preprocessing is part of the saved sklearn pipeline to prevent train/serve mismatch. Explain the train/test split and stratified five-fold cross-validation.

## 7. Model development and evaluation
Compare Logistic Regression and Random Forest. Insert a table using values from `artifacts/metrics.json`: CV accuracy, precision, recall, ROC-AUC (mean and standard deviation), and held-out test metrics. Explain the selection rule: highest mean cross-validation ROC-AUC. Discuss the trade-off between precision and recall.

## 8. Experiment tracking with MLflow
Include a screenshot of the MLflow experiment and describe logged parameters, cross-validation/test metrics, model artifact, classification report, confusion matrix, and ROC curve. Include how to start the UI: `mlflow ui --backend-store-uri ./mlruns`.

## 9. Packaging, API, and reproducibility
Describe `requirements.txt`, `artifacts/model.joblib`, `metadata.json`, `/health`, `/predict`, `/metrics`, input validation, output confidence/probabilities, and the medical disclaimer. Include a sample request and response captured from your actual run.

## 10. CI/CD, containerization, deployment, monitoring
Describe the GitHub Actions lint/test/train workflow, Docker build/run verification, Kubernetes deployment and Service, readiness/liveness probes, and monitoring/logging. Insert real screenshots and list commands used. For a cloud deployment, record cloud provider, region, resource names, endpoint URL, security controls, and cleanup steps.

## 11. Limitations, risks, and responsible use
Discuss small/older dataset limitations, missingness, population representativeness, potential bias, calibration, false negatives/positives, data drift, privacy, security, and why predictions must not be used for medical decisions.

## 12. Conclusion and future work
Summarize measured findings and suggest external validation, calibration, model monitoring, data/version lineage, model registry, security hardening, and a more robust deployment strategy.

## Appendix A — Reproduction
List Python version, OS, exact setup/train/test/Docker/Kubernetes commands, and dependency versions. Include commit hash and link to CI run.

## Appendix B — Evidence checklist
- [ ] EDA plots
- [ ] MLflow experiment view
- [ ] Passing tests and linting
- [ ] Successful GitHub Actions run
- [ ] Docker build and running container
- [ ] Swagger `/predict` result
- [ ] Kubernetes pods and service
- [ ] Metrics or dashboard
- [ ] Public endpoint or local access instructions
- [ ] Short video walkthrough
