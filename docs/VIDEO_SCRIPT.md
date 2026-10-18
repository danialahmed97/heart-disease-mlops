# Demo video script (6-8 minutes)

Have ready: terminal in the repo with venv active, VS Code, browser tabs for GitHub, MLflow UI
(:5000), Grafana (:3000), Prometheus (:9090). Run `make monitoring-up` and `make k8s-deploy` beforehand.

**0:00-0:45 - Intro and architecture.** "This is a heart-disease classifier run as a monitored API."
Show the Mermaid diagram in the README. Mention the data (UCI Cleveland, 303 rows), the pipeline
(download -> clean -> train -> MLflow -> package -> CI -> Docker -> Kubernetes -> monitoring).

**0:45-1:45 - Data and EDA.** Open `notebooks/01_eda.ipynb`. Point at the class balance (164/139),
the missing values (only `ca` and `thal`), the correlation heatmap (thal, ca, exang, oldpeak, cp,
thalach strongest) and one insight cell. Show `src/data_processing.py` briefly.

**1:45-3:00 - Features and models.** Open `src/features.py`: scaler + one-hot + passthrough, and the
custom `ClinicalFeatureEngineer` (explain `heart_rate_reserve`). Open `docs/model_comparison.md`:
three models, 5-fold CV, why LogisticRegression won (best CV AUC, highest recall 0.93, selection on
CV not test).

**3:00-4:00 - MLflow.** Run `make train` (or show a finished run). In the MLflow UI show the three
runs, metrics, plots, signature/input example, the `best_model` tag and the registered model.

**4:00-4:45 - Tests and CI.** Run `make test` (36 passed, ~97%). Show the green GitHub Actions run:
lint -> test -> train -> docker, and the uploaded artifacts.

**4:45-5:45 - API and Docker.** `make docker-run`; curl `/health`, `/predict`, a 422 case, and
`/predict/batch`. Show `docker exec heart-api id` is non-root, and the JSON log line.

**5:45-6:45 - Kubernetes.** `kubectl get pods,svc,hpa -n heart-disease`; show probes, resource limits
and the HPA in `k8s/`. Show the port-forward `/predict` call from `make k8s-deploy`; mention the Helm chart.

**6:45-7:45 - Monitoring.** Run `make load-test`. Show Prometheus target UP, then the Grafana dashboard:
request rate, p95 latency, error rate (the deliberate 422s), predictions by class, confidence buckets.

**7:45-8:00 - Wrap-up.** Recap originality: custom transformer, recall-aware selection, batch endpoint,
HPA, custom metrics. Mention limitations (small dataset, one hold-out of 61 rows) and next steps.
