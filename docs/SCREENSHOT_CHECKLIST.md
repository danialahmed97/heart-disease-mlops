# Screenshot checklist

Save every image in `screenshots/` with the exact file name. Run commands from the repo root with
the venv active (`source .venv/bin/activate`).

| # | File | Run first | What must be visible |
|---|---|---|---|
| 1 | `eda_notebook.png` | `jupyter notebook notebooks/01_eda.ipynb` (or open in VS Code) | Correlation heatmap + insight cell |
| 2 | `mlflow_runs.png` | `python src/train.py && mlflow ui --backend-store-uri ./mlruns` -> http://localhost:5000 | Experiment `heart-disease-classification` with 3 runs and metric columns |
| 3 | `mlflow_best_run.png` | Click the LogisticRegression run | Params, metrics, artifacts (plots/, model/), `best_model` tag |
| 4 | `mlflow_model_registry.png` | MLflow UI -> Models | `heart-disease-classifier` version 1 with alias `champion` |
| 5 | `pytest_coverage.png` | `make test` | `36 passed` and the coverage table (~97%) |
| 6 | `lint_ok.png` | `make lint` | flake8 finishing with no output |
| 7 | `github_actions_green.png` | Open https://github.com/danialahmed97/heart-disease-mlops/actions | Latest run with 4 green jobs |
| 8 | `github_actions_artifacts.png` | Open that run, scroll to Artifacts | `training-outputs`, `coverage-report` |
| 8b | `github_actions_failed.png` | Open https://github.com/danialahmed97/heart-disease-mlops/actions/runs/39131770605 | Failed "Unit tests + coverage" job, `assert 0 == 2`, later jobs skipped (the demo branch is deleted but the run page stays) |
| 9 | `docker_build.png` | `make docker-build` | Successful build, final `naming to ... heart-disease-api:latest` |
| 10 | `docker_curl.png` | `make docker-run` in one terminal; `curl -s localhost:8000/health; curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d @sample_request.json` | Both JSON responses |
| 11 | `api_docs.png` | Open http://localhost:8000/docs | Swagger UI listing the endpoints |
| 12 | `api_422.png` | `curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"age":999}'` | Validation error JSON |
| 13 | `k8s_pods.png` | `make k8s-deploy` then `kubectl get pods,svc,hpa -n heart-disease` | 2 Running pods, LoadBalancer service, HPA |
| 14 | `k8s_predict.png` | Tail of the `make k8s-deploy` output | `/health` and `/predict` through the service |
| 15 | `helm_install.png` | `helm lint helm/heart-disease-api` | `0 chart(s) failed` |
| 16 | `prometheus_targets.png` | `make monitoring-up`, open http://localhost:9090/targets | `heart-disease-api` target UP |
| 17 | `grafana_dashboard.png` | `make load-test`, then http://localhost:3000/d/heart-disease-api (admin/admin) | All 5 panels with data |
| 18 | `api_logs.png` | `tail -n 5 logs/api.log` (local) or `docker logs hd-api \| tail` | JSON lines with latency_ms and prediction |

Tip: run `make load-test` twice a minute apart so rate-based panels have enough points.

After adding screenshots, run `make report` to rebuild `docs/REPORT.docx` and `docs/REPORT.pdf` with the images embedded.
