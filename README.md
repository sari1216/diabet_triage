# Diabetes Triage – Progression Risk Service

A small ML service that predicts short-term diabetes progression as a continuous
risk score, so triage nurses can prioritise follow-up calls. Trained on the
scikit-learn Diabetes dataset as a stand-in for de-identified EHR data.

- `GET /health` – service status and model version
- `POST /predict` – progression score (higher = higher risk)

## Run the released image

```bash
docker pull ghcr.io/sari1216/diabet_triage:v0.2
docker run -it -p 80:80 ghcr.io/sari1216/diabet_triage:v0.2
```

Replace `v0.2` with `v0.1` to run the baseline. Stop one version before starting
the other, since both use port 80.

## API

### Health check

```bash
curl http://localhost:80/health
```

Response:

```json
{"status": "ok", "model_version": "v0.2"}
```

### Prediction – sample payload

```bash
curl -X POST http://localhost:80/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03, "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02, "s5": 0.02, "s6": -0.001}'
```

Response:

```json
{"prediction": 198.75}
```

All ten features (`age, sex, bmi, bp, s1, s2, s3, s4, s5, s6`) are required and
must be numbers.

### Error response structure

Invalid input returns **HTTP 422** with a JSON body. Example when `s6` is missing:

```json
{
  "detail": [
    {
      "type": "missing",
      "loc": ["body", "s6"],
      "msg": "Field required",
      "input": {"age": 0.02, "sex": -0.044, "...": "..."}
    }
  ]
}
```

- `type` – kind of error, e.g. `missing` (field not sent), `float_parsing`
  (not a number) or `json_invalid` (malformed JSON)
- `loc` – where the error is; the last element is the field name
- `msg` – human-readable description
- `input` – the input that was received

## Reproduce locally

Requires Python 3.10.

```bash
pip install -r requirements.txt -r requirements-dev.txt
python train.py           # trains the model, writes pipeline.joblib and metrics.json
ruff check .              # lint
pytest -v                 # unit tests
uvicorn app:app --reload  # API at http://localhost:8000 (docs at /docs)
```

Training is reproducible: the train/test split and the model use
`random_state=42`, and all package versions are pinned in the requirements files.

To compare the candidate models for v0.2:

```bash
python compare_models.py
```

### Build and run the Docker image locally

```bash
docker build -t diabetes-triage .
docker run -p 80:80 diabetes-triage
```

The model is trained during the build, so the image is self-contained.

## CI/CD

- **CI** (`.github/workflows/ci.yml`) runs on every push and pull request:
  lint (ruff), training, unit tests (pytest), and uploads the model and
  metrics as an artifact.
- **Release** (`.github/workflows/release.yml`) runs on tags `v*`: lint and
  tests, builds the Docker image, smoke-tests the running container
  (`/health`, `/predict` and a 422 check), pushes the image to GHCR and
  publishes a GitHub Release with the metrics and changelog.

To release a new version:

```bash
git tag v0.3
git push origin v0.3
```

## Model versions

| Version | Model            | Test RMSE |
|---------|------------------|-----------|
| v0.1    | LinearRegression | 53.85     |
| v0.2    | LassoCV          | 52.92     |

See [CHANGELOG.md](CHANGELOG.md) for details.

## Project structure

```
train.py              training and metrics
app.py                FastAPI service
compare_models.py     model comparison for v0.2
test/test_api.py      unit tests
Dockerfile            container image
.github/workflows/    CI and release pipelines
CHANGELOG.md          changes and metrics per version
```