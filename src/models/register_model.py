import mlflow
import os
from pathlib import Path

# --- Authenticate with DagsHub ---
os.environ["MLFLOW_TRACKING_USERNAME"] = os.getenv("DAGSHUB_USERNAME", "")
os.environ["MLFLOW_TRACKING_PASSWORD"] = os.getenv("DAGSHUB_TOKEN", "")

# --- Configure MLflow to use DagsHub ---
mlflow.set_tracking_uri("https://dagshub.com/laxmikantbabaleshwar07/zomato-swiggy-delivery-time-prediction.mlflow")

# --- Set the experiment name ---
mlflow.set_experiment("DVC Pipeline")

# --- Paths ---
root_path = Path(__file__).resolve().parents[2]
model_file = root_path / "models" / "delivery_time_pred_model.pkl"

# --- Log artifact only (no model registry) ---
with mlflow.start_run(run_name="register_model"):
    mlflow.log_artifact(model_file, artifact_path="models")
    print(f" Model logged successfully to DagsHub under experiment 'dvc_pipeline': {model_file}")
