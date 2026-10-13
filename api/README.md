# VQC-ECG API

This service connects the static GitHub Pages website to the Python/Qiskit model.

## What it accepts

- **Raw signal CSV/TXT:** one numeric ECG channel in the first column, with the sampling rate supplied by the user. The API uses WFDB XQRS to find candidate beats and computes the same eight summary features used in the training notebook.
- **Feature CSV:** one row per beat with columns `f1` through `f8`. Additional columns such as `label` and `record` are ignored.

## Required model artifact

Place `model_bundle.pkl` in this `api/` folder. Create it from the *same trained notebook run* with keys:
`vqc`, `scaler`, `pca`, and `angle_scaler`.
The file is loaded using cloudpickle. Only load a model bundle you created yourself; serialized Python objects are not safe to load from untrusted sources.

## Run locally

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Check `http://127.0.0.1:8000/health` and visit `http://127.0.0.1:8000/docs`.

## Deploy on Render

Create a **Web Service** from this repository. Set **Root Directory** to `api`, **Build Command** to `pip install -r requirements.txt`, and **Start Command** to `uvicorn main:app --host 0.0.0.0 --port $PORT`. Deploy only after uploading `model_bundle.pkl` into this folder.

This educational prototype is not a medical device and must not be used for diagnosis or treatment.


## Export the trained bundle from Colab

Run the following in the same Colab runtime **after the cell that fits `vqc` and the preprocessing objects**. It saves and round-trip-tests the fitted VQC plus the exact training-fitted transformations.

```python
from pathlib import Path
import cloudpickle
import numpy as np
from google.colab import files

required = ["vqc", "scaler", "pca", "angle_scaler", "X_test_q"]
missing = [name for name in required if name not in globals()]
if missing:
    raise RuntimeError(
        "Run the notebook training/preprocessing cells first. Missing: "
        + ", ".join(missing)
    )

bundle = {
    "vqc": vqc,
    "scaler": scaler,
    "pca": pca,
    "angle_scaler": angle_scaler,
    "feature_names": [f"f{i}" for i in range(1, 9)],
}

artifact_path = Path("/content/model_bundle.pkl")
with artifact_path.open("wb") as f:
    cloudpickle.dump(bundle, f)

# Round-trip test: load the saved file and run inference on prepared examples.
with artifact_path.open("rb") as f:
    check = cloudpickle.load(f)
test_output = np.asarray(check["vqc"].predict(X_test_q[:2])).astype(int).ravel()

print("Artifact saved:", artifact_path, "bytes:", artifact_path.stat().st_size)
print("Test predictions after reload:", test_output.tolist())
files.download(str(artifact_path))
```

If the smoke-test fails because the dummy row is outside the fitted transform's input distribution or the model object cannot be serialized, don't upload the artifact yet. Share the error so the export can be adjusted. The bundle must be made from the same trained model and its fitted scalers/PCA; do not recreate unfitted scalers in the API.
