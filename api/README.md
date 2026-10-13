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
