# SafeText AI - Hate Speech Detection

SafeText AI is a small full-stack text classification project. Enter a message in the React interface and the Python API uses a trained Scikit-learn pipeline to classify it as **Hate Speech Detected** or **No Hate Speech Detected**. It does not save submissions and does not use a database or external AI service.

## How the model works

The training script reads the real `labeled_data.csv` in the project root. The existing script verifies the meanings of its `class` values: `0` is Hate Speech, `1` is Offensive Language, and `2` is Neither. For the website, class `0` maps to `hate_speech`; classes `1` and `2` map to `not_hate_speech`.

The current training run loaded 24,783 records and retained 24,783 after removing missing/empty text, unmapped labels, and exact duplicate text/label pairs. The data was split with stratification into 19,826 training records and 4,957 held-out test records. A TF-IDF vectorizer and balanced Logistic Regression classifier are fitted together in one pipeline, so the vectorizer learns from training text only.

The current held-out evaluation produced macro F1 **0.6952** and accuracy **0.92**. Hate-speech precision was **0.35** and recall was **0.57**; the classes are imbalanced, so those class-specific results matter more than accuracy alone. These figures are from the dataset and split used for this model and are not a guarantee of real-world performance. The displayed hate percentage is the model's `predict_proba()` value for the verified `hate_speech` class, selected by its class label. It is an estimate and has not been calibrated to represent a guaranteed real-world probability.

The original [Hate Speech Detection.py](Hate%20Speech%20Detection.py) remains unchanged. Its `tweet`/`class` columns and class mapping informed this implementation. Its existing NLTK cleaning and Decision Tree workflow remain available as the original script; the web model uses TF-IDF and Logistic Regression as requested.

## Project structure

```text
SafeText-AI/
|-- backend/
|   |-- main.py
|   |-- predictor.py
|   |-- train_model.py
|   |-- requirements.txt
|   |-- models/
|   |   `-- hate_speech_pipeline.joblib  (required by Render)
|   `-- tests/
|       `-- test_api.py
|-- frontend/
|   |-- index.html
|   |-- package.json
|   |-- vite.config.js
|   |-- .env.example
|   `-- src/
|       |-- App.jsx
|       |-- api.js
|       |-- index.css
|       |-- main.jsx
|       `-- components/
|           |-- PredictionResult.jsx
|           `-- TextInput.jsx
|-- labeled_data.csv
|-- Hate Speech Detection.py
|-- .gitignore
`-- README.md
```

The trained model artifact is 4.61 MiB and is intended to be committed with the application so Render can load it. The dataset is excluded from Git because redistribution terms have not been verified. Keep the CSV locally for retraining; on a new machine, supply the real dataset before running the training command. The API returns a clear unavailable-model response if the model artifact is absent.

## Dataset and columns

Put the real CSV at the project root as `labeled_data.csv`, or pass its path with `--data`. By default the trainer reads text from `tweet`, labels from `class`, and treats `0` as hate and `1,2` as not hate. Confirm any different dataset's label meanings before supplying custom mappings. For example, if the verified dataset has text in `message`, labels in `category`, and `HATE` means hate while `OFFENSIVE` and `NEITHER` mean not hate:

```powershell
python backend/train_model.py --data .\data\messages.csv --text-column message --target-column category --hate-labels HATE --not-hate-labels OFFENSIVE,NEITHER
```

Missing/blank text, missing or unmapped labels, and exact duplicate text/label pairs are excluded. If the specified file or columns are absent, training stops with an explanatory error instead of creating data.

## Run on Windows PowerShell

Run these commands from the project root.

### 1. Create and activate the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in this terminal, allow it only for the current process and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 2. Install backend dependencies and train

```powershell
python -m pip install -r backend/requirements.txt
python backend/train_model.py --data labeled_data.csv
```

Training prints the cleaned record counts, class distribution, classification report, macro F1, and confusion matrix, then writes `backend/models/hate_speech_pipeline.joblib`. Training is separate from prediction; the API loads and caches the saved pipeline and does not retrain on each request.

### 3. Start the backend

In the project root, with the virtual environment active:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

The health endpoint is [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health). Interactive API documentation is at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Start the frontend

Open a second terminal and run:

```powershell
cd frontend
npm install
npm run dev
```

Open [http://127.0.0.1:5173](http://127.0.0.1:5173). The Vite server uses that fixed local address and the frontend defaults to the API at `http://127.0.0.1:8000`. To use a different backend URL, copy `.env.example` to `.env` in `frontend`, set `VITE_API_BASE_URL`, and restart Vite.

### 5. Use the application

Enter or choose an example message, select **Check text**, and read the binary result and estimated hate probability. The percentage bar uses the API's probability value and is only a model estimate. Select **Clear** to reset the input and result. Offensive language and Neither are grouped together as **No Hate Speech Detected**.

## Hosting configuration

The root-level `render.yaml` deploys the backend from the repository root. It installs `backend/requirements.txt`, starts `backend.main:app` on the hosting-provided `$PORT`, and checks `/health`. The trained model file must be included in the repository for Render; no training dataset or database is needed at runtime.

The root-level `netlify.toml` sets the frontend base directory to `frontend`, builds with `npm run build`, and publishes `dist`. After Render provides the backend URL, set `VITE_API_BASE_URL` in the Netlify site's production environment to that exact origin (for example, `https://service-name.onrender.com`, without `/predict`). Then redeploy the frontend. Production builds intentionally fail closed rather than silently sending requests to localhost if this variable is missing.

After Netlify provides its site origin, set Render's `FRONTEND_ORIGINS` environment variable to a comma-separated allowlist containing that exact origin and the two local development origins, for example `https://your-site.netlify.app,http://localhost:5173,http://127.0.0.1:5173`. Do not leave the example domain in production. Save the environment change and redeploy/restart the Render service. The API always retains the two local origins and adds the explicitly configured origins.

The current local workspace has no Git repository, GitHub/Render/Netlify CLI authentication, or public visibility confirmation for the requested repository. Deployment therefore still requires authenticated account access and, if the GitHub repository does not already exist, a public/private visibility choice.

## Tests and checks

From the project root, with the virtual environment active:

```powershell
python -m pytest backend/tests/test_api.py -q
python -m py_compile backend/train_model.py backend/predictor.py backend/main.py backend/tests/test_api.py
```

Build the frontend from its directory:

```powershell
cd frontend
npm run build
```

The backend tests cover the root and health endpoints, blank and oversized input, response shape, probability selection by the actual `hate_speech` class label, both binary outcomes, and missing model behavior.

## Troubleshooting

- **Model not trained / HTTP 503:** From the project root, activate `.venv`, install `backend/requirements.txt`, and run `python backend/train_model.py --data labeled_data.csv`.
- **Dataset or column not found:** Check the CSV path and headers. Pass `--data`, `--text-column`, and `--target-column` as needed. Verify the dataset's class meanings and set `--hate-labels` / `--not-hate-labels` explicitly.
- **Frontend says backend offline:** Keep the Uvicorn terminal running. Check `http://127.0.0.1:8000/health` and confirm `VITE_API_BASE_URL` if customized. A production build requires this variable to be configured.
- **Vite dependencies are missing:** Run `npm install` inside `frontend`.
- **PowerShell cannot activate `.venv`:** Use the current-process execution policy command shown above, or invoke `.venv\Scripts\python.exe` directly.

## Technologies

React, Vite, Tailwind CSS, Lucide React, FastAPI, Pydantic, Uvicorn, Pandas, NumPy, Scikit-learn, Joblib, and Pytest. There is no database, user account system, prediction history, or third-party inference API.