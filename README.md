# AutoML Studio — Transparent AutoML Workspace

AutoML Studio is a React + FastAPI app for dataset profiling, preprocessing, train/test splitting, model comparison, Top 3 ranking, metric visualizations, and plain-language model-family reasoning.

## Included

- `frontend/`: React + Vite + Recharts dashboard.
- `backend/`: FastAPI API using pandas, NumPy and scikit-learn.
- `.github/workflows/deploy-frontend.yml`: builds the React frontend and deploys it to GitHub Pages.
- `.github/workflows/python-tests.yml`: runs backend tests.
- `render.yaml`: Render service configuration for the API.

## Run locally on Windows

Open two terminals from the `automl-studio` project root.

### Terminal 1 — API

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

API docs: http://127.0.0.1:8000/docs

### Terminal 2 — React

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL, usually http://localhost:5173. The frontend uses `http://127.0.0.1:8000` for the API during local development.

## Test the backend

From the `backend` directory, install the testing dependencies and run:

```powershell
pip install pytest httpx
pytest -q
```

## Publish the frontend and API

GitHub Pages hosts static frontend assets only; the Python API must be deployed separately. A free Render service can be used for a prototype, subject to Render's current free-tier limits and sleep behavior.

### 1. Create a GitHub repository

Create a **public** repository named `automl-studio` under your GitHub account. Do not initialize the repository with a README or license because this project already contains files.

From the extracted project root (`automl-studio`), run:

```powershell
git init
git add .
git commit -m "Rename and prepare AutoML Studio for deployment"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/automl-studio.git
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username. If Git says a remote already exists, run `git remote -v` and check it before changing it.

### 2. Deploy the Python API on Render

1. Sign into Render and create a Web Service from your `automl-studio` GitHub repository.
2. Set **Root Directory** to `backend`.
3. Build command: `pip install -r requirements.txt`.
4. Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
5. Health-check path: `/api/health` (if shown).
6. After deployment, copy the API URL, e.g. `https://automl-studio-api.onrender.com`.
7. In the Render service's environment settings, set `CORS_ORIGINS` to `http://localhost:5173,http://127.0.0.1:5173,https://YOUR_USERNAME.github.io` and redeploy. Replace the username.

The repository includes `render.yaml` as a starting point; confirm the settings in the Render dashboard before deploying.

### 3. Configure the frontend to call the API

1. In GitHub, open **Settings → Secrets and variables → Actions → Variables**.
2. Add a repository variable named `VITE_API_URL`.
3. Set its value to your exact Render API URL, without a trailing slash.
4. Open **Settings → Pages** and set the source to **GitHub Actions**.
5. Open **Actions** and verify the `Deploy AutoML Studio frontend to GitHub Pages` workflow succeeds.

The Vite build uses `/automl-studio/`, matching the repository name. If you choose a different repo name, update `frontend/vite.config.js` and the workflow/build configuration to match.

After success, your website will be:

```text
https://YOUR_USERNAME.github.io/automl-studio/
```

### 4. Future updates

After editing files, push changes to `main`:

```powershell
git add .
git commit -m "Describe the update"
git push
```

GitHub Actions redeploys the frontend. Render will redeploy the API when backend files are pushed if automatic deployments are enabled.

## Important scope and safety notes

- The app supports structured tables and text columns, with supervised text training requiring labels. It does not currently train directly on images, audio, PDFs, or Word documents.
- For large files, the app profiles the dataset but samples at most 30,000 cleaned rows for model comparison. This is disclosed in the UI.
- Avoid uploading sensitive or regulated datasets to a public demo. Use only data you have permission to process.
- A high holdout metric is not proof of production performance. Check target correctness, leakage, class imbalance, repeated entities, time drift, and external validation.
- Free cloud services can have limited RAM and may sleep when idle. Test large datasets locally first.
