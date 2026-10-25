# AutoML Studio — Publish it now

This repository is prepared for:
- GitHub repository name: `automl-studio`
- Frontend URL: `https://YOUR_USERNAME.github.io/automl-studio/`
- Frontend: GitHub Pages using GitHub Actions
- Python API: Render Web Service (separate from GitHub Pages)

## 1. Create the GitHub repository

On GitHub, create a **public** repository named `automl-studio`. Leave “Add a README”, `.gitignore`, and license unchecked because the project already includes them.

## 2. Push this project

Open PowerShell in the extracted `automl-studio` folder (the folder containing `README.md`) and run:

```powershell
git init
git add .
git commit -m "Publish AutoML Studio"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/automl-studio.git
git push -u origin main
```

Replace `YOUR_USERNAME`. If a remote already exists, inspect it using `git remote -v` before adding or changing anything. Git may open a browser for GitHub authentication.

## 3. Deploy the Python API

In Render, create a Web Service from the GitHub repository:
- Root directory: `backend`
- Build command: `pip install -r requirements.txt`
- Start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
- Health path: `/api/health` (if the field is available)

After the API deploys, copy its URL, for example `https://your-automl-api.onrender.com`.

In the Render service environment variables, set `CORS_ORIGINS` to:

```text
http://localhost:5173,http://127.0.0.1:5173,https://YOUR_USERNAME.github.io
```

Replace `YOUR_USERNAME`, save, and redeploy the API.

## 4. Set the frontend API URL and publish Pages

On GitHub, open **Settings → Secrets and variables → Actions → Variables → New repository variable**.
- Name: `VITE_API_URL`
- Value: the Render API URL you copied (no trailing slash)

Then open **Settings → Pages** and select **GitHub Actions** as the source. In **Actions**, inspect the “Deploy AutoML Studio frontend to GitHub Pages” run. If the first run happened before you configured `VITE_API_URL`, rerun it from Actions after adding the variable.

When the workflow succeeds, open:

```text
https://YOUR_USERNAME.github.io/automl-studio/
```

## 5. Test and update

Backend tests:

```powershell
cd backend
pip install -r requirements.txt pytest httpx
pytest -q
```

For future updates from the repo root:

```powershell
git add .
git commit -m "Describe your change"
git push
```

## Notes

- GitHub Pages hosts the React frontend; it does not run Python. The Render API is required for uploading data and training models.
- Render free web services can sleep after 15 minutes of inactivity and may take about a minute to wake up. Free-tier limits can change.
- Do not upload sensitive data to the public demo. Large datasets may exceed free cloud memory limits; run them locally where possible.
