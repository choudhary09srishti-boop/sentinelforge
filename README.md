# SentinelForge

Privacy-conscious AI coding platform: MCP server, local RAG, coding agent, sandbox, web app.
Work in progress for the Dexter Platform AI/ML Engineer assessment.
See docs/architecture.md.

## Setup (Windows PowerShell, Python 3.11)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt -r requirements-dev.txt
Copy-Item .env.example .env
```

## Run the API

```powershell
uvicorn app.main:create_app --factory --app-dir backend --reload
```

Docs: http://127.0.0.1:8000/docs  |  Health: http://127.0.0.1:8000/health

## Checks

```powershell
.\check.ps1
```

## Dependencies

Edit `requirements.in`, then run `pip-compile requirements.in --output-file requirements.txt --strip-extras`.
