# CallSeal MVP

CallSeal turns a business conversation into a traceable shared understanding. This MVP supports transcript analysis, optional local Whisper transcription, term-by-term review, secure public confirmation, and a persisted Conversation Agreement Record.

Product, software, design, feature, and verification specifications are indexed in [docs/README.md](docs/README.md).

## Run

```powershell
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\uvicorn app.main:app --reload
```

Open http://127.0.0.1:8000 and sign in with `demo@callseal.local` / `Demo123!`.

Or run the PostgreSQL-backed stack with Docker:

```powershell
docker compose up --build
```

## Permanent validation deployment

The included `render.yaml` deploys the Docker application and a PostgreSQL database as a Render Blueprint. Push this repository to GitHub, open the Render dashboard, select **New > Blueprint**, connect the repository, and apply the blueprint. Render generates `CALLSEAL_SECRET`, injects `DATABASE_URL`, enables secure cookies, and assigns a public HTTPS `onrender.com` URL.

The free Render web service may sleep after inactivity, and free Render PostgreSQL expires after 30 days. Use it for evaluation only; upgrade the database before storing durable pilot data.

SQLite is the zero-setup default. Set `DATABASE_URL=postgresql+psycopg://...` (and install `psycopg[binary]`) for PostgreSQL. Set `CALLSEAL_SECRET` in every non-demo deployment. Audio uploads use `faster-whisper` when installed; transcript input always works without it.

## Architecture

- `app/main.py`: FastAPI routes, persistence, auth, ownership checks, sharing, confirmations
- `app/ai.py`: validated schemas and provider abstraction; conservative offline heuristic implementation
- `app/static/`: dependency-free single-page UI
- `tests/`: agreement-intelligence regression tests

The heuristic provider is intentionally deterministic for demos and tests. A production open-weight provider should implement `AIProvider.analyze_conversation` and return the same validated `ConversationAnalysis` schema.

CallSeal helps participants document their shared understanding of a conversation. It does not determine legal enforceability or provide legal advice.
