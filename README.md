# crewai-env

## Setup

```bash
python3 -m venv .
bin/pip install -r requirements.txt
```

## Run

```bash
bin/python main.py
```

## Environment variables

The agents read their config from environment variables (see [app/config.py](app/config.py)):

- `AGENTS_API_BASE` — API base URL (default: `http://localhost:20128/v1`)
- `AGENTS_API_KEY` — API key
- `AGENTS_MODEL` — model name
- `AGENTS_TEMPERATURE` — sampling temperature (default: `0.1`)

Each agent listed in `AGENTS` can also be overridden individually with a `<PREFIX>_API_BASE`, `<PREFIX>_API_KEY`, `<PREFIX>_MODEL`, `<PREFIX>_TEMPERATURE` set.
