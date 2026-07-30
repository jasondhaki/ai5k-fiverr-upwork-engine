"""
Loads .env exactly once, at import time - the single place environment
variables (GITHUB_TOKEN, ANTHROPIC_API_KEY, GROQ_API_KEY, LLM_PROVIDER, ...)
get resolved from a local .env file into os.environ.

Imported at the top of app/__init__.py, so any `import app...` - the app
server, a script, a test - loads .env before any other app code runs,
regardless of which module happens to be imported first. Individual modules
that read an env var (github_parser.py, extractor.py) no longer call
load_dotenv() themselves.
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()
