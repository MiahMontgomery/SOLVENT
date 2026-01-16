"""Solvent Local-Agent Stack entrypoint and API."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException
from langchain_ollama import ChatOllama

from browser_use import Agent, Browser, BrowserConfig

DEFAULT_MODEL = os.environ.get("SOLVENT_OLLAMA_MODEL", "llama3.2")
SCRUB_GOOGLE_TASK = (
    "Navigate to myactivity.google.com, click the 'Delete' button, select "
    "'All Time', and confirm. Then navigate to Google Photos and archive all "
    "photos older than 1 year."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class SolventScrubber:
    """Runs local browser-use scrubbing tasks with persistent identity."""

    def __init__(self, model: Optional[str] = None, profile_dir: Optional[Path] = None) -> None:
        self.model = model or DEFAULT_MODEL
        self.profile_dir = profile_dir or (Path(__file__).resolve().parent / "solvent_user_profile")
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    def _build_browser_config(self) -> BrowserConfig:
        config_kwargs: dict[str, Any] = {
            "headless": False,
            "user_data_dir": str(self.profile_dir),
        }
        try:
            params = set(inspect.signature(BrowserConfig).parameters)
        except (TypeError, ValueError):
            params = set()

        if "use_persistent_context" in params:
            config_kwargs["use_persistent_context"] = True
        elif "persistent_context" in params:
            config_kwargs["persistent_context"] = True
        elif "persist_context" in params:
            config_kwargs["persist_context"] = True

        return BrowserConfig(**config_kwargs)

    def _build_llm(self, model: Optional[str] = None) -> ChatOllama:
        return ChatOllama(model=model or self.model)

    async def _close_browser(self, browser: Browser) -> None:
        for method_name in ("close", "stop"):
            closer = getattr(browser, method_name, None)
            if closer:
                if inspect.iscoroutinefunction(closer):
                    await closer()
                else:
                    closer()
                return

    async def scrub_google_identity(self, model: Optional[str] = None) -> dict[str, Any]:
        async with self._lock:
            llm = self._build_llm(model)
            browser = Browser(config=self._build_browser_config())
            agent = Agent(task=SCRUB_GOOGLE_TASK, llm=llm, browser=browser)
            try:
                result = await agent.run()
            finally:
                await self._close_browser(browser)
        return {"status": "completed", "result": result}


SCRUBBER = SolventScrubber()
app = FastAPI(title="Solvent Local-Agent Stack", version="0.1.0")


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/scrub/google")
async def scrub_google(model: Optional[str] = None) -> dict[str, Any]:
    try:
        return await SCRUBBER.scrub_google_identity(model=model)
    except Exception as exc:  # pragma: no cover - passthrough for API response
        logger.exception("Scrub failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Solvent scrubbing tasks.")
    parser.add_argument("--serve", action="store_true", help="Run the FastAPI server.")
    parser.add_argument("--model", default=None, help="Override the Ollama model.")
    parser.add_argument("--host", default="0.0.0.0", help="API host when --serve is set.")
    parser.add_argument("--port", type=int, default=8000, help="API port when --serve is set.")
    return parser.parse_args()


def _run_once(model: Optional[str]) -> None:
    logger.info("Starting Google identity scrub.")
    result = asyncio.run(SCRUBBER.scrub_google_identity(model=model))
    logger.info("Scrub finished: %s", result)


def _run_api(host: str, port: int) -> None:
    import uvicorn

    uvicorn.run("main_agent:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    args = _parse_args()
    if args.serve:
        _run_api(host=args.host, port=args.port)
    else:
        _run_once(model=args.model)
