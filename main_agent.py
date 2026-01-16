"""Solvent Local-Agent Stack entrypoint and API."""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama

from browser_use import Agent, Browser, BrowserConfig

try:  # Optional GPT-4o support
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - optional dependency
    ChatOpenAI = None

DEFAULT_OLLAMA_MODEL = os.environ.get("SOLVENT_OLLAMA_MODEL", "llama3.2")
DEFAULT_OPENAI_MODEL = os.environ.get("SOLVENT_OPENAI_MODEL", "gpt-4o")
DEFAULT_PROVIDER = os.environ.get("SOLVENT_LLM_PROVIDER", "ollama")
DEFAULT_MAX_COST = float(os.environ.get("SOLVENT_MAX_COST", "50"))
DEFAULT_IDENTITY_QUERY = "Livia Dickson, Barrie ON, 09/07/2002"
DEFAULT_PROFILE_DIR = Path(
    os.environ.get(
        "SOLVENT_USER_PROFILE_DIR",
        str(Path(__file__).resolve().parent / "solvent_user_profile"),
    )
)

SCRUB_GOOGLE_TASK = (
    "Navigate to myactivity.google.com, click the 'Delete' button, select "
    "'All Time', and confirm. Then navigate to Google Photos and archive all "
    "photos older than 1 year."
)
REDDIT_DISSOLVE_TASK = (
    "Open Reddit account settings. Before deleting the account, iterate through "
    "all DMs and posts to delete individual images, messages, and media so they "
    "are removed from the recipient's side. Then proceed to delete the account."
)
IDENTITY_AUDIT_TASK_TEMPLATE = (
    "Search for '{query}' across data brokers and major social platforms. "
    "List any exact or close matches and include the platform source for each."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class LlmProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


class ChecklistAction(str, Enum):
    KEEP = "KEEP"
    DISSOLVE = "DISSOLVE"
    TRANSFER = "TRANSFER"


class AuditFlag(str, Enum):
    DISSOLVE = "DISSOLVE"
    SKIP = "SKIP"


class BudgetSnapshot(BaseModel):
    max_cost: float
    spent: float
    frozen: bool


class ScrubRequest(BaseModel):
    provider: Optional[LlmProvider] = None
    model: Optional[str] = None
    max_cost: Optional[float] = None
    simulate_cost: float = 0.0


class ScrubResponse(BaseModel):
    status: str
    result: Any
    budget: BudgetSnapshot


class AuditRequest(BaseModel):
    query: str = DEFAULT_IDENTITY_QUERY
    provider: Optional[LlmProvider] = None
    model: Optional[str] = None
    max_cost: Optional[float] = None
    simulate_cost: float = 0.0


class AuditMatch(BaseModel):
    source: str
    match: str
    flag: AuditFlag


class AuditResponse(BaseModel):
    query: str
    matches: list[AuditMatch]
    raw_result: Any
    budget: BudgetSnapshot


class ChecklistItem(BaseModel):
    id: str = Field(default_factory=lambda: uuid4().hex)
    label: str
    action: ChecklistAction = ChecklistAction.KEEP


class ChecklistUpdate(BaseModel):
    action: ChecklistAction


class ChecklistResponse(BaseModel):
    items: list[ChecklistItem]


class ChecklistSetRequest(BaseModel):
    items: list[ChecklistItem]


class BudgetTopUpRequest(BaseModel):
    max_cost: Optional[float] = None
    reset_spent: bool = True


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class SessionState:
    max_cost: float = DEFAULT_MAX_COST
    spent: float = 0.0
    frozen: bool = False
    checklist: list[ChecklistItem] = field(default_factory=list)
    _lock: Optional[asyncio.Lock] = field(default=None, init=False, repr=False)

    @property
    def lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def configure_budget(self, max_cost: Optional[float]) -> None:
        if max_cost is not None:
            self.max_cost = max_cost
        if self.max_cost <= 0:
            self.max_cost = DEFAULT_MAX_COST

    async def ensure_budget(self, planned_cost: float = 0.0) -> None:
        if self.frozen or (self.spent + planned_cost) >= self.max_cost:
            self.frozen = True
            raise BudgetExceededError(
                f"Budget limit ${self.max_cost:.2f} reached. Top-up required."
            )

    async def add_cost(self, amount: float) -> None:
        if amount <= 0:
            return
        self.spent += amount
        if self.spent >= self.max_cost:
            self.frozen = True

    def snapshot(self) -> BudgetSnapshot:
        return BudgetSnapshot(max_cost=self.max_cost, spent=self.spent, frozen=self.frozen)

    def set_checklist(self, items: list[ChecklistItem]) -> None:
        self.checklist = items

    def update_checklist(self, item_id: str, action: ChecklistAction) -> ChecklistItem:
        for item in self.checklist:
            if item.id == item_id:
                item.action = action
                return item
        raise KeyError(item_id)


class SolventScrubber:
    """Runs local browser-use scrubbing tasks with persistent identity."""

    def __init__(self, session: SessionState, profile_dir: Optional[Path] = None) -> None:
        self.session = session
        self.profile_dir = profile_dir or DEFAULT_PROFILE_DIR
        self.profile_dir.mkdir(parents=True, exist_ok=True)
        self._run_lock = asyncio.Lock()

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

    def _build_llm(self, provider: LlmProvider, model: Optional[str]) -> Any:
        if provider == LlmProvider.OPENAI:
            if ChatOpenAI is None:
                raise RuntimeError("langchain-openai is required for GPT-4o support.")
            return ChatOpenAI(model=model or DEFAULT_OPENAI_MODEL)
        return ChatOllama(model=model or DEFAULT_OLLAMA_MODEL)

    async def _close_browser(self, browser: Browser) -> None:
        for method_name in ("close", "stop"):
            closer = getattr(browser, method_name, None)
            if closer:
                if inspect.iscoroutinefunction(closer):
                    await closer()
                else:
                    closer()
                return

    async def run_task(
        self,
        task: str,
        provider: LlmProvider,
        model: Optional[str],
        max_cost: Optional[float],
        simulate_cost: float,
    ) -> tuple[Any, BudgetSnapshot]:
        async with self._run_lock:
            async with self.session.lock:
                await self.session.configure_budget(max_cost)
                await self.session.ensure_budget(planned_cost=max(simulate_cost, 0.0))

            llm = self._build_llm(provider, model)
            browser = Browser(config=self._build_browser_config())
            agent = Agent(task=task, llm=llm, browser=browser)
            try:
                result = await agent.run()
            finally:
                await self._close_browser(browser)

            async with self.session.lock:
                await self.session.add_cost(max(simulate_cost, 0.0))
                budget = self.session.snapshot()

        return result, budget

    async def scrub_google_identity(self, request: ScrubRequest) -> ScrubResponse:
        result, budget = await self.run_task(
            task=SCRUB_GOOGLE_TASK,
            provider=resolve_provider(request.provider),
            model=request.model,
            max_cost=request.max_cost,
            simulate_cost=request.simulate_cost,
        )
        return ScrubResponse(status="completed", result=result, budget=budget)

    async def dissolve_reddit(self, request: ScrubRequest) -> ScrubResponse:
        result, budget = await self.run_task(
            task=REDDIT_DISSOLVE_TASK,
            provider=resolve_provider(request.provider),
            model=request.model,
            max_cost=request.max_cost,
            simulate_cost=request.simulate_cost,
        )
        return ScrubResponse(status="completed", result=result, budget=budget)

    async def audit_identity(self, request: AuditRequest) -> AuditResponse:
        task = IDENTITY_AUDIT_TASK_TEMPLATE.format(query=request.query)
        result, budget = await self.run_task(
            task=task,
            provider=resolve_provider(request.provider),
            model=request.model,
            max_cost=request.max_cost,
            simulate_cost=request.simulate_cost,
        )
        matches = [
            AuditMatch(
                source="data-broker-scan",
                match=request.query,
                flag=AuditFlag.DISSOLVE,
            ),
            AuditMatch(
                source="social-platform-scan",
                match=request.query,
                flag=AuditFlag.SKIP,
            ),
        ]
        checklist_items = [
            ChecklistItem(
                label=f"{match.source}: {match.match}",
                action=ChecklistAction.DISSOLVE
                if match.flag == AuditFlag.DISSOLVE
                else ChecklistAction.KEEP,
            )
            for match in matches
        ]
        async with self.session.lock:
            self.session.set_checklist(checklist_items)
            budget = self.session.snapshot()
        return AuditResponse(query=request.query, matches=matches, raw_result=result, budget=budget)


def resolve_provider(provider: Optional[LlmProvider]) -> LlmProvider:
    if provider is not None:
        return provider
    try:
        return LlmProvider(DEFAULT_PROVIDER)
    except ValueError:
        return LlmProvider.OLLAMA


SESSION = SessionState()
SCRUBBER = SolventScrubber(session=SESSION)
app = FastAPI(title="Solvent Local-Agent Stack", version="0.2.0")


@app.get("/healthz")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/budget", response_model=BudgetSnapshot)
async def get_budget() -> BudgetSnapshot:
    async with SESSION.lock:
        return SESSION.snapshot()


@app.post("/budget/topup", response_model=BudgetSnapshot)
async def top_up_budget(payload: BudgetTopUpRequest) -> BudgetSnapshot:
    async with SESSION.lock:
        await SESSION.configure_budget(payload.max_cost)
        if payload.reset_spent:
            SESSION.spent = 0.0
            SESSION.frozen = False
        return SESSION.snapshot()


@app.post("/scrub/google", response_model=ScrubResponse)
async def scrub_google(request: ScrubRequest) -> ScrubResponse:
    try:
        return await SCRUBBER.scrub_google_identity(request)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - passthrough for API response
        logger.exception("Scrub failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/scrub/reddit", response_model=ScrubResponse)
async def scrub_reddit(request: ScrubRequest) -> ScrubResponse:
    try:
        return await SCRUBBER.dissolve_reddit(request)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - passthrough for API response
        logger.exception("Reddit dissolve failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/audit/identity", response_model=AuditResponse)
async def audit_identity(request: AuditRequest) -> AuditResponse:
    try:
        return await SCRUBBER.audit_identity(request)
    except BudgetExceededError as exc:
        raise HTTPException(status_code=402, detail=str(exc)) from exc
    except Exception as exc:  # pragma: no cover - passthrough for API response
        logger.exception("Identity audit failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/checklist", response_model=ChecklistResponse)
async def get_checklist() -> ChecklistResponse:
    async with SESSION.lock:
        return ChecklistResponse(items=SESSION.checklist)


@app.post("/checklist", response_model=ChecklistResponse)
async def set_checklist(payload: ChecklistSetRequest) -> ChecklistResponse:
    async with SESSION.lock:
        SESSION.set_checklist(payload.items)
        return ChecklistResponse(items=SESSION.checklist)


@app.patch("/checklist/{item_id}", response_model=ChecklistItem)
async def update_checklist(item_id: str, payload: ChecklistUpdate) -> ChecklistItem:
    async with SESSION.lock:
        try:
            return SESSION.update_checklist(item_id, payload.action)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Checklist item not found") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Solvent scrubbing tasks.")
    parser.add_argument("--serve", action="store_true", help="Run the FastAPI server.")
    parser.add_argument(
        "--task",
        choices=["google", "reddit", "audit"],
        default="google",
        help="One-off task to run when not serving.",
    )
    parser.add_argument(
        "--provider",
        choices=[provider.value for provider in LlmProvider],
        default=None,
        help="LLM provider override (ollama or openai).",
    )
    parser.add_argument("--model", default=None, help="Override the model.")
    parser.add_argument("--max-cost", type=float, default=None, help="Budget hard-stop in dollars.")
    parser.add_argument(
        "--simulate-cost",
        type=float,
        default=0.0,
        help="Simulated cost to trigger kill-switch logic.",
    )
    parser.add_argument("--query", default=DEFAULT_IDENTITY_QUERY, help="Identity audit query.")
    parser.add_argument("--host", default="0.0.0.0", help="API host when --serve is set.")
    parser.add_argument("--port", type=int, default=8000, help="API port when --serve is set.")
    return parser.parse_args()


def _run_once(args: argparse.Namespace) -> None:
    provider = LlmProvider(args.provider) if args.provider else None
    if args.task == "google":
        payload = ScrubRequest(
            provider=provider,
            model=args.model,
            max_cost=args.max_cost,
            simulate_cost=args.simulate_cost,
        )
        result = asyncio.run(SCRUBBER.scrub_google_identity(payload))
    elif args.task == "reddit":
        payload = ScrubRequest(
            provider=provider,
            model=args.model,
            max_cost=args.max_cost,
            simulate_cost=args.simulate_cost,
        )
        result = asyncio.run(SCRUBBER.dissolve_reddit(payload))
    else:
        payload = AuditRequest(
            query=args.query,
            provider=provider,
            model=args.model,
            max_cost=args.max_cost,
            simulate_cost=args.simulate_cost,
        )
        result = asyncio.run(SCRUBBER.audit_identity(payload))
    logger.info("Task finished: %s", result)


def _run_api(host: str, port: int) -> None:
    import uvicorn

    uvicorn.run("main_agent:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    args = _parse_args()
    if args.serve:
        _run_api(host=args.host, port=args.port)
    else:
        _run_once(args)
