"""FastAPI webhook server with HMAC verification and async agent dispatch."""

import hashlib
import hmac
import json
from typing import TYPE_CHECKING

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse

from dev_agent.webhooks.handlers import handle_github, handle_gitlab, WebhookTask

if TYPE_CHECKING:
    from dev_agent.agent.harness import AgentHarness
    from dev_agent.config import Settings


def _verify_github_signature(payload_bytes: bytes, signature: str, secret: str) -> bool:
    if not secret or not signature:
        return not secret  # skip check if no secret configured
    expected = "sha256=" + hmac.new(secret.encode(), payload_bytes, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)


def _verify_gitlab_token(token: str, secret: str) -> bool:
    if not secret:
        return True
    return hmac.compare_digest(secret, token)


async def _dispatch_task(harness: "AgentHarness", task: WebhookTask) -> str:
    """Run the agent for a webhook task and collect the final response."""
    result = ""
    async for event in harness.run(task.prompt, thread_id=task.thread_id, workspace=task.workspace):
        if event["type"] == "done":
            result = event["payload"]
    return result


def create_app(
    harness: "AgentHarness",
    default_workspace: str = ".",
    settings: "Settings | None" = None,
) -> FastAPI:
    secret = settings.webhooks.secret if settings else ""

    app = FastAPI(title="Dev Agent Webhook Server", version="0.1.0")

    @app.get("/health")
    async def health():
        return {"status": "ok", "agent": "dev-agent"}

    @app.post("/webhook/github")
    async def github_webhook(request: Request, background_tasks: BackgroundTasks):
        payload_bytes = await request.body()
        signature = request.headers.get("X-Hub-Signature-256", "")

        if not _verify_github_signature(payload_bytes, signature, secret):
            raise HTTPException(status_code=401, detail="Invalid signature")

        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        task = handle_github(payload, default_workspace)
        if task is None:
            return JSONResponse({"status": "ignored", "reason": "unhandled event type"})

        background_tasks.add_task(_dispatch_task, harness, task)
        return JSONResponse({
            "status": "accepted",
            "thread_id": task.thread_id,
            "event_type": task.event_type,
        })

    @app.post("/webhook/gitlab")
    async def gitlab_webhook(request: Request, background_tasks: BackgroundTasks):
        token = request.headers.get("X-Gitlab-Token", "")

        if not _verify_gitlab_token(token, secret):
            raise HTTPException(status_code=401, detail="Invalid token")

        payload_bytes = await request.body()
        try:
            payload = json.loads(payload_bytes)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        task = handle_gitlab(payload, default_workspace)
        if task is None:
            return JSONResponse({"status": "ignored", "reason": "unhandled event type"})

        background_tasks.add_task(_dispatch_task, harness, task)
        return JSONResponse({
            "status": "accepted",
            "thread_id": task.thread_id,
            "event_type": task.event_type,
        })

    @app.post("/webhook/run")
    async def run_webhook(request: Request, background_tasks: BackgroundTasks):
        """Generic endpoint: POST JSON with {prompt, workspace?, thread_id?}."""
        payload_bytes = await request.body()
        try:
            body = json.loads(payload_bytes)
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        prompt = body.get("prompt", "").strip()
        if not prompt:
            raise HTTPException(status_code=422, detail="'prompt' field is required")

        workspace = body.get("workspace", default_workspace)
        thread_id = body.get("thread_id") or harness.new_thread()

        from dev_agent.webhooks.handlers import WebhookTask
        task = WebhookTask(prompt=prompt, workspace=workspace, thread_id=thread_id,
                           event_type="manual", source="api")
        background_tasks.add_task(_dispatch_task, harness, task)
        return JSONResponse({"status": "accepted", "thread_id": thread_id})

    return app
