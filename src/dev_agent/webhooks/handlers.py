"""GitHub and GitLab event handlers — convert payloads to agent prompts."""

import uuid
from dataclasses import dataclass
from typing import Any


@dataclass
class WebhookTask:
    prompt: str
    workspace: str
    thread_id: str
    event_type: str
    source: str


def handle_github(payload: dict[str, Any], workspace: str) -> WebhookTask | None:
    """Map a GitHub webhook payload to an agent task."""
    action = payload.get("action", "")
    thread_id = str(uuid.uuid4())

    # Pull Request events
    if "pull_request" in payload:
        pr = payload["pull_request"]
        number = pr.get("number", "?")
        title = pr.get("title", "")
        body = pr.get("body", "") or ""
        base = pr.get("base", {}).get("ref", "main")
        head = pr.get("head", {}).get("ref", "")
        diff_url = pr.get("diff_url", "")

        prompt = (
            f"A GitHub Pull Request event was received (action: {action}).\n"
            f"PR #{number}: {title}\n"
            f"Base: {base} ← Head: {head}\n"
            f"Description: {body[:500]}\n"
            f"Diff URL: {diff_url}\n\n"
            "Please review this PR: check for obvious bugs, style issues, "
            "missing tests, and security concerns. Summarise your findings."
        )
        return WebhookTask(prompt=prompt, workspace=workspace, thread_id=thread_id,
                           event_type="pull_request", source="github")

    # Push events
    if "commits" in payload and "ref" in payload:
        ref = payload.get("ref", "")
        branch = ref.replace("refs/heads/", "")
        commits = payload.get("commits", [])
        commit_lines = "\n".join(
            f"- {c.get('id', '')[:7]} {c.get('message', '').splitlines()[0]}"
            for c in commits[:10]
        )
        prompt = (
            f"A push was made to branch '{branch}' on GitHub.\n"
            f"Commits:\n{commit_lines}\n\n"
            "Analyse the commit messages and identify any concerning patterns "
            "(e.g. force pushes, large binary files, sensitive data, broken CI)."
        )
        return WebhookTask(prompt=prompt, workspace=workspace, thread_id=thread_id,
                           event_type="push", source="github")

    # Issue events
    if "issue" in payload:
        issue = payload["issue"]
        number = issue.get("number", "?")
        title = issue.get("title", "")
        body = issue.get("body", "") or ""
        prompt = (
            f"GitHub issue #{number} was {action}: {title}\n"
            f"Body: {body[:800]}\n\n"
            "Triage this issue: classify its type (bug/feature/question), "
            "suggest a fix approach if it's a bug, or ask clarifying questions."
        )
        return WebhookTask(prompt=prompt, workspace=workspace, thread_id=thread_id,
                           event_type="issue", source="github")

    return None


def handle_gitlab(payload: dict[str, Any], workspace: str) -> WebhookTask | None:
    """Map a GitLab webhook payload to an agent task."""
    kind = payload.get("object_kind", "")
    thread_id = str(uuid.uuid4())

    if kind == "merge_request":
        attrs = payload.get("object_attributes", {})
        iid = attrs.get("iid", "?")
        title = attrs.get("title", "")
        description = attrs.get("description", "") or ""
        source_branch = attrs.get("source_branch", "")
        target_branch = attrs.get("target_branch", "main")
        state = attrs.get("state", "")
        prompt = (
            f"GitLab Merge Request !{iid} ({state}): {title}\n"
            f"Branch: {source_branch} → {target_branch}\n"
            f"Description: {description[:500]}\n\n"
            "Review this MR: look for bugs, missing tests, and improvement suggestions."
        )
        return WebhookTask(prompt=prompt, workspace=workspace, thread_id=thread_id,
                           event_type="merge_request", source="gitlab")

    if kind == "push":
        branch = payload.get("ref", "").replace("refs/heads/", "")
        commits = payload.get("commits", [])
        commit_lines = "\n".join(
            f"- {c.get('id', '')[:7]} {c.get('message', '').splitlines()[0]}"
            for c in commits[:10]
        )
        prompt = (
            f"GitLab push to '{branch}'.\nCommits:\n{commit_lines}\n\n"
            "Summarise the changes and flag any concerns."
        )
        return WebhookTask(prompt=prompt, workspace=workspace, thread_id=thread_id,
                           event_type="push", source="gitlab")

    if kind == "issue":
        attrs = payload.get("object_attributes", {})
        iid = attrs.get("iid", "?")
        title = attrs.get("title", "")
        description = attrs.get("description", "") or ""
        prompt = (
            f"GitLab issue #{iid}: {title}\n"
            f"Description: {description[:800]}\n\n"
            "Triage and suggest next steps."
        )
        return WebhookTask(prompt=prompt, workspace=workspace, thread_id=thread_id,
                           event_type="issue", source="gitlab")

    return None
