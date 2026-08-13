"""Claude adapter for the Metis-Agent optimizer.

Routes every LLM call through the Claude Agent SDK, which drives the local
`claude` CLI as a subprocess using its own subscription login. No
ANTHROPIC_API_KEY is required (and the CLI's OAuth login is used).

The optimizer is synchronous, so this module exposes a plain blocking
`invoke(system, user)` and owns the async bridge internally.
"""

import asyncio
import os
import threading

try:
    from claude_agent_sdk import (
        AssistantMessage,
        ClaudeAgentOptions,
        TextBlock,
        ThinkingBlock,
        query,
    )
    HAS_SDK = True
except Exception:  # SDK not installed
    HAS_SDK = False

# Model alias resolved by the CLI (e.g. "sonnet", "opus", "haiku"), or a full id.
DEFAULT_MODEL = os.environ.get("CLAUDE_MODEL", "sonnet")


def _options(system, model, max_turns):
    kwargs = dict(
        system_prompt=system,
        model=model,
        tools=[],                 # expose no tools -> single-shot text answer
        allowed_tools=[],
        max_turns=max_turns,
        setting_sources=[],       # ignore repo CLAUDE.md / settings / skills
        permission_mode="bypassPermissions",
    )
    try:
        kwargs["max_thinking_tokens"] = 4096   # let the agent reason before answering
    except Exception:
        pass
    return ClaudeAgentOptions(**kwargs)


async def _acall(system, user, model, max_turns):
    """Return (assistant_text, thinking_text) for one system+user exchange."""
    text_parts, think_parts = [], []
    async for message in query(prompt=user, options=_options(system, model, max_turns)):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                elif isinstance(block, ThinkingBlock):
                    think_parts.append(getattr(block, "thinking", "") or "")
    return "".join(text_parts), "".join(think_parts)


def _run_sync(coro):
    """Run a coroutine to completion from synchronous (possibly threaded) code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box = {}

    def runner():
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:
            box["error"] = exc

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def invoke(system, user, model=None, max_turns=1):
    """Blocking call: returns (assistant_text, thinking_text)."""
    if not HAS_SDK:
        raise ImportError(
            "claude-agent-sdk is not installed. Run `pip install claude-agent-sdk` "
            "and make sure the `claude` CLI is installed and logged in."
        )
    return _run_sync(_acall(system, user, model or DEFAULT_MODEL, max_turns))
