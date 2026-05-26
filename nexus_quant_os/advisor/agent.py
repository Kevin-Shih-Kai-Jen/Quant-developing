"""
nexus_quant_os/advisor/agent.py — Antigravity Agent Configuration
===================================================================

Defines the NexusAdvisor class, encapsulating the Google Antigravity SDK
Agent for conversational financial advice.

The Agent is kept alive across multiple chat() calls using an async
context manager, preserving ToolContext state (e.g., personal_context)
across turns.

Author : Nexus Quant OS — Advisor Division
"""

from __future__ import annotations

from pathlib import Path
from typing import AsyncGenerator

from google.antigravity import Agent, LocalAgentConfig
from google.antigravity.types import TemplatedSystemInstructions

from nexus_quant_os.advisor.tools import (
    get_current_portfolio,
    get_market_snapshot,
    record_personal_context,
)

# Project root: three levels up from this file
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ADVISOR_MEMORY_DIR = _PROJECT_ROOT / "data" / "advisor_memory"


class NexusAdvisor:
    """Wrapper around the Google Antigravity Agent for Financial Advice.

    Usage (async context manager — recommended)::

        async with NexusAdvisor(conversation_id=prev_id) as advisor:
            reply = await advisor.chat("What's my portfolio?")
            print(reply)

    The context manager keeps the SDK Agent alive across multiple calls,
    preserving ToolContext state (e.g., personal_context) between turns.
    """

    def __init__(self, conversation_id: str | None = None) -> None:
        """Initializes the advisor agent configuration.

        Args:
            conversation_id: Optional ID to resume a previous conversation.
        """
        _ADVISOR_MEMORY_DIR.mkdir(parents=True, exist_ok=True)

        # Fix S6: initialize conversation_id in __init__
        self.conversation_id: str | None = conversation_id

        # Define the persona using TemplatedSystemInstructions
        persona = (
            "You are Nexus, a professional, calm, and highly analytical quantitative "
            "financial advisor. You are part of the Nexus Quant OS ecosystem.\n\n"
            "CRITICAL DIRECTIVES:\n"
            "1. You have access to the user's primary quantitative portfolio via the "
            "`get_current_portfolio` tool. This is a simulation account.\n"
            "2. If the user uploads an image of ANOTHER brokerage account or asks about "
            "assets outside the quant system, analyze it IN MEMORY ONLY. Do NOT attempt "
            "to save those external assets into the quantitative database.\n"
            "3. Use `record_personal_context` to remember life events (e.g., military "
            "service, vacations, risk tolerance changes) so you can factor them into "
            "future advice.\n"
            "4. Always prioritize risk management and drawdown protection.\n"
            "5. If you need current market prices, use `get_market_snapshot`."
        )

        system_instructions = TemplatedSystemInstructions(
            identity=persona
        )

        self.config = LocalAgentConfig(
            system_instructions=system_instructions,
            tools=[
                get_current_portfolio,
                record_personal_context,
                get_market_snapshot,
            ],
            save_dir=str(_ADVISOR_MEMORY_DIR),
            conversation_id=conversation_id,
        )

        # Fix C6: Agent will be created once and reused across calls
        self._agent: Agent | None = None

    async def __aenter__(self) -> "NexusAdvisor":
        """Start the SDK Agent (kept alive for multi-turn conversations)."""
        self._agent = Agent(self.config)
        await self._agent.__aenter__()
        self.conversation_id = self._agent.conversation_id
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Gracefully shut down the SDK Agent."""
        if self._agent is not None:
            await self._agent.__aexit__(exc_type, exc_val, exc_tb)
            self._agent = None

    async def chat(self, content: str | list) -> str:
        """Sends a message to the agent and returns the full string response.

        If the advisor was opened via ``async with``, the existing agent
        is reused (preserving ToolContext state).  Otherwise a temporary
        agent is created and destroyed (legacy one-shot mode for the API).

        Args:
            content: The message text, or a list containing text and Image objects.

        Returns:
            The complete response text from the agent.
        """
        if self._agent is not None:
            # Persistent mode — reuse the existing agent
            response = await self._agent.chat(content)
            return await response.text()

        # Fallback one-shot mode (for HTTP API where context manager is impractical)
        async with Agent(self.config) as agent:
            self.conversation_id = agent.conversation_id
            response = await agent.chat(content)
            return await response.text()

    async def chat_stream(self, content: str | list) -> AsyncGenerator[str, None]:
        """Sends a message and yields the response as a stream of chunks.

        Args:
            content: The message text, or a list containing text and Image objects.

        Yields:
            String chunks of the agent's response as they are generated.
        """
        if self._agent is not None:
            response = await self._agent.chat(content)
            async for chunk in response:
                yield chunk
            return

        # Fallback one-shot mode
        async with Agent(self.config) as agent:
            self.conversation_id = agent.conversation_id
            response = await agent.chat(content)
            async for chunk in response:
                yield chunk
