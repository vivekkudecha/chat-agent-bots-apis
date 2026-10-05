from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal
import uuid

from app.models.conversation import Message


@dataclass
class TemporalContext:
    """
    Temporal awareness signals between current message and previous conversation turns.
    """
    elapsed_seconds: float | None = None
    last_active_at: datetime | None = None
    gap_type: Literal[
        "new_session",
        "active_dialogue",
        "short_break",
        "same_day",
        "recent_days",
        "long_absence",
    ] = "new_session"
    prompt_cue: str | None = None


@dataclass
class ConversationSummary:
    """
    Structured rolling summary of older conversation messages.
    Compacted from dialogue history to maintain episodic context without token bloat.
    """
    topic: str = ""
    user_intent: str = ""
    key_decisions: list[str] = field(default_factory=list)
    entities_and_facts: dict[str, str] = field(default_factory=dict)
    pending_actions: list[str] = field(default_factory=list)
    last_summarized_message_id: str | None = None
    total_summarized_messages: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic": self.topic,
            "user_intent": self.user_intent,
            "key_decisions": self.key_decisions,
            "entities_and_facts": self.entities_and_facts,
            "pending_actions": self.pending_actions,
            "last_summarized_message_id": self.last_summarized_message_id,
            "total_summarized_messages": self.total_summarized_messages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ConversationSummary:
        if not data:
            return cls()
        return cls(
            topic=data.get("topic", ""),
            user_intent=data.get("user_intent", ""),
            key_decisions=list(data.get("key_decisions", [])),
            entities_and_facts=dict(data.get("entities_and_facts", {})),
            pending_actions=list(data.get("pending_actions", [])),
            last_summarized_message_id=data.get("last_summarized_message_id"),
            total_summarized_messages=int(data.get("total_summarized_messages", 0)),
        )

    def is_empty(self) -> bool:
        return (
            not self.topic
            and not self.user_intent
            and not self.key_decisions
            and not self.entities_and_facts
            and not self.pending_actions
        )

    def to_prompt_text(self) -> str:
        """
        Formats the structured summary into a clean, context-dense text block for the LLM.
        """
        if self.is_empty():
            return ""

        parts: list[str] = []
        if self.topic:
            parts.append(f"Primary Topic: {self.topic}")
        if self.user_intent:
            parts.append(f"User Goal/Intent: {self.user_intent}")
        if self.key_decisions:
            decisions_str = "\n".join(f"- {d}" for d in self.key_decisions)
            parts.append(f"Key Decisions & Agreements:\n{decisions_str}")
        if self.entities_and_facts:
            facts_str = "\n".join(f"- {k}: {v}" for k, v in self.entities_and_facts.items())
            parts.append(f"Key Entities & Stated Facts:\n{facts_str}")
        if self.pending_actions:
            pending_str = "\n".join(f"- {a}" for a in self.pending_actions)
            parts.append(f"Pending/Unresolved Actions:\n{pending_str}")

        return "\n\n".join(parts)


@dataclass
class MemoryContext:
    """
    Unified memory context assembled for the current agent graph and prompt builder.
    """
    conversation_id: uuid.UUID
    temporal: TemporalContext = field(default_factory=TemporalContext)
    summary: ConversationSummary | None = None
    working_history: list[Message] = field(default_factory=list)
    total_conversation_messages: int = 0
