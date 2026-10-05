from __future__ import annotations

from datetime import datetime, timezone
import math

from app.ai.memory.schemas import TemporalContext
from app.models.conversation import Message


class TemporalEngine:
    """
    Computes time deltas and generates conversational temporal cues
    so the AI model understands elapsed time between chat turns.
    """

    @staticmethod
    def calculate(
        last_message: Message | None,
        now: datetime | None = None,
    ) -> TemporalContext:
        if not last_message or not last_message.created_at:
            return TemporalContext(gap_type="new_session")

        current_time = now or datetime.now(timezone.utc)
        last_time = last_message.created_at

        # Ensure both are timezone-aware for comparison
        if last_time.tzinfo is None:
            last_time = last_time.replace(tzinfo=timezone.utc)
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=timezone.utc)

        delta = current_time - last_time
        elapsed_seconds = max(0.0, delta.total_seconds())

        return TemporalEngine._format_cue(elapsed_seconds, last_time)

    @staticmethod
    def _format_cue(elapsed_seconds: float, last_time: datetime) -> TemporalContext:
        minutes = elapsed_seconds / 60.0
        hours = minutes / 60.0
        days = hours / 24.0

        date_str = last_time.strftime("%b %d, %Y")
        time_str = last_time.strftime("%I:%M %p UTC")

        # Case 1: Continuous / active conversation (< 15 mins)
        if elapsed_seconds < 900:
            return TemporalContext(
                elapsed_seconds=elapsed_seconds,
                last_active_at=last_time,
                gap_type="active_dialogue",
                prompt_cue=None,
            )

        # Case 2: Short break (15 mins to 2 hours)
        if elapsed_seconds < 7200:
            rounded_mins = int(math.ceil(minutes))
            cue = (
                f"[Session Context: User returned after ~{rounded_mins} minutes. "
                "Maintain conversational flow naturally.]"
            )
            return TemporalContext(
                elapsed_seconds=elapsed_seconds,
                last_active_at=last_time,
                gap_type="short_break",
                prompt_cue=cue,
            )

        # Case 3: Same day / later in the day (2 to 24 hours)
        if elapsed_seconds < 86400:
            rounded_hours = int(math.ceil(hours))
            cue = (
                f"[Session Context: User returned after ~{rounded_hours} hours "
                f"(last active: {time_str}). Context of earlier dialogue is preserved.]"
            )
            return TemporalContext(
                elapsed_seconds=elapsed_seconds,
                last_active_at=last_time,
                gap_type="same_day",
                prompt_cue=cue,
            )

        # Case 4: Recent days (1 to 7 days)
        if days < 7.0:
            rounded_days = int(math.ceil(days))
            unit = "day" if rounded_days == 1 else "days"
            cue = (
                f"[Session Context: User returned after {rounded_days} {unit} "
                f"(last active: {date_str}). Acknowledge return naturally if helpful, "
                "prioritizing their immediate request.]"
            )
            return TemporalContext(
                elapsed_seconds=elapsed_seconds,
                last_active_at=last_time,
                gap_type="recent_days",
                prompt_cue=cue,
            )

        # Case 5: Long absence (> 7 days)
        rounded_days = int(math.ceil(days))
        cue = (
            f"[Session Context: User returned after an extended absence of {rounded_days} days "
            f"(last active: {date_str}). Previous context is summarized below for orientation.]"
        )
        return TemporalContext(
            elapsed_seconds=elapsed_seconds,
            last_active_at=last_time,
            gap_type="long_absence",
            prompt_cue=cue,
        )
