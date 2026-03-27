"""
Cognee Store Module

Cognee knowledge graph interface for storing and querying memory events.
"""

import logging
import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Any
import json
import os

from config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CogneeMemoryStore:
    """
    Cognee-based memory store for events.

    Provides async interface for storing and querying memory events.
    Uses local SQLite backend for MVP.
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize Cognee memory store.

        Args:
            db_path: Path to SQLite database (default: from config)
        """
        self.db_path = db_path or config.db_path
        self.events: List[Dict] = []
        self._initialized = True

        # Load existing events from log file
        self._load_from_log()

        logger.info(f"CogneeMemoryStore initialized with db_path: {self.db_path}")

    def _load_from_log(self):
        """Load events from the log file (deduplicates by event_id)."""
        log_path = getattr(config, "log_path", "data/session.log")
        if os.path.exists(log_path):
            try:
                existing_ids = {e.get("event_id") for e in self.events}
                new_events = []
                with open(log_path, "r") as f:
                    for line in f:
                        if line.strip():
                            try:
                                event = json.loads(line)
                                eid = event.get("event_id")
                                if eid not in existing_ids:
                                    new_events.append(event)
                                    existing_ids.add(eid)
                            except json.JSONDecodeError:
                                pass
                self.events.extend(new_events)
                if new_events:
                    logger.info(f"Loaded {len(new_events)} new events from log (total: {len(self.events)})")
            except Exception as e:
                logger.warning(f"Failed to load log: {e}")

    def _run_async(self, coro):
        """Run async function in sync context."""
        try:
            return asyncio.run(coro)
        except RuntimeError:
            # Already in async context
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(coro)

    async def add_event(self, event_dict: Dict[str, Any]) -> str:
        """
        Store a memory event.

        Args:
            event_dict: Event data with fields:
                - timestamp: ISO8601 string
                - frame_index: int
                - surprise_score: float
                - description: str
                - latent_hash: str
                - context_window: list of floats

        Returns:
            Event ID
        """
        # Ensure required fields
        if "timestamp" not in event_dict:
            event_dict["timestamp"] = datetime.now().isoformat()

        if "event_id" not in event_dict:
            event_dict["event_id"] = (
                f"event_{len(self.events)}_{int(datetime.now().timestamp())}"
            )

        # Store event
        self.events.append(event_dict)

        # Also persist to file for durability
        self._persist_event(event_dict)

        logger.info(
            f"Added event {event_dict['event_id']} with surprise {event_dict.get('surprise_score', 0):.3f}"
        )

        return event_dict["event_id"]

    def _persist_event(self, event: Dict):
        """Persist event to JSON file."""
        try:
            os.makedirs(os.path.dirname(config.log_path) or "data", exist_ok=True)
            with open(config.log_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.warning(f"Failed to persist event: {e}")

    async def query(self, question: str) -> str:
        """
        Query memory with natural language.

        Args:
            question: Natural language question

        Returns:
            Answer string
        """
        if not self.events:
            return "No events in memory to query."

        # Get recent events for context
        recent = self.get_recent_events_sync(n=10)

        # Format context
        context_parts = []
        for event in recent:
            context_parts.append(
                f"Frame {event.get('frame_index', '?')}: "
                f"surprise={event.get('surprise_score', 0):.3f}, "
                f"description={event.get('description', 'N/A')}"
            )

        context = "\n".join(context_parts)

        # For MVP, return formatted context
        # In full implementation, this would use cognee's graph query
        response = f"Based on {len(self.events)} stored events:\n\n{context}\n\nAnswer: {question}"

        logger.info(f"Query processed: {question[:50]}...")

        return response

    async def get_recent_events(self, n: int = 10) -> List[Dict]:
        """
        Get recent events.

        Args:
            n: Number of events to retrieve

        Returns:
            List of recent events
        """
        return self.events[-n:] if self.events else []

    def get_recent_events_sync(self, n: int = 10) -> List[Dict]:
        """Synchronous version of get_recent_events."""
        return self.events[-n:] if self.events else []

    async def get_summary(self) -> str:
        """
        Get summary of all stored events.

        Returns:
            Summary string
        """
        if not self.events:
            return "No events stored in memory."

        surprise_scores = [e.get("surprise_score", 0) for e in self.events]
        avg_surprise = sum(surprise_scores) / len(surprise_scores)

        summary = f"Memory Summary:\n"
        summary += f"- Total events: {len(self.events)}\n"
        summary += f"- Average surprise: {avg_surprise:.3f}\n"
        summary += f"- Time range: {self.events[0].get('timestamp', '?')} to {self.events[-1].get('timestamp', '?')}\n"

        return summary

    def get_summary_sync(self) -> str:
        """Synchronous version of get_summary."""
        if not self.events:
            return "No events stored in memory."

        surprise_scores = [e.get("surprise_score", 0) for e in self.events]
        avg_surprise = sum(surprise_scores) / len(surprise_scores)

        summary = f"Memory Summary:\n"
        summary += f"- Total events: {len(self.events)}\n"
        summary += f"- Average surprise: {avg_surprise:.3f}\n"

        return summary

    def get_all_events(self) -> List[Dict]:
        """Get all stored events."""
        return self.events.copy()

    def clear(self):
        """Clear all events."""
        self.events.clear()
        logger.info("Memory store cleared")


def test_cognee_store():
    """Test function for standalone testing."""
    logger.info("Testing CogneeMemoryStore...")

    store = CogneeMemoryStore()

    # Test add event
    import asyncio

    async def test_add():
        event_id = await store.add_event(
            {
                "timestamp": datetime.now().isoformat(),
                "frame_index": 0,
                "surprise_score": 0.85,
                "description": "Test event - sudden motion detected",
                "latent_hash": "abc123",
                "context_window": [0.1, 0.2, 0.15],
            }
        )
        logger.info(f"Added event with ID: {event_id}")

        # Add more events
        for i in range(1, 5):
            await store.add_event(
                {
                    "timestamp": datetime.now().isoformat(),
                    "frame_index": i,
                    "surprise_score": 0.3 + i * 0.1,
                    "description": f"Test event {i}",
                    "latent_hash": f"hash_{i}",
                    "context_window": [0.1] * 3,
                }
            )

        # Query
        result = await store.query("What happened in the session?")
        logger.info(f"Query result: {result[:200]}...")

        # Get recent
        recent = await store.get_recent_events(n=3)
        logger.info(f"Recent events: {len(recent)}")

        # Summary
        summary = await store.get_summary()
        logger.info(f"Summary: {summary}")

    asyncio.run(test_add())

    return store


if __name__ == "__main__":
    test_cognee_store()
