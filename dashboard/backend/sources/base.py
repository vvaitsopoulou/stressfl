"""DataSource abstraction.

Every data source — mock, replay, fitbit_live — implements the same async
interface. This is what lets us flip the backend's input without touching
main.py or the frontend.

A source is an async iterator of ServerMessage instances. It runs forever
(or until the session ends). The session loop in main.py just consumes
from it and forwards to the WebSocket.

The contract is small on purpose: one method, one return type. We don't
need plugins or registries yet as the source is chosen by config at
startup. If we ever want runtime switching (e.g. "fall back to replay if
Fitbit goes stale for 5 min"), we add a wrapper source that composes
two real ones.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from backend.protocol import ServerMessage


class DataSource(ABC):
    """Abstract async source of protocol messages."""

    @abstractmethod
    def stream(self) -> AsyncIterator[ServerMessage]:
        """Yield messages forever, in protocol-cadence order.

        The first message yielded MUST be a HelloMessage. After that,
        the source emits hr_sample, prediction, quality_update, and
        system_event messages at protocol-defined cadences.

        The source is responsible for its own timing (asyncio.sleep).
        The caller treats the iterator as opaque.

        Cancellation: if the consumer cancels the task, the source
        should clean up promptly. asyncio.CancelledError will propagate
        through any in-flight asyncio.sleep, which is what we want.
        """
        ...
