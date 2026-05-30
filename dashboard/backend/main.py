"""FastAPI app entry point.

One endpoint: GET /ws/live  →  WebSocket that streams the v1 protocol.

The data source is selected via an env var:
    DATA_SOURCE=mock   (default — synthetic generator)
    DATA_SOURCE=replay (not implemented yet)
    DATA_SOURCE=fitbit (not implemented yet)

There's also a tiny GET /health endpoint for sanity-checking the server
is up without opening a WebSocket.

Per-connection lifecycle:
  1. Client connects to /ws/live.
  2. Server constructs a fresh DataSource for that connection.
  3. Server iterates source.stream(), serialising and sending each
     message as a single JSON frame.
  4. On disconnect (client close, server shutdown, or any error), the
     stream task is cancelled, which propagates CancelledError into
     the source's asyncio.sleep and lets it clean up.

There is no client→server message handling in v1
We don't read from the socket — only write.
"""

from __future__ import annotations

import asyncio
import contextlib
from contextlib import asynccontextmanager
import logging
import os

from backend.sources import DataSource, MockSource
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

logger = logging.getLogger("dashboard.backend")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


# Source selection


def make_source() -> DataSource:
    """Construct a DataSource based on env vars.

    Per-connection called once for each new WebSocket.
    """
    kind = os.environ.get("DATA_SOURCE", "mock").lower()
    if kind == "mock":
        # MOCK_TIME_SCALE lets devs run faster than real-time. e.g.
        # MOCK_TIME_SCALE=10 = 10 sim seconds per wall-clock second.
        time_scale = float(os.environ.get("MOCK_TIME_SCALE", "1.0"))
        seed = int(os.environ.get("MOCK_SEED", "42"))
        return MockSource(seed=seed, time_scale=time_scale)
    if kind in ("replay", "fitbit"):
        raise NotImplementedError(
            f"DATA_SOURCE={kind} is not implemented yet. Use DATA_SOURCE=mock."
        )
    raise ValueError(f"Unknown DATA_SOURCE: {kind!r}")


# App


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info(
        "Backend starting up. DATA_SOURCE=%s", os.environ.get("DATA_SOURCE", "mock")
    )
    yield
    logger.info("Backend shutting down.")


app = FastAPI(
    title="Stress Dashboard Backend",
    description="WebSocket server implementing the v1 stress-prediction protocol.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints


@app.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "data_source": os.environ.get("DATA_SOURCE", "mock"),
    }


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    """Stream v1 protocol messages until client disconnects."""
    await websocket.accept()
    client = websocket.client
    logger.info("WebSocket connected: %s", client)

    try:
        source = make_source()
    except (NotImplementedError, ValueError) as exc:
        logger.error("Failed to construct data source: %s", exc)
        await websocket.close(code=1011, reason=str(exc))
        return

    try:
        async for msg in source.stream():
            payload = msg.model_dump_json()
            await websocket.send_text(payload)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client: %s", client)
    except asyncio.CancelledError:
        logger.info("WebSocket task cancelled: %s", client)
        raise
    except Exception:
        logger.exception("Unhandled error on WebSocket %s", client)
        with contextlib.suppress(Exception):
            await websocket.close(code=1011, reason="internal error")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "backend.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
        log_level="info",
    )
