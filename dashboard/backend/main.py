"""FastAPI app entry point.

One endpoint: GET /ws/live  →  WebSocket that streams the v1 protocol.

The data source is selected per-connection. Priority:
    1. ?source= query param on the WebSocket URL  (the dashboard toggle)
    2. DATA_SOURCE env var
    3. "mock" (default)

Recognised sources:
    mock    — synthetic generator (default)
    replay  — pre-recorded Fitbit pilot, real models   (env-configurable)
    fitbit  — live Fitbit (not implemented yet)

Replay configuration (env vars, read per-connection):
    REPLAY_JSON_PATH        path to the Fitbit intraday JSON
                            (default: data/fitbit/fitbit_intraday_2026-05-03.json)
    REPLAY_CHECKPOINT_DIR   dir holding kd_global_best.pt / no_kd_global_best.pt
                            (default: KD_CHECKPOINT_DIR_CEN from kd_config)
    REPLAY_TIME_SCALE       1.0 = real-time (default); >1 compresses for rehearsal

There is no client→server message handling in v1. We don't read from the
socket — only write. The toggle works by reconnecting with a new
?source= value, not by sending a control message.
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


def make_source(source: str | None = None) -> DataSource:
    """Construct a DataSource. Called once per WebSocket connection.

    Parameters
    ----------
    source : str | None
        Per-connection override (from the ?source= query param). If None,
        falls back to the DATA_SOURCE env var, then "mock".
    """
    kind = (source or os.environ.get("DATA_SOURCE", "mock")).lower()

    if kind == "mock":
        # MOCK_TIME_SCALE lets devs run faster than real-time. e.g.
        # MOCK_TIME_SCALE=10 = 10 sim seconds per wall-clock second.
        time_scale = float(os.environ.get("MOCK_TIME_SCALE", "1.0"))
        seed = int(os.environ.get("MOCK_SEED", "42"))
        return MockSource(seed=seed, time_scale=time_scale)

    if kind == "replay":
        # Lazy import: only pull in torch / src.kd when replay is requested,
        # so a mock-only server has no heavy dependencies at startup.
        from backend.sources.replay import ReplaySource

        json_path = os.environ.get(
            "REPLAY_JSON_PATH", "data/fitbit/fitbit_intraday_2026-05-03.json"
        )
        checkpoint_dir = os.environ.get("REPLAY_CHECKPOINT_DIR")  # None → kd_config default
        time_scale = float(os.environ.get("REPLAY_TIME_SCALE", "1.0"))
        return ReplaySource(
            json_path=json_path,
            checkpoint_dir=checkpoint_dir,
            time_scale=time_scale,
        )

    if kind == "fitbit":
        raise NotImplementedError(
            f"DATA_SOURCE={kind} is not implemented yet. Use mock or replay."
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
    source_param = websocket.query_params.get("source")
    logger.info("WebSocket connected: %s (source=%s)", client, source_param)

    try:
        source = make_source(source_param)
    except (NotImplementedError, ValueError, FileNotFoundError, RuntimeError) as exc:
        logger.error("Failed to construct data source: %s", exc)
        await websocket.close(code=1011, reason=str(exc)[:120])
        return
    except Exception:
        logger.exception("Unexpected error constructing data source")
        await websocket.close(code=1011, reason="source construction failed")
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
