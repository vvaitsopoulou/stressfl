# Stress Dashboard — Backend (v0.1)

WebSocket server implementing the v1 protocol described in
`websocket_protocol_v1.md`. Currently ships only a **mock data source**
that satisfies the protocol with synthetic data, no real Fitbit, no
real models. This is the day-one scaffold, the real Fitbit poller and
real model inference plug in later behind the `DataSource` abstraction.

## Layout

```
backend/
├── main.py              # FastAPI app, /ws/live endpoint
├── protocol.py          # Pydantic models for every message type
├── sources/
│   ├── base.py          # DataSource ABC — the swappable seam
│   └── mock.py          # Synthetic generator
├── poker.html           # Throwaway HTML for eyeballing the stream
├── requirements.txt
└── README.md            # this file
```

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

From the **parent** directory of `backend/` (so Python imports resolve):

```bash
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Or, equivalently:

```bash
python -m backend.main
```

It should display:

```
[INFO] dashboard.backend: Backend starting up. DATA_SOURCE=mock
INFO:     Uvicorn running on http://127.0.0.1:8000
```

## Verify it's working

### Sanity check — `/health`

```bash
curl http://127.0.0.1:8000/health
# {"status":"ok","data_source":"mock"}
```

### Eyeball the stream — `poker.html`

Open `backend/poker.html` directly in the browser (it's pure HTML/JS,
no build step). Click **Connect**. The following messages show up

- One `hello` immediately
- `hr_sample` once per second
- `prediction` every 8 seconds (window_id 0, 1, 2, …)
- `quality_update` every 30 seconds
- `system_event` (fitbit_sync) every minute

Hover any row to see the full JSON in the tooltip.

### Run faster than real-time during development

The mock source has a time-scale knob. `MOCK_TIME_SCALE=10` runs ten
simulated seconds per wall-clock second:

```bash
MOCK_TIME_SCALE=10 python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

`MOCK_SEED` controls determinism , same seed, identical stream.
Default is `42`.

## Configuration

| Env var            | Default | Meaning                                              |
| ------------------ | ------- | ---------------------------------------------------- |
| `DATA_SOURCE`      | `mock`  | `mock`, `replay` (NYI), `fitbit` (NYI)               |
| `MOCK_TIME_SCALE`  | `1.0`   | Sim-seconds per wall-clock-second (mock only)        |
| `MOCK_SEED`        | `42`    | RNG seed for determinism (mock only)                 |

## Adding a new data source

1. Subclass `DataSource` in `backend/sources/yours.py`.
2. Implement `async def stream(self) -> AsyncIterator[ServerMessage]`.
   First message MUST be a `HelloMessage`.
3. Add a branch in `make_source()` in `main.py`.

## What's missing (deliberate, for later)

- **Real Fitbit poller.** Will live in `sources/fitbit_live.py`. OAuth
  token handling, intraday endpoint polling every 60 s, sample
  diff-and-batch logic.
- **JSON replay source.** `sources/replay.py`. Reads a recorded session
  JSONL file and emits at original cadence (or accelerated).
- **Real model inference.** Currently `MockSource._make_prediction`
  fabricates plausible outputs. The real version will load three
  PyTorch checkpoints, hold a feature buffer, and run forward passes.
- **Tests.** Smoke tests are in this README; proper pytest suite is
  follow-up work.
