"""Data sources for the dashboard backend.

Each source implements the DataSource async-iterator interface and can be
selected per-connection in main.make_source().

Note: ReplaySource is imported lazily (inside make_source) rather than
re-exported here, because it pulls in torch / src.kd. Importing it at
package-import time would force those heavy deps on a mock-only server.
Import it directly where needed:

    from backend.sources.replay import ReplaySource
"""

from backend.sources.base import DataSource
from backend.sources.mock import MockSource

__all__ = ["DataSource", "MockSource"]
