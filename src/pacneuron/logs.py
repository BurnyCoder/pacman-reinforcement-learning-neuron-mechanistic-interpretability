"""Timestamped logging to the terminal and to logs/<timestamp>_<name>.log.

Local: one call `get_logger(name)` returns a logger that writes ISO timestamps to both sinks.
Global: every phase of the pipeline logs through this so a run can be audited afterwards.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
from datetime import datetime

from pacneuron.config import LOGS_DIR

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def get_logger(name: str = "pacneuron") -> logging.Logger:
    """Return a logger; the first call attaches a stdout handler and a per-run log file."""
    global _configured
    if mp.current_process().name != "MainProcess":  # spawned workers re-import modules: no per-worker log files
        return logging.getLogger(name)
    if not _configured:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        # One log file per process start, named by start time so runs never overwrite each other.
        logfile = LOGS_DIR / f"{datetime.now():%Y%m%d_%H%M%S}_{name.replace('.', '_')}.log"
        handlers = [logging.StreamHandler(sys.stdout), logging.FileHandler(logfile, encoding="utf-8")]
        # force=True replaces handlers that imported libraries may have installed on the root logger.
        logging.basicConfig(level=logging.INFO, format=_FORMAT, handlers=handlers, force=True)
        _configured = True
    return logging.getLogger(name)
