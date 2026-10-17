"""Lock-free-in-spirit ring buffer for the OPEN-ITEMS-NODE1 investigation (item 1, item 2).
No threading.Lock is used anywhere in this module. list.append() and list-reference
reassignment are both single bytecode-level operations that run to completion without a
context switch under CPython's GIL, so concurrent appends from multiple threads in the
same process cannot interleave mid-append the way they could with an unlocked C struct in a
language without a GIL -- this is "lock-free" in the sense asked for (no explicit lock
object, no blocking acquire on the hot path a kvtimer/iter call takes), not a claim that
Python itself has no serialization at all. A daemon thread flushes the buffer to a per-pid
file every FLUSH_INTERVAL_S seconds (so results are retrievable via `kubectl exec cat` without
needing to kill the pod to force an atexit handler to run); atexit and SIGTERM handlers are
also registered as a final best-effort flush.
"""
import atexit
import json
import os
import signal
import sys
import threading
import time

_buf = []
_lock_free_marker = True  # no threading.Lock object anywhere in this module, by design
MAX_ENTRIES = 4_000_000
FLUSH_INTERVAL_S = 1.0
_PATH = f"/tmp/aim345_ring_{os.getpid()}.jsonl"
# time.perf_counter() has an arbitrary, per-process origin -- not wall-clock epoch -- so a bare
# "ts" value cannot by itself be bucketed into a seed's own epoch window (windows.txt). Record
# one origin pairing of both clocks at import time; downstream analysis converts
# epoch = ORIGIN_EPOCH + (ts - ORIGIN_PERF) for every later entry.
ORIGIN_PERF = time.perf_counter()
ORIGIN_EPOCH = time.time()


def rec(event, room=-1, ts=None, **kw):
    if len(_buf) < MAX_ENTRIES:
        entry = {"event": event, "room": room, "ts": ts if ts is not None else time.perf_counter()}
        if kw:
            entry.update(kw)
        _buf.append(entry)


def _flush_once():
    global _buf
    if not _buf:
        return
    # Swap the list reference rather than mutating in place -- a single STORE_FAST/STORE_GLOBAL,
    # so any append() racing this swap either lands fully in the old list (and gets flushed this
    # round) or fully in the new one (and gets flushed next round); no explicit lock needed.
    pending, _buf = _buf, []
    try:
        with open(_PATH, "a") as f:
            for entry in pending:
                f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _flush_loop():
    while True:
        time.sleep(FLUSH_INTERVAL_S)
        _flush_once()


def _dump_and_exit(*_):
    _flush_once()
    os._exit(0)


_thread = threading.Thread(target=_flush_loop, name="aim345-ring-flush", daemon=True)
_thread.start()
atexit.register(_flush_once)
try:
    signal.signal(signal.SIGTERM, _dump_and_exit)
except Exception:
    pass

# Emit the clock-origin pairing as the first real record too, so a reader of the dumped file
# alone (no separate side-channel) can still recover it without reaching into this module's
# globals from outside the process.
rec("CLOCK_ORIGIN", -1, ts=ORIGIN_PERF, origin_epoch=ORIGIN_EPOCH, pid=os.getpid())
