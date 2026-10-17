"""Atomic, NFS-safe file writes shared by labctl and runner_agent (DESIGN.md section 7.3).

Every writer in the in-cluster runner owns a disjoint set of filenames (labctl writes the shared
lifecycle files; each shard writes only its own shard-indexed files), so the only thing this module
has to guarantee is that one writer's own write is never observed half-written, and that a
retransmitted rename reporting failure after actually succeeding ("false rename failure") is not
mistaken for a real one.
"""
import errno
import json
import os
import socket
import uuid


def atomic_write(path, data):
    tmp = path.parent / f"{path.name}.tmp.{socket.gethostname()}.{uuid.uuid4().hex[:8]}"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        os.replace(tmp, path)
    except OSError as exc:
        if exc.errno in (errno.ENOENT, errno.EEXIST) and path.exists() and path.read_text() == data:
            return  # Retransmitted rename reported failure after the content already landed.
        raise


def atomic_write_json(path, obj):
    atomic_write(path, json.dumps(obj, indent=2))


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else None
