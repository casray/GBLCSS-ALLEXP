"""Small runtime utilities shared by preparation and independent training arms."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
from pathlib import Path


def validation_due(epoch: int, final_epoch: int, interval: int) -> bool:
    return epoch % interval == 0 or epoch == final_epoch


class Checkpoints:
    def __init__(self, patience_checks: int):
        self.patience_checks = patience_checks
        self.primary = self.secondary = -float("inf")
        self.primary_state = self.secondary_state = None
        self.primary_epoch = self.secondary_epoch = None
        self.bad_checks = 0

    def observe(self, epoch, primary, secondary, state):
        if primary > self.primary:
            self.primary, self.primary_epoch = primary, epoch
            self.primary_state = copy.deepcopy(state)
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        if secondary is not None and secondary > self.secondary:
            self.secondary, self.secondary_epoch = secondary, epoch
            self.secondary_state = copy.deepcopy(state)

    @property
    def stop(self):
        return self.bad_checks >= self.patience_checks


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def array_hash(*arrays):
    import numpy as np
    h = hashlib.sha256()
    for array in arrays:
        a = np.ascontiguousarray(array)
        h.update(str((a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class Stage:
    def __init__(self, times, name, sync=None):
        self.times, self.name, self.sync = times, name, sync

    def __enter__(self):
        if self.sync:
            self.sync()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync:
            self.sync()
        self.times[self.name] = self.times.get(self.name, 0.0) + time.perf_counter() - self.start
