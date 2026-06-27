"""Atomic JSON persistence helpers for Mistikguard.

Pure standard library — no external dependencies. Writes are atomic
(temp file + fsync + os.replace) so a crash mid-write cannot corrupt
the target file.
"""
import json
import os
import time


def safe_load_json(path, default):
    for attempt in range(3):
        try:
            if os.path.exists(path):
                with open(path, "r") as f:
                    return json.load(f)
            return default
        except Exception:
            if attempt < 2:
                time.sleep(0.12)
                continue
    return default


def safe_save_json(path, data):
    tmp = None
    try:
        tmp = f"{path}.tmp.{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        return True
    except Exception:
        try:
            if tmp and os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        return False
