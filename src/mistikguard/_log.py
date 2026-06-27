"""Internal logging shim for Mistikguard, backed by stdlib logging.

Consumers configure verbosity their own way:
    import logging
    logging.getLogger("mistikguard").setLevel(logging.DEBUG)
"""
import logging

_log = logging.getLogger("mistikguard")


def dprint(*args, **kwargs):
    """Debug-level log. Accepts print-style args."""
    if _log.isEnabledFor(logging.DEBUG):
        msg = " ".join(str(a) for a in args)
        _log.debug(msg)


def warn(source, message):
    """Warning-level log. Replaces the old mistik_health.record_error path."""
    _log.warning("[%s] %s", source, message)
