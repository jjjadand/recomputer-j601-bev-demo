"""Encode frame payloads without spending the GPU's idle time on Python JSON loops."""
import json

try:
    import ujson
except ImportError:
    ujson = None


def dumps(value):
    if ujson is not None:
        return ujson.dumps(value, allow_nan=False, escape_forward_slashes=False).encode()
    return json.dumps(value, separators=(',', ':'), allow_nan=False).encode()


def loads(value):
    return ujson.loads(value) if ujson is not None else json.loads(value)
