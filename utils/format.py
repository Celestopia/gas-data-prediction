import argparse
import json
import numpy as np


def dict2namespace(config):
    """Recursively convert a dictionary to a argparse.Namespace object."""
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def namespace2dict(ns):
    """Convert a Namespace object to a regular dictionary."""
    if isinstance(ns, argparse.Namespace):
        d = vars(ns)
        return {k: namespace2dict(v) for k, v in d.items()}
    elif isinstance(ns, dict):
        return {k: namespace2dict(v) for k, v in ns.items()}
    elif isinstance(ns, list):
        return [namespace2dict(x) for x in ns]
    else:
        return ns


def get_namespace_value(namespace, keys):
    """Get the value of a nested namespace with a series of keys."""
    value = namespace
    for key in keys:
        value = getattr(value, key)
    return value


def set_namespace_value(namespace, keys, value):
    """Set the value of a nested namespace with a series of keys."""
    for key in keys[:-1]:
        namespace = getattr(namespace, key)
    setattr(namespace, keys[-1], value)
    return namespace


class NumpyEncoder(json.JSONEncoder):
    """Customized json encoder for numpy array data."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist() # Convert to list for json serialization.
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return json.JSONEncoder.default(self, obj)
