"""Extend JSON encoding/decoding to work with `numpy.ndarray` and NEMS objects.

When encoding `obj` with `name = type(obj).__name__`, if an `encode_{name}`
method is found then that method will be used. Otherwise, the json package's
default encoding method will be used (`json.JSONEncoder.default`).

Warnings
--------
Subclasses of NEMS base classes should avoid overwriting `to_json` and
`from_json` whenever possible. NEMSEncoder will use the `to_json` method of a
subclass if overwritten, but NEMSDecoder will only ever use the `from_json`
method of the base class, which could cause confusion and/or mismatch between
methods.

If a subclass truly needs customized to/from methods, there are two options:
1) Corresponding `encode_{SubClassName}` and `decode_{SubClassName}` methods
   should also be added to NEMSEncoder and NEMSDecoder, respectively, so that
   the correct `from_json` method can be found by the decoder. Alternatively,
   add the imported subclass to the top-level `_NEMS_classes_to_encode` variable.
2) The base class `from_json` method should make use of a the subclass name
   (either from the default '__nems_json_subtype__' key or a defined alias) to
   re-direct `BaseClass.from_json(json)` to `SubClass.from_json(json)`.
   See `nems.layers.base.Layer.from_json` for an example.

Notes
-----
With the exception of objects from external libraries like `np.ndarray`,
`encode_{name}` and `decode_{name}` methods should simply reference
`obj.to_json()` or `cls.from_json(obj)` (along with, of course, defining the
`to_json` and `from_json` methods for that object's base class). The simplest
way to accomplish this is to add the imported class to `_NEMS_classes_to_encode`.

This ensures that the JSON encoder/decoder code does not depend on details of
individual classes, as they may change. Similarly, individual classes don't need
to worry about encoding/decoding details: they just need to return/load from a
dictionary that is suitable for standard `json.dumps`, i.e. it contains only
strings, lists, ints, etc. Additionally, classes don't need to worry about
encoding of other nems objects. I.e. a Phi instance contains a dict with
Parameter instances; however, since both classes implement `to_json`,
`json.dumps` will happily handle the details of encoding all Parameters before
storing them in the encoded Phi instance.

References
----------
.. [1] https://mathspp.com/blog/custom-json-encoder-and-decoder

Examples
--------
To encode data stored in a `numpy.ndarray` as JSON:
>>> my_data = np.arange(1, 5)
>>> encoded_data = json.dumps(my_data, cls=NEMSEncoder)
>>> encoded_data
'{"data": [1, 2, 3, 4], "dtype": "int32", "shape": [4],
    "__nems_json_type__": "ndarray"}'

>>> decoded_data = json.loads(encoded_data, cls=NEMSDecoder)
>>> decoded_data
array([1, 2, 3, 4])

"""

import os
import re
import json
import inspect
from functools import partialmethod

import numpy as np

from nems.distributions.base import Distribution
from nems.layers.base import Layer, Phi, Parameter
from nems.models.base import Model


_NEMS_classes_to_encode = [
    Distribution, Layer, Phi, Parameter, Model
    ]


class NEMSEncoder(json.JSONEncoder):
    """An extended JSON encoder. See `help(nems.tools.json)` for docs."""

    def default(self, obj):
        # Iterate through object's class, immediate parent class(es), then
        # grandparent class(es), and so on. Stop when one of these class names
        # has a corresponding `encode_{name}` method.
        # (This means NEMS objects can subclassed as many times as desired, and
        #  this method will still find the correct base class).
        inheritance_list = inspect.getmro(type(obj))
        for cls in inheritance_list:
            encoder = getattr(self, f"encode_{cls.__name__}", None)
            if encoder is not None:
                encoded = encoder(obj)
                encoded["__nems_json_type__"] = cls.__name__
                encoded["__nems_json_subtype__"] = inheritance_list[0].__name__
                return encoded
        else:
            super().default(obj)

    def encode_ndarray(self, array):
        data = array.tolist()
        return {'data': data, 'dtype': str(array.dtype), 'shape': array.shape}


class NEMSDecoder(json.JSONDecoder):
    """An extended JSON decoder. See `help(nems.tools.json)` for docs."""

    def __init__(self, **kwargs):
        kwargs["object_hook"] = self.object_hook
        super().__init__(**kwargs)

    def object_hook(self, obj):
        try:
            name = obj["__nems_json_type__"]
            decoder = getattr(self, f"decode_{name}")
        except (KeyError, AttributeError):
            return obj
        else:
            return decoder(obj)

    def decode_ndarray(self, obj):
        return np.asarray(obj['data'], dtype=obj['dtype']).reshape(obj['shape'])


# Monkey-patch generic encode/decode methods for listed NEMS classes.
@staticmethod
def _generic_NEMS_encoder(obj):
    return obj.to_json()
@staticmethod
def _generic_NEMS_decoder(obj, cls=None):
    return cls.from_json(obj)

for cls in _NEMS_classes_to_encode:
    name = cls.__name__
    decoder = partialmethod(_generic_NEMS_decoder, cls=cls)
    setattr(NEMSEncoder, f'encode_{name}', _generic_NEMS_encoder)
    setattr(NEMSDecoder, f'decode_{name}', decoder)


# Convenience functions so that front-end users don't need to remember to use
# cls=NEMSEncoder/Decoder.
def nems_to_json(obj):
    """Encode `obj` as a json string. Supports ndarray and NEMS objects."""
    return json.dumps(obj, cls=NEMSEncoder)

def nems_from_json(obj):
    """Decode json `obj`. Supports ndarray and NEMS objects."""
    return json.loads(obj, cls=NEMSDecoder)


def save_model(model, filepath):
    """Save a Model to `filepath` as a json file.

    TODO: refactor using pathlib, options to use different permissions.

    Parameters
    ----------
    model: nems.models.base.Model
        NEMS model object to save.
    filepath : string
        Full path including file name and parent directories.
    
    Notes
    -----
    Currently this function is hard-coded to set broad permissions for
    the saved file.

    """
    data = nems_to_json(model)
    dirpath = os.path.dirname(filepath)

    if not os.path.exists(dirpath):
        os.makedirs(dirpath, mode=0o0777)
    with open(filepath, mode='w+') as f:
        f.write(data)
        f.close()
        try:
            os.chmod(filepath, 0o666)
        except PermissionError:
            # File should already exist with the correct permissions
            pass


def load_model(filepath):
    """Load a Model from a json file saved at `filepath`.

    Parameters
    ----------
    filepath : string
        Full path to the location of the saved Model, including file name.

    Returns
    -------
    model : nems.models.base.Model
        A NEMS Model object.

    """
    with open(filepath, mode='r') as f:
        data = f.read()

    model = nems_from_json(data)

    return model


def generate_model_filepath(model, basepath=""):
    """
    return filepath for saving data based on model.name, but filtered for
    invalid characters and truncated with a hash if the name is too long for
    the file system
    :param model: nems model
    :param basepath: path where save directory should be created
    :return: filepath: string <basepath>/<model.name>/model.json
    """
    guess = model.name

    # remove problematic characters
    guess = re.sub('[/]', '_', guess)
    guess = re.sub('[:]', '', guess)
    guess = re.sub('[,]', '', guess)

    if len(guess) > 100:
        # If modelname is too long, causes filesystem errors.
        guess = guess[:80] + '...' + str(hash(guess))
    return os.path.join(basepath, guess, 'model.json')
