"""Computational frameworks for fitting Models.

To include: SciPy (default), TensorFlow, ... (others)

NOTE: There are no public API imports here so that backend dependencies are
      only imported if used. For example, users don't need to install TensorFlow
      unless they want to fit models using `Model.fit(..., backend='tf')`.

"""

# TODO: better way to do this. For now, new backends just have to add on to
#       this if/else chain.
def get_backend(name):
    """Import and return a Backend subclass corresponding to `name`."""
    name = name.lower()
    if name == 'scipy':
        from .scipy import SciPyBackend
        backend = SciPyBackend
    elif (name == 'tf') or (name == 'tensorflow'):
        from .tf import TensorFlowBackend
        backend = TensorFlowBackend
    else:
        raise NotImplementedError(f"Unrecognized backend: {name}.")

    return backend
