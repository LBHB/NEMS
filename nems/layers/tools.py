"""Misc. tools for supporting Layer implementations."""


def pop_shape(keyword_options):
    """Parse first keyword option that begins with a digit as a shape.

    Parameters
    ----------
    keyword_options : list of str.
        Result of `keyword.split('.').

    Returns
    -------
    shape : tuple or None

    Examples
    --------
    >>> keyword = 'test.5'
    >>> pop_shape(keyword.split('.'))
    (5,)

    >>> keyword = 'test.more.1x2x3'
    >>> pop_shape(keyword.split('.'))
    (1,2,3)

    >>> keyword = 'test.3.2.1'
    >>> pop_shape(keyword.split('.'))
    (3,)
    
    """

    for i, op in enumerate(keyword_options):
        if (op[0].isdigit()):
            dims = op.split('x')
            shape = tuple([int(d) for d in dims])
            break
    else:
        shape = None
    
    if shape is not None: _ = keyword_options.pop(i)

    return shape


def require_shape(layer, kwargs, minimum_ndim=0, maximum_ndim=None):
    """Raise `TypeError` if `shape` is not specified."""

    shape = kwargs.get('shape', None)
    layer_name = type(layer).__name__
    message = None

    if not isinstance(shape, (tuple, list)):
        message = f'Layer {layer_name} requires a `shape` kwarg.'
    elif len(shape) < minimum_ndim:
        message = (f'Layer {layer_name}.shape must have at least '
                   f'{minimum_ndim} dimensions.')
    elif (maximum_ndim is not None) and (len(shape) > maximum_ndim):
        message = (f'Layer {layer_name}.shape must have no more than '
                   f'{maximum_ndim} dimensions.')

    if message is not None:
        raise TypeError(message)
