import ast

import numpy as np

from nems.registry import layer
from .base import Layer

class NumPy(Layer):
    """Applies a NumPy operation to inputs.
    
    Parameters
    ----------
    operation : str
        Name of `np.{operation}` to apply to data, ex: 'concatenate'.
    op_args : N-tuple; optional.
        Positional arguments to supply to `np.{operation}`. These do not
        include arrays supplied through `inputs` in `evaluate`.
    op_kwargs : dict; optional.
        Keyword arguments to supply to `np.{operation}`.
    input, output, name : see docs for `Layer`.
    
    Returns
    -------
    ndarray or list of ndarray (see docs for {operation} at numpy.org).

    Notes
    -----
    Unlike most Layer subclasses, NumPy Layers will not pass additional
    keyword arguments to `Layer.__init__`. Instead, they are passed to
    `np.{concatenate}`. Additionally, note that `parameters`, `bounds`,
    `priors`, and `shape` cannot be set through initialization and have no
    effect since NumPy Layers do not have any fittable parameters.

    Examples
    --------
    >>> x = np.arange(8).reshape(2,4)
    >>> x
    array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7]])
    >>> split = NumPy('split', 2, axis=0)
    >>> split.evaluate(x)
    [array([[0, 1, 2, 3]]), array([[4, 5, 6, 7]])]

    """

    def __init__(self, operation, *op_args, input=None, output=None, name=None,
                 **op_kwargs):

        if name is None:
            name = f'np.{operation}'
        super().__init__(input=input, output=output, name=name)

        self._operation = getattr(np, operation)
        self._args = op_args
        op_kwargs = {} if op_kwargs is None else op_kwargs
        self._kwargs = op_kwargs

    def evaluate(self, *inputs):
        """Apply `np.{operation}` to inputs."""
        return self._operation(*inputs, *self._args, **self._kwargs)

    @layer('np')
    def from_keyword(keyword):
        """Construct NumPy Layer from keyword.

        Keyword options
        ---------------
        {operation} : Name of numpy operation to use, ex: 'split', 'sum'.
            NOTE: This must be the first option, ex: `np.split.op2.op3...`
        {value} : Positional arguments for `np.{operation}`.
            Only literal, non-alpha value types are supported, such as `int`,
            `tuple`, or `float` (see docs for `ast.literal_eval`). For an alpha
            string argument (ex: 'full', 'same'), instantiate directly from
            the NumPy Layer class.
        {key}{value} : Keyword arguments for `np.{operation}`.
            Only keys that are all alpha characters (like `axis`, `shape`, etc)
            are supported. For kwarg names with underscores, numeric characters,
            etc, instantiate the Layer directly.
        See also
        --------
        Layer.from_keyword
        
        """
        # TODO: want to be able to do 'np.concatenate.axis1'
        options = keyword.split('.')
        operation = options[1]  # NOTE: this is hard-coded

        args = []
        kwargs = {}
        for op in options[2:]:
            key = ''
            while op[0].isalpha():
                key += op[0]
                op = op[1:]
            val = ast.literal_eval(op)
            if key == '':
                # treat as positional argument
                args.append(val)
            else:
                # keyword argument
                kwargs[key] = val

        return NumPy(operation, *args, **kwargs)
