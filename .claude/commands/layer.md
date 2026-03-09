# NEMS Layer Development Guide

You are helping develop a new `Layer` subclass for the NEMS library. Read the
relevant source files below before making changes, then apply the contracts and
conventions documented here.

## Key source files

- `nems/layers/base.py` — `Layer`, `Phi`, `Parameter` base classes
- `nems/layers/filter.py` — `FiniteImpulseResponse`, `STRF` (multi-parameter examples)
- `nems/layers/weight_channels.py` — `WeightChannels` (clean reference implementation)
- `nems/layers/nonlinearity.py` — `StaticNonlinearity`, activation layers
- `nems/registry.py` — `@layer` decorator and `KeywordRegistry`
- `nems/models/base.py` — `Model.evaluate`, `Model._evaluate_layer`
- `nems/backends/tf/backend.py` — `TensorFlowBackend._build`
- `nems/backends/tf/layer_tools.py` — `NemsKerasLayer` base class

---

## Method contracts

### `__init__`

```python
class MyLayer(Layer):
    def __init__(self, my_arg=default, **kwargs):
        self.my_attr = my_arg                        # 1. store custom attrs first
        require_shape(self, kwargs, minimum_ndim=2)  # 2. validate/extract shape
        super().__init__(**kwargs)                   # 3. call super (triggers initial_parameters)
        # 4. optional: modify priors/bounds after super (self.parameters now exists)
```

- `require_shape` **must** be called before `super().__init__()` — it extracts
  `kwargs['shape']` and sets `self.shape`, which `initial_parameters` needs.
- Custom attributes used inside `initial_parameters` **must** also be set before
  `super().__init__()`.
- Accepted kwargs passed through to `Layer.__init__`: `input`, `output`,
  `parameters`, `priors`, `bounds`, `name`, `regularizer`.

---

### `initial_parameters`

Returns a `Phi` wrapping one or more `Parameter` objects. Called automatically
by `Layer.__init__`.

```python
from nems.layers.base import Phi, Parameter
from nems.distributions import Normal, HalfNormal

def initial_parameters(self):
    shape = self.shape          # set by require_shape in __init__
    prior = Normal(mean=np.zeros(shape), sd=np.ones(shape))
    param = Parameter(name='coefficients', shape=shape, prior=prior,
                      bounds=(-np.inf, np.inf))
    return Phi(param)
```

**`Parameter` signature:**
```python
Parameter(
    name='coefficients',        # unique within the Layer
    shape=(T, N),               # tuple of ints
    prior=Normal(...),          # nems.distributions.Distribution
    bounds=(-np.inf, np.inf),   # (lower, upper) scalars; used by scipy fitter
    zero_to_epsilon=False,      # replace 0-bounds with float32 epsilon
    initial_value='mean',       # 'mean' | 'sample' | scalar | ndarray
)
```

**Gotchas:**
- Return `Phi(param)`, not a bare `Parameter` — Phi is the container.
- `prior.shape` must equal `Parameter.shape` (use `np.zeros(shape)` for array priors).
- Multiple parameters: `Phi(param_a, param_b)`.

---

### `evaluate`

The forward pass. Called by `Model.evaluate` for each layer in sequence.

```python
def evaluate(self, input):
    # input: np.ndarray shaped (T, N) — time × channels
    output = np.tensordot(input, self.coefficients, axes=(1, 0))
    return output  # shape (T, M)
```

**Data shape convention:** `(T, N)` where T = time bins, N = channels.
With samples: `(S, T, N)`. Never `(N, T)`.

**How `Model.evaluate` feeds this method:**
1. It maintains a `data_dict` that starts with the raw input arrays.
2. For each layer, `layer.data_map.get_inputs(data_dict)` resolves
   `Layer.input` to `(args, kwargs)` passed to `evaluate`.
3. If `Layer.input is None`, the previous layer's output is passed.
4. The return value is stored back in `data_dict` under `Layer.output`.
5. A returned `(T,)` array is auto-promoted to `(T, 1)`.

**Gotchas:**
- Never modify input arrays in-place — other layers may use the same data.
- Return `np.ndarray` or a list/tuple of `np.ndarray`s (one per output key).
- If `Layer.output` is a list, return a matching list of arrays.

---

### `from_keyword`

Static constructor registered with the global `KeywordRegistry`.

```python
from nems.registry import layer
from nems.layers.tools import require_shape, pop_shape

@layer('mykey')
@staticmethod
def from_keyword(keyword):
    options = keyword.split('.')   # e.g. ['mykey', '18x4', 'n', 'l2d3']
    shape = pop_shape(options)     # extracts '18x4' → (18, 4), mutates list
    kwargs = {'shape': shape}
    for op in options[1:]:
        if op == 'n':
            kwargs['normalize'] = True
        elif op.startswith('s'):
            kwargs['stride'] = int(op[1:])
        elif op.startswith('l2'):
            kwargs['regularizer'] = op
    return MyLayer(**kwargs)
```

**Rules:**
- Decorated with `@layer('head')` where `head` is an alphanumeric string.
- Must be a `@staticmethod` (no `self`).
- Receives the full keyword string: `'mykey.18x4.option'`.
- Must return a Layer instance.
- Registered in `nems.registry.keyword_lib`; enables `Model.from_keywords('...-mykey.18x4-...')`.

---

### `as_tensorflow_layer`

Returns a `NemsKerasLayer` subclass that mirrors `evaluate` using TensorFlow ops.
Only required when `backend='tf'` support is needed.

```python
def as_tensorflow_layer(self, input_shape, **kwargs):
    import tensorflow as tf                          # always import inside method
    from nems.backends.tf import NemsKerasLayer

    class MyLayerTF(NemsKerasLayer):
        def call(self, inputs):
            # self.coefficients is a tf.Variable (auto-registered by NemsKerasLayer)
            return tf.tensordot(inputs, self.coefficients, axes=[[2], [0]])

    return MyLayerTF(self, **kwargs)
```

**How the TF backend calls this:**

`TensorFlowBackend._build` iterates over model layers:
```python
input_shape = tuple(layer_inputs.shape)       # KerasTensor shape
tf_layer = nems_layer.as_tensorflow_layer(input_shape, ...)
last_output = tf_layer(layer_inputs)          # Keras symbolic call → traces call()
```

`NemsKerasLayer.__init__` auto-converts every NEMS `Parameter` into a
`tf.Variable` and attaches it as `self.<param_name>`. So if `initial_parameters`
defines `'coefficients'`, then `self.coefficients` is a `tf.Variable` inside
`call`.

**Passing state into the inner class:**

Capture numpy values or scalars in the closure before the class definition:

```python
def as_tensorflow_layer(self, input_shape, **kwargs):
    import tensorflow as tf
    from nems.backends.tf import NemsKerasLayer

    normalize = self.normalize          # capture in closure
    n_outputs  = input_shape[-1]

    class MyLayerTF(NemsKerasLayer):
        def call(self, inputs):
            if normalize:               # closure variable, not self.normalize
                ...
    return MyLayerTF(self, **kwargs)
```

**Keras symbolic-tracing gotchas:**

During model build, `call` is traced with `KerasTensor` (not real data).
Operations that call `.numpy()` or `__array__()` will raise:
> `NotImplementedError: numpy() is only available when eager execution is enabled.`

- ❌ `self.coefficients.numpy()` — fails during tracing
- ❌ Passing a `tf.Variable` directly to a `@tf.function` — triggers `__array__()`
- ✅ `tf.convert_to_tensor(self.coefficients, dtype=tf.float32)` — safe
- ✅ `tf.einsum(subscript, inputs, self.coefficients)` — safe
- ✅ `tf.tensordot(inputs, self.coefficients, axes=...)` — safe when `axes` is a Python list

**NumPy → TensorFlow op mapping:**

| NumPy                            | TensorFlow equivalent                          |
|----------------------------------|------------------------------------------------|
| `np.tensordot(a, b, axes)`       | `tf.tensordot(a, b, axes=axes)`                |
| `np.einsum('ij,jk->ik', a, b)`   | `tf.einsum('ij,jk->ik', a, b)`                 |
| `np.flip(a, axis=0)`             | `tf.reverse(a, axis=[0])`                      |
| `np.reshape(a, shape)`           | `tf.reshape(a, shape)` (`-1` for batch dim)    |
| `np.concatenate([a,b], axis=1)`  | `tf.concat([a, b], axis=1)`                    |
| `np.pad(a, paddings)`            | `tf.pad(a, paddings)`                          |
| `np.where(cond, x, y)`           | `tf.where(cond, x, y)`                         |
| `np.maximum(a, 0)`               | `tf.nn.relu(a)`                                |
| `a + b` (broadcast)              | same syntax, but both must be TF tensors       |

---

## Minimal complete example

```python
import numpy as np
from nems.layers.base import Layer, Phi, Parameter
from nems.layers.tools import require_shape, pop_shape
from nems.registry import layer
from nems.distributions import Normal


class Gain(Layer):
    """Element-wise scaling layer: output = input * gain."""

    def __init__(self, **kwargs):
        require_shape(self, kwargs, minimum_ndim=1, maximum_ndim=1)
        super().__init__(**kwargs)

    def initial_parameters(self):
        n = self.shape[0]
        prior = Normal(mean=np.ones(n), sd=np.ones(n))
        return Phi(Parameter('gain', shape=(n,), prior=prior,
                             bounds=(0, np.inf)))

    def evaluate(self, input):
        # input: (T, N), self.gain: (N,)
        return input * self.gain   # broadcast over T

    @layer('gain')
    @staticmethod
    def from_keyword(keyword):
        options = keyword.split('.')
        shape = pop_shape(options)
        return Gain(shape=shape)

    def as_tensorflow_layer(self, input_shape, **kwargs):
        import tensorflow as tf
        from nems.backends.tf import NemsKerasLayer

        class GainTF(NemsKerasLayer):
            def call(self, inputs):
                return inputs * self.gain   # tf.Variable broadcast

        return GainTF(self, **kwargs)
```

Invoke via keyword string: `Model.from_keywords('wc.18x4-gain.4-dexp.1')`

---

## Common failure modes

| Symptom | Likely cause |
|---------|-------------|
| `AttributeError: 'MyLayer' has no attribute 'shape'` | `require_shape` called after `super().__init__`, or omitted |
| `TypeError` wrapping Phi | Returning a bare `Parameter` instead of `Phi(param)` |
| Shape mismatch in subsequent layers | `evaluate` returning `(T,)` instead of `(T, N)` |
| Silent wrong results in multi-layer models | Modifying input array in-place |
| `KeywordMissingError` | Missing `@layer('head')` decorator or wrong head string |
| `NotImplementedError: numpy() only in eager` | Passing `tf.Variable` to `@tf.function` or calling `.numpy()` in `call()` |
| Gradients missing for a parameter in TF | Parameter name mismatch between `initial_parameters` and `call` usage |
| `input_shape` key in kwargs warning | `as_tensorflow_layer` passing `input_shape` to `NemsKerasLayer` — it is popped automatically |
