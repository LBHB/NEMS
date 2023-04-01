from os.path import dirname, basename, isfile, join
import glob
import importlib
import logging

from nems0.registry import xform, xmodule

log = logging.getLogger(__name__)

class NemsModule(object):
    """
    NemsModule parent object

    Backward compatible with former module design, so everything important
    goes in self.data_dict. Also, the object acts like a dict in many ways
    (i.e., iterating, indexing, get/set)

    Four methods that will likely be over-loaded for a child class:
    - description
    - keyword
    - eval
    - tflayer
    """

    def __init__(self, **options):
        """
        options is dictionary in the format of old module dictionary.

        Standard properties:
        fn
        fn_kwargs
        fn_coefficients - this only exists for some modules. Like if you specify weight channels with a gaussian or fir with a damped oscillator.
        plot_fns
        plot_fn_idx
        prior  -- make sure not to evaluate the distributions
        bounds
        phi

        """
        options['fn'] = options.get('fn',str(self.__module__) + '.NemsModule.eval')
        options['fn_kwargs'] = options.get('fn_kwargs', {'i': 'pred', 'o': 'pred'})
        options['fn_coefficients'] = options.get('fn_coefficients', None)
        options['plot_fns'] = options.get('plot_fns',
                                           ['nems0.plots.api.mod_output',
                                            'nems0.plots.api.spectrogram_output',
                                            'nems0.plots.api.pred_resp'])
        options['plot_fn_idx'] = options.get('plot_fn_idx', 2)
        options['prior'] = options.get('prior', {})
        options['bounds'] = options.get('bounds', {})
        options['phi'] = options.get('phi', None)

        #log.info(options)

        self.data_dict = options.copy()

    # begin generic properties. Unlikely to need to over-load for a specific module
    def __eval__(self, **options):
        self.eval(**options)

    def __getitem__(self, key=None, default=None):
        """
        Return information from self.data_dict. Acts like a dictionary for
        backward compatibility with old module format

        :param key: key in data_dict
        :return: self.data_dict[key]
        :raises ValueError: Raised if `key` out of bounds or not one of the above.
        """
        try:
            if key is None:
                return self.data_dict
            else:
                return self.data_dict.get(key, default)
        except IndexError:
            raise ValueError('key %s not supported', key)

    def __setitem__(self, key, val):
        """
        Update the data_dict of the module.

        :param int key: data_dict key
        :param val: data_dict value
        :return: self, updated.
        :raises ValueError: If unable to set
        """
        try:
            # Try converting types like np.int64 instead of just
            # throwing an error.
            self.data_dict[key] = val
        except ValueError:
            raise ValueError('key {} not supported'.format(key))
        return self

    # act like a dictionary for iteration
    def __iter__(self):
        return iter(self.data_dict)

    def keys(self):
        return self.data_dict.keys()

    def items(self):
        return self.data_dict.items()

    def values(self):
        return self.data_dict.values()

    def __repr__(self):
        """
        Overloaded repr.

        :return str: string form of data_dict
        """
        return str(self.data_dict)

    def __str__(self):
        """
        Overloaded str.

        :return str: string form of data_dict
        """
        return str(self.data_dict)

    def get(self, key=None, default=None):
        return self.__getitem__(key, default)

    def set(self, key, val):
        return self.__setitem__(key, val)

    @property
    def phi(self):
        return self.data_dict['phi']

    @property
    def prior(self):
        return self.data_dict['prior']

    @property
    def bounds(self):
        return self.data_dict['bounds']

    # begin module-specific properties
    def description(self):
        """
        String description
        """
        return "Null module: Simple pass-through."

    @xmodule('null')
    def keyword(keyword):
        """
        Placeholder for keyword registry
        """
        return NemsModule()

    def eval(self, rec):
        """
        Placeholder. Null function returns empty signal list
        """
        return []

    def tflayer(self):
        """
        layer definition for TF spec
        """
        #import tf-relevant code only here, to avoid dependency
        return []

mods = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in mods if isfile(f) and not f.endswith('__init__.py')]
for a in __all__:
    importlib.import_module(__name__ + "." + a)
del mods
del a
