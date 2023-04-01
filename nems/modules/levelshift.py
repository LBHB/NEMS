import numpy as np
import re

from nems0.modules import NemsModule
from nems0.registry import xmodule


def levelshift(rec, i, o, level, **kwargs):
    '''
    Parameters
    ----------
    level : a scalar to add to every element of the input signal.
    '''
    fn = lambda x: x + level
    return [rec[i].transform(fn, o)]

class levelshift_new(NemsModule):
    """
    Add a constant to a NEMS signal
    """
    def __init__(self, **options):
        """
        set options to defaults if not supplied. pass to super() to add to data_dict
        """
        options['fn'] = options.get('fn', str(self.__module__) + '.levelshift_new')
        options['fn_kwargs'] = options.get('fn_kwargs', {'i': 'pred', 'o': 'pred'})
        options['plot_fns'] = options.get('plot_fns',
                                             ['nems0.plots.api.mod_output',
                                              'nems0.plots.api.spectrogram_output',
                                              'nems0.plots.api.pred_resp'])
        options['plot_fn_idx'] = options.get('plot_fn_idx', 2)
        options['prior'] = options.get('prior', {'level': ('Normal', {'mean': np.zeros([1, 1]),
                                           'sd': np.ones([1, 1])})})
        options['bounds'] = options.get('bounds', {})
        super().__init__(**options)

    def description(self):
        """
        String description (include phi values?)
        """
        return "Add a constant to each of N channels"

    @xmodule('lvl2')
    def keyword(kw):
        '''
        Generate and register default modulespec for the levelshift module.

        Parameters
        ----------
        kw : str
            Expected format: r'^lvl\.(\d{1,})$'

        Options
        -------
        None
        '''
        options = kw.split('.')
        required = '.'.join(options[:2])
        pattern = re.compile(r'^lvl2\.?(\d{1,})$')
        parsed = re.match(pattern, required)
        try:
            n_shifts = int(parsed.group(1))
        except TypeError:
            raise ValueError("Got a TypeError when parsing lvl keyword, "
                             "make sure keyword has the form: \n"
                             "lvl.{n_shifts}.\n"
                             "keyword given: %s" % kw)

        return levelshift_new(prior={'level': ('Normal', {'mean': np.zeros([n_shifts, 1]),
                                                          'sd': np.ones([n_shifts, 1])})})

    def eval(self, rec, i, o, level, **kw_args):
        '''
        Parameters
        ----------
        level : a scalar to add to every element of the input signal.
        '''
        fn = lambda x: x + level
        return [rec[i].transform(fn, o)]


    def tflayer(self):
        """
        layer definition for TF spec
        """
        #import tf-relevant code only here, to avoid dependency
        return []
