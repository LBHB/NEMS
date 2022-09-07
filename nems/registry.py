# TODO: should this go in nems.layers directory instead?
#       or maybe nems.tools?

import logging
import inspect
import re

log = logging.getLogger(__name__)


class KeywordMissingError(Exception):
    """Raised when a keyword lookup fails."""
    def __init__(self, value):
        default_message = f'"{value}" not found in keyword registry.'
        super().__init__(default_message)


class Keyword:
    """Maps a Layer to a string representation.
    
    See also
    --------
    For additional information on keyword formatting:
    nems.layers.base.Layer.from_keyword

    """

    def __init__(self, key, parse, source_string=None):
        """
        Parameters
        ----------
        key : str
            A short-hand description of a Layer with a particular set of
            options.
        parse : callable
            Function that parses the full keyword string. In most cases this
            should be the `from_keyword` method of the relevant Layer.
        source_string : str, optional
            Location (i.e., python layer or file) where `parse` was defined.

        Example
        -------
        `key = wc.g` would map to WeightChannels(parameterization='gaussian')

        """
        self.key = key
        self.parse = parse
        self.source_string = source_string if source_string is not None else ''

    def __repr__(self):
        return str(self.parse(self.key))



class KeywordRegistry:
    '''A collection of Keywords registered by the `@layer` decorator.

    A `KeywordRegistry` behaves similar to a dictionary, except that
    `KeywordRegistry[key_string]` will trigger a function call with `key_string`
    as the only argument. The function to call is specified by the Keyword
    instance referenced by the leading portion of the key: either before the
    first period or until the first non-alpha character.
    
    Example
    -------
        def simple_keyword(kw):
            return kw.split('.')
        kws = KeywordRegistry(name='test')
        kws['mykey'] = simple_keyword

        In[0]: kws['mykey.test']
        Out[0]: ['mykey', 'test']

    '''

    def __init__(self, name='registry', warn_on_overwrite=True,
                 set_obj_name=False):
        """
        Parameters
        ----------
        name : str
            Just a name to distinguish the registry's purpose, incase multiple
            registries are defined.
        warn_on_overwrite : bool
            If True, raise a RuntimeWarning when a keyword is overwritten to
            help debug name clashes.
        set_obj_name : bool
            If true, set `obj.name = kw_head` during indexing.

        """
        self.keywords = {}
        self.list = []
        self.name = name
        self.warn_on_overwrite = warn_on_overwrite
        self.set_obj_name = set_obj_name

    def __getitem__(self, kw_string):
        kw = self.lookup(kw_string)
        obj = kw.parse(kw_string)

        if (self.set_obj_name):
            # Type check is done this way to avoid circular import issues.
            inheritance_list = inspect.getmro(type(obj))
            checks = [cls.__name__.split('.')[-1] != 'Layer'
                      for cls in inheritance_list]
            if all(checks):
                raise TypeError(
                    '@layer functions must return a Layer instance.'
                    )
            if obj._name is None:
                kw_head = kw_string.split('.')[0]
                obj._name = kw_head

        return obj

    def __setitem__(self, kw_head, parse, source=None):
        if kw_head in self.keywords and self.warn_on_overwrite:
            raise RuntimeWarning(
                f'Keyword: {kw_head} overwritten in registry: {self.name}'
                )
        self.keywords[kw_head] = Keyword(kw_head, parse, source_string=source)
        self.list.append(f'{kw_head}:    {source}')

    def kw_head(self, kw_string):
        """Identifies the leading portion of a keyword string.
        
        Parameters
        ----------
        kw_string : str

        Returns
        -------
        kw_head : str
            Substring of `kw_string` that comes before the first period. If no
            period is present, this will be the first substring of  alpha-only
            characters instead.
        
        """

        # if the full kw_string is in the registry as-is, then it's a
        # backwards-compatibility alias and overrides the normal kw head rule.
        if kw_string in self.keywords:
            return kw_string
        # Look for '.' first. If not present, use first alpha-only string
        # as head instead.
        h = kw_string.split('.')
        if len(h) == 1:
            # no period, do regex for first all-alpha string
            alpha = re.compile('^[a-zA-Z]*')
            kw_head = re.match(alpha, kw_string).group(0)
        else:
            kw_head = h[0]
        return kw_head

    def lookup(self, kw_string):
        """Find `kw_string` in the registry using the leading portion.
        
        See also
        --------
        `KeywordRegistry.kw_head`
        
        """
        kw_head = self.kw_head(kw_string)
        if kw_head not in self.keywords:
            raise KeywordMissingError(kw_head)

        return self.keywords[kw_head]

    def source(self, kw_string):
        """Fetch the definition location for `kw_string`."""
        return self.lookup(kw_string).source_string

    def register_function(self, obj, name=None, obj_type=None):
        """Add function `obj` to the registry.
        
        Parameters
        ----------
        obj : callable
        name : str; optional.
            Name of KeyWord to store in registry. Defaults to `obj.__name__`.
        obj_type : str; optional.
            If specified, raise TypeError if `type(obj).__name__` does not match
            `obj_type`. String specification is used to avoid circular imports
            for decorators.

        """

        if name is None:
            name = obj.__name__
        log.debug("%s lib registering function: %s", self.name, name)
        
        try:
            # Default to module rather than file, to maintain portability.
            location = str(obj.__module__) + "." + obj.__name__
        except AttributeError:
            location = str(obj.__name__)
            
        self.keywords[name] = Keyword(name, obj, location)
        self.list.append(f'{name}:    {location}')

    def __iter__(self):
        """Iterate over registered keywords."""
        return self.keywords.__iter__()

    def __next__(self):
        """Iterate over registered keywords."""
        return self.keywords.__next__()

    def info(self, kw=None):
        """Look up sources of registered keyword(s).

        Parameters
        ----------
        kw : str or None

        Returns
        -------
        dict
            Contains source information for keywords containing string `kw`,
            or all keywords if `kw` is None.
    
        """
        kw_list = list(self.keywords.keys())
        s = {}
        for k in kw_list:
            if (kw is None) or kw in k:
                s[k] = self.source(k)
        return s


# Create registries here so they can be updated as new layers are imported.
keyword_lib = KeywordRegistry(
    name="layers", warn_on_overwrite=True, set_obj_name=True
    )

def layer(name=None):
    """Decorator for `Layer.from_keyword()` methods.
    
    Returns
    -------
    decorator : function
        Registers a decorated function in `keyword_lib`.

    Examples
    --------
    >>> @layer('cool')
    >>> def from_keyword(keyword):
    ...     # parse keyword into some Layer options
    ...     option1, option2 = my_kw_parser(keyword)
    ...     return CoolLayer(option1, option2)
    
    """
    def decorator(func):
        # If func has an underlying __func__, use that instead to avoid
        # error from staticmethod not having __module__ or __name__ attributes.
        func = getattr(func, '__func__', func)

        # Raise error if func doesn't accept any arguments.
        try:
            args = inspect.getfullargspec(func)
            _ = args[0]
        except TypeError:
            raise TypeError(
                f'Keyword {name} must accept a keyword string '
                'as its first argument.'
            )

        # If there aren't any syntax errors, register the keyword.
        keyword_lib.register_function(func, name=name, obj_type='Layer')
        return func

    return decorator
