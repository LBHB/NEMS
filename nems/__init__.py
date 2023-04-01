import logging
import ast
import os

log = logging.getLogger(__name__)
NEMS_PATH = os.path.abspath(os.path.dirname(__file__) + '/..')


def load_config():
    # Load the default settings
    from os import environ, path, utime
    from .configs import defaults as config

    # leave defaults.py off the end of path
    configs_path = path.dirname(path.abspath(config.__file__))
    # import configs/settings.py for user-specified overrides.
    # If it doesn't exist, create a dummy file in its place that
    # the user can fill in later.
    try:
        from .configs import settings
    except ImportError:
        settings_path = path.join(configs_path, 'settings.py')
        # this should be equivalent to
        # `touch path/to/configs/settings.py`
        with open(settings_path, 'a'):
            utime(settings_path, None)
        log.info("No settings.py found in configs directory,"
                 " generating blank file ... ")
        try:
            from .configs import settings
        except ImportError:
            # if it still doesn't work, file wasn't created correctly
            # (known issue on MacOS), so just leave settings as a blank object
            log.info("Could not create settings.py in configs directory ...")
            settings = object()

    for s in config.__dir__():
        if s.startswith('__') or not (s == s.upper()):
            # Ignore python magic variables and any that are not in
            # all caps.
            pass
        elif s == s.upper():
            if s in environ:
                # If it's already in the environment, don't need
                # to do anything else.
                pass
            elif hasattr(settings, s):
                log.info("Found setting: %s in %s, adding "
                         "value to environment ... ", s, settings.__name__)
                d = getattr(settings, s)
                if d is None:
                    d = ''
                environ[s] = str(d)
            else:
                log.info("No value specified for: %s. Using default value "
                         "in %s", s, config.__name__)
                d = getattr(config, s)
                if d is None:
                    d = ''
                environ[s] = str(d)
            setattr(config, s, environ[s])

    config.init_settings()
    return config



_config = load_config()

def get_settings():
    return {k: getattr(_config, k) for k in dir(_config) if k == k.upper()}


def get_setting(setting=None):
    '''
    Get value of setting.
    '''
    s = getattr(_config, setting)
    # Necessary since environment variables can only hold strings,
    # but config settings some times need to be other types.
    # NOTE: Will not work for dictionaries, but tested fine so far
    #       with strings, lists, ints, floats, and booleans.
    try:
        # s is something other than a string
        s = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        # setting is just a string or empty list
        pass
    return s
