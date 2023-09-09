import logging, datetime
import logging.config

# Folder to store log file in
NEMS_LOG_ROOT = '/tmp/nems'

# Filename to save log file in
NEMS_LOG_FILENAME = datetime.datetime.now().strftime('NEMS %Y-%m-%d %H%M%S.log')

# Format for messages saved to file
NEMS_LOG_FILE_FORMAT = '[%(relativeCreated)d %(thread)d %(name)s - %(levelname)s] %(message)s'

# Format for messages printed to console
NEMS_LOG_CONSOLE_FORMAT = '[%(name)s %(levelname)s] %(message)s'

# Logging level for file
NEMS_LOG_FILE_LEVEL = 'DEBUG'

# Logging level for console
NEMS_LOG_CONSOLE_LEVEL = 'DEBUG'

# if logging is already set up, don't set it up again.
x = logging.getLogger('root')
if len(x.handlers)==0:
    config = {
        'version': 1,
        'formatters': {
            'file': {'format': NEMS_LOG_FILE_FORMAT},
            'console': {'format': NEMS_LOG_CONSOLE_FORMAT},
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'formatter': 'console',
                'level': NEMS_LOG_CONSOLE_LEVEL,
            },
        },
        'loggers': {
            '__main__': {'level': 'INFO'},
            '': {'level': 'INFO'},
            'nems_db': {'level': 'INFO'},
            'nems': {'level': 'INFO'},
            'nems0.analysis.fit_basic': {'level': 'INFO'},
            'fontTools': {'level': 'WARNING'}
        },
        'root': {
            'handlers': ['console'],
        },
    }
    logging.config.dictConfig(config)

