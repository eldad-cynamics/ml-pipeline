import logging.config
import traceback
import os
from .logstruct import Logstruct
from logzio.flusher import LogzioFlusher

_ = Logstruct

logzio_cred = os.environ['LOGZIO_CRED'] if 'LOGZIO_CRED' in os.environ else None
if logzio_cred is not None:
    print('running with logzio logger')
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'logzioFormat': {
                'validate': False
            }
        },
        'handlers': {
            'logzio': {
                'class': 'logzio.handler.LogzioHandler',
                'level': 'INFO',
                'formatter': 'logzioFormat',
                'token': logzio_cred,
                'logs_drain_timeout': 5,
                'url': 'https://listener.logz.io:8071',
                'backup_logs': False
            }
        },
        'loggers': {
            '': {
                'level': 'DEBUG',
                'handlers': ['logzio'],
                'propogate': True
            }
        }
    }
    logging.config.dictConfig(logging_config)
logger = logging.getLogger("pythonLogger")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console_handler)

event_id = None

@LogzioFlusher(logger)
def info(message, account_id, **kwargs):
    try:
        logger.info(_(message, 'info', account_id, eventId=event_id, **kwargs))
    except Exception as ex:
        print(f'issue with sending log. ex={ex}. stacktrace={str(traceback.format_exc())}')

@LogzioFlusher(logger)
def warning(message, account_id, **kwargs):
    try:
        logger.warning(_(message, "warning", account_id, eventId=event_id, **kwargs))
    except Exception as ex:
        print(f'issue with sending log. ex={ex}')

@LogzioFlusher(logger)
def error(message, account_id, stackTrace, **kwargs):
    try:
        logger.error(
            _(message, 'error', account_id, str(traceback.format_exc()), eventId=event_id,
              **kwargs))
    except Exception as ex:
        print(f'issue with sending log. ex={ex}. stacktrace={str(traceback.format_exc())}')

@LogzioFlusher(logger)
def fatal(message, **kwargs):
    try:
        logger.fatal(_(message, "fatal", eventId=event_id, **kwargs))
    except Exception as ex:
        print(f'issue with sending log. ex={ex}')

@LogzioFlusher(logger)
def debug(message, account_id, **kwargs):
    try:
        logger.debug(_(message, 'debug', account_id, eventId=event_id, **kwargs))
    except Exception as ex:
        print(f'issue with sending log. ex={ex}')


def set_event_id(new_event_id=None):
    global event_id
    event_id = new_event_id

