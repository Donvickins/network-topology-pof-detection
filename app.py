import sys

sys.dont_write_bytecode = True

import uvicorn
import os
import logging
from logging.config import dictConfig
from colorama import just_fix_windows_console
from core.utils.constants import HOST, PORT, BEARER_TOKEN

# Suppress Ultralytics' initial setup logs
os.environ['ULTRALYTICS_LOGGING'] = 'CRITICAL'

from core.utils.logger_config import LOG_CONFIG
from core.utils.logger_config import ensure_logs_dir

if __name__ == '__main__':
    if sys.platform == 'win32':
        just_fix_windows_console()
    
    ensure_logs_dir()  
    dictConfig(LOG_CONFIG)
    
    def handle_uncaught_exception(exc_type, exc_value, exc_traceback):
        """
        Global handler for uncaught exceptions that logs the exception
        and then calls the default sys.excepthook.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.getLogger("").critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_uncaught_exception

    uvicorn.run(
        "core.api:app",
        host=HOST,
        port=PORT,
        log_config=LOG_CONFIG
    )