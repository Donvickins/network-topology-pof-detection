import uvicorn
import os
import sys
import logging
from logging.config import dictConfig
from colorama import just_fix_windows_console

# Suppress Ultralytics' initial setup logs
os.environ['ULTRALYTICS_LOGGING'] = 'CRITICAL'

from core.utils.logger_config import LOG_CONFIG

if __name__ == '__main__':
    if sys.platform == 'win32':
        just_fix_windows_console()
        
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

    # Assign the custom handler to the system's exception hook
    sys.excepthook = handle_uncaught_exception

    uvicorn.run(
        "core.api:app",
        host="0.0.0.0",
        port=5500,
        log_config=LOG_CONFIG
    )