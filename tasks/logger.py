import logging
import os
import sys
import urllib.parse
import urllib.request
import warnings
from typing import Optional, TextIO

import tqdm


class TQDMHandler(logging.Handler):
    """tqdm 사용시 ``logging`` 모듈과 같이 사용가능한 handler입니다.
    .. code-block:: python
        import time
        import logging
        import sys
        import tqdm
        from pnlp.utils import TQDMHandler
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        handler = TQDMHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(handler)
        for i in tqdm.tqdm(range(1000)):
            logger.info("안녕")
            time.sleep(0.1)
    """

    def __init__(self, stream: Optional[TextIO] = None):
        super().__init__()
        if stream is None:
            stream = sys.stdout

        self.stream = stream

    def flush(self):
        self.acquire()
        try:
            if self.stream and hasattr(self.stream, "flush"):
                self.stream.flush()
        finally:
            self.release()

    def emit(self, record: logging.LogRecord):
        try:
            message = self.format(record)
            tqdm.tqdm.write(message, self.stream)
            self.flush()
        except RecursionError:
            raise
        except Exception:
            self.handleError(record)


def get_logger(log_path: str = None):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    tqdm_handler = TQDMHandler(sys.stdout)
    tqdm_handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(tqdm_handler)

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        logger.addHandler(file_handler)

    return logger
